"""One-to-many (dense, pre-NMS) head wrapper for YOLO26.

Why this exists: Ultralytics' normal inference path collapses every detection down to a
single (class, probability) pair. That's fine for drawing boxes, but not enough for what
TTA/UQ and the XAI methods need -- the full per-class probability vector for every
surviving detection (e.g. to measure prediction entropy, or to explain a specific class).
This module exposes that richer, pre-collapse output instead.

How: YOLO26's head (`Detect`) internally computes two versions of its predictions -- a
dense "one2many" version (every anchor gets a prediction, several anchors can claim the
same object) and a sparse "one2one" version (matched one-to-one at train time, which is
what makes YOLO26 "NMS-free" at inference). Ultralytics only exposes the one2one version
by default. Setting `model.model.end2end = False` before calling the model switches it to
return the raw one2many tensor instead -- the exact mechanism `model.predict(...,
end2end=False)` uses internally, not a hack. Because the one2many output was never trained
for one-to-one matching, it's redundant (several boxes per real object) and needs NMS --
here, a class-agnostic NMS that keeps the full per-class score vector of each surviving box,
instead of Ultralytics' own NMS, which keeps only the winning class.

Precondition -- do not call this on a model that has been `.fuse()`-d: fusing (which
`model.predict()` and `model.val()` both do automatically) deletes the very conv layers
(`cv2`/`cv3`) this function reads from. After fusing, this function doesn't crash -- it
silently returns zero detections, which looks like "the model found nothing" instead of an
error. If you need both normal inference and this dense output, load two separate `YOLO(...)`
instances: one you never call `.predict()`/`.val()` on, one you use only for those.
"""

from dataclasses import dataclass

import torch
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.ops import scale_boxes, xywh2xyxy


@dataclass
class DenseDetections:
    """Per-image detections retaining the full per-class probability vector.

    boxes: (N, 4) xyxy coordinates in the ORIGINAL image's pixel space (already rescaled
        back from the preprocessed/letterboxed input), N = survivors after NMS.
    class_probs: (N, nc) sigmoid score per class, for each surviving box (not argmax-only).
    max_conf: (N,) class_probs.amax(-1), used for ranking/thresholding, kept for convenience.
    """

    boxes: torch.Tensor
    class_probs: torch.Tensor
    max_conf: torch.Tensor


@torch.no_grad()
def get_one2many_predictions(
    model,
    imgs: torch.Tensor,
    orig_shapes: list[tuple[int, int]],
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
) -> list[DenseDetections]:
    """Dense (one2many) inference with a class-preserving NMS, in original-image coordinates.

    Args:
        model: loaded `ultralytics.YOLO` instance (any checkpoint, not yet `.fuse()`-d).
        imgs: preprocessed batch (B, 3, H, W) -- letterboxed/normalized as Ultralytics
            expects (e.g. via `model.predictor.preprocess(...)`, not reimplemented here).
        orig_shapes: list of (height, width) of each original image, same order as `imgs`
            was built from -- needed to rescale boxes back out of the letterboxed frame.
        conf_thres: max-class-confidence filter applied before NMS (cheap; drops the bulk
            of near-zero-confidence anchors up front).
        iou_thres: IoU threshold for the class-agnostic suppression.

    Returns:
        One `DenseDetections` per image in the batch, boxes in original-image pixel space.
    """
    detection_model = model.model  
    detection_model.eval()
    detection_model.to(imgs.device)  
    num_classes = detection_model.model[-1].nc  

    prev_end2end = detection_model.end2end
    detection_model.end2end = False  # force the dense one2many path
    try:
        raw, _ = detection_model(imgs)  
    finally:
        detection_model.end2end = prev_end2end  # leave the model exactly as we found it

    raw = raw.permute(0, 2, 1)  
    img1_shape = imgs.shape[2:]  # (H, W) of the letterboxed input, shared by the whole batch

    detections = []
    for pred, orig_shape in zip(raw, orig_shapes):  
        boxes_xywh, class_probs = pred.split([4, num_classes], dim=-1)
        boxes_xyxy = xywh2xyxy(boxes_xywh)  # decode_bboxes gives xywh when end2end=False 

        max_conf = class_probs.amax(dim=-1)
        keep = max_conf > conf_thres
        boxes_xyxy, class_probs, max_conf = boxes_xyxy[keep], class_probs[keep], max_conf[keep]

        keep_idx = TorchNMS.nms(boxes_xyxy, max_conf, iou_thres)

        kept_boxes = scale_boxes(img1_shape, boxes_xyxy[keep_idx].clone(), orig_shape)  # back to original-image pixels
        # (same call Ultralytics itself makes in models/yolo/detect/predict.py right after inference)

        detections.append(DenseDetections(
            boxes=kept_boxes,
            class_probs=class_probs[keep_idx],
            max_conf=max_conf[keep_idx],
        ))
    return detections