"""
conflict_checker.py
====================
Detects whether a sampled warehouse image triggers a conflict: an object close
enough (relative box area) and centered enough in the frame to require the
robot to stop and replan. Detection reuses the dense (one2many) YOLO26 head
used throughout this project, so the triggering detection's box and per-class
probabilities can be handed directly to the explanation methods afterwards,
without re-running inference on the same image.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from xai_benchmark.detection.yolo_head import get_one2many_predictions

# 99th percentile of relative box area (box_area / image_area) over the target dataset's
# annotated instances -- a detection this size or larger counts as "close". Raised from the
# 97th percentile: at 0.056 roughly 1 in 13 sampled images already triggered a conflict,
# still too frequent for a demo bounded by a handful of replans.
NEAR_RELATIVE_AREA_THRESHOLD = 0.1028
# A detection counts as "centered" if its box center falls within the middle 30% of the
# frame on both axes. Narrowed from 50% for the same reason.
CENTER_FRACTION = 0.3

CLASS_NAMES = {
    0: "small load carrier", 1: "forklift", 2: "pallet", 3: "stillage", 4: "pallet truck",
}

@dataclass
class ConflictResult:
    """target_box/target_class/target_class_probs are already in the shape the
    explanation methods expect, ready to pass through unchanged."""
    triggered: bool
    target_box: Optional[Tuple[float, float, float, float]] = None
    target_class: Optional[int] = None
    target_class_probs: Optional[torch.Tensor] = None
    confidence: Optional[float] = None
    relative_area: Optional[float] = None


def load_conflict_model(checkpoint_path: str, device: str = "cuda"):
    """Loads the two YOLO instances check_conflict needs from a single checkpoint:
    one for dense one2many inference, one reserved for preprocessing only.

    Returns (model_dense, model_prep, resolved_device) -- resolved_device must be
    passed to every check_conflict() call made with these two models, since a
    missing GPU is resolved here and not knowable from the models themselves.
    """
    from ultralytics import YOLO
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model_dense = YOLO(str(checkpoint_path))
    model_dense.model.to(device)
    model_dense.model.eval()

    model_prep = YOLO(str(checkpoint_path))
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model_prep.predict([dummy], verbose=False, device=0 if device == "cuda" else "cpu")

    return model_dense, model_prep, device


def check_conflict(model_dense, model_prep, img_bgr: np.ndarray, device: str = "cuda",
                    conf_thres: float = 0.25, iou_thres_nms: float = 0.5,
                    near_threshold: float = NEAR_RELATIVE_AREA_THRESHOLD,
                    center_fraction: float = CENTER_FRACTION) -> ConflictResult:
    """Runs detection on img_bgr and checks whether any surviving box is both close
    (relative area >= near_threshold) and centered (box center within the middle
    center_fraction of the frame). Among qualifying boxes, the largest (closest) wins.
    """
    orig_shape = img_bgr.shape[:2]
    imgs_tensor = model_prep.predictor.preprocess([img_bgr]).to(device)

    dets = get_one2many_predictions(
        model_dense, imgs_tensor, [orig_shape], conf_thres=conf_thres, iou_thres=iou_thres_nms,
    )[0]

    if dets.boxes.shape[0] == 0:
        return ConflictResult(triggered=False)

    boxes = dets.boxes.cpu().numpy()
    h, w = orig_shape
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    rel_areas = areas / (w * h)
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2

    margin = (1 - center_fraction) / 2
    is_centered = (
        (centers_x >= w * margin) & (centers_x <= w * (1 - margin)) &
        (centers_y >= h * margin) & (centers_y <= h * (1 - margin))
    )
    candidates = np.where((rel_areas >= near_threshold) & is_centered)[0]
    if candidates.size == 0:
        return ConflictResult(triggered=False)

    trigger_idx = int(candidates[np.argmax(rel_areas[candidates])])
    class_probs = dets.class_probs[trigger_idx].cpu()

    return ConflictResult(
        triggered=True,
        target_box=tuple(boxes[trigger_idx].tolist()),
        target_class=int(class_probs.argmax().item()),
        target_class_probs=class_probs,
        confidence=float(dets.max_conf[trigger_idx]),
        relative_area=float(rel_areas[trigger_idx]),
    )