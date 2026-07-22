"""Detection-specific bridges for per-instance XAI heatmap evaluation.

Method-agnostic: everything here depends only on "one object = one heatmap (at whatever
native resolution the method produced) + one target box + one target class", not on how
that heatmap was computed..

Two coordinate frames are used, depending on the metric family (see run_evaluation.py):
  - Original image space: feeds quantus_metrics.py's Pointing Game/EBPG/Sparseness, where
    the heatmap is compared directly against the box coordinates stored in object_level.csv.
  - Letterboxed (model input) space: for Deletion/Insertion, which perturb pixels on the 
    actual tensor fed to the detector.
"""

import math
import cv2
import numpy as np
import torch

from xai_benchmark.detection.yolo_head import get_one2many_predictions
from xai_benchmark.uncertainty.tta import box_iou
from xai_benchmark.xai.common import unletterbox_map


def build_box_mask(box_xyxy, height: int, width: int) -> np.ndarray:
    """Rectangular boolean mask, True inside `box_xyxy`, shape (height, width) -- the
    `s_batch` Quantus's localisation metrics expect."""
    x1, y1, x2, y2 = box_xyxy
    mask = np.zeros((height, width), dtype=bool)
    r0, r1 = max(0, int(np.floor(y1))), min(height, int(np.ceil(y2)))
    c0, c1 = max(0, int(np.floor(x1))), min(width, int(np.ceil(x2)))
    mask[r0:r1, c0:c1] = True
    return mask


def reconstruct_heatmap_original(heatmap_raw: np.ndarray, letterbox_shape: tuple,
                                  orig_shape: tuple) -> np.ndarray:
    """Native-resolution heatmap -> full original-image resolution, normalised to [0, 1].
    Used by localisation/complexity metrics."""
    h1, w1 = letterbox_shape
    heatmap_letterboxed = cv2.resize(heatmap_raw, (w1, h1))
    heatmap_orig = unletterbox_map(heatmap_letterboxed, letterbox_shape, orig_shape)
    return heatmap_orig / (heatmap_orig.max() + 1e-12)


def reconstruct_heatmap_letterboxed(heatmap_raw: np.ndarray, letterbox_shape: tuple) -> np.ndarray:
    """Native-resolution heatmap -> letterboxed (model-input) resolution, normalised to
    [0, 1]. Used by faithfulness metrics (Deletion/Insertion), which perturb the tensor
    actually fed to the detector, not the original image."""
    h1, w1 = letterbox_shape
    heatmap_letterboxed = cv2.resize(heatmap_raw, (w1, h1))
    return heatmap_letterboxed / (heatmap_letterboxed.max() + 1e-12)


def best_matching_confidence(target_box, target_class_idx: int, boxes: torch.Tensor,
                              class_probs: torch.Tensor, max_conf: torch.Tensor) -> float:
    """Confidence of the best-matched candidate."""
    if len(boxes) == 0:
        return 0.0
    same_class = class_probs.argmax(dim=-1) == target_class_idx
    if not same_class.any():
        return 0.0
    target_box_t = torch.as_tensor(target_box, dtype=boxes.dtype, device=boxes.device).unsqueeze(0)
    ious = box_iou(target_box_t, boxes[same_class]).squeeze(0)
    overlapping = ious > 0
    if not overlapping.any():
        return 0.0
    return float(max_conf[same_class][overlapping].max())


class DetectorAsClassifier(torch.nn.Module):
    """Bridges a YOLO26 detector to the (model, x_batch) -> (batch, n_classes) classifier
    interface `deletion_insertion_auc` (below) needs for both Deletion and Insertion."""

    def __init__(self, model_dense, num_classes: int, orig_shape: tuple,
                 target_box, target_class_idx: int, conf_thres: float = 0.25, 
                 iou_thres_nms: float = 0.5):
        super().__init__()
        object.__setattr__(self, "model_dense", model_dense)
        self.num_classes = num_classes
        self.orig_shape = orig_shape
        self.target_box = target_box
        self.target_class_idx = target_class_idx
        self.conf_thres = conf_thres
        self.iou_thres_nms = iou_thres_nms
        self.eval()  

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device

        dets = get_one2many_predictions(
            self.model_dense, x, [self.orig_shape] * batch_size,
            conf_thres=self.conf_thres, iou_thres=self.iou_thres_nms,
        )

        out = torch.zeros(batch_size, self.num_classes, device=device)
        for i, det in enumerate(dets):
            conf = best_matching_confidence(
                self.target_box, self.target_class_idx,
                det.boxes.to(device), det.class_probs.to(device), det.max_conf.to(device),
            )
            out[i, self.target_class_idx] = conf
        return out


def _resolve_baseline(x: np.ndarray, perturb_baseline: str) -> np.ndarray:
    """Baseline resolution for deletion_insertion_auc. 
    "black": zeros. 
    "mean": per-channel mean. 
    "blur": Gaussian-blurred version of `x` (following the original RISE paper)."""
    if perturb_baseline == "black":
        return np.zeros_like(x)
    if perturb_baseline == "mean":
        channel_mean = x.mean(axis=(0, 2, 3), keepdims=True)
        return np.broadcast_to(channel_mean, x.shape).copy()
    if perturb_baseline == "blur":
        blurred = np.stack([cv2.GaussianBlur(x[0, c], ksize=(11, 11), sigmaX=5)
                             for c in range(x.shape[1])])
        return blurred[None, ...]
    raise ValueError(f"Unsupported perturb_baseline: {perturb_baseline!r}")


def _normalised_auc(scores: list) -> float:
    """Trapezoidal rule divided by the number of intervals, so the result 
    doesn't scale with however many steps were used."""
    arr = np.asarray(scores)
    return float((arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1))


def deletion_insertion_auc(model_as_classifier: DetectorAsClassifier, x_batch: np.ndarray,
                            heatmap: np.ndarray, target_class_idx: int, step: int = 1000,
                            del_baseline: str = "black", ins_baseline: str = "blur",
                            device: str = "cpu") -> dict:
    """D-Deletion and D-Insertion in one pass, following the design of D-CRISP's own
    reference implementation (itself adapted from the original RISE evaluation code).

    Args:
        x_batch: (1, C, H, W) letterboxed input tensor, as fed to the detector.
        heatmap: (H, W) saliency map, already resized to x_batch's H, W and normalised.
        step: pixels perturbed per iteration.

    Returns:
    {"deletion_auc": float, "insertion_auc": float} -- AUC of the target's confidence-score
    curve (`_normalised_auc`: divided by the number of intervals so the result doesn't depend on step count. 
    NOT divided by the object's original confidence), both curves evaluated over the SAME step count.
    """
    assert x_batch.shape[0] == 1, "one object (one image) at a time."
    _, channels, height, width = x_batch.shape
    assert heatmap.shape == (height, width), (
        f"heatmap must match x_batch's spatial size, got {heatmap.shape} vs ({height}, {width})"
    )
    n_pixels = height * width
    n_steps = math.ceil(n_pixels / step)
    order = np.argsort(-heatmap.reshape(-1))  

    def predict(flat_array: np.ndarray) -> float:
        x_input = torch.as_tensor(flat_array.reshape(1, channels, height, width),
                                   dtype=torch.float32, device=device)
        return model_as_classifier(x_input)[0, target_class_idx].item()

    x_flat = x_batch.reshape(channels, n_pixels)
    del_running = x_flat.copy()
    del_target = _resolve_baseline(x_batch, del_baseline).reshape(channels, n_pixels)
    ins_running = _resolve_baseline(x_batch, ins_baseline).reshape(channels, n_pixels)
    ins_target = x_flat

    deletion_scores = [predict(del_running)]   # step 0: original image, nothing removed
    insertion_scores = [predict(ins_running)]  # step 0: fully occluded/blurred canvas

    for i in range(n_steps):
        idx = order[i * step: (i + 1) * step]
        del_running[:, idx] = del_target[:, idx]
        deletion_scores.append(predict(del_running))
        ins_running[:, idx] = ins_target[:, idx]
        insertion_scores.append(predict(ins_running))

    return {"deletion_auc": _normalised_auc(deletion_scores),
            "insertion_auc": _normalised_auc(insertion_scores)}