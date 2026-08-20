"""Shared geometry helpers for per-instance XAI heatmaps.

XAI method that explains one detection at a time and computes its heatmap in the
model's preprocessed (letterboxed) coordinate frame needs to map it back to the original
image before it can be compared against ground-truth/bounding-box annotations.
"""

import cv2
import numpy as np


def unletterbox_map(map_letterboxed: np.ndarray, img1_shape: tuple, img0_shape: tuple) -> np.ndarray:
    """Undo Ultralytics' letterbox padding on a 2D map (heatmap, mask, ...).
    Args:
        map_letterboxed: (h1, w1) array in the model's preprocessed (letterboxed) frame.
        img1_shape: (h1, w1) of the letterboxed frame.
        img0_shape: (h0, w0) of the original image.
    Returns:
        (h0, w0) array, cropped and resized back to the original image's resolution.
    """
    h1, w1 = img1_shape
    h0, w0 = img0_shape
    gain = min(h1 / h0, w1 / w0)
    pad_x = round((w1 - round(w0 * gain)) / 2 - 0.1)
    pad_y = round((h1 - round(h0 * gain)) / 2 - 0.1)
    cropped = map_letterboxed[pad_y:h1 - pad_y, pad_x:w1 - pad_x]
    return cv2.resize(cropped, (w0, h0))


def letterbox_map_box(box_xyxy: list, img0_shape: tuple, img1_shape: tuple) -> list:
    """Original-image box -> letterboxed (model input) box.

    Forward counterpart of unletterbox_map (above): reuses the exact same gain/pad
    convention (own code, this project, mirrored from unletterbox_map's own derivation
    of Ultralytics' letterbox transform), applied to a box instead of a 2D map, and
    without the crop-back step since a box has no interior pixels to resample.
    Needed because build_box_mask (detection_metrics.py) was, until now, only ever
    called in original-image space; the extended faithfulness/localisation metrics that
    operate directly on the letterboxed tensor (detection_metrics.py::rank_accuracy_at_k,
    outside_box_deletion) need a box mask in that same letterboxed frame.

    Args:
        box_xyxy: [x1, y1, x2, y2] in the original image's coordinate frame.
        img0_shape: (h0, w0) of the original image.
        img1_shape: (h1, w1) of the letterboxed frame.

    Returns:
        [x1, y1, x2, y2] in the letterboxed frame.
    """
    h0, w0 = img0_shape
    h1, w1 = img1_shape
    gain = min(h1 / h0, w1 / w0)
    pad_x = round((w1 - round(w0 * gain)) / 2 - 0.1)
    pad_y = round((h1 - round(h0 * gain)) / 2 - 0.1)
    x1, y1, x2, y2 = box_xyxy
    return [x1 * gain + pad_x, y1 * gain + pad_y, x2 * gain + pad_x, y2 * gain + pad_y]