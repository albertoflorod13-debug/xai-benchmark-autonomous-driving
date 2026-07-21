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