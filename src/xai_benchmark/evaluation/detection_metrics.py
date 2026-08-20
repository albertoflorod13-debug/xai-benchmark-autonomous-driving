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

def minimal_subset(scores: list, k_threshold: float) -> float:
    """Fraction (in [0, 1]) of the Deletion curve's steps needed before `scores` first drops
    below `k_threshold`. 0 if it starts already below; 1 if it never does (worst case: the
    object survives full occlusion). Ported from the reference D-CRISP repo's own
    metrics/utils.py::minimal_subset. Use k_threshold=conf_thres here: DetectorAsClassifier
    already hard-floors `scores` to exactly 0.0 the moment the real confidence drops below
    conf_thres (the box gets filtered out before it ever reaches best_matching_confidence), so
    this is not an arbitrary substitute for the reference's own conf_thre-tied 0.7 -- it's the
    same self-consistent choice adapted to our conf_thres=0.25.
    """
    arr = np.asarray(scores)
    below = np.nonzero(arr < k_threshold)[0]
    if len(below) == 0:
        return 1.0
    return float(below[0] / (len(arr) - 1))


def _predict_at_k(x_batch: np.ndarray, order: np.ndarray, k: int, baseline: np.ndarray,
                   classifier: DetectorAsClassifier, target_class_idx: int,
                   channels: int, height: int, width: int, device: str) -> float:
    """Confidence after deleting exactly the top-k pixels of `order`, evaluated at an
    arbitrary k -- not tied to deletion_insertion_auc's step-multiple checkpoints.
    Stateless: reconstructs the perturbed tensor from scratch each call, independent of
    any running accumulator. Own code, this project; shared by refine_minimal_subset_k
    and outside_box_deletion below.
    """
    x_flat = x_batch.reshape(channels, height * width).copy()
    baseline_flat = baseline.reshape(channels, height * width)
    idx = order[:k]
    x_flat[:, idx] = baseline_flat[:, idx]
    x_input = torch.as_tensor(x_flat.reshape(1, channels, height, width),
                               dtype=torch.float32, device=device)
    return classifier(x_input)[0, target_class_idx].item()


def refine_minimal_subset_k(x_batch: np.ndarray, heatmap: np.ndarray,
                             classifier: DetectorAsClassifier, target_class_idx: int,
                             k_lo: int, k_hi: int, conf_thres: float,
                             del_baseline: str = "black", device: str = "cpu",
                             tol: int = 1) -> int:
    """Bisection refinement of the coarse (step-quantised) Min-Subset crossing point
    down to pixel precision.

    Own numerical method, this project: root-finding by bisection on
    g(k) = confidence(k) - conf_thres, NOT on confidence(k) directly -- the sign change
    required by bisection is guaranteed at the endpoints by construction of the coarse
    bracket (k_lo: last coarse checkpoint with confidence >= conf_thres, so g(k_lo) >= 0;
    k_hi: first coarse checkpoint with confidence < conf_thres, so g(k_hi) < 0), since
    k_lo/k_hi come directly from minimal_subset's own crossing index on the SAME
    deletion curve already computed by deletion_insertion_auc. This is the discrete
    analogue of bisection: a binary search over the monotone predicate
    P(k) = "confidence(k) < conf_thres", assumed locally monotone within [k_lo, k_hi]
    (a window of at most `step` pixels from the coarse pass -- a much safer assumption
    than global monotonicity of the whole curve). Guaranteed to converge in
    ceil(log2(k_hi - k_lo)) evaluations, unlike Newton-Raphson (needs an analytic
    derivative, which a black-box detector forward pass does not have, and has no
    global convergence guarantee).

    Args:
        k_lo, k_hi: coarse bracket in pixel counts (NOT step indices), e.g.
            k_lo = (i-1)*step, k_hi = i*step where i = minimal_subset's crossing index.
        tol: stop once the bracket width is <= tol pixels (1 = pixel-exact).

    Returns:
        int, the refined k: the smallest pixel count in [k_lo, k_hi] for which
        confidence(k) < conf_thres.
    """
    _, channels, height, width = x_batch.shape
    order = np.argsort(-heatmap.reshape(-1))
    baseline = _resolve_baseline(x_batch, del_baseline)
    while k_hi - k_lo > tol:
        k_mid = (k_lo + k_hi) // 2
        score = _predict_at_k(x_batch, order, k_mid, baseline, classifier,
                               target_class_idx, channels, height, width, device)
        if score < conf_thres:
            k_hi = k_mid
        else:
            k_lo = k_mid
    return k_hi


def rank_accuracy_at_k(heatmap: np.ndarray, box_mask: np.ndarray, k: int) -> dict:
    """Relevance Rank Accuracy with k decoupled from
    |box_mask|, reported both as box coverage and as selection precision.

    Adapted from quantus.RelevanceRankAccuracy.evaluate_batch (quantus/metrics/
    localisation/relevance_rank_accuracy.py): that implementation
    hard-ties k = s_batch.sum() with no exposed k parameter. 
    Own extension, this project: k is instead the exact causal
    minimal-subset size from refine_minimal_subset_k, so the metric measures "of the
    pixels causally necessary to sustain the detection, what fraction lies inside the
    box" instead of "of the box-sized top-k pixels, what fraction lies inside the box".

    Returns two variants, not one:
      - "rank_accuracy_box" = |top-k ∩ box_mask| / |box_mask|. The original RRA formula,
        denominator kept as |box_mask| (not k) -- a value of 1.0 means the whole box was
        covered by the causally-necessary subset, a value < 1.0 with a generous k is
        evidence of genuine spillover, not a denominator artifact.
      - "rank_accuracy_precision" = |top-k ∩ box_mask| / k. Own extension, this project,
        added after cross-method validation showed k varies by an order of magnitude across XAI
        methods -- with "rank_accuracy_box" alone, a method with a much larger
        k covers a small box almost by construction regardless of ranking quality, which
        makes cross-method comparisons at very different k magnitudes favour the
        larger-k method. "rank_accuracy_precision" is comparable regardless of k's
        absolute size; "rank_accuracy_box" remains the right choice for the original,
        single-method question ("what fraction of the box does the causal subset
        cover?"). Report both, do not average them.

    Args:
        heatmap: (H, W) saliency map, same resolution/coordinate frame as box_mask
            (letterboxed, since k comes from the letterboxed deletion curve).
        box_mask: (H, W) boolean mask, same frame as heatmap.
        k: number of top-ranked pixels to select, independent of |box_mask|.

    Returns:
        {"rank_accuracy_box": float, "rank_accuracy_precision": float}, both in [0, 1],
        or both nan if the box is empty or k <= 0.
    """
    box_size = int(box_mask.sum())
    if box_size == 0 or k <= 0:
        return {"rank_accuracy_box": float("nan"), "rank_accuracy_precision": float("nan")}
    ranked = np.argsort(-heatmap.reshape(-1))
    hits = int(box_mask.reshape(-1)[ranked[:k]].sum())
    return {"rank_accuracy_box": float(hits / box_size), "rank_accuracy_precision": float(hits / k)}


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
    {"deletion_auc": float, "insertion_auc": float, "deletion_scores": list[float]} -- AUC of
    the target's confidence-score curve (`_normalised_auc`: divided by the number of intervals
    so the result doesn't depend on step count. NOT divided by the object's original
    confidence), both curves evaluated over the SAME step count. `deletion_scores` is the raw
    per-step curve (step 0 = original image), exposed for `minimal_subset`.
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
            "insertion_auc": _normalised_auc(insertion_scores),
            "deletion_scores": deletion_scores}


def outside_box_deletion(model_as_classifier: DetectorAsClassifier, x_batch: np.ndarray,
                          heatmap: np.ndarray, box_mask: np.ndarray,
                          target_class_idx: int, step: int = 1000,
                          del_baseline: str = "black", device: str = "cpu") -> dict:
    """Restricted variant of D-Deletion (see deletion_insertion_auc above): only pixels
    OUTSIDE box_mask are ever perturbed; pixels inside the box are never touched.

    Tests whether the detector's confidence causally depends on the context pixels the
    heatmap ranks as important -- i.e. whether a heatmap's spillover beyond the target
    box reflects genuine model use of context or is an artifact of the explainability method.

    Own code, this project: reuses the exact perturbation/AUC machinery of
    deletion_insertion_auc above, restricted to a
    fixed pixel subset. Motivated by the documented finding that object detectors rely
    on context beyond the target box.

    CAUTION: a confidence drop here is consistent with genuine context use, but cannot
    by itself rule out an out-of-distribution masking artifact -- deleting pixels
    (inside or outside the box) creates inputs the detector never saw in training. Report as
    suggestive causal evidence, not proof.

    Args:
        x_batch: (1, C, H, W) letterboxed input tensor.
        heatmap: (H, W) saliency map, same letterboxed resolution as x_batch.
        box_mask: (H, W) boolean mask, True inside the target box, in the same
            letterboxed frame as heatmap/x_batch -- build with
            common.letterbox_map_box + build_box_mask, not the original-image-space
            mask used elsewhere for RRA/EBPG.

    Returns:
        {"outside_box_deletion_auc": float, "outside_box_confidence_drop": float}
        (drop = score at step 0 minus score after the last outside-box pixel removed).
    """
    _, channels, height, width = x_batch.shape
    outside_pixels = np.flatnonzero(~box_mask.reshape(-1))
    ranked_outside = outside_pixels[np.argsort(-heatmap.reshape(-1)[outside_pixels])]
    n_steps = math.ceil(len(ranked_outside) / step)

    def predict(flat_array: np.ndarray) -> float:
        x_input = torch.as_tensor(flat_array.reshape(1, channels, height, width),
                                   dtype=torch.float32, device=device)
        return model_as_classifier(x_input)[0, target_class_idx].item()

    x_flat = x_batch.reshape(channels, height * width)
    running = x_flat.copy()
    baseline = _resolve_baseline(x_batch, del_baseline).reshape(channels, height * width)

    scores = [predict(running)]  # step 0: original image, nothing removed
    for i in range(n_steps):
        idx = ranked_outside[i * step: (i + 1) * step]
        running[:, idx] = baseline[:, idx]
        scores.append(predict(running))

    return {"outside_box_deletion_auc": _normalised_auc(scores),
            "outside_box_confidence_drop": scores[0] - scores[-1]}