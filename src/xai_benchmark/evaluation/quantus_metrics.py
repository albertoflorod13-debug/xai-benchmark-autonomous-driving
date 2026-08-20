"""Thin wrappers around Quantus 0.6.0 metric classes for the localisation/complexity
metrics that stayed on Quantus's own implementation -- one function per metric,
translating our (heatmap, box) format into the (model, x_batch, y_batch, a_batch,
s_batch) signature every quantus.Metric subclass expects, with model=None since
a_batch/s_batch are always precomputed here.

Method-agnostic: these functions only need an already-reconstructed heatmap and, 
where relevant, a boolean box mask. 
No knowledge of SSGrad-CAM++, D-CRISP, or any other XAI method's internals.
"""

import warnings
import numpy as np
import quantus


def _to_batch(array_2d: np.ndarray) -> np.ndarray:
    """(H, W) -> (1, 1, H, W), the shape every quantus.Metric expects for a_batch/s_batch."""
    return array_2d[None, None, ...]

def _non_degenerate(heatmap: np.ndarray) -> np.ndarray:
    """Quantus's own consistency check (`assert_attributions`) rejects a constant attribution
    map (all-zero, all-one, or a single repeated value) since ordering-based metrics are
    undefined for it. D-CRISP's `explain_image` can legitimately return an all-zero heatmap
    when an object never appears under any mask. Replace
    with random noise so the metric stays well-defined instead of crashing.
    """
    if np.all(heatmap == heatmap.flat[0]):
        return np.random.default_rng().random(heatmap.shape).astype(np.float32)
    return heatmap


def pointing_game(heatmap: np.ndarray, box_mask: np.ndarray) -> float:
    """Does the single max-attribution pixel fall inside `box_mask`?"""
    heatmap = _non_degenerate(heatmap)
    metric = quantus.PointingGame(normalise=False, abs=False,
                                   disable_warnings=True, display_progressbar=False)
    a_batch, s_batch = _to_batch(heatmap.astype(np.float32)), _to_batch(box_mask.astype(bool))
    x_batch = np.zeros_like(a_batch)
    scores = metric(model=None, x_batch=x_batch, y_batch=np.array([0]),
                     a_batch=a_batch, s_batch=s_batch, channel_first=True)
    return float(scores[0])


def energy_based_pointing_game(heatmap: np.ndarray, box_mask: np.ndarray) -> float:
    """EBPG: fraction of the heatmap's total attribution energy that falls inside the
    target's bounding box.
     """
    heatmap = _non_degenerate(heatmap)
    metric = quantus.AttributionLocalisation(weighted=False, positive_attributions=False,
                                              abs=True, normalise=False,
                                              disable_warnings=True, display_progressbar=False)
    a_batch, s_batch = _to_batch(heatmap.astype(np.float32)), _to_batch(box_mask.astype(bool))
    x_batch = np.zeros_like(a_batch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        scores = metric(model=None, x_batch=x_batch, y_batch=np.array([0]),
                         a_batch=a_batch, s_batch=s_batch, channel_first=True)
    return float(scores[0])


def relevance_rank_accuracy(heatmap: np.ndarray, box_mask: np.ndarray) -> float:
    """Fraction of the top-k highest-ranked attribution pixels that fall inside `box_mask`,
    with k self-adapting to each object's own box size (k = |box_mask|), unlike
    quantus.TopKIntersection's fixed global k. Ranking-based: only pixel ORDER matters, not
    magnitude, so unlike EBPG it is not diluted by the scattered low-intensity background
    energy that random-mask methods (D-CRISP) leave outside the target box.
    """
    heatmap = _non_degenerate(heatmap)
    metric = quantus.RelevanceRankAccuracy(abs=False, normalise=False,
                                            disable_warnings=True, display_progressbar=False)
    a_batch, s_batch = _to_batch(heatmap.astype(np.float32)), _to_batch(box_mask.astype(bool))
    x_batch = np.zeros_like(a_batch)
    scores = metric(model=None, x_batch=x_batch, y_batch=np.array([0]),
                     a_batch=a_batch, s_batch=s_batch, channel_first=True)
    return float(scores[0])


def sparseness(heatmap: np.ndarray) -> float:
    """Gini index of the flattened, sorted heatmap: 0 if attribution is spread evenly
    across every pixel, 1 if it is concentrated on a single pixel. 
    Needs only the heatmap. A complexity-axis metric."""
    heatmap = _non_degenerate(heatmap)
    metric = quantus.Sparseness(normalise=False, disable_warnings=True, display_progressbar=False)
    a_batch = _to_batch(heatmap.astype(np.float32))
    x_batch = np.zeros_like(a_batch)
    scores = metric(model=None, x_batch=x_batch, y_batch=np.array([0]),
                     a_batch=a_batch, channel_first=True)
    return float(scores[0])



