"""Random-noise "explainer": the chance-level baseline for this benchmark.

Generates i.i.d. Uniform(0, 1) heatmaps that carry no information about the image or the
target object, the explicit point is to establish what every metric in this project's
evaluation pipeline scores for an explanation that explains nothing. Protocol follows
Skliarov, El Shawi, Dhaoui & Ahmed, "A comparative evaluation of explainability techniques
for image data", Scientific Reports 15:41898 (2025): "a random saliency map generator that
produces noise-based attribution maps normalized to [0, 1]... generated with the same
spatial resolution and normalization settings as the evaluated methods." 

Every downstream metric function is already method-agnostic and only assumes a 2D array of
the right shape. This module therefore does not replicate DCRISP's or SSGradCAMPP's
constructor, `.explain()` signature, or any of their image/target-dependent machinery: it
only produces the one thing every reconstruction step consumes -- a (H, W) array at the
model's own letterboxed resolution, already in [0, 1].
"""

import numpy as np


class RandomExplainer:
    """Chance-level explainer: `heatmap(letterbox_shape)` ignores everything about the
    image and the target object by design, returning fresh i.i.d. Uniform(0, 1) noise on
    every call.

    Unlike DCRISP/SSGradCAMPP, this class has no dependency on the detector, the image
    content, or the target box/class, constructing it only needs a seed, and calling
    `heatmap` only needs the spatial shape to match.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def heatmap(self, letterbox_shape: tuple[int, int]) -> np.ndarray:
        """(H, W) float32 array, i.i.d. Uniform(0, 1), already in the [0, 1] range every
        downstream metric expects, no separate max-normalisation
        needed.

        A fresh call always returns an independent draw: this is what makes it a valid
        chance-level reference for the perturbation-based robustness metrics too
        (Max-Sensitivity, RIS, ROS, RRS), not only for the fidelity/localisation family --
        every repetition inside stability.evaluate_perturbation_stability's loop must call
        this again, never cache and reuse the first draw.
        """
        return self.rng.random(letterbox_shape, dtype=np.float32)
