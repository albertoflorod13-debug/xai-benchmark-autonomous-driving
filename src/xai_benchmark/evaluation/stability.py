"""Perturbation-based robustness/stability framework for D-CRISP and SSGrad-CAM++.

Adapts four classification-only metrics to object detection, following the same class+IoU
matching principle already used in this project to adapt Deletion/Insertion/Min-Subset to
D-Deletion/D-Insertion/D-Min-Subset ("On the black-box explainability
of object detection models for safe and trustworthy industrial applications"; see detection_metrics.py::best_matching_confidence /
DetectorAsClassifier, which this module reuses the same principle from -- generalised here with
a configurable IoU threshold, see `_matched_confidence`).

Metrics implemented:
  - Max-Sensitivity: Yeh, Hsieh, Suggala, Inouye, Ravikumar, "On the (In)fidelity and
    Sensitivity for Explanations", NeurIPS 2019, arXiv:1901.09392.
    Implemented literally (max Frobenius-norm difference, NOT divided by ||e_x||) -- this is a
    deliberate deviation from quantus.metrics.robustness.max_sensitivity.MaxSensitivity, which
    divides by norm_denominator(a_batch). We skip that extra normalisation because our heatmaps
    are already uniformly max-normalised to [0, 1] by explain_image, 
    so an extra relative-scale correction isn't needed here the way it is for Quantus's
    general-purpose classifiers with wildly different explanation magnitudes.
  - Relative Input Stability (RIS) and Relative Output Stability (ROS): Agarwal, Johnson,
    Pawelczyk, Krishna, Saxena, Zitnik, Lakkaraju, "Rethinking Stability for Attribution-based
    Explanations", arXiv:2203.06877. The eps_min-guarded
    elementwise-relative-change pattern follows the exact numerically-safe implementation of
    quantus.metrics.robustness.relative_input_stability.RelativeInputStability / .
    relative_output_stability.RelativeOutputStability.
  - Relative Representation Stability (RRS), SSGrad-CAM++ only: same paper. Not
    implemented for D-CRISP: D-CRISP is a genuine black-box method (get_one2many_predictions
    only, no partial forward pass, no hook), and this paper explicitly design ROS as their
    own recommended alternative to RRS "for black-box ML models" -- we follow that
    recommendation rather than instrumenting D-CRISP's model call from scratch.

The "unchanged prediction" gate required by all four metrics' own mathematical definitions
(explicitly part of the max's domain: "for all x' in N_x s.t. y_x = y_x'") is
replaced, for detection, by: does a detection matching the target's class (top-1) and
overlapping its box by IoU >= iou_thres_gt still exist in the perturbed image? iou_thres_gt=0.5
is the same GT-matching convention already used everywhere else in this project (tta.py,
ssgradcampp.yaml, tta_uq.yaml) -- deliberately not D-Deletion's gamma=0 (too permissive for
"is this still recognisably the same object", D-Deletion needs it permissive to track a smooth
decay curve as pixels are progressively removed, we don't) nor SSGradCAMPP._locate_anchor's
0.999 (that threshold reproduces an exact known detection, a different, stricter task).
"""

from typing import Callable, Optional

import numpy as np
import torch

from xai_benchmark.detection.yolo_head import get_one2many_predictions
from xai_benchmark.uncertainty.tta import box_iou

EPS_MIN = 1e-6  # zero-division guard, same value as quantus.RelativeInputStability's default


def _perturb_image_uniform(img_bgr: np.ndarray, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Additive uniform noise in [-radius, radius] on the [0, 1]-scaled image, clipped back to
    a valid uint8 BGR image. `radius` is a fraction of the pixel range, matching the
    neighbourhood-radius convention of Max-Sensitivity's paper and the
    perturbation mechanism RIS. ROS and RRS. Operates on the
    raw BGR image, not a preprocessed tensor, because DCRISP.explain() and SSGradCAMPP.explain()
    both take img_bgr as their entry point -- one perturbation serves both methods.
    """
    noise = rng.uniform(-radius, radius, size=img_bgr.shape).astype(np.float32)
    perturbed = img_bgr.astype(np.float32) / 255.0 + noise
    return (np.clip(perturbed, 0.0, 1.0) * 255.0).astype(np.uint8)


def _matched_confidence(model_dense, model_prep, img_bgr: np.ndarray, orig_shape: tuple[int, int],
                         target_box, target_class_idx: int, conf_thres: float,
                         iou_thres_nms: float, iou_thres_gt: float) -> float:
    """Re-detects `img_bgr` and returns the confidence of whichever detection best matches the
    target object (top-1 class == target_class_idx, IoU(target_box, candidate) >= iou_thres_gt),
    or 0.0 if none matches. This is our "prediction unchanged" gate and also supplies h(x) for
    Relative Output Stability (a single scalar is enough: h(x) has exactly one non-zero entry
    at target_class_idx, so ||h(x)-h(x')||_p over the full per-class vector reduces to
    |conf(x)-conf(x')|).

    Same class+IoU matching principle as detection_metrics.py::best_matching_confidence (reused
    for D-Deletion/D-Insertion), generalised here with a configurable
    `iou_thres_gt` instead of that function's hardcoded gamma=0 -- see module docstring.
    """
    imgs_tensor = model_prep.predictor.preprocess([img_bgr])
    dets = get_one2many_predictions(
        model_dense, imgs_tensor, [orig_shape], conf_thres=conf_thres, iou_thres=iou_thres_nms,
    )[0]

    same_class = dets.class_probs.argmax(dim=-1) == target_class_idx
    if not same_class.any():
        return 0.0
    target_box_t = torch.as_tensor(target_box, dtype=dets.boxes.dtype, device=dets.boxes.device).unsqueeze(0)
    ious = box_iou(target_box_t, dets.boxes[same_class]).squeeze(0)
    matching = ious >= iou_thres_gt
    if not matching.any():
        return 0.0
    return float(dets.max_conf[same_class][matching].max())


def _relative_change_norm(original: np.ndarray, perturbed: np.ndarray, eps_min: float) -> float:
    """L2 norm of the elementwise relative change (original-perturbed)/original, guarded
    against division by zero. Shared numerator for RIS/ROS/RRS (all three use this exact form for the explanation term) 
    and shared denominator for RIS/RRS (whose denominators are ALSO relative, unlike ROS's). 
    Numerically-safe pattern verified against quantus.metrics.robustness.relative_input_stability.
    RelativeInputStability.relative_input_stability_objective (Quantus 0.6.0).
    """
    diff = original - perturbed
    diff = diff / (original + (original == 0) * eps_min)
    return float(np.linalg.norm(diff))


def evaluate_perturbation_stability(
    explain_fn: Callable[[np.ndarray], np.ndarray],
    model_dense, model_prep,
    img_bgr: np.ndarray, orig_shape: tuple[int, int],
    target_box, target_class_idx: int,
    heatmap: np.ndarray,
    n_samples: int = 50, radius: float = 0.1,
    conf_thres: float = 0.25, iou_thres_nms: float = 0.5, iou_thres_gt: float = 0.5,
    eps_min: float = EPS_MIN, seed: Optional[int] = None,
    representation_fn: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
    representation: Optional[np.ndarray] = None,
    return_raw: bool = False,
) -> dict:
    """Max-Sensitivity + RIS + ROS (and RRS, if `representation_fn` is given) for one target
    object, method-agnostic via `explain_fn`.

    Generates `n_samples` independent random perturbations of `img_bgr` (not a progressive
    trajectory like D-Deletion -- each sample is an independent attempt),
    discards any sample where the target object is no longer matched, and
    computes all metrics from the surviving samples in a single pass -- avoids re-perturbing
    and re-explaining once per metric.

    Args:
        explain_fn: given a perturbed BGR image, returns its (H, W) heatmap for this same
            target object. Callers pass a small closure around DCRISP.explain() or
            SSGradCAMPP.explain().
        model_dense, model_prep: same pair of YOLO instances used everywhere else in this
            project.
        heatmap: the already computed (H, W) heatmap for the unperturbed image, not
            recomputed here, callers already have it from the original explanation run.
        representation_fn: optional. Given a perturbed BGR image, returns the internal
            representation L(x') for RRS, or None if the target is lost (e.g. SSGradCAMPP's
            own _locate_anchor already raises in that case). Leave None for D-CRISP (no RRS).
        representation: L(x) for the unperturbed image, required iff representation_fn is set.

    Returns:
        dict with n_samples, n_valid (how many perturbations kept the target matched),
        n_rejected_gate (samples discarded by the shared class+IoU gate, iou_thres_gt),
        n_rejected_explain and rejected_explain_best_ious (samples where explain_fn raised --
        for SSGradCAMPP this is _locate_anchor's 0.90 anchor gate; rejected_explain_best_ious
        holds each rejection's best_iou_same_class, or None if the exception didn't carry one),
        max_sensitivity, relative_input_stability, relative_output_stability, and, only if
        representation_fn was given: relative_representation_stability, n_valid_rrs (<=
        n_valid, RRS has its own independent anchor gate inside representation_fn),
        n_rejected_representation and rejected_representation_best_ious (same pattern as
        explain_fn's, for representation_fn's failures). All metric values are NaN if the
        corresponding sample count is 0.

        If `return_raw` is True, also includes the per-repetition lists `sensitivities`,
        `ris_values`, `ros_values`, `rrs_values` (this last one empty if `representation_fn`
        was not given) -- for diagnostic inspection of individual repetitions. NOT used to
        derive the random-noise baseline's n=5/n=20 aggregates: those come from two separate
        calls (n_samples=5 and n_samples=20, same `seed`) rather than slicing a single run's
        raw lists, precisely because these lists only contain the repetitions that survived
        the gate/explain_fn checks (compacted, not indexed by loop iteration) -- see
        run_evaluation.py's random_baseline branch. False by default: does not change the
        result dict, or any behaviour, for existing callers (D-CRISP, SSGrad-CAM++).
    """

    rng = np.random.default_rng(seed)
    h_x = _matched_confidence(model_dense, model_prep, img_bgr, orig_shape, target_box,
                               target_class_idx, conf_thres, iou_thres_nms, iou_thres_gt)

    sensitivities, ris_values, ros_values, rrs_values = [], [], [], []
    n_rejected_gate = 0
    rejected_explain_best_ious = []
    rejected_representation_best_ious = []

    for _ in range(n_samples):
        img_perturbed = _perturb_image_uniform(img_bgr, radius, rng)

        h_x_perturbed = _matched_confidence(model_dense, model_prep, img_perturbed, orig_shape,
                                             target_box, target_class_idx, conf_thres,
                                             iou_thres_nms, iou_thres_gt)
        if h_x_perturbed == 0.0:
            n_rejected_gate += 1
            continue 

        try:
            heatmap_perturbed = explain_fn(img_perturbed)
        except Exception as e:
            rejected_explain_best_ious.append(getattr(e, "best_iou_same_class", None))
            continue 

        sensitivities.append(float(np.linalg.norm(heatmap - heatmap_perturbed)))  

        ris_numerator = _relative_change_norm(heatmap, heatmap_perturbed, eps_min)
        ris_denominator = max(_relative_change_norm(img_bgr.astype(np.float32), img_perturbed.astype(np.float32), eps_min), eps_min)
        ris_values.append(ris_numerator / ris_denominator)

        ros_denominator = max(abs(h_x - h_x_perturbed), eps_min)  
        ros_values.append(ris_numerator / ros_denominator) 

        if representation_fn is not None:
            try:
                representation_perturbed = representation_fn(img_perturbed)
            except Exception as e:
                rejected_representation_best_ious.append(getattr(e, "best_iou_same_class", None))
                representation_perturbed = None
            if representation_perturbed is not None:
                rrs_numerator = _relative_change_norm(heatmap, heatmap_perturbed, eps_min)
                rrs_denominator = max(_relative_change_norm(representation, representation_perturbed, eps_min), eps_min)
                rrs_values.append(rrs_numerator / rrs_denominator)

    n_valid = len(sensitivities)
    result = {
        "n_samples": n_samples,
        "n_valid": n_valid,
        "n_rejected_gate": n_rejected_gate,
        "n_rejected_explain": len(rejected_explain_best_ious),
        "rejected_explain_best_ious": rejected_explain_best_ious,
        "max_sensitivity": max(sensitivities) if n_valid else float("nan"),
        "relative_input_stability": max(ris_values) if n_valid else float("nan"),
        "relative_output_stability": max(ros_values) if n_valid else float("nan"),
    }
    if representation_fn is not None:
        result["relative_representation_stability"] = max(rrs_values) if rrs_values else float("nan")
        result["n_valid_rrs"] = len(rrs_values)
        result["n_rejected_representation"] = len(rejected_representation_best_ious)
        result["rejected_representation_best_ious"] = rejected_representation_best_ious
    if return_raw:
        result["sensitivities"] = sensitivities
        result["ris_values"] = ris_values
        result["ros_values"] = ros_values
        result["rrs_values"] = rrs_values
    return result


def ssgradcampp_representation(explainer, img_bgr: np.ndarray, orig_shape: tuple[int, int],
                                target_box, target_class_idx: int) -> Optional[np.ndarray]:
    """L(x) for Relative Representation Stability (RRS): the activation
    A_k at the anchor location that reproduces the target detection, i.e. exactly the same
    quantity SSGrad-CAM++'s own formula uses as A_k.

    Reuses SSGradCAMPP's own anchor-relocation machinery instead of a new matching gate. Unlike
    a naive None-on-failure design, this does NOT catch `_locate_anchor`'s failure internally: it
    lets AnchorNotFoundError (xai/ssgradcampp.py) propagate to the caller, so
    evaluate_perturbation_stability can both discard the sample AND record how close the best
    same-class anchor got to the 0.90 threshold (`best_iou_same_class`). The `Optional[np.ndarray]` return type is kept
    for compatibility with the general `representation_fn` contract in
    evaluate_perturbation_stability, which tolerates either a None return or a raised exception
    as "target lost".
    """
    imgs_tensor = explainer.model_prep.predictor.preprocess([img_bgr]).to(explainer.device)
    anchor_idx = explainer._locate_anchor(imgs_tensor, orig_shape, target_box, target_class_idx)
    scale_idx, row, col, _, _ = explainer._anchor_index_to_cell(anchor_idx)
    explainer._dense_forward_with_grad(imgs_tensor)  # populates self._activations via the hooks from _register_hooks
    return explainer._activations[scale_idx][0, :, row, col].detach().cpu().numpy()