"""Test-Time Augmentation (TTA) for uncertainty quantification.

Method adapted from ElHayani, M. (2024). Real-Time Object Detection Uncertainty Quantification using 
Augmented Images for Autonomous Vehicles (Master's thesis). Technical University of Munich (TUM), Germany.
Requires the one-to-many detection head (see detection/yolo_head.py).

Pipeline per image: 1 unaugmented "anchor" pass + N_TTA augmented passes
(thesis Eq. 4.3). Each TTA pass is matched against the anchor's own boxes by
IoU >= iou_thres_tta_anchor (thesis Sec. 4.5.2) -- the same mechanism is
reused for both pixel-level (Sec. 4.5.1) and component-level (Sec. 4.5.2)
uncertainty.

The transformations used are those obtained from the study carried out 
in notebooks/tta_augmention_selection.ipynb.


The ground-truth matching (correct / wrong_class / false_positive) is our
own addition, not from the thesis, needed to study whether uncertainty
correlates with real detection quality.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from xai_benchmark.detection.yolo_head import get_one2many_predictions
from xai_benchmark.data.kitti_dataset import KITTI_CLASSES, load_kitti_yolo_bboxes


GAMMA = 1.25
CONTRAST_FACTOR = 1.30

_color_jitter = T.ColorJitter(brightness=0.5, hue=0.3)
_elastic = T.ElasticTransform(alpha=(50.0, 200.0))


def color_jitter(img: np.ndarray) -> np.ndarray:
    rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return cv2.cvtColor(np.array(_color_jitter(rgb)), cv2.COLOR_RGB2BGR)


def elastic_transform(img: np.ndarray) -> np.ndarray:
    rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return cv2.cvtColor(np.array(_elastic(rgb)), cv2.COLOR_RGB2BGR)


def contrast_enhance(img: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=CONTRAST_FACTOR, beta=0)


def gamma_correction(img: np.ndarray) -> np.ndarray:
    lut = ((np.arange(256) / 255.0) ** (1 / GAMMA) * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def flip_horizontal(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def equalize(img: np.ndarray) -> np.ndarray:
    return cv2.merge([cv2.equalizeHist(c) for c in cv2.split(img)])


AUGMENTATIONS = {
    "colour_jitter": color_jitter,
    "elastic_transform": elastic_transform,
    "contrast_enhancing": contrast_enhance,
    "gamma_correction": gamma_correction,
    "flip_horizontal": flip_horizontal,
    "equalization": equalize,
}
AUG_NAMES = list(AUGMENTATIONS.keys())


def sample_complex_augmentation(rng) -> list:
    """3 different transformations, random order (thesis 4.4.3)."""
    return rng.sample(AUG_NAMES, 3)


def apply_complex_augmentation(img: np.ndarray, aug_names: list) -> np.ndarray:
    out = img
    for name in aug_names:
        out = AUGMENTATIONS[name](out)
    return out


def generate_tta_batch(img_bgr: np.ndarray, rng, n_tta: int):
    """index 0 = anchor (without augmentation); 1..n_tta = augmented (thesis Sec. 4.4.3)."""
    batch_imgs, batch_augs = [img_bgr], [[]]
    for _ in range(n_tta):
        augs = sample_complex_augmentation(rng)
        batch_imgs.append(apply_complex_augmentation(img_bgr, augs))
        batch_augs.append(augs)
    return batch_imgs, batch_augs


def flip_boxes_x(boxes_xyxy: torch.Tensor, img_w: int) -> torch.Tensor:
    out = boxes_xyxy.clone()
    out[:, [0, 2]] = img_w - boxes_xyxy[:, [2, 0]]
    return out


def revert_geometric_augs(det, aug_names: list, img_w: int):
    if "flip_horizontal" in aug_names and len(det.boxes):
        det.boxes = flip_boxes_x(det.boxes, img_w)
    return det


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU. a: (N,4), b: (M,4) xyxy -> (N,M)."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def renormalize_probs(class_probs: torch.Tensor) -> torch.Tensor:
    """Independent sigmoids per class do not sum to 1.
    Renormalized so that, together with the 0%/100% background used in thesis (4.5.1),
    the complete vector is a valid probability distribution."""
    return class_probs / class_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)


def run_image_tta(model_dense, model_prep, img_bgr: np.ndarray, rng, n_tta: int,
                   conf_thres: float, iou_thres_nms: float):
    """model_dense: only for get_one2many_predictions, never .predict()/.val()
    (It merges the model and breaks the one2many head). model_prep: only for
    preprocessing (already merged, not used for dense inference)."""
    orig_shape = img_bgr.shape[:2]
    batch_imgs, batch_augs = generate_tta_batch(img_bgr, rng, n_tta)
    imgs_tensor = model_prep.predictor.preprocess(batch_imgs)
    dets = get_one2many_predictions(model_dense, imgs_tensor, [orig_shape] * len(batch_imgs),
                                     conf_thres=conf_thres, iou_thres=iou_thres_nms)
    for det, augs in zip(dets, batch_augs):
        revert_geometric_augs(det, augs, orig_shape[1])
        det.boxes = det.boxes.cpu()
        det.class_probs = renormalize_probs(det.class_probs).cpu()
    return dets[0], dets[1:], orig_shape


def match_tta_to_anchor(anchor, tta_passes, iou_thres: float):
    """Thesis 4.5.2: For each anchor box, the box with the highest IoU in each pass
    (if it exceeds the threshold). None if that pass found nothing for this object."""
    n_anchor = len(anchor.boxes)
    matches = [[None] * len(tta_passes) for _ in range(n_anchor)]
    for k, det in enumerate(tta_passes):
        if len(det.boxes) == 0 or n_anchor == 0:
            continue
        best_iou, best_idx = box_iou(anchor.boxes, det.boxes).max(dim=1)
        for j in range(n_anchor):
            if best_iou[j] >= iou_thres:
                matches[j][k] = int(best_idx[j])
    return matches


#  Component and pixel level (Eqs. 4.4-4.6, Sec. 4.5.1/4.5.2)

def union_canvas(anchor, tta_passes, matches, obj_idx):
    boxes = [anchor.boxes[obj_idx]] + [tta_passes[k].boxes[m] for k, m in enumerate(matches[obj_idx]) if m is not None]
    b = torch.stack(boxes)
    return (int(b[:, 0].min().floor()), int(b[:, 1].min().floor()),
            int(b[:, 2].max().ceil()), int(b[:, 3].max().ceil()))


def component_level_uncertainty(anchor, tta_passes, matches, obj_idx, num_classes):
    x1, y1, x2, y2 = union_canvas(anchor, tta_passes, matches, obj_idx)
    H, W = y2 - y1, x2 - x1
    if H <= 0 or W <= 0:
        return None

    boxes = [anchor.boxes[obj_idx]]
    probs = [anchor.class_probs[obj_idx]]
    for k, m in enumerate(matches[obj_idx]):
        if m is not None:
            boxes.append(tta_passes[k].boxes[m])
            probs.append(tta_passes[k].class_probs[m])
    boxes_t, probs_t = torch.stack(boxes), torch.stack(probs)
    M = boxes_t.shape[0]

    yy, xx = torch.meshgrid(torch.arange(y1, y2), torch.arange(x1, x2), indexing="ij")
    inside = ((xx[None] >= boxes_t[:, 0].view(M, 1, 1)) & (xx[None] < boxes_t[:, 2].view(M, 1, 1)) &
              (yy[None] >= boxes_t[:, 1].view(M, 1, 1)) & (yy[None] < boxes_t[:, 3].view(M, 1, 1)))

    uniform = torch.full((num_classes,), 1.0 / num_classes)
    contrib = torch.where(inside.unsqueeze(-1),
                           probs_t.view(M, 1, 1, num_classes).expand(M, H, W, num_classes),
                           uniform.view(1, 1, 1, num_classes).expand(M, H, W, num_classes))
    mean_probs = contrib.mean(dim=0)
    entropy_map = -(mean_probs.clamp(min=1e-12) * mean_probs.clamp(min=1e-12).log()).sum(-1)

    n_inside = inside.sum(dim=0)
    intersection_mask = n_inside == M
    border_mask = (n_inside > 0) & ~intersection_mask

    return {
        "n_matched": M,
        "class_uncertainty": entropy_map[intersection_mask].mean().item() if intersection_mask.any() else 0.0,
        "localization_uncertainty": entropy_map[border_mask].mean().item() if border_mask.any() else 0.0,
    }


def pixel_level_uncertainty(tta_passes, matches, obj_idx, canvas, num_classes):
    x1, y1, x2, y2 = canvas
    H, W = y2 - y1, x2 - x1
    if H <= 0 or W <= 0:
        return None

    n_tta = len(tta_passes)
    yy, xx = torch.meshgrid(torch.arange(y1, y2), torch.arange(x1, x2), indexing="ij")
    vectors = torch.zeros(n_tta, H, W, num_classes + 1)
    vectors[..., -1] = 1.0

    for k, det in enumerate(tta_passes):
        m = matches[obj_idx][k]
        if m is None:
            continue
        box = det.boxes[m]
        inside = (xx >= box[0]) & (xx < box[2]) & (yy >= box[1]) & (yy < box[3])
        probs = det.class_probs[m]
        vectors[k][inside] = torch.cat([probs, probs.new_zeros(1)])

    mean_vec = vectors.mean(dim=0)
    H_total = -(mean_vec.clamp(min=1e-12) * mean_vec.clamp(min=1e-12).log()).sum(-1)
    per_pass_H = -(vectors.clamp(min=1e-12) * vectors.clamp(min=1e-12).log()).sum(-1)
    U_a = per_pass_H.mean(dim=0)
    U_e = (H_total - U_a).clamp(min=0)

    return {"H_total": H_total, "U_a": U_a, "U_e": U_e}


def match_detections_to_gt(anchor, gt_labels: list, iou_thres: float):
    """Match each anchor box against the actual GT labels.
    Threshold 0.5 = mAP50 convention, not from the thesis, 
    is an addition of my own to study the correlation between uncertainty/correctness.
    Returns a list per anchor box of
    (gt_class or None, iou, 'correct'|'wrong_class'|'false_positive'), and the
    set of GT indices found (to count false negatives separately)."""
    n_anchor = len(anchor.boxes)
    if not gt_labels or n_anchor == 0:
        return [(None, 0.0, "false_positive")] * n_anchor, set()

    gt_boxes = torch.tensor([g[1:] for g in gt_labels], dtype=torch.float32)
    gt_classes = [g[0] for g in gt_labels]

    results = []
    matched_gt = set()
    best_iou, best_idx = box_iou(anchor.boxes, gt_boxes).max(dim=1)
    for j in range(n_anchor):
        if best_iou[j] >= iou_thres:
            gt_idx = int(best_idx[j])
            matched_gt.add(gt_idx)
            gt_class = gt_classes[gt_idx]
            pred_class = int(anchor.class_probs[j].argmax())
            label = "correct" if pred_class == gt_class else "wrong_class"
            results.append((gt_class, float(best_iou[j]), label))
        else:
            results.append((None, float(best_iou[j]), "false_positive"))
    return results, matched_gt


def reduce_object_to_row(image_id: str, obj_idx: int, anchor, comp: dict, pix: dict, gt_match: tuple):
    """Reduction to flat rows (avoids accumulating [H,W] maps of 1496 images in memory)"""
    gt_class, gt_iou, correctness = gt_match
    box = anchor.boxes[obj_idx].tolist()
    pred_class = int(anchor.class_probs[obj_idx].argmax())
    return {
        "image_id": image_id, "object_idx": obj_idx,
        "pred_class": pred_class, "pred_class_name": KITTI_CLASSES.get(pred_class, "?"),
        "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
        "n_matched": comp["n_matched"],
        "pixel_H_total_mean": pix["H_total"].mean().item(), "pixel_H_total_max": pix["H_total"].max().item(),
        "pixel_Ua_mean": pix["U_a"].mean().item(), "pixel_Ua_max": pix["U_a"].max().item(),
        "pixel_Ue_mean": pix["U_e"].mean().item(), "pixel_Ue_max": pix["U_e"].max().item(),
        "comp_class_uncertainty": comp["class_uncertainty"],
        "comp_localization_uncertainty": comp["localization_uncertainty"],
        "gt_class": gt_class, "gt_class_name": KITTI_CLASSES.get(gt_class, "none"),
        "gt_iou": gt_iou, "correctness": correctness,
    }


def aggregate_image_uncertainty(scores: list, boxes: list, method: str) -> float:
    """scores: pixel_H_total_mean per object. boxes: anchor boxes [x1,y1,x2,y2],
    used to weight by size in weighted_mean (thesis 4.6)."""
    if not scores:
        return 0.0
    scores = np.asarray(scores, dtype=np.float64)
    if method == "max":
        return float(scores.max())
    if method == "mean":
        return float(scores.mean())
    if method == "weighted_mean":
        areas = np.array([max((b[2] - b[0]) * (b[3] - b[1]), 1.0) for b in boxes])
        return float(np.average(scores, weights=np.log(areas)))
    raise ValueError(f"Unknown aggregation method: {method}")


def process_image_for_uq(model_dense, model_prep, img_bgr: np.ndarray, gt_labels: list,
                          image_id: str, rng, cfg: dict):
    """Complete image pipeline: TTA -> anchor matching/TTA ->
    anchor matching/GT -> pixel and component uncertainty per
    object -> flat rows. Returns (object_rows, image_row)."""
    anchor, tta_passes, _ = run_image_tta(
        model_dense, model_prep, img_bgr, rng,
        n_tta=cfg["tta"]["n_tta"], conf_thres=cfg["detection"]["conf_thres"],
        iou_thres_nms=cfg["detection"]["iou_thres_nms"],
    )
    n_gt = len(gt_labels)
    methods = cfg["aggregation"]["methods"]

    if len(anchor.boxes) == 0:
        image_row = {"image_id": image_id, "n_anchor_objects": 0, "n_gt_objects": n_gt,
                     "n_false_negatives": n_gt}
        image_row.update({f"agg_{m}": 0.0 for m in methods})
        return [], image_row

    matches = match_tta_to_anchor(anchor, tta_passes, cfg["matching"]["iou_thres_tta_anchor"])
    gt_matches, matched_gt_idx = match_detections_to_gt(anchor, gt_labels, cfg["matching"]["iou_thres_gt"])
    num_classes = anchor.class_probs.shape[-1]

    object_rows = []
    for j in range(len(anchor.boxes)):
        canvas = union_canvas(anchor, tta_passes, matches, j)
        pix = pixel_level_uncertainty(tta_passes, matches, j, canvas, num_classes)
        comp = component_level_uncertainty(anchor, tta_passes, matches, j, num_classes)
        if pix is None or comp is None:
            continue
        object_rows.append(reduce_object_to_row(image_id, j, anchor, comp, pix, gt_matches[j]))

    agg_scores = [r["pixel_H_total_mean"] for r in object_rows]
    agg_boxes = [[r["x1"], r["y1"], r["x2"], r["y2"]] for r in object_rows]
    image_row = {
        "image_id": image_id,
        "n_anchor_objects": len(anchor.boxes),
        "n_gt_objects": n_gt,
        "n_false_negatives": n_gt - len(matched_gt_idx),
    }
    image_row.update({f"agg_{m}": aggregate_image_uncertainty(agg_scores, agg_boxes, m) for m in methods})

    return object_rows, image_row