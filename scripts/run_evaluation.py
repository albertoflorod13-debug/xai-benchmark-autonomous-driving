"""CLI entry point: evaluation pipeline for per-instance XAI heatmaps.

Reads the object_level.csv + heatmaps/*.npz already produced by a method's own generation
script. Computes, per object:
  - pointing_game, ebpg, relevance_rank_accuracy, sparseness -- cheap, no re-inference,
    computed for EVERY object (quantus_metrics.py).
  - deletion_auc, insertion_auc, minimal_subset, plus the extended fidelity/localisation
    metrics (min_subset_k_pixels_exact, rank_accuracy_at_min_subset, rank_accuracy_precision,
    outside_box_deletion_auc, outside_box_confidence_drop) -- expensive (re-runs the detector
    many times per object via DetectorAsClassifier, detection_metrics.py), computed only for a
    random sample (--faithfulness-sample-size).
  - stability/robustness under input perturbation (stability.py): Max-Sensitivity, RIS, ROS,
    and RRS (SSGrad-CAM++ only) -- instantiates a second, dedicated explainer per method (its
    own iou_match_thres for SSGrad-CAM++, a fresh DCRISP for D-CRISP), computed only for
    objects belonging to an image-level sample (evaluation.stability.image_sample_size in the
    method's config), never object-level, so no image contributes a partial set of objects.
Writes one row per object to <method>/eval_metrics.csv. Never plots anything -- see
notebooks/xai_evaluation_analysis.ipynb for that.

Usage:
    python scripts/run_evaluation.py --method ssgradcampp
    python scripts/run_evaluation.py --method ssgradcampp --limit 20 --faithfulness-sample-size 10
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from xai_benchmark.evaluation import detection_metrics, quantus_metrics, stability
from xai_benchmark.xai import common
from xai_benchmark.xai.dcrisp import DCRISP
from xai_benchmark.xai.ssgradcampp import SSGradCAMPP
from xai_benchmark.detection.yolo_head import get_one2many_predictions
from xai_benchmark.uncertainty.tta import box_iou

METHOD_CONFIGS = {
    "ssgradcampp": REPO_ROOT / "configs" / "xai" / "ssgradcampp.yaml",
    "dcrisp": REPO_ROOT / "configs" / "xai" / "dcrisp.yaml",
}

EVAL_FIELDS = [
    "image_id", "obj_idx", "pred_class", "pred_class_name",
    "gt_class", "gt_class_name", "gt_iou", "correctness",
    "pointing_game", "ebpg", "relevance_rank_accuracy", "sparseness",
    "in_faithfulness_sample", "deletion_auc", "insertion_auc", "minimal_subset",
    "min_subset_k_pixels_exact", "rank_accuracy_at_min_subset", "rank_accuracy_precision",
    "outside_box_deletion_auc", "outside_box_confidence_drop",
    "in_stability_sample", "stability_n_valid", "stability_n_rejected_gate",
    "stability_n_rejected_explain", "stability_max_sensitivity", "stability_ris",
    "stability_ros", "stability_rrs", "stability_n_valid_rrs",
    "stability_n_rejected_representation",
]


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=sorted(METHOD_CONFIGS),
                   help="Which method's already-generated heatmaps to evaluate.")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N images worth of objects (quick tests).")
    p.add_argument("--faithfulness-sample-size", type=int, default=None,
                   help="Overrides evaluation.faithfulness_sample_size from the config.")
    p.add_argument("--seed", type=int, default=None,
                   help="Overrides evaluation.seed from the config.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(METHOD_CONFIGS[args.method])
    eval_cfg = cfg["evaluation"]

    checkpoint = REPO_ROOT / cfg["checkpoint"]
    assert checkpoint.exists(), f"Checkpoint no encontrado: {checkpoint}"
    object_csv_path = REPO_ROOT / cfg["output"]["object_level_csv"]
    heatmaps_dir = REPO_ROOT / cfg["output"]["heatmaps_dir"]
    assert object_csv_path.exists(), (
        f"{object_csv_path} does not exist"
    )

    step = eval_cfg["step"]
    sample_size_cfg = args.faithfulness_sample_size or eval_cfg["faithfulness_sample_size"]
    seed = args.seed if args.seed is not None else eval_cfg["seed"]
    save_every = eval_cfg.get("save_every", 100)
    conf_thres = cfg["detection"]["conf_thres"]
    iou_thres_nms = cfg["detection"]["iou_thres_nms"]
    iou_thres_gt = cfg["matching"]["iou_thres_gt"]
    margin = cfg["mask"].get("margin", 0)  
    stability_cfg = eval_cfg["stability"]
    stability_radius = stability_cfg["radius"]
    stability_n_samples = stability_cfg["n_samples"]
    stability_seed = stability_cfg["seed"]
    eval_csv_path = REPO_ROOT / eval_cfg["eval_metrics_csv"]
    eval_csv_path.parent.mkdir(parents=True, exist_ok=True)
    val_images_dir = REPO_ROOT / cfg["data"]["val_images_dir"]

    device = cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    from ultralytics import YOLO
    model_dense = YOLO(str(checkpoint))     # never .predict()/.val() -- see yolo_head.py
    model_prep = YOLO(str(checkpoint))      # only used for preprocessing
    model_dense.model.to(device)
    model_dense.model.eval()
    num_classes = model_dense.model.model[-1].nc

    if getattr(model_prep, "predictor", None) is None:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model_prep.predict([dummy], verbose=False, device=0 if device == "cuda" else "cpu")

    if args.method == "ssgradcampp":
        stability_explainer = SSGradCAMPP.from_checkpoint(
            checkpoint, device=device, iou_match_thres=stability_cfg["iou_match_thres"],
            eps=cfg["numerics"]["eps"],
        )
    else:  # dcrisp
        stability_explainer = DCRISP.from_checkpoint(
            checkpoint, device=device, conf_thres=conf_thres, iou_thres_nms=iou_thres_nms,
            gpu_batch=cfg["mask"]["gpu_batch"], n_masks=cfg["mask"]["n_masks"],
            alpha=cfg["mask"]["alpha"], resolution=cfg["mask"]["resolution"],
            p1=cfg["mask"]["p1"], num_levels=cfg["mask"]["num_levels"],
        )

    df = pd.read_csv(object_csv_path, dtype={"image_id": str})
    if args.limit:
        image_ids = df["image_id"].unique()[: args.limit]
        df = df[df["image_id"].isin(image_ids)].reset_index(drop=True)
        print(f"TEST MODE: {len(image_ids)} imagenes, {len(df)} objetos")
    else:
        print(f"Evaluating {df['image_id'].nunique()} images, {len(df)} objects")

    rng = random.Random(seed)
    sample_size = min(sample_size_cfg, len(df))
    faithfulness_idx = set(rng.sample(list(df.index), sample_size))
    print(f"Deletion/Insertion on a sample of {sample_size}/{len(df)} objects "
          f"(root {seed}).")

    stability_rng = random.Random(stability_seed)
    all_image_ids = list(df["image_id"].unique())
    stability_image_sample_size = min(stability_cfg["image_sample_size"], len(all_image_ids))
    stability_images = set(stability_rng.sample(all_image_ids, stability_image_sample_size))
    print(f"Robustness/stability on {stability_image_sample_size}/{len(all_image_ids)} images "
          f"(seed {stability_seed}).")

    start_time = time.time()
    n_done = 0
    stability_errors = []

    done_ids = set()
    if eval_csv_path.exists():
        done_ids = set(pd.read_csv(eval_csv_path, dtype={"image_id": str})["image_id"].unique())
        print(f"Resuming: {len(done_ids)} images already evaluated, skipping them.")

    file_mode = "a" if eval_csv_path.exists() else "w"
    with open(eval_csv_path, file_mode, newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=EVAL_FIELDS)
        if file_mode == "w":
            writer.writeheader()

        for img_counter, (image_id, group) in enumerate(df.groupby("image_id", sort=False), start=1):
            if image_id in done_ids:
                continue
            img_path = val_images_dir / f"{image_id}.png"
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"Could not be read {img_path}")
                continue
            orig_shape = img_bgr.shape[:2]

            npz_path = heatmaps_dir / f"{image_id}.npz"
            if not npz_path.exists():
                print(f"Does not exist {npz_path}")
                continue

            x_tensor = model_prep.predictor.preprocess([img_bgr]).to(device)
            letterbox_shape = tuple(x_tensor.shape[2:])

            is_stability_image = image_id in stability_images
            dets0 = None
            if is_stability_image and args.method == "dcrisp":
                dets0 = get_one2many_predictions(
                    model_dense, x_tensor, [orig_shape], conf_thres=conf_thres, iou_thres=iou_thres_nms,
                )[0]

            needs_reference = any(idx in faithfulness_idx for idx in group.index)
            x_np = x_tensor.cpu().numpy().astype(np.float32) if needs_reference else None

            with np.load(npz_path) as heatmaps_npz:
                for idx, row in group.iterrows():
                    obj_idx = int(row["obj_idx"])
                    key = f"obj_{obj_idx}"
                    if key not in heatmaps_npz:
                        print(f"WARNING: {image_id} obj {obj_idx} no heatmap saved")
                        continue
                    heatmap_raw = heatmaps_npz[key]
                    target_box = [row["x1"], row["y1"], row["x2"], row["y2"]]
                    target_class = int(row["pred_class"])

                    heatmap_orig = detection_metrics.reconstruct_heatmap_original(
                        heatmap_raw, letterbox_shape, orig_shape)
                    box_mask = detection_metrics.build_box_mask(target_box, *orig_shape)

                    out_row = {
                        "image_id": image_id, "obj_idx": obj_idx,
                        "pred_class": target_class, "pred_class_name": row["pred_class_name"],
                        "gt_class": row["gt_class"], "gt_class_name": row["gt_class_name"],
                        "gt_iou": row["gt_iou"], "correctness": row["correctness"],
                        "pointing_game": quantus_metrics.pointing_game(heatmap_orig, box_mask),
                        "ebpg": quantus_metrics.energy_based_pointing_game(heatmap_orig, box_mask),
                        "relevance_rank_accuracy": quantus_metrics.relevance_rank_accuracy(
                            heatmap_orig, box_mask),
                        "sparseness": quantus_metrics.sparseness(heatmap_orig),
                        "in_faithfulness_sample": idx in faithfulness_idx,
                        "deletion_auc": "", "insertion_auc": "", "minimal_subset": "",
                        "min_subset_k_pixels_exact": "", "rank_accuracy_at_min_subset": "",
                        "rank_accuracy_precision": "", "outside_box_deletion_auc": "",
                        "outside_box_confidence_drop": "",
                        "in_stability_sample": False, "stability_n_valid": "",
                        "stability_n_rejected_gate": "", "stability_n_rejected_explain": "",
                        "stability_max_sensitivity": "", "stability_ris": "", "stability_ros": "",
                        "stability_rrs": "", "stability_n_valid_rrs": "",
                        "stability_n_rejected_representation": "",
                    }

                    if idx in faithfulness_idx:
                        classifier = detection_metrics.DetectorAsClassifier(
                            model_dense, num_classes, orig_shape, target_box, target_class,
                            conf_thres=conf_thres, iou_thres_nms=iou_thres_nms,
                        )
                        heatmap_lb = detection_metrics.reconstruct_heatmap_letterboxed(
                            heatmap_raw, letterbox_shape)
                        aucs = detection_metrics.deletion_insertion_auc(
                            classifier, x_np, heatmap_lb, target_class,
                            step=eval_cfg["step"], device=device,
                        )
                        out_row["deletion_auc"] = aucs["deletion_auc"]
                        out_row["insertion_auc"] = aucs["insertion_auc"]
                        out_row["minimal_subset"] = detection_metrics.minimal_subset(
                            aucs["deletion_scores"], k_threshold=conf_thres)

                        target_box_lb = common.letterbox_map_box(target_box, orig_shape, letterbox_shape)
                        box_mask_lb = detection_metrics.build_box_mask(target_box_lb, *letterbox_shape)

                        scores_arr = np.asarray(aucs["deletion_scores"])
                        below = np.flatnonzero(scores_arr < conf_thres)
                        n_pixels_lb = letterbox_shape[0] * letterbox_shape[1]
                        if below.size > 0:
                            k_hi = min(int(below[0]) * step, n_pixels_lb)
                            k_lo = max(k_hi - step, 0)
                        else:
                            k_lo = k_hi = n_pixels_lb  

                        k_exact = detection_metrics.refine_minimal_subset_k(
                            x_np, heatmap_lb, classifier, target_class, k_lo, k_hi, conf_thres, device=device,
                        )
                        out_row["min_subset_k_pixels_exact"] = k_exact
                        rank_acc = detection_metrics.rank_accuracy_at_k(heatmap_lb, box_mask_lb, k_exact)
                        out_row["rank_accuracy_at_min_subset"] = rank_acc["rank_accuracy_box"]
                        out_row["rank_accuracy_precision"] = rank_acc["rank_accuracy_precision"]

                        outside = detection_metrics.outside_box_deletion(
                            classifier, x_np, heatmap_lb, box_mask_lb, target_class,
                            step=eval_cfg["step"], device=device,
                        )
                        out_row["outside_box_deletion_auc"] = outside["outside_box_deletion_auc"]
                        out_row["outside_box_confidence_drop"] = outside["outside_box_confidence_drop"]

                    if image_id in stability_images:
                        out_row["in_stability_sample"] = True
                        obj_seed = stability_seed + obj_idx
                        try:
                            if args.method == "dcrisp":
                                target_box_t = torch.as_tensor(
                                    [target_box], dtype=dets0.boxes.dtype, device=dets0.boxes.device)
                                ious0 = box_iou(target_box_t, dets0.boxes).squeeze(0)
                                best0 = int(ious0.argmax())
                                assert float(ious0[best0]) > 0.99, "original detection not recovered"
                                target_class_probs = dets0.class_probs[best0].detach().cpu()

                                result0 = stability_explainer.explain(
                                    img_bgr, [(target_box, target_class, target_class_probs)])[0]
                                heatmap0 = detection_metrics.reconstruct_heatmap_letterboxed(
                                    result0.heatmap_raw, letterbox_shape)

                                def explain_fn(img_perturbed, _b=target_box, _c=target_class,
                                               _p=target_class_probs):
                                    return stability_explainer.explain(img_perturbed, [(_b, _c, _p)])[0].heatmap_raw

                                stab = stability.evaluate_perturbation_stability(
                                    explain_fn, model_dense, model_prep, img_bgr, orig_shape,
                                    target_box, target_class, heatmap0,
                                    n_samples=stability_n_samples, radius=stability_radius,
                                    conf_thres=conf_thres, iou_thres_nms=iou_thres_nms,
                                    iou_thres_gt=iou_thres_gt, seed=obj_seed,
                                )
                            else:  # ssgradcampp
                                result0 = stability_explainer.explain(
                                    img_bgr, target_box, target_class, margin=margin)
                                heatmap0 = detection_metrics.reconstruct_heatmap_letterboxed(
                                    result0.heatmap_raw, letterbox_shape)
                                representation0 = stability_explainer._activations[result0.scale_idx][
                                    0, :, result0.row, result0.col].detach().cpu().numpy()

                                def explain_fn(img_perturbed, _b=target_box, _c=target_class):
                                    r = stability_explainer.explain(img_perturbed, _b, _c, margin=margin)
                                    return detection_metrics.reconstruct_heatmap_letterboxed(
                                        r.heatmap_raw, letterbox_shape)

                                def representation_fn(img_perturbed, _b=target_box, _c=target_class):
                                    return stability.ssgradcampp_representation(
                                        stability_explainer, img_perturbed, orig_shape, _b, _c)

                                stab = stability.evaluate_perturbation_stability(
                                    explain_fn, model_dense, model_prep, img_bgr, orig_shape,
                                    target_box, target_class, heatmap0,
                                    n_samples=stability_n_samples, radius=stability_radius,
                                    conf_thres=conf_thres, iou_thres_nms=iou_thres_nms,
                                    iou_thres_gt=iou_thres_gt, seed=obj_seed,
                                    representation_fn=representation_fn, representation=representation0,
                                )

                            out_row["stability_n_valid"] = stab["n_valid"]
                            out_row["stability_n_rejected_gate"] = stab["n_rejected_gate"]
                            out_row["stability_n_rejected_explain"] = stab["n_rejected_explain"]
                            out_row["stability_max_sensitivity"] = stab["max_sensitivity"]
                            out_row["stability_ris"] = stab["relative_input_stability"]
                            out_row["stability_ros"] = stab["relative_output_stability"]
                            if args.method == "ssgradcampp":
                                out_row["stability_rrs"] = stab["relative_representation_stability"]
                                out_row["stability_n_valid_rrs"] = stab["n_valid_rrs"]
                                out_row["stability_n_rejected_representation"] = stab["n_rejected_representation"]
                        except Exception as e:
                            stability_errors.append((image_id, obj_idx, repr(e)))

                    writer.writerow(out_row)
                    n_done += 1

            if img_counter % save_every == 0:
                f_out.flush()
                elapsed = time.time() - start_time
                print(f"[{img_counter} images] {n_done} objects evaluated, "
                      f"{elapsed:.0f}s elapsed")

    if stability_errors:
        print(f"Stability errors on {len(stability_errors)} objects (see list below):")
        for e in stability_errors[:20]:
            print(" ", e)
    print(f"Finished in {time.time() - start_time:.0f}s. Objects evaluated: {n_done}. "
          f"Saved in {eval_csv_path}")

if __name__ == "__main__":
    main()