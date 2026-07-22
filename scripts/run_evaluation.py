"""CLI entry point: evaluation pipeline for per-instance XAI heatmaps.

Reads the object_level.csv + heatmaps/*.npz already produced by a method's own generation
script.
Computes, per object:
  - pointing_game, ebpg, sparseness  -- cheap, no re-inference, computed for EVERY object.
  - deletion_auc, insertion_auc      -- expensive (re-runs the detector n_steps times per
    object), computed only for a random sample (--faithfulness-sample-size).
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

from xai_benchmark.evaluation import detection_metrics, quantus_metrics

METHOD_CONFIGS = {
    "ssgradcampp": REPO_ROOT / "configs" / "xai" / "ssgradcampp.yaml",
}

EVAL_FIELDS = [
    "image_id", "obj_idx", "pred_class", "pred_class_name",
    "gt_class", "gt_class_name", "gt_iou", "correctness",
    "pointing_game", "ebpg", "sparseness",
    "in_faithfulness_sample", "deletion_auc", "insertion_auc",
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

    start_time = time.time()
    n_done = 0

    with open(eval_csv_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=EVAL_FIELDS)
        writer.writeheader()

        for img_counter, (image_id, group) in enumerate(df.groupby("image_id", sort=False), start=1):
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
                        "sparseness": quantus_metrics.sparseness(heatmap_orig),
                        "in_faithfulness_sample": idx in faithfulness_idx,
                        "deletion_auc": "", "insertion_auc": "",
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

                    writer.writerow(out_row)
                    n_done += 1

            if img_counter % save_every == 0:
                f_out.flush()
                elapsed = time.time() - start_time
                print(f"[{img_counter} images] {n_done} objects evaluated, "
                      f"{elapsed:.0f}s elapsed")

    print(f"Finished in {time.time() - start_time:.0f}s. Objects evaluated: {n_done}. "
          f"Saved in {eval_csv_path}")


if __name__ == "__main__":
    main()