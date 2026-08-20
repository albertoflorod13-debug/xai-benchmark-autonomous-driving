"""CLI entry point: D-CRISP explanation generation.

Iterates through the KITTI validation set, obtains the detections for each image (using
`get_one2many_predictions`), and generates D-CRISP heatmaps for every detected object in ONE
call per image (`DCRISP.explain`, `xai/dcrisp.py`) -- not one call per object, per the
single-mask-pass optimization already validated as lossless. Saves:
- one .npz file per image with the raw (letterboxed-frame) heatmaps.
- one row per object in `object_level.csv`, same GT-matching criteria as
  `results/ssgradcampp/object_level.csv` / `results/tta_uq/object_level.csv`.

Configuration: `configs/xai/dcrisp.yaml`.
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from xai_benchmark.data.kitti_dataset import KITTI_CLASSES, load_kitti_yolo_bboxes
from xai_benchmark.detection.yolo_head import get_one2many_predictions
from xai_benchmark.uncertainty.tta import match_detections_to_gt
from xai_benchmark.xai.dcrisp import DCRISP

OBJECT_FIELDS = [
    "image_id", "obj_idx", "pred_class", "pred_class_name", "confidence",
    "x1", "y1", "x2", "y2",
    "letterbox_h", "letterbox_w",
    "gt_class", "gt_class_name", "gt_iou", "correctness",
]

KNOWN_IMAGE_IDS = ["004018", "005663", "000408", "000106"] 

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-size", type=int, default=None,
                   help="Evaluate a random sample of N images instead of the full set. "
                        "Same --seed across alpha runs => same images for all of them.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for --sample-size's sampling (keep fixed across alpha runs).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N images (for quick tests, no randomisation)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(REPO_ROOT / "configs" / "xai" / "dcrisp.yaml")

    val_images_dir = REPO_ROOT / cfg["data"]["val_images_dir"]
    val_labels_dir = REPO_ROOT / cfg["data"]["val_labels_dir"]
    checkpoint = REPO_ROOT / cfg["checkpoint"]
    assert checkpoint.exists(), f"Checkpoint not found: {checkpoint}"

    val_images = sorted(val_images_dir.glob("*.png"))
    assert val_images, f"No images were found in {val_images_dir}"
    if args.sample_size:
        known_images = [p for p in val_images if p.stem in KNOWN_IMAGE_IDS]
        assert len(known_images) == len(KNOWN_IMAGE_IDS), (
            f"Expected {len(KNOWN_IMAGE_IDS)} known images, found {len(known_images)}"
        )
        candidate_images = [p for p in val_images if p.stem not in KNOWN_IMAGE_IDS]
        sampled_images = random.Random(args.seed).sample(candidate_images, args.sample_size)
        val_images = sorted(known_images + sampled_images)
        print(f"SAMPLE MODE: {len(val_images)} images "
              f"({len(known_images)} known + {len(sampled_images)} random, seed={args.seed})")
    elif args.limit:
        val_images = val_images[: args.limit]
        print(f"TEST MODE: {len(val_images)} images")
    else:
        print(f"Processing {len(val_images)} images of {val_images_dir}")

    device = cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    conf_thres = cfg["detection"]["conf_thres"]
    iou_thres_nms = cfg["detection"]["iou_thres_nms"]
    iou_thres_gt = cfg["matching"]["iou_thres_gt"]
    mask_cfg = cfg["mask"]

    explainer = DCRISP.from_checkpoint(
        checkpoint, device=device,
        conf_thres=conf_thres, iou_thres_nms=iou_thres_nms,
        gpu_batch=mask_cfg["gpu_batch"], n_masks=mask_cfg["n_masks"],
        alpha=mask_cfg["alpha"], resolution=mask_cfg["resolution"],
        p1=mask_cfg["p1"], num_levels=mask_cfg["num_levels"],
    )

    heatmaps_dir = REPO_ROOT / cfg["output"]["heatmaps_dir"]
    object_csv_path = REPO_ROOT / cfg["output"]["object_level_csv"]
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    object_csv_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"D-CRISP run: alpha={mask_cfg['alpha']} -> {heatmaps_dir.parent}")

    done_ids = {p.stem for p in heatmaps_dir.glob("*.npz")}
    if done_ids:
        print(f"Resuming: {len(done_ids)} images already processed, skipping them.")

    save_every = cfg["output"]["save_every"]
    start_time = time.time()
    n_objects_total = 0

    file_mode = "a" if object_csv_path.exists() else "w"
    with open(object_csv_path, file_mode, newline="", encoding="utf-8") as f_obj:
        writer = csv.DictWriter(f_obj, fieldnames=OBJECT_FIELDS)
        if file_mode == "w":
            writer.writeheader()

        for i, img_path in enumerate(val_images):
            image_id = img_path.stem
            if image_id in done_ids:
                continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"Could not be read {img_path}")
                continue
            img_h, img_w = img_bgr.shape[:2]

            txt_path = val_labels_dir / f"{image_id}.txt"
            gt_labels = (
                load_kitti_yolo_bboxes(str(txt_path), img_h, img_w, return_class=True, box_format="xyxy")
                if txt_path.exists() else []
            )

            imgs_tensor = explainer.model_prep.predictor.preprocess([img_bgr])
            letterbox_h, letterbox_w = imgs_tensor.shape[2:]

            dets = get_one2many_predictions(
                explainer.model_dense, imgs_tensor, [(img_h, img_w)],
                conf_thres=conf_thres, iou_thres=iou_thres_nms,)[0]
            dets.boxes = dets.boxes.cpu()
            dets.class_probs = dets.class_probs.cpu()
            dets.max_conf = dets.max_conf.cpu()

            if len(dets.boxes) == 0:
                continue

            gt_matches, _ = match_detections_to_gt(dets, gt_labels, iou_thres_gt)

            targets = [
                (dets.boxes[obj_idx].tolist(), int(dets.class_probs[obj_idx].argmax()),
                 dets.class_probs[obj_idx])
                for obj_idx in range(len(dets.boxes))
            ]
            try:
                results = explainer.explain(img_bgr, targets)
            except Exception as e:
                print(f"WARNING: {image_id} omitted ({e})")
                continue

            heatmaps_out = {}
            for obj_idx, result in enumerate(results):
                confidence = float(dets.max_conf[obj_idx])
                gt_class, gt_iou, correctness = gt_matches[obj_idx]

                heatmaps_out[f"obj_{obj_idx}"] = result.heatmap_raw.astype(np.float32)
                writer.writerow({
                    "image_id": image_id, "obj_idx": obj_idx,
                    "pred_class": result.target_class,
                    "pred_class_name": KITTI_CLASSES.get(result.target_class, "?"),
                    "confidence": confidence,
                    "x1": result.target_box[0], "y1": result.target_box[1],
                    "x2": result.target_box[2], "y2": result.target_box[3],
                    "letterbox_h": letterbox_h, "letterbox_w": letterbox_w,
                    "gt_class": gt_class, "gt_class_name": KITTI_CLASSES.get(gt_class, "none"),
                    "gt_iou": gt_iou, "correctness": correctness,
                })
                n_objects_total += 1

            if heatmaps_out:
                np.savez_compressed(heatmaps_dir / f"{image_id}.npz", **heatmaps_out)

            if (i + 1) % save_every == 0 or (i + 1) == len(val_images):
                f_obj.flush()
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(val_images) - (i + 1))
                print(f"[{i + 1}/{len(val_images)}] {n_objects_total} objects explained, "
                      f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    print(f"Finished in {time.time() - start_time:.0f}s. Objects explained: {n_objects_total}. "
          f"Saved in:\n  {object_csv_path}\n  {heatmaps_dir}")


if __name__ == "__main__":
    main()