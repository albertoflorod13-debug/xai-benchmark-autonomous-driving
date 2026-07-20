"""CLI entry point: uncertainty quantification via TTA.

It iterates through the entire KITTI value set, calculates pixel uncertainty (Thesis, eqs. 4.4-4.6)
and component uncertainty (Thesis, Sec. 4.5.2) for each object detected by the anchor, 
matches it against the actual GT labels (see tta.py), and saves two tables:
one row per object (object_level.csv) and one row per image with the added 
score (image_level.csv).

Config: configs/tta_uq.yaml
"""

import csv
import random
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from xai_benchmark.data.kitti_dataset import load_kitti_yolo_bboxes
from xai_benchmark.uncertainty.tta import process_image_for_uq

OBJECT_FIELDS = [
    "image_id", "object_idx", "pred_class", "pred_class_name", "x1", "y1", "x2", "y2",
    "n_matched", "pixel_H_total_mean", "pixel_H_total_max", "pixel_Ua_mean", "pixel_Ua_max",
    "pixel_Ue_mean", "pixel_Ue_max", "comp_class_uncertainty", "comp_localization_uncertainty",
    "gt_class", "gt_class_name", "gt_iou", "correctness",
]


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                    help="Process only N images")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(REPO_ROOT / "configs" / "tta_uq.yaml")
    seed = cfg["tta"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed) 

    val_images_dir = REPO_ROOT / cfg["data"]["val_images_dir"]
    val_labels_dir = REPO_ROOT / cfg["data"]["val_labels_dir"]
    checkpoint = REPO_ROOT / cfg["checkpoint"]
    assert checkpoint.exists(), f"Checkpoint not found: {checkpoint}"

    val_images = sorted(val_images_dir.glob("*.png"))
    assert val_images, f"No images were found in {val_images_dir}"

    if args.limit:
        random.seed(seed)  
        val_images = random.sample(val_images, args.limit) 
        print(f"VERIFICATION MODE: {len(val_images)} images")
    else:
        print(f"Processing {len(val_images)} images of {val_images_dir}")

    model_dense = YOLO(str(checkpoint))   
    model_prep = YOLO(str(checkpoint))    
    dummy = cv2.imread(str(val_images[0]))
    model_prep.predict([dummy], verbose=False, device=0)  

    object_csv_path = REPO_ROOT / cfg["output"]["object_level_csv"]
    image_csv_path = REPO_ROOT / cfg["output"]["image_level_csv"]
    object_csv_path.parent.mkdir(parents=True, exist_ok=True)
    image_csv_path.parent.mkdir(parents=True, exist_ok=True)

    image_fields = (["image_id", "n_anchor_objects", "n_gt_objects", "n_false_negatives"] +
                     [f"agg_{m}" for m in cfg["aggregation"]["methods"]])

    save_every = cfg["output"]["save_every"]
    start_time = time.time()
    n_objects_total = 0

    with open(object_csv_path, "w", newline="", encoding="utf-8") as f_obj, \
         open(image_csv_path, "w", newline="", encoding="utf-8") as f_img:

        obj_writer = csv.DictWriter(f_obj, fieldnames=OBJECT_FIELDS)
        img_writer = csv.DictWriter(f_img, fieldnames=image_fields)
        obj_writer.writeheader()
        img_writer.writeheader()

        for i, img_path in enumerate(val_images):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"AVISO: no se pudo leer {img_path}, se omite")
                continue
            img_h, img_w = img_bgr.shape[:2]
            txt_path = val_labels_dir / f"{img_path.stem}.txt"
            gt_labels = (load_kitti_yolo_bboxes(str(txt_path), img_h, img_w,
                                                 return_class=True, box_format="xyxy")
                         if txt_path.exists() else [])

            object_rows, image_row = process_image_for_uq(
                model_dense, model_prep, img_bgr, gt_labels, img_path.stem, rng, cfg,
            )

            for row in object_rows:
                obj_writer.writerow(row)
            img_writer.writerow(image_row)
            n_objects_total += len(object_rows)

            if (i + 1) % save_every == 0 or (i + 1) == len(val_images):
                f_obj.flush()
                f_img.flush()
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(val_images) - (i + 1))
                print(f"[{i + 1}/{len(val_images)}] {n_objects_total} objetos, "
                      f"{elapsed:.0f}s transcurridos, ETA {eta:.0f}s")

    print(f"Finished in {time.time() - start_time:.0f}s. Objects: {n_objects_total}. Saved in:\n"
          f"  {object_csv_path}\n  {image_csv_path}")


if __name__ == "__main__":
    main()