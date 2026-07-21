"""CLI entry point: SSGrad-CAM++ explanation generation.

It iterates through the entire KITTI value set, obtains the detections for each image 
(using `get_one2many_predictions`), and generates the SSGrad-CAM++ heatmap for each image 
(using `SSGradCAMPP.explain`, `xai/ssgradcampp.py`). It saves:
- one .npz file per image with the raw heatmap (not rescaled to the full image).
- one row per object in `object_level.csv`, with the same matching against the actual GT as 
`results/tta_uq/object_level.csv`. 

Configuration: `configs/xai/ssgradcampp.yaml`
"""

import argparse
import csv
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
from xai_benchmark.xai.ssgradcampp import SSGradCAMPP

OBJECT_FIELDS = [
    "image_id", "obj_idx", "pred_class", "pred_class_name", "confidence",
    "x1", "y1", "x2", "y2", "scale_idx", "row", "col",
    "letterbox_h", "letterbox_w",
    "gt_class", "gt_class_name", "gt_iou", "correctness",
]


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N images (for quick tests)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(REPO_ROOT / "configs" / "xai" / "ssgradcampp.yaml")

    val_images_dir = REPO_ROOT / cfg["data"]["val_images_dir"]
    val_labels_dir = REPO_ROOT / cfg["data"]["val_labels_dir"]
    checkpoint = REPO_ROOT / cfg["checkpoint"]
    assert checkpoint.exists(), f"Checkpoint not found: {checkpoint}"

    val_images = sorted(val_images_dir.glob("*.png"))
    assert val_images, f"No images were found in {val_images_dir}"
    if args.limit:
        val_images = val_images[: args.limit]
        print(f"TEST MODE: {len(val_images)} images")
    else:
        print(f"Processing {len(val_images)} images of {val_images_dir}")

    device = cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    explainer = SSGradCAMPP.from_checkpoint(
        checkpoint, device=device,
        iou_match_thres=cfg["detection"]["iou_match_thres"],
        eps=cfg["numerics"]["eps"],
    )
    margin = cfg["mask"]["margin"]
    conf_thres = cfg["detection"]["conf_thres"]
    iou_thres_nms = cfg["detection"]["iou_thres_nms"]
    iou_thres_gt = cfg["matching"]["iou_thres_gt"]

    heatmaps_dir = REPO_ROOT / cfg["output"]["heatmaps_dir"]
    object_csv_path = REPO_ROOT / cfg["output"]["object_level_csv"]
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    object_csv_path.parent.mkdir(parents=True, exist_ok=True)

    save_every = cfg["output"]["save_every"]
    start_time = time.time()
    n_objects_total = 0

    with open(object_csv_path, "w", newline="", encoding="utf-8") as f_obj:
        writer = csv.DictWriter(f_obj, fieldnames=OBJECT_FIELDS)
        writer.writeheader()

        for i, img_path in enumerate(val_images):
            image_id = img_path.stem
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
                explainer.model, imgs_tensor, [(img_h, img_w)],
                conf_thres=conf_thres, iou_thres=iou_thres_nms,)[0]
            dets.boxes = dets.boxes.cpu()
            dets.class_probs = dets.class_probs.cpu()
            dets.max_conf = dets.max_conf.cpu()

            if len(dets.boxes) == 0:
                continue

            gt_matches, _ = match_detections_to_gt(dets, gt_labels, iou_thres_gt)
            heatmaps_out = {}

            for obj_idx in range(len(dets.boxes)):
                target_box = dets.boxes[obj_idx].tolist()
                target_class = int(dets.class_probs[obj_idx].argmax())
                confidence = float(dets.max_conf[obj_idx])
                gt_class, gt_iou, correctness = gt_matches[obj_idx]

                try:
                    result = explainer.explain(img_bgr, target_box, target_class, margin=margin)
                except RuntimeError as e:
                    print(f"WARNING: {image_id} obj {obj_idx} omitted ({e})")
                    continue

                heatmaps_out[f"obj_{obj_idx}"] = result.heatmap_raw.astype(np.float32)
                writer.writerow({
                    "image_id": image_id, "obj_idx": obj_idx,
                    "pred_class": target_class, "pred_class_name": KITTI_CLASSES.get(target_class, "?"),
                    "confidence": confidence,
                    "x1": target_box[0], "y1": target_box[1], "x2": target_box[2], "y2": target_box[3],
                    "scale_idx": result.scale_idx, "row": result.row, "col": result.col,
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
    device = cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}")