"""CLI entry point: generates the random-noise baseline's heatmaps.

Writes one heatmaps_dir/<image_id>.npz per image, with one obj_{obj_idx} key per object,
each holding a (H, W) float32 array of i.i.d. Uniform(0, 1) noise at the model's own
letterboxed resolution for that image.

Reads output.object_level_csv from random_baseline.yaml, which must already be a copy of
either dcrisp's or ssgradcampp's own object_level.csv. This script assumes that population, it does not run
its own detection pass, and therefore needs no model_dense at all.

Usage:
    python scripts/run_random_baseline_heatmaps.py
    python scripts/run_random_baseline_heatmaps.py --limit 20
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from xai_benchmark.xai.random_explainer import RandomExplainer

CONFIG_PATH = REPO_ROOT / "configs" / "xai" / "random_baseline.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Only the first N images (quick tests).")
    return p.parse_args()


def main():
    args = parse_args()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint = REPO_ROOT / cfg["checkpoint"]
    assert checkpoint.exists(), f"Checkpoint no encontrado: {checkpoint}"
    object_csv_path = REPO_ROOT / cfg["output"]["object_level_csv"]
    assert object_csv_path.exists(), (
        f"{object_csv_path} does not exist -- copy dcrisp's or ssgradcampp's own "
        f"object_level.csv there first (verified identical, plan_fase_final.md Section 9)."
    )
    heatmaps_dir = REPO_ROOT / cfg["output"]["heatmaps_dir"]
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir = REPO_ROOT / cfg["data"]["val_images_dir"]
    save_every = cfg["output"].get("save_every", 100)
    noise_seed = cfg["noise"]["seed"]

    device = cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    from ultralytics import YOLO
    model_prep = YOLO(str(checkpoint))  # only used for preprocessing (letterbox_shape)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model_prep.predict([dummy], verbose=False, device=0 if device == "cuda" else "cpu")

    df = pd.read_csv(object_csv_path, dtype={"image_id": str})
    if args.limit:
        image_ids = df["image_id"].unique()[: args.limit]
        df = df[df["image_id"].isin(image_ids)].reset_index(drop=True)
        print(f"TEST MODE: {len(image_ids)} imagenes, {len(df)} objetos")
    else:
        print(f"Generating {df['image_id'].nunique()} images, {len(df)} objects")

    done_ids = {p.stem for p in heatmaps_dir.glob("*.npz")}
    if done_ids:
        print(f"Resuming: {len(done_ids)} images already generated, skipping them.")

    n_done = 0
    for img_counter, (image_id, group) in enumerate(df.groupby("image_id", sort=False), start=1):
        if image_id in done_ids:
            continue
        img_path = val_images_dir / f"{image_id}.png"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Could not be read {img_path}")
            continue

        x_tensor = model_prep.predictor.preprocess([img_bgr]).to(device)
        letterbox_shape = tuple(x_tensor.shape[2:])

        heatmaps = {}
        for idx, row in group.iterrows():
            obj_idx = int(row["obj_idx"])
            explainer = RandomExplainer(seed=(noise_seed, idx))
            heatmaps[f"obj_{obj_idx}"] = explainer.heatmap(letterbox_shape)
        np.savez(heatmaps_dir / f"{image_id}.npz", **heatmaps)
        n_done += len(heatmaps)

        if img_counter % save_every == 0:
            print(f"[{img_counter} images] {n_done} objects generated")

    print(f"Finished. Objects generated: {n_done}. Saved in {heatmaps_dir}")


if __name__ == "__main__":
    main()