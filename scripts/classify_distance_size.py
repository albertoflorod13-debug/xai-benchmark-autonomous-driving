"""Classifies every object detected by our model (YOLO26) by its real distance
to the camera (KITTI ground truth) and by the size of its bounding box.

Inputs:
- results/ssgradcampp/object_level.csv
    All objects detected by our model on the val set (columns image_id,
    obj_idx, x1, y1, x2, y2 in original-image pixel coordinates).
- data/data_object_label_2/training/label_2/<image_id>.txt
    Original KITTI labels (ground truth), one line per real object.
    Columns 5-8 = 2D box (left, top, right, bottom). Column 14 = real
    distance z in meters (part of "location x,y,z in camera coordinates").
    Format verified against the official KITTI devkit.

Both the distance and the box area are turned into a 3-class label
(near/medium/far and small/medium/large) with K-Means (k=3), the same
criterion already used in xai_evaluation_analysis.ipynb for box size: the 3
clusters are relabeled from smallest to largest mean value.

Output:
- results/object_distance_size.csv with columns:
    image_id, obj_idx, distance_m, dist_classify, size_bin, size_classify
"""
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from xai_benchmark.uncertainty.tta import box_iou

DETECTIONS_CSV = REPO_ROOT / "results" / "ssgradcampp" / "object_level.csv"
LABEL_2_DIR = REPO_ROOT / "data" / "data_object_label_2" / "training" / "label_2"
OUTPUT_CSV = REPO_ROOT / "results" / "object_distance_size.csv"

IOU_THRES_GT = 0.5  # same criterion as ssgradcampp.yaml / tta_uq.yaml / dcrisp.yaml


def load_kitti_ground_truth(label_2_path: Path):
    """Reads a KITTI label_2 file and returns, for every real object, its 2D box and its real
    distance: a list of tuples (x1, y1, x2, y2, distance_m)."""
    ground_truth_objects = []
    with open(label_2_path, encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            object_type = fields[0]
            if object_type == "DontCare":
                continue
            x1, y1, x2, y2 = (float(v) for v in fields[4:8])
            distance_m = float(fields[13])  # The object's depth is in column 14
            ground_truth_objects.append((x1, y1, x2, y2, distance_m))
    return ground_truth_objects


def find_real_distance_per_detection(detections: pd.DataFrame) -> pd.Series:
    """For every detected object, finds the real KITTI object with the highest
    IoU in the same image and returns its real distance. If no real object
    reaches IOU_THRES_GT, returns NaN."""
    distance_per_detection = pd.Series(index=detections.index, dtype=float)

    for image_id, detections_of_this_image in detections.groupby("image_id"):
        label_2_path = LABEL_2_DIR / f"{image_id}.txt"
        if not label_2_path.exists():
            continue

        ground_truth_objects = load_kitti_ground_truth(label_2_path)
        if not ground_truth_objects:
            continue

        detected_boxes = torch.tensor(
            detections_of_this_image[["x1", "y1", "x2", "y2"]].to_numpy(), dtype=torch.float32
        )
        gt_boxes = torch.tensor([obj[:4] for obj in ground_truth_objects], dtype=torch.float32)
        gt_distances = [obj[4] for obj in ground_truth_objects]

        best_iou, best_gt_idx = box_iou(detected_boxes, gt_boxes).max(dim=1)

        for row_position, row_index in enumerate(detections_of_this_image.index):
            if best_iou[row_position] >= IOU_THRES_GT:
                distance_per_detection[row_index] = gt_distances[best_gt_idx[row_position]]

    return distance_per_detection


def classify_with_kmeans(values: pd.Series, class_names: list) -> pd.Series:
    """Generic 3-class K-Means classification (k=3, random_state=0, n_init=10). 
    The 3 clusters are relabeled with
    class_names according to their mean value, from smallest to largest.
    Rows with a missing (NaN) value are left unclassified (NaN)."""
    classification = pd.Series(index=values.index, dtype=object)
    valid = values.notna()

    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(values[valid].to_numpy().reshape(-1, 1))

    mean_value_per_cluster = values[valid].groupby(kmeans.labels_).mean()
    cluster_order = mean_value_per_cluster.sort_values().index.tolist()
    name_per_cluster = dict(zip(cluster_order, class_names))

    classification[valid] = [name_per_cluster[label] for label in kmeans.labels_]
    return classification


def main():
    detections = pd.read_csv(DETECTIONS_CSV, dtype={"image_id": str})

    detections["distance_m"] = find_real_distance_per_detection(detections)
    detections["dist_classify"] = classify_with_kmeans(detections["distance_m"], ["near", "medium", "far"])

    detections["size_bin"] = (detections["x2"] - detections["x1"]) * (detections["y2"] - detections["y1"])
    detections["size_classify"] = classify_with_kmeans(detections["size_bin"], ["small", "medium", "large"])

    result = detections[["image_id", "obj_idx", "distance_m", "dist_classify", "size_bin", "size_classify"]]
    result.to_csv(OUTPUT_CSV, index=False)

    print(f"{len(result)} objects processed.")
    print(f"Real distance found for {result['distance_m'].notna().sum()} of them.")
    print("Distance classification counts:")
    print(result["dist_classify"].value_counts())
    print("Size classification counts:")
    print(result["size_classify"].value_counts())
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()