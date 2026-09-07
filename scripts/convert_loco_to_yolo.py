"""CLI entry point: converts LOCO's COCO-format annotations to a YOLO-ready dataset.

Reads loco-all-v1.json (COCO format) plus the raw subset-1..subset-5 image folders as
downloaded from https://github.com/tum-fml/loco, and produces the standard YOLO layout
Ultralytics expects: images/{train,val}/*.jpg + labels/{train,val}/*.txt + a dataset yaml.

Not built on ultralytics.data.converter.convert_coco: LOCO's category ids are sparse
({3, 5, 7, 10, 11}, not a contiguous 0- or 1-based range), and convert_coco's own class
index is `category_id - 1` (cls91to80=False branch) -- applied here that would yield class
indices {2, 4, 6, 9, 10}, not {0..4}, silently breaking training. convert_coco also expects
images already sitting in images/<split>/, which is not the case here (LOCO's own subset
folders nest inconsistently by recording session and camera). Both problems are solved here
in one auditable pass instead of pre/post-processing around convert_coco.

Official LOCO benchmark split (Mayershofer et al., ICMLA 2020, Sec. IV-B): train = subsets
{2, 3, 5}, val = subsets {1, 4} -- by warehouse, not a random image-level split.
"""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCO_ROOT = REPO_ROOT / "data" / "LOCO"
JSON_PATH = LOCO_ROOT / "loco-all-v1.json"
IMAGES_SEARCH_ROOT = LOCO_ROOT  # subset-1 .. subset-5 live directly under here

# Ascending-original-id order, deterministic: {old_category_id: new_yolo_class}.
CATEGORY_REMAP = {3: 0, 5: 1, 7: 2, 10: 3, 11: 4}
CLASS_NAMES = {
    0: "small_load_carrier",
    1: "forklift",
    2: "pallet",
    3: "stillage",
    4: "pallet_truck",
}

TRAIN_SUBSETS = {"2", "3", "5"}
VAL_SUBSETS = {"1", "4"}

SUBSET_RE = re.compile(r"subset-(\d+)")


def build_filename_index(search_root: Path) -> dict[str, Path]:
    """Maps every *.jpg filename under search_root to its real path, recursively.

    Needed because subset-1 stores images flat while subset-2..5 nest them by recording
    session and camera (inconsistently named: Kinect, RealSense, Cam1/cam1, ...). Filenames
    are timestamps and were verified globally unique across all 5,593 downloaded images, so
    this lookup is unambiguous -- but we assert that here too, rather than trust it silently.
    """
    index: dict[str, Path] = {}
    for jpg_path in search_root.glob("subset-*/**/*.jpg"):
        name = jpg_path.name
        assert name not in index, f"Duplicate filename found on disk: {name} ({index[name]} vs {jpg_path})"
        index[name] = jpg_path
    return index


def main() -> None:
    assert JSON_PATH.exists(), f"{JSON_PATH} does not exist -- place loco-all-v1.json there first."

    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    actual_cat_ids = {c["id"] for c in data["categories"]}
    assert actual_cat_ids == set(CATEGORY_REMAP), (
        f"loco-all-v1.json's category ids {sorted(actual_cat_ids)} do not match the ids this "
        f"script was verified against {sorted(CATEGORY_REMAP)} -- do not proceed blindly, the "
        f"CATEGORY_REMAP table above must be re-derived for a different LOCO release."
    )

    file_index = build_filename_index(IMAGES_SEARCH_ROOT)

    annotations_by_image = defaultdict(list)
    for ann in data["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    for split in ("train", "val"):
        (LOCO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (LOCO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_images = {"train": 0, "val": 0}
    n_instances = {"train": 0, "val": 0}
    class_counts = {"train": {}, "val": {}}

    for img in data["images"]:
        m = SUBSET_RE.search(img["path"])
        assert m, f"Image {img['file_name']} has no 'subset-N' in its path field: {img['path']}"
        subset = m.group(1)
        if subset in TRAIN_SUBSETS:
            split = "train"
        elif subset in VAL_SUBSETS:
            split = "val"
        else:
            raise ValueError(f"Unexpected subset '{subset}' for image {img['file_name']}")

        file_name = img["file_name"]
        assert file_name in file_index, f"{file_name} listed in JSON but not found on disk"
        src_path = file_index[file_name]

        dst_image_path = LOCO_ROOT / "images" / split / file_name
        shutil.copy2(src_path, dst_image_path)

        w, h = img["width"], img["height"]
        lines = []
        for ann in annotations_by_image[img["id"]]:
            x, y, bw, bh = ann["bbox"]
            assert bw > 0 and bh > 0, f"Non-positive bbox size in annotation {ann['id']}"
            cls = CATEGORY_REMAP[ann["category_id"]]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            nw, nh = bw / w, bh / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            class_counts[split][cls] = class_counts[split].get(cls, 0) + 1

        label_path = LOCO_ROOT / "labels" / split / (Path(file_name).stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        n_images[split] += 1
        n_instances[split] += len(lines)

    yaml_path = LOCO_ROOT / "loco_local.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {LOCO_ROOT}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for idx, name in CLASS_NAMES.items():
            f.write(f"  {idx}: {name}\n")

    for split in ("train", "val"):
        print(f"{split}: {n_images[split]} images, {n_instances[split]} instances, "
              f"class breakdown {class_counts[split]}")
    print(f"Wrote {yaml_path}")

    assert n_images["train"] + n_images["val"] == len(data["images"])
    assert n_instances["train"] + n_instances["val"] == len(data["annotations"])
    print("Self-check OK: image and instance totals match the source JSON exactly.")


if __name__ == "__main__":
    main()