"""CLI entry point: YOLO26 fine-tuning on LOCO, Stage 1.

Reuses the exact same two-stage recipe already validated on KITTI (see
train_finetune_stage1.py) -- this run is not meant to find LOCO-optimal
hyperparameters, only to produce a working LOCO detector for the industrial
use case (path planning + explainability demo). Stage 1: freeze=10 freezes
the backbone (layers 0-9), leaves C2PSA + neck + head trainable. Base config
(batch=16, cls_pw=0.5, rect=False) is the winning KITTI config from Pruebas
1-5, carried over unchanged per that decision.

Starts from the same fresh COCO-pretrained weights as the KITTI run (not
from the KITTI-finetuned checkpoint) -- this is an independent fine-tuning
target, not a continuation of the KITTI model.

Writes to finetune_stage1_freeze10_loco/ (distinct name from KITTI's
finetune_stage1_freeze10/) so this run never overwrites the completed KITTI
checkpoint that the whole XAI benchmark depends on.
"""

import time
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent

LOCO_YAML_LOCAL = REPO_ROOT / "data" / "LOCO" / "loco_local.yaml"
assert LOCO_YAML_LOCAL.exists(), (
    f"{LOCO_YAML_LOCAL} does not exist -- run scripts/convert_loco_to_yolo.py first."
)

PRETRAINED = REPO_ROOT / "models" / "pretrained" / "yolo26n.pt"
RUNS_DIR = REPO_ROOT / "results" / "runs" / "detect"


def main() -> None:
    model = YOLO(str(PRETRAINED))  # fresh COCO-pretrained weights

    start = time.perf_counter()
    model.train(
        data=str(LOCO_YAML_LOCAL),
        epochs=25,
        imgsz=640,
        batch=16,
        rect=False,
        cls_pw=0.5,
        freeze=10,
        patience=10,
        workers=4,
        project=str(RUNS_DIR),
        name="finetune_stage1_freeze10_loco",
        exist_ok=True,
        device=0,
    )
    elapsed = time.perf_counter() - start
    print(f"Stage 1 (LOCO) completed in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
