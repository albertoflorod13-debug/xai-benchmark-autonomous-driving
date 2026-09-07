"""CLI entry point: YOLO26 fine-tuning on LOCO, Stage 2.

Reuses the exact same two-stage recipe already validated on KITTI (see
train_finetune_stage2.py). Stage 2: unfreeze all layers, continue from
Stage 1's (LOCO) best.pt, low lr0 with optimizer set explicitly
(optimizer=auto would silently ignore a manual lr0). momentum=0.9 set
explicitly too, matching Ultralytics' own auto-AdamW branch (see
train_finetune_stage2.py's docstring for the full rationale -- unchanged
here, carried over verbatim).

Writes to finetune_stage2_unfrozen_loco/ (distinct name from KITTI's
finetune_stage2_unfrozen/) so this run never overwrites the completed KITTI
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

STAGE1_CHECKPOINT = REPO_ROOT / "results" / "runs" / "detect" / "finetune_stage1_freeze10_loco" / "weights" / "best.pt"
assert STAGE1_CHECKPOINT.exists(), f"Stage 1 (LOCO) checkpoint not found: {STAGE1_CHECKPOINT}"

RUNS_DIR = REPO_ROOT / "results" / "runs" / "detect"


def main() -> None:
    model = YOLO(str(STAGE1_CHECKPOINT))

    start = time.perf_counter()
    model.train(
        data=str(LOCO_YAML_LOCAL),
        epochs=25,
        imgsz=640,
        batch=16,
        rect=False,
        cls_pw=0.5,
        optimizer="AdamW",
        lr0=0.001,
        momentum=0.9,
        patience=10,
        workers=4,
        project=str(RUNS_DIR),
        name="finetune_stage2_unfrozen_loco",
        exist_ok=True,
        device=0,
    )
    elapsed = time.perf_counter() - start
    print(f"Stage 2 (LOCO) completed in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
