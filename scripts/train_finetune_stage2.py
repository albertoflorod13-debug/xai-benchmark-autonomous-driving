"""CLI entry point: YOLO26 fine-tuning on KITTI, Stage 2.

Stage 2 of the two-stage plan: unfreeze all layers,
continue from Stage 1's best.pt, low lr0 with optimizer set explicitly (optimizer=auto would
silently ignore a manual lr0). momentum=0.9 set explicitly too: when optimizer
is not "auto", Ultralytics passes momentum straight into AdamW's betas=(momentum, 0.999)
(engine/trainer.py::build_optimizer) -- the config default (0.937) is meant for SGD, not Adam;
Ultralytics' own auto-AdamW branch uses 0.9, so we match that here instead of leaving 0.937.
"""

import time
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent

KITTI_YAML_LOCAL = REPO_ROOT / "data" / "kitti" / "kitti_local.yaml"
assert KITTI_YAML_LOCAL.exists(), (
    f"{KITTI_YAML_LOCAL} does not exist -- generate it first (see notebooks/tests.ipynb, "
    "cell 'Nota tecnica: ruta del dataset KITTI')."
)

STAGE1_CHECKPOINT = REPO_ROOT / "results" / "runs" / "detect" / "finetune_stage1_freeze10" / "weights" / "best.pt"
assert STAGE1_CHECKPOINT.exists(), f"Stage 1 checkpoint not found: {STAGE1_CHECKPOINT}"

RUNS_DIR = REPO_ROOT / "results" / "runs" / "detect"


def main() -> None:
    model = YOLO(str(STAGE1_CHECKPOINT))

    start = time.perf_counter()
    model.train(
        data=str(KITTI_YAML_LOCAL),
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
        name="finetune_stage2_unfrozen",
        exist_ok=True,
        device=0,
    )
    elapsed = time.perf_counter() - start
    print(f"Stage 2 completed in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()