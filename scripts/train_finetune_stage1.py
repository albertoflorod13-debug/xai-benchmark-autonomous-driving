"""CLI entry point: YOLO26 fine-tuning on KITTI, Stage 1.

Stage 1 of the two-stage plan: freeze=10 freezes the
backbone (layers 0-9), leaves C2PSA + neck + head trainable (14 of 24 layers). epochs=25 and
patience=10 (early stop if no fitness gain for 10 epochs) follow the official two-stage worked example from docs.ultralytics.com/guides/finetuning-guide 
Base config (batch=16, cls_pw=0.5,rect=False) is the winning config from Pruebas 1-5.
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

PRETRAINED = REPO_ROOT / "models" / "pretrained" / "yolo26n.pt"
RUNS_DIR = REPO_ROOT / "results" / "runs" / "detect"


def main() -> None:
    model = YOLO(str(PRETRAINED))  # fresh COCO-pretrained weights

    start = time.perf_counter()
    model.train(
        data=str(KITTI_YAML_LOCAL),
        epochs=25,     
        imgsz=640,
        batch=16,       
        rect=False,    
        cls_pw=0.5,     
        freeze=10,   
        patience=10,    
        workers=4,
        project=str(RUNS_DIR),
        name="finetune_stage1_freeze10",
        exist_ok=True,
        device=0,     
    )
    elapsed = time.perf_counter() - start
    print(f"Stage 1 completed in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()