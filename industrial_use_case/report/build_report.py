"""
build_report.py
=================
Builds a single self-contained HTML report from a finished demo run
(results/industrial_use_case/session.json): for every conflict the demo
encountered, shows the original camera frame, the D-CRISP and SSGrad-CAM++
heatmaps for the object that triggered it, and each method's fidelity/
localisation metrics side by side.
"""

import base64
import json
import sys
from pathlib import Path
from ultralytics import YOLO

import cv2
import numpy as np
import torch
import yaml
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from xai_benchmark.evaluation import detection_metrics, quantus_metrics
from xai_benchmark.xai.dcrisp import DCRISP
from xai_benchmark.xai.ssgradcampp import SSGradCAMPP
from industrial_use_case.execution.conflict_checker import CLASS_NAMES
from industrial_use_case.report.route_visualizer import build_route_visualization_html

SESSION_PATH = REPO_ROOT / "results" / "industrial_use_case" / "session.json"
LOCO_CHECKPOINT = REPO_ROOT / "models" / "finetuned_loco" / "best.pt"
DCRISP_CONFIG = REPO_ROOT / "configs" / "xai" / "dcrisp.yaml"
SSGRADCAMPP_CONFIG = REPO_ROOT / "configs" / "xai" / "ssgradcampp.yaml"
OUTPUT_HTML = REPO_ROOT / "results" / "industrial_use_case" / "report" / "report.html"

PLOTLY_VERSION = "2.27.0"
PLOTLY_URLS = [
    f"https://cdnjs.cloudflare.com/ajax/libs/plotly.js/{PLOTLY_VERSION}/plotly-cartesian.min.js",
    f"https://cdnjs.cloudflare.com/ajax/libs/plotly.js/{PLOTLY_VERSION}/plotly.min.js",
]
PLOTLY_LOCAL_PATH = OUTPUT_HTML.parent / "plotly.min.js"


def _ensure_plotly_bundle() -> None:
    """Downloads Plotly.js next to the report once, so the report renders
    without depending on the viewer's browser reaching a CDN at view-time."""
    if PLOTLY_LOCAL_PATH.exists():
        return
    PLOTLY_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in PLOTLY_URLS:
        try:
            urllib.request.urlretrieve(url, PLOTLY_LOCAL_PATH)
            return
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Could not download Plotly.js: {last_error}")


METRIC_LABELS = [
    ("pointing_game", "Pointing Game"),
    ("ebpg", "Energy-Based Pointing Game"),
    ("relevance_rank_accuracy", "Relevance Rank Accuracy"),
    ("sparseness", "Sparseness"),
    ("deletion_auc", "Deletion AUC"),
    ("insertion_auc", "Insertion AUC"),
    ("minimal_subset", "Minimal Subset"),
]

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Report: Path Planning + XAI</title>
<script src="plotly.min.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #f7f7f9;
          color: #1c1c1e; margin: 0; padding: 2rem; }}
  h1 {{ text-align: center; font-weight: 600; margin-bottom: 0.25rem; }}
  .subtitle {{ text-align: center; color: #6b6b70; margin-bottom: 2.5rem; }}
  section.conflict, section.route-viz {{ background: #fff; border-radius: 12px;
                       box-shadow: 0 1px 4px rgba(0,0,0,0.08);
                       padding: 1.5rem 2rem; margin: 0 auto 2.5rem auto; max-width: 900px; }}
  h2 {{ margin-top: 0; font-size: 1.2rem; }}
  p.meta {{ color: #6b6b70; font-size: 0.9rem; margin-top: -0.5rem; }}
  figure {{ margin: 0; text-align: center; }}
  figure.original img {{ max-width: 100%; border-radius: 8px; }}
  figure figcaption {{ font-size: 0.85rem; color: #6b6b70; margin-top: 0.3rem; }}
  .heatmaps {{ display: flex; gap: 1.5rem; justify-content: center; margin: 1.2rem 0; }}
  .heatmaps figure {{ flex: 1; }}
  .heatmaps img {{ max-width: 100%; border-radius: 8px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #e5e5ea; }}
  th {{ color: #6b6b70; font-weight: 600; }}
  .tabs {{ display: flex; gap: 0.5rem; margin-bottom: 1rem; }}
  .tab-btn {{ padding: 0.4rem 1rem; border: 1px solid #d0d0d5; border-radius: 8px; background: #f0f0f3;
              color: #1c1c1e; cursor: pointer; font-size: 0.9rem; }}
  .tab-btn.active {{ background: #1c1c1e; color: #fff; border-color: #1c1c1e; }}
  @media print {{
    .tabs {{ display: none !important; }}
    #route-tab-anim {{ display: none !important; }}
    #route-tab-route {{ display: block !important; height: 520px !important; }}
  }}
</style>
</head>
<body>
<h1>Report: Path Planning + XAI</h1>
<p class="subtitle">{summary}</p>
{route_viz}
{sections}
</body>
</html>
"""


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _encode_png(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    assert ok, "PNG encoding failed"
    return base64.b64encode(buf).decode("ascii")


def _heatmap_overlay(img_bgr: np.ndarray, heatmap: np.ndarray, box_xyxy, alpha: float = 0.45) -> np.ndarray:
    """`img_bgr` with `heatmap` ([0,1], original-image resolution) blended on top
    (jet colormap) and the target box outlined in white."""
    heatmap_u8 = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(colored, alpha, img_bgr, 1 - alpha, 0)
    x1, y1, x2, y2 = (int(round(v)) for v in box_xyxy)
    cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return blended


def _compute_metrics(heatmap_orig: np.ndarray, heatmap_raw: np.ndarray, box_xyxy, target_class: int,
                      classifier, x_np: np.ndarray, letterbox_shape: tuple, orig_shape: tuple,
                      conf_thres: float, step: int, device: str) -> dict:
    box_mask_orig = detection_metrics.build_box_mask(box_xyxy, *orig_shape)
    heatmap_lb = detection_metrics.reconstruct_heatmap_letterboxed(heatmap_raw, letterbox_shape)
    aucs = detection_metrics.deletion_insertion_auc(
        classifier, x_np, heatmap_lb, target_class, step=step, device=device)

    return {
        "pointing_game": quantus_metrics.pointing_game(heatmap_orig, box_mask_orig),
        "ebpg": quantus_metrics.energy_based_pointing_game(heatmap_orig, box_mask_orig),
        "relevance_rank_accuracy": quantus_metrics.relevance_rank_accuracy(heatmap_orig, box_mask_orig),
        "sparseness": quantus_metrics.sparseness(heatmap_orig),
        "deletion_auc": aucs["deletion_auc"],
        "insertion_auc": aucs["insertion_auc"],
        "minimal_subset": detection_metrics.minimal_subset(aucs["deletion_scores"], k_threshold=conf_thres),
    }


def _metrics_table_html(dcrisp_metrics: dict, ssg_metrics: dict) -> str:
    rows = "\n".join(
        f"<tr><td>{label}</td><td>{dcrisp_metrics[key]:.4f}</td><td>{ssg_metrics[key]:.4f}</td></tr>"
        for key, label in METRIC_LABELS
    )
    return (
        "<table><thead><tr><th>Metric</th><th>D-CRISP</th><th>SSGrad-CAM++</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _conflict_section_html(idx: int, entry: dict, original_b64: str, dcrisp_b64: str, ssg_b64: str,
                            dcrisp_metrics: dict, ssg_metrics: dict) -> str:
    conflict = entry["conflict"]
    class_name = CLASS_NAMES.get(conflict["target_class"], "unknown")
    position_str = f"({entry['position'][0]:.2f}, {entry['position'][1]:.2f})"
    return f"""
    <section class="conflict">
      <h2>Conflict {idx} &mdash; step {entry['step_index']}</h2>
      <p class="meta">Object: {class_name} &middot; confidence: {conflict['confidence']:.2f}
         &middot; relative area: {conflict['relative_area']:.3f} &middot; position: {position_str}</p>
      <figure class="original">
        <img src="data:image/png;base64,{original_b64}" alt="Original frame">
        <figcaption>Original frame</figcaption>
      </figure>
      <div class="heatmaps">
        <figure>
          <img src="data:image/png;base64,{dcrisp_b64}" alt="D-CRISP heatmap">
          <figcaption>D-CRISP</figcaption>
        </figure>
        <figure>
          <img src="data:image/png;base64,{ssg_b64}" alt="SSGrad-CAM++ heatmap">
          <figcaption>SSGrad-CAM++</figcaption>
        </figure>
      </div>
      {_metrics_table_html(dcrisp_metrics, ssg_metrics)}
    </section>
    """


def main() -> None:
    _ensure_plotly_bundle()
    with open(SESSION_PATH, "r", encoding="utf-8") as f:
        session = json.load(f)

    conflicts = session["conflict_log"]
    route_viz = build_route_visualization_html(session)
    sections = []

    if conflicts:
        dcrisp_cfg = _load_yaml(DCRISP_CONFIG)
        ssg_cfg = _load_yaml(SSGRADCAMPP_CONFIG)
        device = dcrisp_cfg["device"]
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        conf_thres = dcrisp_cfg["detection"]["conf_thres"]
        iou_thres_nms = dcrisp_cfg["detection"]["iou_thres_nms"]
        step = dcrisp_cfg["evaluation"]["step"]

        model_dense = YOLO(str(LOCO_CHECKPOINT))
        model_prep = YOLO(str(LOCO_CHECKPOINT))
        model_dense.model.to(device)
        model_dense.model.eval()
        num_classes = model_dense.model.model[-1].nc
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model_prep.predict([dummy], verbose=False, device=0 if device == "cuda" else "cpu")

        dcrisp_kwargs = dict(
            conf_thres=conf_thres, iou_thres_nms=iou_thres_nms,
            gpu_batch=dcrisp_cfg["mask"]["gpu_batch"], n_masks=dcrisp_cfg["mask"]["n_masks"],
            alpha=dcrisp_cfg["mask"]["alpha"], resolution=dcrisp_cfg["mask"]["resolution"],
            p1=dcrisp_cfg["mask"]["p1"], num_levels=dcrisp_cfg["mask"]["num_levels"],
        )
        ssgradcampp = SSGradCAMPP.from_checkpoint(
            str(LOCO_CHECKPOINT), device=device, iou_match_thres=ssg_cfg["detection"]["iou_match_thres"],
            eps=ssg_cfg["numerics"]["eps"],
        )
        ssg_margin = ssg_cfg["mask"]["margin"]

        for idx, entry in enumerate(conflicts, start=1):
            img_bgr = cv2.imread(entry["image_path"])
            if img_bgr is None:
                print(f"Could not read {entry['image_path']}, skipping conflict {idx}")
                continue
            orig_shape = img_bgr.shape[:2]

            conflict = entry["conflict"]
            target_box = tuple(conflict["target_box"])
            target_class = conflict["target_class"]
            target_class_probs = torch.tensor(conflict["target_class_probs"])

            x_tensor = model_prep.predictor.preprocess([img_bgr]).to(device)
            letterbox_shape = tuple(x_tensor.shape[2:])
            x_np = x_tensor.cpu().numpy().astype(np.float32)

            dcrisp = DCRISP(model_dense, model_prep, device=device, **dcrisp_kwargs)
            dcrisp_result = dcrisp.explain(img_bgr, [(target_box, target_class, target_class_probs)])[0]
            ssg_result = ssgradcampp.explain(img_bgr, target_box, target_class, margin=ssg_margin)

            classifier = detection_metrics.DetectorAsClassifier(
                model_dense, num_classes, orig_shape, target_box, target_class,
                conf_thres=conf_thres, iou_thres_nms=iou_thres_nms,
            )
            dcrisp_metrics = _compute_metrics(
                dcrisp_result.heatmap, dcrisp_result.heatmap_raw, target_box, target_class,
                classifier, x_np, letterbox_shape, orig_shape, conf_thres, step, device,
            )
            ssg_metrics = _compute_metrics(
                ssg_result.heatmap, ssg_result.heatmap_raw, target_box, target_class,
                classifier, x_np, letterbox_shape, orig_shape, conf_thres, step, device,
            )

            original_b64 = _encode_png(img_bgr)
            dcrisp_b64 = _encode_png(_heatmap_overlay(img_bgr, dcrisp_result.heatmap, target_box))
            ssg_b64 = _encode_png(_heatmap_overlay(img_bgr, ssg_result.heatmap, target_box))

            sections.append(_conflict_section_html(
                idx, entry, original_b64, dcrisp_b64, ssg_b64, dcrisp_metrics, ssg_metrics,
            ))
    else:
        print("No conflicts recorded in this session -- report will only show the route.")

    summary = (
        f"{len(conflicts)} conflict(s) &middot; {session['num_replans']} replan(s) &middot; "
        f"{session['total_steps']} step(s) &middot; target reached: {session['reached_target']}"
    )
    html = PAGE_TEMPLATE.format(summary=summary, route_viz=route_viz, sections="\n".join(sections))

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report written to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()