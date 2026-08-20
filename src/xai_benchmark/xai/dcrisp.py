"""D-CRISP mask generation, adapted to YOLO26 on KITTI.

Extracted and adapted from https://github.com/aklein1995/D-CRISP, license CC BY-NC 4.0. Citation:
  Andres, A. and Del Ser, J. "D-CRISP: Explaining Object Detectors by Combining Randomized
  and Segment-based Perturbations." European Conference on Artificial Intelligence (ECAI), 2025.


This module only covers mask generation: the D-CRISP class combining these masks with the detector 
(compute_similarity, explain_image) composing DCRISPMaskGenerator as an attribute.

Two mask families, combined per Eq. 1 of the reference paper (N_r = alpha*N random, N_s = (1-alpha)*N
segmentation masks):
  - RISE masks: image-agnostic -- generated once, cached, and reused
    across every image, which is D-CRISP's main efficiency argument.
  - SLIC masks: content-dependent, regenerated per image.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import slic
from skimage.transform import resize
from dataclasses import dataclass

from xai_benchmark.xai.common import unletterbox_map
from xai_benchmark.uncertainty.tta import box_iou
from xai_benchmark.detection.yolo_head import get_one2many_predictions


class DCRISPMaskGenerator:
    """Generates and caches the two mask families D-CRISP combines.

    letterbox_shape must match the model's fixed preprocessed input size. This constancy is exactly what
    makes caching RISE masks across images valid.
    """

    SEGMENT_LEVELS = [50, 100, 200, 400, 800, 1600]

    def __init__(self, letterbox_shape: tuple[int, int]):
        self.letterbox_shape = letterbox_shape
        self.rise_masks: torch.Tensor | None = None  

    def generate_rise_masks(self, n_masks: int, resolution: int, p1: float) -> torch.Tensor:
        """RISE-style masks: a random (resolution x resolution) binary grid, upsampled with
        bilinear interpolation and randomly cropped back to letterbox_shape.

        Args:
            n_masks: number of masks to generate.
            resolution: grid side length before upsampling.
            p1: probability of a grid cell being kept visible (not occluded).

        Returns:
            (n_masks, 1, H, W) float32 tensor on CPU, values in [0, 1].
        """
        h, w = self.letterbox_shape
        cell_h, cell_w = math.ceil(h / resolution), math.ceil(w / resolution)
        up_h, up_w = (resolution + 1) * cell_h, (resolution + 1) * cell_w

        grid = (np.random.rand(n_masks, resolution, resolution) < p1).astype(np.float32)
        masks = np.empty((n_masks, h, w), dtype=np.float32)
        for i in range(n_masks):
            upsampled = resize(grid[i], (up_h, up_w), order=1, mode="reflect", anti_aliasing=False)
            off_h, off_w = np.random.randint(0, cell_h), np.random.randint(0, cell_w)
            masks[i] = upsampled[off_h : off_h + h, off_w : off_w + w]

        self.rise_masks = torch.from_numpy(masks).unsqueeze(1)
        return self.rise_masks

    def generate_slic_masks(self, img_rgb_uint8: np.ndarray, n_masks: int, p1: float,
                             num_levels: int) -> torch.Tensor:
        """Segmentation-based masks: at each of the first `num_levels` SLIC granularities in
        SEGMENT_LEVELS, grows a randomly shuffled subset of superpixels until their combined
        area reaches p1 * image area.

        Args:
            img_rgb_uint8: (H, W, 3) uint8 RGB image, already in the letterboxed frame.
            n_masks: masks requested. Actual count may be a few short of this if it doesn't
                divide evenly across num_levels -- use the returned tensor's own shape, not
                n_masks, as the true count.
            p1: target fraction of image area kept visible per mask.
            num_levels: how many of the 6 predefined granularities to use.

        Returns:
            (masks_per_level * num_levels, 1, H, W) float32 tensor on CPU.
        """
        h, w = self.letterbox_shape
        target_pixels = p1 * h * w
        masks_per_level = n_masks // num_levels
        if masks_per_level == 0:
            return torch.empty(0, 1, h, w)

        masks = []
        for n_segments in self.SEGMENT_LEVELS[:num_levels]:
            segments = slic(img_rgb_uint8, n_segments=n_segments)
            labels = np.unique(segments)
            for _ in range(masks_per_level):
                np.random.shuffle(labels)
                mask, covered = np.zeros((h, w), dtype=np.float32), 0
                for label in labels:
                    if covered >= target_pixels:
                        break
                    cell = segments == label
                    mask[cell] = 1.0
                    covered += cell.sum()
                masks.append(mask)

        return torch.from_numpy(np.stack(masks)).unsqueeze(1)

    def generate_combined_masks(self, img_rgb_uint8: np.ndarray, n_total: int, alpha: float,
                                 resolution: int, p1: float, num_levels: int,
                                 rise_cache_path: Path | None = None) -> torch.Tensor:
        """Combines RISE and SLIC masks in `alpha` proportion. RISE masks are generated only once
        and reused on every subsequent call.

        Args:
            rise_cache_path: optional .npy path to persist/reload RISE masks across separate
                script runs. Not needed within a single run; useful only for faster dev iteration.

        Returns:
            (n_rise + actual_n_slic, 1, H, W) float32 tensor.
        """
        n_rise = int(n_total * alpha)

        if self.rise_masks is not None:
            pass  # already cached in memory from an earlier image in this run
        elif rise_cache_path is not None and rise_cache_path.exists():
            self.rise_masks = torch.from_numpy(np.load(rise_cache_path)).float()
        else:
            self.generate_rise_masks(n_rise, resolution, p1)
            if rise_cache_path is not None:
                np.save(rise_cache_path, self.rise_masks.numpy())

        slic_masks = self.generate_slic_masks(img_rgb_uint8, n_total - n_rise, p1, num_levels)
        return torch.cat([self.rise_masks, slic_masks], dim=0)

def compute_similarity(target_box, target_class_probs: torch.Tensor, boxes: torch.Tensor,
                        class_probs: torch.Tensor, max_conf: torch.Tensor) -> float:
    """D-CRISP's similarity score (Eq. 6 of the paper): IoU(target, candidate) x
    cosine(P_target, P_candidate) x max_conf(candidate), maximised over every candidate
    detection found in one masked image.

    Args:
        target_box: (4,) xyxy, the object being explained (from the unperturbed image).
        target_class_probs: (nc,) full per-class sigmoid vector of the target detection.
        boxes: (M, 4) xyxy, candidate detections found in ONE masked image.
        class_probs: (M, nc) their per-class sigmoid vectors.
        max_conf: (M,) their max-class confidence.

    Returns:
        Similarity in [0, 1]; 0.0 if no candidate overlaps the target box at all.
    """
    if len(boxes) == 0:
        return 0.0
    target_box_t = torch.as_tensor(target_box, dtype=boxes.dtype, device=boxes.device).unsqueeze(0)
    ious = box_iou(target_box_t, boxes).squeeze(0)
    overlapping = ious > 0
    if not overlapping.any():
        return 0.0

    target_probs = target_class_probs.to(class_probs.device, class_probs.dtype)
    cosine = F.cosine_similarity(class_probs[overlapping], target_probs.unsqueeze(0), dim=-1)
    scores = ious[overlapping] * cosine.clamp(min=0) * max_conf[overlapping]
    return float(scores.max())

@torch.no_grad()
def explain_image(model_dense, imgs_tensor: torch.Tensor, orig_shape: tuple[int, int],
                   masks: torch.Tensor, targets: list[tuple[torch.Tensor, torch.Tensor]],
                   conf_thres: float, iou_thres_nms: float, gpu_batch: int,
                   device: str) -> list[np.ndarray]:
    """Builds one saliency map per target object in an image from a single pass over `masks`,
    instead of one pass per object (Eq. 7: per-mask max similarity via compute_similarity;
    Eq. 8: sum of similarity-weighted masks, max-normalised). Lossless: the model's output on
    a masked image never depends on which target we explain, so every target reuses the same
    cached predictions instead of triggering its own forward pass.

    Args:
        model_dense: YOLO instance, never `.predict()`/`.val()`-ed (see yolo_head.py).
        imgs_tensor: (1, 3, H, W) preprocessed (letterboxed) tensor of the current image.
        orig_shape: (height, width) of the original image, for box rescaling.
        masks: (n_masks, 1, H, W) CPU tensor from generate_combined_masks. n_masks is read
            from masks.shape[0], not assumed from a config value.
        targets: (target_box_xyxy, target_class_probs) per object, from
            get_one2many_predictions on the UNMASKED image (computed by the caller).
        gpu_batch: masks processed per forward pass; only this chunk moves to GPU.

    Returns:
        One (H, W) float32 array per target, max-normalised to [0, 1] (all-zero if the
        object never appeared under any mask).
    """
    if not targets:
        return []

    h, w = masks.shape[-2:]
    n_masks = masks.shape[0]  # source of truth 
    sum_maps = torch.zeros(len(targets), h, w)

    for i in range(0, n_masks, gpu_batch):
        masks_cpu = masks[i : i + gpu_batch]                    
        stack = masks_cpu.to(device) * imgs_tensor               
        dets_per_mask = get_one2many_predictions(
            model_dense, stack, [orig_shape] * stack.shape[0],
            conf_thres=conf_thres, iou_thres=iou_thres_nms,
        )
        for j, dets in enumerate(dets_per_mask):
            boxes, class_probs, max_conf = dets.boxes.cpu(), dets.class_probs.cpu(), dets.max_conf.cpu()
            for t, (target_box, target_probs) in enumerate(targets):
                sim = compute_similarity(target_box, target_probs, boxes, class_probs, max_conf)
                if sim > 0:
                    sum_maps[t] += sim * masks_cpu[j, 0]

    saliency_maps = []
    for m in sum_maps:
        peak = m.max()
        saliency_maps.append((m / peak).numpy() if peak > 0 else m.numpy())
    return saliency_maps

@dataclass
class DCRISPResult:
    """heatmap: (H0,W0) normalised [0,1], original-image coordinates.
    heatmap_raw: (H,W) letterboxed frame, already max-normalised by explain_image.
    Unlike SSGrad-CAM++'s heatmap_raw (a small per-scale grid), here it's full letterboxed
    resolution since masks act at pixel level, not on a feature grid."""
    target_class: int
    target_box: list
    heatmap: np.ndarray
    heatmap_raw: np.ndarray


class DCRISP:
    """D-CRISP explainer for a pre-loaded YOLO26 model."""

    def __init__(self, model_dense, model_prep, device: str = "cuda",
                 conf_thres: float = 0.25, iou_thres_nms: float = 0.5, gpu_batch: int = 50,
                 n_masks: int = 500, alpha: float = 0.25, resolution: int = 16,
                 p1: float = 0.25, num_levels: int = 5):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model_dense = model_dense
        self.model_prep = model_prep
        self.conf_thres = conf_thres
        self.iou_thres_nms = iou_thres_nms
        self.gpu_batch = gpu_batch
        self.n_masks = n_masks
        self.alpha = alpha
        self.resolution = resolution
        self.p1 = p1
        self.num_levels = num_levels

        self.model_dense.model.to(self.device)
        self.model_dense.model.eval()

        if getattr(self.model_prep, "predictor", None) is None:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model_prep.predict([dummy], verbose=False, device=0 if self.device == "cuda" else "cpu")

        self.mask_generator = None  # built on first call to explain(), once letterbox_shape is known

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device: str = "cuda", **kwargs) -> "DCRISP":
        """Load both required YOLO instances from a single checkpoint."""
        from ultralytics import YOLO
        model_dense = YOLO(str(checkpoint_path))
        model_prep = YOLO(str(checkpoint_path))
        return cls(model_dense, model_prep, device=device, **kwargs)

    def explain(self, img_bgr: np.ndarray,
                targets: list[tuple[list, int, torch.Tensor]]) -> list[DCRISPResult]:
        """Explains every target object in one image with a single mask pass.

        Args:
            img_bgr: (H0, W0, 3) BGR image, original resolution.
            targets: (target_box_xyxy, target_class_idx, target_class_probs) per object, all
                in original-image space -- computed by the caller via get_one2many_predictions
                on the unmasked image.

        Returns:
            One DCRISPResult per target, same order as `targets`.
        """
        orig_shape = img_bgr.shape[:2]
        imgs_tensor = self.model_prep.predictor.preprocess([img_bgr]).to(self.device)
        letterbox_shape = tuple(imgs_tensor.shape[2:])

        if self.mask_generator is None:
            self.mask_generator = DCRISPMaskGenerator(letterbox_shape)
        assert self.mask_generator.letterbox_shape == letterbox_shape, (
            f"Letterboxed shape changed ({self.mask_generator.letterbox_shape} -> "
            f"{letterbox_shape}) -- cached RISE masks would silently misalign."
        )

        letterboxed_rgb = (imgs_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        masks = self.mask_generator.generate_combined_masks(
            letterboxed_rgb, self.n_masks, self.alpha, self.resolution, self.p1, self.num_levels,
        )

        explain_targets = [(box, probs) for box, _, probs in targets]
        heatmaps_lb = explain_image(
            self.model_dense, imgs_tensor, orig_shape, masks, explain_targets,
            self.conf_thres, self.iou_thres_nms, self.gpu_batch, self.device,
        )

        results = []
        for (target_box, target_class, _), heatmap_lb in zip(targets, heatmaps_lb):
            heatmap_orig = unletterbox_map(heatmap_lb, letterbox_shape, orig_shape)
            results.append(DCRISPResult(
                target_class=target_class, target_box=list(target_box),
                heatmap=heatmap_orig / (heatmap_orig.max() + 1e-12),
                heatmap_raw=heatmap_lb,
            ))
        return results