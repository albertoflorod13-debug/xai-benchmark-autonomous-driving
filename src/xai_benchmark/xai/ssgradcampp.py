"""SSGrad-CAM++ (Toshinori Yamauchi, Spatial Sensitive Grad-CAM++: Towards High-Quality Visual 
Explanations for Object Detectors via Weighted Combination of Gradient Maps, 
https://www.sciencedirect.com/science/article/pii/S1077314226000251) adapted to YOLO26

Chosen white-box method per the project guide over G-CAME/Score-CAM/DiffCAM/ODSmoothGrad. 
Full technical design (target layer, M^{k,det}, the Grad-CAM++ exponential trick for 
higher-order derivatives), empirically validated in notebooks/ss_gradcampp_test.ipynb

Requires the one-to-many detection head machinery from detection/yolo_head.py (end2end=False), 
but reimplemented here WITHOUT @torch.no_grad() (gradients must flow for the backward pass this method needs).
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils.ops import scale_boxes, xywh2xyxy
from ultralytics.utils.tal import make_anchors

from xai_benchmark.uncertainty.tta import box_iou
from xai_benchmark.xai.common import unletterbox_map

EPS_DEFAULT = 1e-8


@dataclass
class SSGradCAMPPResult:
    """target_box/target_class: what was requested to be explained.
    scale_idx/row/col: grid cell (from the 3 YOLO26 scales) that originated the detection, automatically discovered.
    heatmap: (H0,W0) normalized [0,1], already in the coordinates of the original image (letterbox removed).
    heatmap_raw: (H_s,W_s), exactly as it comes from the formula, without normalization or letterbox removal.
    w_k: weights per channel (Eq. 7, SSGrad-CAM++ paper)."""

    target_class: int
    target_box: list
    scale_idx: int
    row: int
    col: int
    heatmap: np.ndarray
    heatmap_raw: np.ndarray
    w_k: torch.Tensor


class SSGradCAMPP:
    """SSGrad-CAM++ explainer for a pre-loaded YOLO26 model.

    It doesn't calculate detections itself: it receives an ALREADY detected box/class 
    (e.g., via get_one2many_predictions, detection/yolo_head.py) and generates the heat 
    map for that specific instance. Designed to be invoked one image/instance at a time.

    `model` and `model_prep` must be TWO distinct `YOLO` instances loaded from the same 
    checkpoint (same pattern as uncertainty/tta.py): `model` should never go through `.predict()`/`.val()` 
    (they merge, and merging deletes cv2/cv3 -- see yolo_head.py) """

    def __init__(self, model, model_prep, device: str = "cuda",
                 iou_match_thres: float = 0.999, eps: float = EPS_DEFAULT):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model = model
        self.model_prep = model_prep
        self.iou_match_thres = iou_match_thres
        self.eps = eps

        self.model.model.to(self.device)
        self.model.model.eval()
        self.model.model.requires_grad_(True)  

        if getattr(self.model_prep, "predictor", None) is None:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model_prep.predict([dummy], verbose=False, device=0 if self.device == "cuda" else "cpu")

        detect_head = self.model.model.model[-1]
        assert detect_head.__class__.__name__ == "Detect", (
            f"The Detect head was expected as the final layer, but it was obtained {detect_head.__class__.__name__}"
        )
        self.detect_head = detect_head
        self.num_scales = detect_head.nl
        self.num_classes = detect_head.nc

        self._activations = [None] * self.num_scales
        self._logits = [None] * self.num_scales
        self._hook_handles = []
        self._register_hooks()

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device: str = "cuda", **kwargs) -> "SSGradCAMPP":
        """Load both necessary YOLO instances from a single checkpoint."""
        from ultralytics import YOLO
        model_dense = YOLO(str(checkpoint_path))
        model_prep = YOLO(str(checkpoint_path))
        return cls(model_dense, model_prep, device=device, **kwargs)

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def _register_hooks(self):
        for s in range(self.num_scales):
            branch = self.detect_head.cv3[s]  # cv3[s][1]: penultima conv (capa objetivo, criterio que usa G-CAME en YOLOX)
            self._hook_handles.append(branch[1].register_forward_hook(self._make_activation_hook(s)))
            self._hook_handles.append(branch[2].register_forward_hook(self._make_logit_hook(s))) 

    def _make_activation_hook(self, scale_idx: int):
        def _hook(module, inp, out):
            if out.requires_grad:  
                out.retain_grad()
            self._activations[scale_idx] = out
        return _hook

    def _make_logit_hook(self, scale_idx: int):
        def _hook(module, inp, out):
            self._logits[scale_idx] = out
        return _hook

    def _dense_forward_with_grad(self, imgs_tensor: torch.Tensor) -> torch.Tensor:
        """Replicate get_one2many_predictions (yolo_head.py), without its @torch.no_grad() 
        Repopulate self._activations/self._logits via hooks."""
        detection_model = self.model.model
        prev_end2end = detection_model.end2end
        detection_model.end2end = False
        try:
            raw, _ = detection_model(imgs_tensor)
        finally:
            detection_model.end2end = prev_end2end
        return raw.permute(0, 2, 1)  # (1, anclas, 4+nc)

    def _locate_anchor(self, imgs_tensor, orig_shape, target_box_xyxy, target_class_idx) -> int:
        raw = self._dense_forward_with_grad(imgs_tensor)[0]
        boxes_xywh, class_probs = raw.split([4, self.num_classes], dim=-1)
        boxes_xyxy_letterboxed = xywh2xyxy(boxes_xywh)
        img1_shape = imgs_tensor.shape[2:]
        boxes_xyxy_orig = scale_boxes(img1_shape, boxes_xyxy_letterboxed.clone().detach(), orig_shape)

        target = torch.as_tensor(target_box_xyxy, dtype=boxes_xyxy_orig.dtype,
                                  device=boxes_xyxy_orig.device).unsqueeze(0)
        ious = box_iou(boxes_xyxy_orig, target).squeeze(-1)
        pred_class = class_probs.detach().argmax(dim=-1)
        candidates = ((ious >= self.iou_match_thres) & (pred_class == target_class_idx)).nonzero(as_tuple=True)[0]

        if len(candidates) == 0:
            raise RuntimeError(
                "No anchor reproduces the given detection."
            )
        if len(candidates) > 1:
            best = class_probs[candidates, target_class_idx].argmax()
            return int(candidates[best])
        return int(candidates[0])

    def _anchor_index_to_cell(self, anchor_idx: int):
        offset = 0
        for s in range(self.num_scales):
            _, _, H_s, W_s = self._activations[s].shape
            n_s = H_s * W_s
            if anchor_idx < offset + n_s:
                local_idx = anchor_idx - offset
                row, col = divmod(local_idx, W_s)  
                return s, row, col, H_s, W_s
            offset += n_s
        raise ValueError(f"anchor_idx {anchor_idx} out of range (total anchors: {offset})")

    @staticmethod
    def _build_instance_mask(H_s: int, W_s: int, row: int, col: int, margin: int, device) -> torch.Tensor:
        mask = torch.zeros(H_s, W_s, device=device)
        r0, r1 = max(0, row - margin), min(H_s, row + margin + 1)
        c0, c1 = max(0, col - margin), min(W_s, col + margin + 1)
        mask[r0:r1, c0:c1] = 1.0
        return mask


    def explain(self, img_bgr: np.ndarray, target_box_xyxy, target_class_idx: int,
                margin: int = 0) -> SSGradCAMPPResult:
        """Generates the SSGrad-CAM++ map for a known detection. 
        margin=0 uses a single grid cell as M^{k,det}"""
        orig_shape = img_bgr.shape[:2]
        imgs_tensor = self.model_prep.predictor.preprocess([img_bgr]).to(self.device)

        self.model.model.zero_grad(set_to_none=True)
        anchor_idx = self._locate_anchor(imgs_tensor, orig_shape, target_box_xyxy, target_class_idx)
        scale_idx, row, col, H_s, W_s = self._anchor_index_to_cell(anchor_idx)
        M_mask = self._build_instance_mask(H_s, W_s, row, col, margin, device=self._activations[scale_idx].device)

        self.model.model.zero_grad(set_to_none=True)
        self._dense_forward_with_grad(imgs_tensor)  
        S_c = self._logits[scale_idx][0, target_class_idx, row, col]
        S_c.backward()

        grad_A = self._activations[scale_idx].grad[0]
        A_k = self._activations[scale_idx][0].detach()
        if grad_A is None:
            raise RuntimeError("Non-propagated gradient.")

        grad_abs = grad_A.abs()
        S_k_spatial = grad_abs / grad_abs.amax(dim=(1, 2), keepdim=True).clamp(min=1e-12)  # Eq. 6, paper SSGrad-CAM++ 

        grad2, grad3 = grad_A ** 2, grad_A ** 3
        masked_activation_sum = (A_k * M_mask.unsqueeze(0)).sum(dim=(1, 2), keepdim=True)
        denom = 2 * grad2 * M_mask.unsqueeze(0) + masked_activation_sum * grad3
        alpha = grad2 / (denom + self.eps)  # Eq. 12, paper SSGrad-CAM++
        w_k = (alpha * grad_A.clamp(min=0)).sum(dim=(1, 2))  # Eq. 7, paper SSGrad-CAM++

        L_c_det = F.relu((w_k.view(-1, 1, 1) * A_k * S_k_spatial).sum(dim=0))  # Eq. 4, paper SSGrad-CAM++
        heatmap_raw = L_c_det.detach().cpu().numpy()

        img1_shape = imgs_tensor.shape[2:]
        heatmap_letterboxed = cv2.resize(heatmap_raw, (img1_shape[1], img1_shape[0]))
        heatmap_orig = unletterbox_map(heatmap_letterboxed, img1_shape, orig_shape)
        heatmap_norm = heatmap_orig / (heatmap_orig.max() + 1e-12)

        return SSGradCAMPPResult(
            target_class=target_class_idx, target_box=list(target_box_xyxy),
            scale_idx=scale_idx, row=row, col=col,
            heatmap=heatmap_norm, heatmap_raw=heatmap_raw, w_k=w_k.detach().cpu(),
        )