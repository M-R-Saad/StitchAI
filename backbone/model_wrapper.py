"""
Unified interface: image + reference bank (+ category) -> anomaly score + heatmap.

This is THE CORE module (whitepaper Section 4.2 / project structure doc). It must stay
identical no matter which category is being scored - category-specific behavior comes
only from `prompts.py` and the reference bank passed in, never from branching model code
per category here.

Implementation notes (Phase 1):
- The model (CLIP backbone + AnomalyCLIP prompt learner + checkpoint) is loaded ONCE in
  `load()`, not per-request - loading takes real time (confirmed during manual CLI
  testing), so the backend should call `load()` once at startup and reuse the instance.
- Scoring is currently ZERO-SHOT only (mirrors the official test_one_example.py
  behavior we validated manually on real WFDD fabric images). `reference_bank` is
  accepted and stored on the result for now but not yet used to adjust the score -
  few-shot calibration is a planned refinement, not implemented yet. When it is added,
  it slots into `score_image()` as an additional comparison step; the public interface
  (this class's methods) should not need to change.
- Category-specific text prompts come from `prompts.py`, not from anything here.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class AnomalyResult:
    score: float                    # image-level anomaly score, 0..1
    heatmap: Optional[np.ndarray]   # pixel-level anomaly heatmap (H x W), or None
    category: str
    verdict: str                    # "normal" | "anomalous" (set by caller after thresholding)


class AnomalyCLIPWrapper:
    """
    Thin wrapper around the vendored AnomalyCLIP implementation in
    backbone/anomalyclip/. All AnomalyCLIP-specific plumbing (loading the checkpoint,
    calling its forward pass, converting output tensors) lives inside this class so the
    rest of the app only ever calls `load()` once and then `score_image()` per request.
    """

    # These must match whatever checkpoint_path in config.yaml was trained with.
    DEFAULT_FEATURES_LIST = [6, 12, 18, 24]
    DEFAULT_IMAGE_SIZE = 518
    DEFAULT_DEPTH = 9
    DEFAULT_N_CTX = 12
    DEFAULT_T_N_CTX = 4
    DEFAULT_FEATURE_MAP_LAYER = [0, 1, 2, 3]
    DEFAULT_SIGMA = 4

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None
        self._prompt_learner = None
        self._text_features = None
        self._preprocess = None
        self._loaded = False

    def load(self):
        """Load model weights + build text prompt embeddings. Call once at backend
        startup - this is the slow step (downloading/loading the CLIP backbone and
        checkpoint), so it must not run per-request."""
        import sys

        import torch

        # backbone/anomalyclip/ is a vendored git clone, not an installed package -
        # its modules (AnomalyCLIP_lib, prompt_ensemble, utils) only import correctly
        # if that folder is on sys.path.
        anomalyclip_dir = str(Path(__file__).parent / "anomalyclip")
        if anomalyclip_dir not in sys.path:
            sys.path.insert(0, anomalyclip_dir)

        import AnomalyCLIP_lib
        from prompt_ensemble import AnomalyCLIP_PromptLearner
        from utils import get_transform

        if self.checkpoint_path is None:
            raise ValueError(
                "checkpoint_path is not set - point backbone/config.yaml's "
                "model.checkpoint_path at a real .pth file, e.g. "
                "backbone/anomalyclip/checkpoints/9_12_4_multiscale/epoch_15.pth"
            )

        anomalyclip_params = {
            "Prompt_length": self.DEFAULT_N_CTX,
            "learnabel_text_embedding_depth": self.DEFAULT_DEPTH,
            "learnabel_text_embedding_length": self.DEFAULT_T_N_CTX,
        }

        model, _ = AnomalyCLIP_lib.load(
            "ViT-L/14@336px", device=self.device, design_details=anomalyclip_params
        )
        model.eval()

        class _Args:
            image_size = self.DEFAULT_IMAGE_SIZE

        self._preprocess, _ = get_transform(_Args())

        prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), anomalyclip_params)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        prompt_learner.to(self.device)
        model.to(self.device)
        model.visual.DAPM_replace(DPAM_layer=20)

        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
        text_features = model.encode_text_learn(
            prompts, tokenized_prompts, compound_prompts_text
        ).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self._model = model
        self._prompt_learner = prompt_learner
        self._text_features = text_features
        self._loaded = True

    def score_image(
        self,
        image_path: str,
        category: str,
        reference_bank: List[str],
    ) -> AnomalyResult:
        """
        Run AnomalyCLIP on a single image for the given category.

        Args:
            image_path: path to the uploaded/inference image.
            category: one of "fabric" | "safety" | "machinery".
            reference_bank: paths to that category's few-shot "normal" reference
                images. Accepted for interface stability; not yet used to adjust the
                score (zero-shot only for now - see module docstring).

        Returns:
            AnomalyResult with score in [0, 1] and a heatmap the same size as the
            (resized) input image.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded - call .load() once before score_image()")

        import sys

        import torch
        from PIL import Image
        from scipy.ndimage import gaussian_filter

        anomalyclip_dir = str(Path(__file__).parent / "anomalyclip")
        if anomalyclip_dir not in sys.path:
            sys.path.insert(0, anomalyclip_dir)
        import AnomalyCLIP_lib

        img = Image.open(image_path).convert("RGB")
        img = self._preprocess(img)
        image = img.reshape(1, 3, self.DEFAULT_IMAGE_SIZE, self.DEFAULT_IMAGE_SIZE).to(
            self.device
        )

        with torch.no_grad():
            image_features, patch_features = self._model.encode_image(
                image, self.DEFAULT_FEATURES_LIST, DPAM_layer=20
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ self._text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            score = float(text_probs.item())

            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= self.DEFAULT_FEATURE_MAP_LAYER[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(
                        patch_feature, self._text_features[0]
                    )
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(
                        similarity[:, 1:, :], self.DEFAULT_IMAGE_SIZE
                    )
                    anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list)
            anomaly_map = anomaly_map.sum(dim=0)
            anomaly_map = torch.stack(
                [
                    torch.from_numpy(gaussian_filter(i, sigma=self.DEFAULT_SIGMA))
                    for i in anomaly_map.detach().cpu()
                ],
                dim=0,
            )
            heatmap = anomaly_map.detach().cpu().numpy()[0]

        # reference_bank is intentionally unused right now - see module docstring.
        return AnomalyResult(score=score, heatmap=heatmap, category=category, verdict="")


def load_reference_bank(category: str, reference_bank_dir: str) -> List[str]:
    """List the few-shot 'normal' reference image paths for a category."""
    d = Path(reference_bank_dir)
    if not d.exists():
        return []
    return sorted(str(p) for p in d.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
