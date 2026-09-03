"""
Routes: image -> model -> threshold -> (optional) explanation -> log.

This is the one place that ties backbone/, explanation/, and storage/ together for a
single request. Category-specific behavior should still only come from config
(reference bank + threshold), never from branching logic added here per category.

The safety category uses a YOLOv8 detector (SafetyDetector) instead of AnomalyCLIP,
because PPE compliance is a semantic object-detection task, not a texture anomaly
(see PROGRESS7.md Phase 3 findings and whitepaper Section 1.2 contingency).
"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from backbone.model_wrapper import AnomalyCLIPWrapper, load_reference_bank
from backbone.safety_detector import SafetyDetector
from explanation.explain_service import explain_anomaly
from storage.db import SessionLocal, init_db
from storage.models import InferenceLog

with open("backbone/config.yaml") as f:
    _CONFIG = yaml.safe_load(f)

_model = AnomalyCLIPWrapper(
    checkpoint_path=_CONFIG["model"].get("checkpoint_path"),
    device=_CONFIG["model"].get("device", "cpu"),
)
# Loaded once here, at process startup (module import time) - NOT per-request. This is
# the slow step (CLIP backbone + checkpoint loading), confirmed during Phase 1 manual
# CLI testing, so the backend must only pay this cost once.
_model.load()

# Safety uses a separate YOLOv8 detector (whitepaper Section 1.2 contingency)
_safety_cfg = _CONFIG["categories"].get("safety", {})
_safety_model = None
if _safety_cfg.get("enabled") and _safety_cfg.get("detection_method") == "yolo":
    _safety_model = SafetyDetector(
        weights_path=_safety_cfg.get("yolo_weights_path"),
        device=_CONFIG["model"].get("device", "cpu"),
    )
    _safety_model.load()

init_db()  # create storage/stitchai.db + tables if they don't exist yet

_HEATMAP_DIR = Path("storage/heatmaps")
_HEATMAP_DIR.mkdir(parents=True, exist_ok=True)


def _save_heatmap(heatmap: np.ndarray, original_image_path: str) -> str:
    """Overlay the raw anomaly map on the original image and save it, mirroring the
    apply_ad_scoremap/visualizer approach from AnomalyCLIP's own test_one_example.py."""
    import cv2

    img_size = _model.DEFAULT_IMAGE_SIZE
    vis = cv2.cvtColor(
        cv2.resize(cv2.imread(original_image_path), (img_size, img_size)),
        cv2.COLOR_BGR2RGB,
    )

    mask = heatmap.copy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask_uint8 = (mask * 255).astype(np.uint8)
    colored = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = (0.5 * vis.astype(float) + 0.5 * colored.astype(float)).astype(np.uint8)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    filename = f"heatmap_{Path(original_image_path).stem}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
    out_path = _HEATMAP_DIR / filename
    cv2.imwrite(str(out_path), overlay)
    return str(out_path)


def _save_safety_annotation(annotated_img: np.ndarray, original_image_path: str) -> str:
    """Save the YOLO-annotated safety image (already has bounding boxes drawn)."""
    import cv2

    filename = f"safety_{Path(original_image_path).stem}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
    out_path = _HEATMAP_DIR / filename
    cv2.imwrite(str(out_path), annotated_img)
    return str(out_path)


def _write_log(category: str, score: float, verdict: str, image_ref: str):
    """Write one InferenceLog row - called for EVERY /infer call, regardless of
    category, so /logs shows one unified audit trail across fabric/safety/machinery
    (whitepaper Section 2.1's 'one log instead of three' claim)."""
    db = SessionLocal()
    try:
        entry = InferenceLog(
            category=category,
            score=score,
            verdict=verdict,
            image_ref=image_ref,
            timestamp=datetime.now(timezone.utc),
        )
        db.add(entry)
        db.commit()
    finally:
        db.close()


def run_inference(image_path: str, category: str, original_filename: str = None) -> dict:
    """
    Full pipeline for one uploaded image. Returns a dict matching
    backend.schemas.InferenceResponse (minus heatmap_url, which the route layer fills
    in - this returns a local file path in `heatmap_path` for the route to serve).
    """
    if category not in _CONFIG["categories"]:
        raise ValueError(f"Unknown category: {category!r}")

    cat_cfg = _CONFIG["categories"][category]
    if not cat_cfg.get("enabled", False):
        raise ValueError(f"Category {category!r} is not enabled yet in backbone/config.yaml")

    reference_bank = load_reference_bank(category, cat_cfg["reference_bank_dir"])

    # Route to the appropriate detector
    if category == "safety" and _safety_model is not None:
        result = _safety_model.score_image(image_path, category, reference_bank)
    else:
        result = _model.score_image(image_path, category, reference_bank)

    threshold = cat_cfg.get("threshold", 0.5)
    verdict = "anomalous" if result.score >= threshold else "normal"

    # Save visualization: YOLO produces an annotated image, AnomalyCLIP produces a heatmap
    if category == "safety" and _safety_model is not None:
        heatmap_path = _save_safety_annotation(result.heatmap, image_path) if result.heatmap is not None else None
    else:
        heatmap_path = _save_heatmap(result.heatmap, image_path) if result.heatmap is not None else None

    # Build explanation with region description from safety detector if available
    explanation = None
    if verdict == "anomalous":
        region_desc = getattr(result, '_region_description', 'a region flagged by the anomaly heatmap')
        explanation = explain_anomaly(category=category, score=result.score, region_description=region_desc)

    log_ref = original_filename if original_filename else image_path
    _write_log(category=category, score=result.score, verdict=verdict, image_ref=log_ref)

    payload = {
        "category": category,
        "verdict": verdict,
        "score": result.score,
        "heatmap_path": heatmap_path,
        "explanation": explanation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return payload
