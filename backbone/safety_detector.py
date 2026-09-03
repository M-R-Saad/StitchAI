"""
Lightweight task-specific visual component for safety/PPE compliance detection
(whitepaper Section 1.2 contingency).

Uses a pre-trained YOLOv8n model (trained on the Construction Site Safety dataset)
to detect PPE violations (NO-Hardhat, NO-Mask, NO-Safety Vest) rather than the shared
AnomalyCLIP backbone, because PPE compliance is a semantic object-presence question —
not a texture anomaly — and AnomalyCLIP's MVTec-AD-trained checkpoint cannot separate
compliant vs. violation scenes (see PROGRESS7.md Phase 3 findings).

The interface mirrors AnomalyCLIPWrapper so the orchestrator can call either model
through the same flow (threshold → explanation → audit log).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backbone.model_wrapper import AnomalyResult

# Class IDs from the dataset (ppe_data.yaml)
CLASS_NAMES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle",
}

VIOLATION_CLASS_IDS = {2, 3, 4}  # NO-Hardhat, NO-Mask, NO-Safety Vest
COMPLIANT_CLASS_IDS = {0, 1, 7}  # Hardhat, Mask, Safety Vest


@dataclass
class SafetyDetection:
    """One detected object from YOLO."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    is_violation: bool


class SafetyDetector:
    """
    Thin wrapper around a YOLOv8 model trained on Construction Site Safety data.

    Same interface pattern as AnomalyCLIPWrapper: call load() once at startup,
    then score_image() per request.
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu",
                 confidence_threshold: float = 0.25):
        self.weights_path = weights_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._loaded = False

    def load(self):
        """Load YOLO weights. Call once at backend startup."""
        from ultralytics import YOLO

        if self.weights_path is None:
            raise ValueError(
                "weights_path is not set — point backbone/config.yaml's "
                "safety.yolo_weights_path at the trained best.pt file"
            )

        if not Path(self.weights_path).exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {self.weights_path}"
            )

        self._model = YOLO(self.weights_path)
        self._loaded = True

    def _run_detection(self, image_path: str) -> List[SafetyDetection]:
        """Run YOLO inference and return structured detections."""
        if not self._loaded:
            raise RuntimeError("Model not loaded — call .load() first")

        results = self._model(image_path, conf=self.confidence_threshold, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                detections.append(SafetyDetection(
                    class_id=cls_id,
                    class_name=CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    is_violation=cls_id in VIOLATION_CLASS_IDS,
                ))

        return detections

    def _draw_annotations(self, image_path: str, detections: List[SafetyDetection]) -> np.ndarray:
        """Draw bounding boxes on the image — red for violations, green for compliant PPE."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            if det.is_violation:
                color = (0, 0, 255)  # red (BGR)
                thickness = 3
            elif det.class_id in COMPLIANT_CLASS_IDS:
                color = (0, 200, 0)  # green
                thickness = 2
            else:
                color = (200, 200, 0)  # cyan-ish for person/other
                thickness = 1

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            label = f"{det.class_name} {det.confidence:.2f}"
            font_scale = 0.5
            font_thickness = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Label background
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        return img

    def score_image(
        self,
        image_path: str,
        category: str,
        reference_bank: List[str],
    ) -> AnomalyResult:
        """
        Run PPE compliance detection on a single image.

        Args:
            image_path: path to the uploaded image.
            category: should be "safety".
            reference_bank: accepted for interface compatibility; not used by YOLO.

        Returns:
            AnomalyResult with:
            - score: max confidence of violation detections (0.0 if none found)
            - heatmap: annotated image with bounding boxes (reuses the heatmap field)
        """
        detections = self._run_detection(image_path)
        annotated_img = self._draw_annotations(image_path, detections)

        violations = [d for d in detections if d.is_violation]

        if violations:
            # Score = max confidence of violation detections
            score = max(d.confidence for d in violations)
        else:
            # No violations found — score 0.0 (well below any reasonable threshold)
            score = 0.0

        # Build a region description for the explanation layer
        if violations:
            violation_summary = ", ".join(
                f"{d.class_name} (conf {d.confidence:.2f})" for d in violations
            )
            region_desc = f"Detected violations: {violation_summary}"
        else:
            region_desc = "No PPE violations detected"

        result = AnomalyResult(
            score=score,
            heatmap=annotated_img,  # reuse heatmap field for the annotated image
            category=category,
            verdict="",  # set by orchestrator after thresholding
        )
        # Attach extra metadata that the orchestrator can use
        result._detections = detections
        result._region_description = region_desc

        return result
