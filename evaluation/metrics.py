"""Detection performance, false-negative rate, localization quality, inference time —
the measures named in whitepaper Section 2.3 (Objectives) and Section 4."""
import time
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def detection_auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Image-level AUROC: y_true is 1=anomalous/0=normal ground truth, y_score the
    predicted anomaly score."""
    return roc_auc_score(y_true, y_score)


def false_negative_rate(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Fraction of true anomalies (y_true==1) predicted as normal (y_pred==0)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    positives = y_true == 1
    if positives.sum() == 0:
        return float("nan")
    return float(((y_pred == 0) & positives).sum() / positives.sum())


def localization_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """IoU between a thresholded predicted heatmap and a ground-truth defect mask, as a
    simple localization-quality proxy."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(intersection / union)


def measure_inference_time(fn, *args, **kwargs) -> float:
    """Wall-clock seconds for a single call to fn(*args, **kwargs)."""
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start
