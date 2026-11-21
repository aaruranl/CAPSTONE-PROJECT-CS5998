"""
How models are scored. Every model is judged the same way:

  1. it gives each group an anomaly SCORE (higher = more suspicious);
  2. the alert THRESHOLD is chosen on the validation set (best F1 there);
  3. that fixed threshold is applied to the test set, which is used only once.

The test set never influences any choice, so test results are an honest estimate
of how the system would do on data it has not seen.
"""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.config import RANDOM_SEED


def best_f1_threshold(y_true, scores) -> tuple[float, float]:
    """The threshold t (alert when score >= t) with the highest F1, and that F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1[:-1]))  # the last point (recall 0) has no threshold
    return float(thresholds[best]), float(f1[best])


def confusion(y_true, y_pred) -> dict:
    """True/false positives/negatives, where 'positive' means 'flagged as abnormal'."""
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    return {
        "tp": int((y_pred & y_true).sum()),    # anomalies caught
        "fp": int((y_pred & ~y_true).sum()),   # false alarms
        "fn": int((~y_pred & y_true).sum()),   # anomalies missed
        "tn": int((~y_pred & ~y_true).sum()),  # normal groups left alone
    }


def f1_score(tp, fp, fn):
    """F1 = 2TP / (2TP + FP + FN): the balance of precision and recall (works on arrays too)."""
    tp, fp, fn = np.asarray(tp, float), np.asarray(fp, float), np.asarray(fn, float)
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)


def f1_confidence_interval(tp, fp, fn, tn, n_boot=2000, seed=RANDOM_SEED) -> tuple[float, float]:
    """95% bootstrap interval for F1: 'how much could F1 move with a different test sample?'

    Re-sampling test groups with replacement only changes how many land in each
    confusion cell, so we draw the four cell counts directly (multinomial), which
    is exact and much faster than re-sampling 115,000 rows 2,000 times.
    """
    counts = np.array([tp, fp, fn, tn], dtype=float)
    draws = np.random.default_rng(seed).multinomial(int(counts.sum()), counts / counts.sum(), size=n_boot)
    f1 = f1_score(draws[:, 0], draws[:, 1], draws[:, 2])
    low, high = np.percentile(f1, [2.5, 97.5])
    return float(low), float(high)


def evaluate(y_true, y_pred, scores=None) -> dict:
    """Test-set metrics for the abnormal class."""
    c = confusion(y_true, y_pred)
    precision = c["tp"] / max(c["tp"] + c["fp"], 1)
    recall = c["tp"] / max(c["tp"] + c["fn"], 1)
    low, high = f1_confidence_interval(c["tp"], c["fp"], c["fn"], c["tn"])
    out = {
        "precision": precision,
        "recall": recall,
        "f1": float(f1_score(c["tp"], c["fp"], c["fn"])),
        "f1_ci_low": low,
        "f1_ci_high": high,
        # threshold-free view: how well the scores RANK abnormal above normal
        "pr_auc": float(average_precision_score(y_true, scores)) if scores is not None else np.nan,
        # DIAGNOSTIC ONLY - best F1 if the threshold were picked on the test set itself
        # (not a valid result; it shows how much is lost because the threshold came from validation)
        "oracle_f1": best_f1_threshold(y_true, scores)[1] if scores is not None else np.nan,
    }
    return {**out, **c}
