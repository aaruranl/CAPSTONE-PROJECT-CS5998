"""
Shared plumbing for the model scripts: load a dataset, cut it into the
train / validation / test parts of one split, and record results the same way
for every model (so the comparison is fair).
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.metrics import best_f1_threshold, confusion, evaluate, f1_score

DATASETS = ("hdfs", "bgl")
SPLITS = {"random": "split_random", "time": "split_time"}  # experiment name -> column

RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
PRED_DIR = RESULTS_DIR / "predictions"  # per-group test predictions (regenerated, not committed)
MODELS_DIR = PROJECT_ROOT / "models"    # trained models (regenerated, not committed)


def load_sequences(dataset: str) -> pd.DataFrame:
    cols = ["event_ids", "label", "split_random", "split_time"]
    if dataset == "bgl":
        cols.append("window_id")
    return pd.read_parquet(PROCESSED_DIR / dataset / "sequences.parquet", columns=cols)


def split_parts(seqs: pd.DataFrame, split: str) -> dict[str, pd.DataFrame]:
    col = SPLITS[split]
    return {part: seqs[seqs[col] == part] for part in ("train", "val", "test")}


def record(dataset, split, model, kind, parts, val_scores, test_scores, seconds,
           threshold=None, is_rule=False) -> dict:
    """Pick the threshold on validation (unless given), apply it to test, save everything.

    Rules give 0/1 instead of a score, so they have a fixed threshold of 1 and no PR-AUC.
    """
    y_val, y_test = parts["val"]["label"].to_numpy(), parts["test"]["label"].to_numpy()
    val_scores, test_scores = np.asarray(val_scores, float), np.asarray(test_scores, float)
    if is_rule:
        threshold = 1.0
    elif threshold is None:
        threshold, _ = best_f1_threshold(y_val, val_scores)
    val_c = confusion(y_val, val_scores >= threshold)
    pred = test_scores >= threshold

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row": parts["test"].index.to_numpy(), "label": y_test, "score": test_scores, "pred": pred}) \
        .to_parquet(PRED_DIR / f"{dataset}_{split}_{model}.parquet", index=False)

    return {
        "dataset": dataset, "split": split, "model": model, "kind": kind,
        "threshold": threshold,
        "val_f1": float(f1_score(val_c["tp"], val_c["fp"], val_c["fn"])),
        **evaluate(y_test, pred, None if is_rule else test_scores),
        "n_test": len(y_test), "n_test_abnormal": int(y_test.sum()),
        "train_seconds": seconds,
    }
