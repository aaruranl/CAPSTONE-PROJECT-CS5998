"""
Shared plumbing for the model scripts: load a dataset, cut it into the
train / validation / test parts of one split, and record results the same way
for every model (so the comparison is fair).
"""
import numpy as np
import pandas as pd

from src.config import BGL_WINDOW_MINUTES, PROCESSED_DIR, PROJECT_ROOT
from src.config import DATASETS as RAW_FILES  # dataset -> raw log file paths
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


def test_period_start(dataset: str) -> int:
    """Log time (unix seconds) at which the time-split test period begins."""
    cols = ["start_time", "split_time"] + (["window_id"] if dataset == "bgl" else [])
    seqs = pd.read_parquet(PROCESSED_DIR / dataset / "sequences.parquet", columns=cols)
    test = seqs[seqs["split_time"] == "test"]
    if dataset == "bgl":  # the start of the first test window
        return int(test["window_id"].min()) * BGL_WINDOW_MINUTES * 60
    return int(test["start_time"].min().timestamp())


def test_period_lines(dataset: str, seqs: pd.DataFrame) -> list[str]:
    """All raw lines that can belong to a test group (they all come after the test period starts)."""
    test = seqs[seqs["split_time"] == "test"]
    path = RAW_FILES[dataset.upper()]["log_file"]
    with open(path, encoding="utf-8", errors="replace") as f:
        if dataset == "hdfs":  # groups are blocks; every line of a test block is at/after its first line
            start = int(test["first_line"].min())
            return [line for i, line in enumerate(f) if i >= start]
        # BGL: keep lines whose 5-minute window is a test window
        windows = set(test["window_id"].astype(str))
        return [line for line in f if (p := line.split(maxsplit=2))[1:2] and p[1].isdigit()
                and str(int(p[1]) // (BGL_WINDOW_MINUTES * 60)) in windows]
