"""
Train / validation / test splitting - the heart of the project's research question.

Two strategies, applied to the SAME groups so the comparison is fair:

  random_split         Shuffle all groups, then cut 70/10/20. The common choice
                       in published papers. Training data contains groups from
                       the whole time range, so the model "sees the future".

  chronological_split  Sort groups by when they started, then cut 70/10/20:
                       oldest 70% -> train, next 10% -> validation,
                       newest 20% -> test. This is what happens in real use.

The random split is stratified (each part keeps the ~same anomaly rate) so that
any difference we see later is caused by *time*, not by an unlucky shuffle.
"""
import numpy as np
import pandas as pd

from src import config

SPLIT_NAMES = np.array(["train", "val", "test"])


def _cut_points(n: int) -> tuple[int, int]:
    n_train = int(round(n * config.TRAIN_RATIO))
    n_val = int(round(n * config.VAL_RATIO))
    return n_train, n_train + n_val


def random_split(labels: np.ndarray, seed: int = config.RANDOM_SEED) -> np.ndarray:
    """Stratified random split. Returns an array of 'train' / 'val' / 'test'."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    out = np.empty(len(labels), dtype=object)
    for value in np.unique(labels):              # do normal and abnormal separately
        idx = np.flatnonzero(labels == value)
        rng.shuffle(idx)
        a, b = _cut_points(len(idx))
        out[idx[:a]], out[idx[a:b]], out[idx[b:]] = "train", "val", "test"
    return out


def chronological_split(order_key: np.ndarray) -> np.ndarray:
    """Time-based split: smallest order_key (oldest) goes to train."""
    order = np.argsort(np.asarray(order_key), kind="stable")
    a, b = _cut_points(len(order))
    out = np.empty(len(order), dtype=object)
    out[order[:a]], out[order[a:b]], out[order[b:]] = "train", "val", "test"
    return out


def split_summary(df: pd.DataFrame, split_col: str, label_col: str = "label",
                  time_col: str = "start_time") -> pd.DataFrame:
    """Size, anomaly count/rate and time range of each part."""
    g = df.groupby(split_col, observed=True)
    summary = pd.DataFrame({
        "groups": g.size(),
        "anomalies": g[label_col].sum(),
        "anomaly_rate_%": (g[label_col].mean() * 100).round(2),
        "first_start": g[time_col].min(),
        "last_start": g[time_col].max(),
    })
    return summary.reindex([s for s in SPLIT_NAMES if s in summary.index])
