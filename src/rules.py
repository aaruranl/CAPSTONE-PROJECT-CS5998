"""
Two simple non-ML rules found during the EDA. A machine-learning model is only
worth its complexity if it beats these (docs/02_eda.md, sections 5 and 10).
"""
import numpy as np
import pandas as pd

from src.config import BGL_WINDOW_MINUTES
from src.eda import event_group_counts

FATAL_LEVELS = ("FATAL", "FAILURE")


def anomaly_only_event_set(train: pd.DataFrame) -> set[int]:
    """Events that occurred in abnormal training groups but never in a normal one."""
    counts = event_group_counts(train)
    only = (counts["groups_normal"] == 0) & (counts["groups_abnormal"] > 0)
    return {int(e) for e in counts.index[only]}


def contains_any(event_lists, events: set[int]) -> np.ndarray:
    """True for every group that contains at least one of `events`."""
    return np.fromiter((not events.isdisjoint(e.tolist()) for e in event_lists), dtype=bool)


def fatal_window_ids(lines: pd.DataFrame, window_minutes: int = BGL_WINDOW_MINUTES) -> np.ndarray:
    """BGL: IDs of the 5-minute windows that contain a FATAL or FAILURE line.

    Uses the same window numbering as preprocessing: window_id = unix_seconds // 300.
    """
    fatal = lines["level"].isin(FATAL_LEVELS).to_numpy()
    seconds = lines["timestamp"].to_numpy()[fatal].astype("datetime64[s]").astype(np.int64)
    return np.unique(seconds // (window_minutes * 60))
