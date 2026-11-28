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


def harmless_fatal_events(train_lines: pd.DataFrame, min_lines: int = 20) -> set[int]:
    """BGL: message types (events) that were FATAL/FAILURE at least `min_lines` times in
    TRAINING but never part of a real alert, e.g. "ciod: Error loading <*> invalid or missing
    program image" (a user's job failed to start, not the machine).

    Found in the error analysis (Milestone 3, step 6). Never-seen messages (event 0) are
    never on the list: what is unknown cannot be known to be harmless.
    """
    fatal = train_lines[train_lines["level"].isin(FATAL_LEVELS) & (train_lines["event_id"] != 0)]
    stats = fatal.groupby("event_id")["is_alert"].agg(["size", "sum"])
    return {int(e) for e in stats.index[(stats["size"] >= min_lines) & (stats["sum"] == 0)]}


def fatal_window_ids_except(lines: pd.DataFrame, harmless: set[int],
                            window_minutes: int = BGL_WINDOW_MINUTES) -> np.ndarray:
    """Like fatal_window_ids, but FATAL/FAILURE lines of a `harmless` message type do not count."""
    return fatal_window_ids(lines[~lines["event_id"].isin(harmless)], window_minutes)
