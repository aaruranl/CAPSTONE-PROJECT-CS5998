"""
Operational measures (Milestone 3, step 6): what a detector means for the people who have
to read its alerts. F1 says how good the decisions are; these say how much WORK they cause.

  * false alarms per day       false alarms / length of the period they come from
  * bundled notifications      alerts grouped into 10-minute periods, as an alerting tool would
  * alert budgets              choose the threshold on validation so that at most B false
                               alarms per day are raised there; is the promise kept on test?
"""
from collections import Counter

import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)


def days_covered(start_times: pd.Series, pad: pd.Timedelta = pd.Timedelta(0)) -> float:
    """Length in days of the period in which these groups STARTED (+ `pad`, e.g. one BGL window)."""
    return (start_times.max() - start_times.min() + pad) / DAY


def split_days(seqs: pd.DataFrame, split_col: str, part: str, pad: pd.Timedelta = pd.Timedelta(0)) -> float:
    """Days of log time a part of a split stands for.

    Time split: the part is one continuous period, so its own length.
    Random split: the part is a random share of ALL groups, spread over the whole log, so
    it stands for (whole log length) x (its share of the groups).
    """
    in_part = seqs[split_col] == part
    if split_col == "split_time":
        return days_covered(seqs.loc[in_part, "start_time"], pad)
    return days_covered(seqs["start_time"], pad) * in_part.mean()


def budget_threshold(normal_scores: np.ndarray, allowed: int) -> np.float64:
    """The lowest threshold (alert if score >= threshold) with at most `allowed` false alarms
    among these normal scores. Ties are respected: equal scores are all in or all out.

    Compare scores with it in float64: a float32 score array would round the threshold back
    down onto a tied score (numpy treats a plain Python float as "weak" and casts it to float32).
    """
    ranked = np.sort(np.asarray(normal_scores, dtype=np.float64))[::-1]
    if allowed >= len(ranked):
        return np.float64(-np.inf)
    return np.nextafter(ranked[allowed], np.float64(np.inf))  # just above the first score that must stay silent


def notifications(end_times: pd.Series, alert: np.ndarray, abnormal: np.ndarray, every: str = "10min") -> dict:
    """Bundle alerts into periods of `every` (one notification per period with any alert).
    A notification is FALSE when none of its alerts is a real anomaly."""
    period = pd.Series(end_times.to_numpy()[alert]).dt.floor(every)
    real = pd.Series(abnormal[alert]).groupby(period.to_numpy()).any()
    return {"notifications": int(len(real)), "false_notifications": int((~real).sum())}


def pattern(events) -> str:
    """The event-count pattern of a group, as count-based models see it, e.g. 'E1x3 E2x1 E5x3'."""
    return " ".join(f"E{e}x{c}" for e, c in sorted(Counter(np.asarray(events).tolist()).items()))
