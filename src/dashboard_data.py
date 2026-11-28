"""
The numbers behind the dashboard (Milestone 3, step 4), kept apart from the page layout
(dashboard/app.py) so that they can be tested.

Almost everything comes from the alert store: what the live system decided. The only
exception is the "check with the true labels" tab, which joins the labels from
preprocessing to GRADE the live system. The labels never reach the live system itself:
it has to work without them, as in real use.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.experiment import RESULTS_DIR

ID_COLUMN = {"hdfs": "block_id", "bgl": "window_id"}
MODEL_NAMES = {"rule_fatal": "FATAL rule", "rule_anomaly_only": "Anomaly-only events rule",
               "logreg": "Logistic Regression", "iforest": "Isolation Forest", "deeplog": "DeepLog (LSTM)",
               "deeplog_ignore_unseen": "DeepLog, never-seen events ignored (diagnostic)"}


@lru_cache(maxsize=2)
def true_labels(dataset: str) -> pd.Series:
    """group ID (text) -> 1 if the group is truly abnormal, else 0 (from preprocessing)."""
    col = ID_COLUMN[dataset]
    seqs = pd.read_parquet(PROCESSED_DIR / dataset / "sequences.parquet", columns=[col, "label"])
    return pd.Series(seqs["label"].to_numpy().astype(int), index=seqs[col].astype(str).to_numpy())


def time_step(frame: pd.DataFrame) -> str:
    """Hours for a short stream (2 days or less of log time), days for a longer one."""
    if frame.empty:
        return "h"
    span = frame["last_seen"].max() - frame["first_seen"].min()
    return "h" if span <= pd.Timedelta(days=2) else "D"


def time_range(frame: pd.DataFrame, status: dict, step: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The log-time span a chart should show: from the first judged event to "now" (whole hours/days)."""
    end = max(frame["last_seen"].max(), pd.Timestamp(status.get("log_time") or frame["last_seen"].max()))
    return frame["first_seen"].min().floor(step), (end + pd.Timedelta(1, step)).floor(step)


def stacked(data: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Add bottom/top columns so bars of several kinds can be drawn on top of each other."""
    data = data.assign(rank=data["kind"].map({k: i for i, k in enumerate(order)})).sort_values(["time", "rank"])
    data["top"] = data.groupby("time")["alerts"].cumsum()
    data["bottom"] = data["top"] - data["alerts"]
    return data.drop(columns="rank")


def headline(frame: pd.DataFrame, status: dict) -> dict:
    """The numbers at the top of the live page."""
    alerts = frame[frame["is_anomaly"]]
    events = int(frame["n_events"].sum())
    delay = (alerts["judged_at"] - alerts["last_seen"]).median() if len(alerts) else None
    return {
        "log_time": status.get("log_time"),
        "lines": status.get("lines", 0),
        "judged": len(frame),
        "alerts": len(alerts),
        "alert_share": len(alerts) / len(frame) if len(frame) else 0.0,
        "unseen_share": frame["n_unseen_events"].sum() / events if events else 0.0,
        "open_groups": status.get("open_groups", 0),
        "not_judged": status.get("not_judged_started_before_watching", 0),
        "median_delay": delay,
    }


def alerts_over_time(frame: pd.DataFrame, step: str, labels: pd.Series | None = None) -> pd.DataFrame:
    """Alerts per hour or day (by when the group's activity ended). With `labels`, each
    alert is also split into "real anomaly" and "false alarm"."""
    alerts = frame[frame["is_anomaly"]].copy()
    alerts["time"] = alerts["last_seen"].dt.floor(step)
    if labels is None:
        return alerts.groupby("time").size().rename("alerts").reset_index()
    truth = alerts["group_id"].map(labels).fillna(0).astype(int)
    alerts["kind"] = np.where(truth == 1, "real anomaly", "false alarm")
    return alerts.groupby(["time", "kind"]).size().rename("alerts").reset_index()


def grade(frame: pd.DataFrame, labels: pd.Series) -> dict:
    """Compare the live decisions with the true labels.

    A group that was cut into pieces (HDFS quiet time) counts once: flagged if any piece was.
    False alarms per day use the log time the stream covered (first event to last judgement).
    """
    if frame.empty:
        return {}
    flagged = frame.groupby("group_id")["is_anomaly"].any()
    truth = labels.reindex(flagged.index).fillna(0).astype(bool)
    tp = int((flagged & truth).sum())
    fp = int((flagged & ~truth).sum())
    fn = int((~flagged & truth).sum())
    tn = int((~flagged & ~truth).sum())
    days = (frame["judged_at"].max() - frame["first_seen"].min()) / pd.Timedelta(days=1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"groups": len(flagged), "abnormal": tp + fn, "caught": tp, "false_alarms": fp, "missed": fn,
            "correct_normal": tn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "days": days, "false_alarms_per_day": fp / days if days > 0 else float("nan")}


def drift_over_time(frame: pd.DataFrame, step: str) -> pd.DataFrame:
    """Share of log messages the parser had never seen in training, per hour or day."""
    f = frame.assign(time=frame["last_seen"].dt.floor(step))
    g = f.groupby("time")[["n_unseen_events", "n_events"]].sum()
    return (g["n_unseen_events"] / g["n_events"]).rename("unseen_share").reset_index()


def unseen_examples(results: list[dict], limit: int = 10) -> list[str]:
    """A few different never-seen messages, most recent first."""
    seen: dict[str, None] = {}
    for r in results:
        for message in r["unseen_examples"]:
            seen.setdefault(message, None)
            if len(seen) == limit:
                return list(seen)
    return list(seen)


def model_results(dataset: str) -> pd.DataFrame:
    """Test F1 of every model, random vs time split (the research question), from step 3 of Milestone 2."""
    r = pd.read_csv(RESULTS_DIR / "results.csv")
    r = r[r["dataset"] == dataset].copy()
    r["model_name"] = r["model"].map(MODEL_NAMES).fillna(r["model"])
    r["split_name"] = r["split"].map({"random": "random split", "time": "time split"})
    return r[["model", "model_name", "split", "split_name", "f1", "f1_ci_low", "f1_ci_high", "precision", "recall"]]
