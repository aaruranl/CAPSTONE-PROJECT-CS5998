"""Tests for the dashboard: its numbers (src/dashboard_data.py) and the whole page (dashboard/app.py),
on a small SQLite alert store, so no network or real stream is needed."""
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src import dashboard_data as dd
from src.store import ResultStore

T = pd.Timestamp("2005-11-09 04:40:00")


def frame(rows):
    """rows: (group, minutes after T when it started, minutes when it ended, events, never-seen, alert)"""
    return pd.DataFrame([{"group_id": g, "first_seen": T + pd.Timedelta(minutes=a), "last_seen": T + pd.Timedelta(minutes=b),
                          "judged_at": T + pd.Timedelta(minutes=b + 5), "n_events": n, "n_unseen_events": u,
                          "score": 0.9 if alert else 0.1, "is_anomaly": alert} for g, a, b, n, u, alert in rows])


def test_grade_counts_a_cut_group_once_and_gives_false_alarms_per_day():
    f = frame([("a", 0, 10, 5, 0, False), ("a", 20, 30, 1, 0, True),   # one group cut in two: flagged once
               ("b", 0, 10, 4, 0, True), ("c", 0, 10, 4, 0, False), ("d", 0, 2875, 4, 0, False)])
    g = dd.grade(f, pd.Series({"a": 1, "b": 0, "c": 1, "d": 0}))
    assert (g["caught"], g["false_alarms"], g["missed"], g["correct_normal"]) == (1, 1, 1, 1)
    assert g["precision"] == 0.5 and g["recall"] == 0.5 and g["groups"] == 4
    assert g["days"] == 2.0 and g["false_alarms_per_day"] == 0.5       # 2 days of log time, 1 false alarm


def test_alerts_over_time_split_by_truth_and_stacked():
    # T is 04:40: a and b end in the 04:00 hour, c in the 06:00 hour
    f = frame([("a", 0, 5, 5, 0, True), ("b", 0, 15, 5, 0, True), ("c", 70, 80, 5, 0, True), ("d", 0, 5, 5, 0, False)])
    plain = dd.alerts_over_time(f, "h")
    assert plain["alerts"].tolist() == [2, 1]
    kinds = dd.alerts_over_time(f, "h", pd.Series({"a": 1, "b": 0, "c": 1}))
    first_hour = dd.stacked(kinds, ["real anomaly", "false alarm"]).query("time == @T.floor('h')")
    assert first_hour[["kind", "bottom", "top"]].values.tolist() == [["real anomaly", 0, 1], ["false alarm", 1, 2]]


def test_headline_drift_and_time_range():
    f = frame([("a", 0, 10, 8, 2, True), ("b", 0, 15, 2, 0, False)])   # both end in the 04:00 hour
    h = dd.headline(f, {"log_time": "2005-11-09 06:00:00", "lines": 10, "open_groups": 3})
    assert (h["judged"], h["alerts"], h["alert_share"], h["unseen_share"]) == (2, 1, 0.5, 0.2)
    assert h["median_delay"] == pd.Timedelta(minutes=5) and h["open_groups"] == 3
    assert dd.drift_over_time(f, "h")["unseen_share"].tolist() == [0.2]
    start, end = dd.time_range(f, {"log_time": "2005-11-09 06:00:00"}, "h")
    assert (start, end) == (T.floor("h"), pd.Timestamp("2005-11-09 07:00:00"))
    assert dd.time_step(f) == "h"


def test_model_results_have_both_splits_with_readable_names():
    r = dd.model_results("bgl")
    assert set(r["split_name"]) == {"random split", "time split"}
    assert "FATAL rule" in set(r["model_name"]) and not r["model_name"].str.contains("_").any()


def result(gid, minute, alert):
    t = (T + pd.Timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M:%S")
    return {"dataset": "bgl", "detector": "rule_fatal", "group_id": gid, "first_seen": t, "last_seen": t,
            "finished_at": t, "n_events": 4, "n_unseen_events": 1, "score": 1.0 if alert else 0.0, "is_anomaly": alert,
            "raw_score": float(alert), "threshold": 1.0, "unseen_examples": ["a new message"],
            "reasons": ["1 line(s) with level FATAL/FAILURE"] if alert else []}


def test_dashboard_page_runs_without_errors(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'alerts.db'}"
    status = {"run_id": "bgl-test-run", "dataset": "bgl", "detector": "rule_fatal", "lines": 12,
              "log_time": "2005-11-09 05:00:00", "watching_since": "2005-11-09 04:40:00", "open_groups": 1,
              "not_judged_started_before_watching": 0}
    ResultStore(url).save(status, [result("1", 0, True), result("2", 5, True), result("3", 10, False)])
    monkeypatch.setenv("DATABASE_URL", url)                                   # the page uses this test store
    monkeypatch.setattr(dd, "true_labels", lambda dataset: pd.Series({"1": 1, "2": 0, "3": 0}))

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py"), default_timeout=60)
    app.run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Alerts"] == "2" and metrics["Groups judged"] == "3" and metrics["Lines read"] == "12"
    assert metrics["Real anomalies caught"] == "1 of 1" and metrics["False alarms"] == "1"
    assert metrics["Log messages never seen in training"] == "25.0%"
