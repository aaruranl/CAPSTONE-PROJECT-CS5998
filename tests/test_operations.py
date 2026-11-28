"""Tests for the operational measures (src/operations.py) and the refined FATAL rule (src/rules.py)."""
import numpy as np
import pandas as pd
import pytest

from src.operations import budget_threshold, days_covered, notifications, pattern, split_days
from src.rules import fatal_window_ids_except, harmless_fatal_events

T0 = pd.Timestamp("2005-11-09 00:00:00")


def test_days_time_split_is_its_own_period_random_split_a_share_of_the_whole():
    seqs = pd.DataFrame({"start_time": [T0 + pd.Timedelta(days=d) for d in range(11)],   # 10 days of log
                         "split_time": ["train"] * 8 + ["test"] * 3,
                         "split_random": ["test", "train"] * 5 + ["train"]})
    assert split_days(seqs, "split_time", "test") == 2.0                    # days 8, 9, 10
    assert split_days(seqs, "split_random", "test") == pytest.approx(10 * 5 / 11)  # 5 of 11 groups over 10 days
    assert days_covered(seqs["start_time"][:1], pd.Timedelta(minutes=5)) == 5 / 1440   # one BGL window


def test_budget_threshold_keeps_ties_together_even_with_float32_scores():
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.1], dtype=np.float32)          # three tied normal scores
    assert (scores >= budget_threshold(scores, 2)).sum() == 1               # 3 tied cannot fit in 2: all out
    assert (scores >= budget_threshold(scores, 4)).sum() == 4
    assert (scores >= budget_threshold(scores, 0)).sum() == 0
    assert (scores >= budget_threshold(scores, 9)).sum() == 5               # budget larger than everything


def test_notifications_bundle_alerts_per_10_minutes():
    ends = pd.Series([T0 + pd.Timedelta(minutes=m) for m in (1, 5, 12, 31)])
    alert = np.array([True, True, True, False])
    abnormal = np.array([False, True, False, False])
    # 00:00-00:10 has a real anomaly; 00:10-00:20 only a false alarm; 00:30 has no alert
    assert notifications(ends, alert, abnormal) == {"notifications": 2, "false_notifications": 1}


def test_pattern_is_what_a_count_model_sees():
    assert pattern([5, 1, 5]) == pattern([1, 5, 5]) == "E1x1 E5x2"


def test_harmless_fatal_messages_are_learned_from_training_only():
    def rows(event, level, alerts, n, second=0):
        return [{"timestamp": T0 + pd.Timedelta(seconds=second + i), "event_id": event, "level": level,
                 "is_alert": int(i < alerts)} for i in range(n)]
    train = pd.DataFrame(rows(10, "FATAL", 0, 25) + rows(11, "FATAL", 1, 25) + rows(12, "FAILURE", 0, 3)
                         + rows(0, "FATAL", 0, 50) + rows(13, "INFO", 0, 30))
    # 10: FATAL 25x, never an alert -> harmless; 11: once an alert; 12: too rare; 0: never-seen; 13: not FATAL
    assert harmless_fatal_events(train, min_lines=20) == {10}
    assert harmless_fatal_events(train, min_lines=3) == {10, 12}
    test = pd.DataFrame(rows(10, "FATAL", 0, 1, second=0) + rows(11, "FATAL", 0, 1, second=600))
    assert fatal_window_ids_except(test, {10}).tolist() == [int(T0.timestamp() + 600) // 300]
