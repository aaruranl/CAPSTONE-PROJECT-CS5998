"""Unit tests for the EDA helper functions, on a tiny hand-made dataset."""
import pandas as pd

from src.eda import anomaly_only_events, event_group_counts, label_conflicts, pattern_key, pattern_overlap

# 6 groups: event 9 only ever appears in abnormal groups
SEQS = pd.DataFrame({
    "event_ids": [[1, 2, 3], [1, 2, 3], [3, 2, 1], [1, 9], [1, 2, 3], [1, 9, 9]],
    "label":     [0,         0,         0,         1,      1,         1],
    "split":     ["train",   "train",   "test",    "train", "test",   "test"],
})


def test_pattern_key_ignores_order_but_not_counts():
    assert pattern_key([1, 2, 3]) == pattern_key([3, 2, 1])
    assert pattern_key([1, 9]) != pattern_key([1, 9, 9])


def test_event_group_counts():
    counts = event_group_counts(SEQS)
    assert counts.loc[1, "groups_normal"] == 3 and counts.loc[1, "groups_abnormal"] == 3
    assert counts.loc[9, "groups_normal"] == 0 and counts.loc[9, "groups_abnormal"] == 2  # counted once per group
    assert counts.loc[9, "abnormal_share_%"] == 100


def test_pattern_overlap():
    out = pattern_overlap(SEQS, "split")
    assert out["test_normal_seen_in_train_%"] == 100      # [3,2,1] has the same counts as [1,2,3]
    assert out["test_abnormal_seen_in_train_%"] == 50     # [1,2,3] seen, [1,9,9] not


def test_label_conflicts():
    out = label_conflicts(SEQS)                            # pattern {1,2,3} occurs as normal AND abnormal
    assert out["patterns_with_both_labels"] == 1
    assert out["abnormal_groups_in_conflict"] == 1


def test_anomaly_only_events_rule():
    out = anomaly_only_events(SEQS, "split")
    assert out["anomaly_only_events"] == [9]
    assert out["test_abnormal_caught_%"] == 50            # catches [1,9,9], misses [1,2,3]
    assert out["test_normal_flagged"] == 0
