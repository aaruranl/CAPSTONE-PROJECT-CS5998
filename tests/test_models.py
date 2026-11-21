"""Tests for step 3: features, metrics, rules and DeepLog (tiny inputs, run in seconds)."""
import numpy as np
import pandas as pd
import pytest

from src.deeplog import END, PAD, UNK, DeepLog, DeepLogConfig, EventVocab, make_windows
from src.features import EventCountFeatures
from src.metrics import best_f1_threshold, confusion, evaluate, f1_confidence_interval
from src.rules import anomaly_only_event_set, contains_any, fatal_window_ids


def arrays(*groups):
    return [np.array(g) for g in groups]


# ------------------------------------------------------------------ features
def test_count_features_use_training_vocabulary_and_other_column():
    f = EventCountFeatures().fit(arrays([3, 5], [9]))
    assert f.feature_names == ["E3", "E5", "E9", "other"]
    counts = f.counts(arrays([5, 5, 9, 12], [3]))
    assert counts.tolist() == [[0, 2, 1, 1], [1, 0, 0, 0]]  # 12 is unknown -> "other"
    assert np.allclose(f.transform(arrays([5, 5]))[0], np.log1p([0, 2, 0, 0]))


# ------------------------------------------------------------------- metrics
def test_best_threshold_and_confusion():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    threshold, f1 = best_f1_threshold(y, scores)
    assert threshold == 0.7 and f1 == 1.0
    assert confusion(y, scores >= 0.6) == {"tp": 2, "fp": 1, "fn": 0, "tn": 2}


def test_f1_interval_contains_estimate_and_is_reproducible():
    m = evaluate(np.array([1] * 40 + [0] * 960), np.array([1] * 30 + [0] * 10 + [1] * 10 + [0] * 950))
    assert m["f1"] == pytest.approx(0.75)
    assert m["f1_ci_low"] < m["f1"] < m["f1_ci_high"]
    assert f1_confidence_interval(30, 10, 10, 950) == f1_confidence_interval(30, 10, 10, 950)


# --------------------------------------------------------------------- rules
def test_anomaly_only_rule():
    train = pd.DataFrame({"event_ids": arrays([1, 2], [1, 3], [1, 4]), "label": [0, 1, 1]})
    events = anomaly_only_event_set(train)
    assert events == {3, 4}
    assert contains_any(arrays([1, 2], [2, 4]), events).tolist() == [False, True]


def test_fatal_windows_use_preprocessing_window_numbering():
    lines = pd.DataFrame({
        "timestamp": pd.to_datetime([1117838400 + 10, 1117838400 + 400], unit="s"),
        "level": pd.Categorical(["FATAL", "INFO"]),
    })
    assert fatal_window_ids(lines).tolist() == [1117838400 // 300]


# ------------------------------------------------------------------- DeepLog
def test_vocab_maps_unknown_events_to_unk():
    vocab = EventVocab([7, 3])
    assert vocab.size == 5
    assert vocab.encode([3, 7, 99]).tolist() == [3, 4, UNK]


def test_windows_are_padded_end_marked_and_deduplicated():
    unique, inverse, counts, lengths = make_windows(arrays([5, 6], [5, 6]), h=2)
    assert lengths.tolist() == [3, 3]  # n events + END
    assert counts.sum() == 6 and len(unique) == 3  # the two groups are identical
    assert unique[inverse[:3]].tolist() == [[PAD, PAD, 5], [PAD, 5, 6], [5, 6, END]]


def test_deeplog_learns_order_and_ending():
    normal = arrays(*([[3, 4, 5, 6]] * 50))
    cfg = DeepLogConfig(window=3, embed_dim=8, hidden=16, layers=1, epochs=150, batch_size=64, lr=0.02)
    model = DeepLog(cfg).fit(normal, log=lambda *_: None)
    scores = model.score(arrays([3, 4, 5, 6], [3, 5, 4, 6], [3, 4], [3, 4, 9, 6]))
    normal_score, swapped, truncated, unseen = scores
    assert normal_score == 0                      # every next event was the top-1 guess
    assert swapped > 0 and truncated > 0           # wrong order / stopped too early are surprising
    assert unseen == model.vocab.size              # an event never seen in training -> maximum score
    assert model.score(arrays([3, 4, 9, 6]), ignore_unseen=True)[0] < unseen
