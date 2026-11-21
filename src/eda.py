"""
Reusable analysis functions for the exploratory data analysis (EDA).

Every function works on a "sequences" table (one row per group, as produced by
step 1) with at least the columns `event_ids` (list of event IDs) and `label`.
"""
import numpy as np
import pandas as pd


def event_group_counts(seqs: pd.DataFrame) -> pd.DataFrame:
    """For every event ID: in how many normal / abnormal groups does it appear (at least once)?

    Also gives `abnormal_share` = of all groups that contain the event, the % that are
    abnormal. An event with a share near 100% is a strong anomaly signal on its own.
    """
    lengths = seqs["event_ids"].map(len).to_numpy()
    flat = np.concatenate(seqs["event_ids"].to_numpy()).astype(np.int64)
    group = np.repeat(np.arange(len(seqs)), lengths)
    n_events = int(flat.max()) + 1
    pairs = np.unique(group * n_events + flat)          # each (group, event) once
    g, e = pairs // n_events, pairs % n_events
    labels = seqs["label"].to_numpy()[g]
    out = pd.DataFrame({
        "groups_normal": np.bincount(e[labels == 0], minlength=n_events),
        "groups_abnormal": np.bincount(e[labels == 1], minlength=n_events),
    })
    out.index.name = "event_id"
    total = out["groups_normal"] + out["groups_abnormal"]
    out["abnormal_share_%"] = (100 * out["groups_abnormal"] / total.where(total > 0)).round(2)
    return out[total > 0]


def pattern_key(event_ids) -> bytes:
    """A fingerprint of a group's event *counts* (order ignored).

    Two groups get the same key exactly when every event occurs the same number of
    times in both, i.e. when count-based models (Logistic Regression, Isolation
    Forest) see identical input vectors.
    """
    return np.sort(np.asarray(event_ids, dtype=np.int32)).tobytes()


def pattern_overlap(seqs: pd.DataFrame, split_col: str) -> dict:
    """% of test groups whose exact count pattern also occurs in the training set.

    A high value means the model can score many test groups by "remembering" a
    training group, so the test is easier than predicting genuinely new behaviour.
    """
    keys = seqs["event_ids"].map(pattern_key)
    train_keys = set(keys[seqs[split_col] == "train"])
    test = seqs[split_col] == "test"
    seen = keys[test].map(lambda k: k in train_keys)
    labels = seqs.loc[test, "label"]
    return {
        "test_normal_seen_in_train_%": round(100 * seen[labels == 0].mean(), 2),
        "test_abnormal_seen_in_train_%": round(100 * seen[labels == 1].mean(), 2),
        "distinct_patterns_total": int(keys.nunique()),
    }


def label_conflicts(seqs: pd.DataFrame) -> dict:
    """Groups whose exact count pattern occurs with BOTH labels.

    A count-based model sees identical inputs for these groups, so it can never
    classify all of them correctly: a hard ceiling on what such models can reach.
    """
    keys = seqs["event_ids"].map(pattern_key)
    per_key = seqs.groupby(keys)["label"].agg(["min", "max"])
    both = set(per_key.index[(per_key["min"] == 0) & (per_key["max"] == 1)])
    # plain Python membership: pandas .isin() converts bytes to fixed-width numpy
    # strings, which silently drops trailing zero bytes and breaks the comparison
    in_conflict = keys.map(lambda k: k in both)
    labels = seqs["label"]
    return {
        "patterns_with_both_labels": len(both),
        "abnormal_groups_in_conflict": int((in_conflict & (labels == 1)).sum()),
        "abnormal_groups_in_conflict_%": round(100 * (in_conflict & (labels == 1)).sum() / max(labels.sum(), 1), 2),
        "normal_groups_in_conflict": int((in_conflict & (labels == 0)).sum()),
    }


def anomaly_only_events(seqs: pd.DataFrame, split_col: str) -> dict:
    """How far does a trivial rule get?

    Rule: 'flag a group if it contains any event that NEVER appeared in a normal
    training group'. The events are chosen from training data only (no leakage),
    then the rule is checked on the test set.
    """
    train = seqs[seqs[split_col] == "train"]
    counts = event_group_counts(train)
    only_abnormal = set(counts.index[(counts["groups_normal"] == 0) & (counts["groups_abnormal"] > 0)])
    test = seqs[seqs[split_col] == "test"]
    flagged = test["event_ids"].map(lambda s: not only_abnormal.isdisjoint(s))
    tp = int((flagged & (test["label"] == 1)).sum())
    fp = int((flagged & (test["label"] == 0)).sum())
    fn = int((~flagged & (test["label"] == 1)).sum())
    return {
        "anomaly_only_events": sorted(int(e) for e in only_abnormal),
        "test_abnormal_caught_%": round(100 * tp / max(tp + fn, 1), 2),   # recall
        "flags_that_are_correct_%": round(100 * tp / max(tp + fp, 1), 2),  # precision
        "test_normal_flagged": fp,
    }
