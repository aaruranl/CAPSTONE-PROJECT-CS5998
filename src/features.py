"""
Turn each group's list of events into a fixed-length vector of numbers, the input
format that the count-based models (Logistic Regression, Isolation Forest) need.

Example - the vocabulary learned from the training groups is [3, 5, 9]:

    events in a group   [5, 5, 9, 12]
    counts              E3=0  E5=2  E9=1  other=1     (12 was never seen in training)
    log(1 + count)      0.00  1.10  0.69  0.69

The order of events is thrown away ("bag of events"); only how often each occurs
is kept. Order is what DeepLog adds later.
"""
import numpy as np


class EventCountFeatures:
    """Event-count vectors with a vocabulary learned from training groups only.

    Events that are not in the vocabulary (new templates, or the parser's event 0
    for unseen messages) are summed into one extra column called "other", so the
    number of columns stays fixed however much the logs change later.
    """

    def fit(self, event_lists) -> "EventCountFeatures":
        self.events = np.unique(np.concatenate(list(event_lists))).astype(np.int64)
        return self

    @property
    def n_features(self) -> int:
        return len(self.events) + 1  # + "other"

    @property
    def feature_names(self) -> list[str]:
        return [f"E{e}" for e in self.events] + ["other"]

    def counts(self, event_lists) -> np.ndarray:
        """Raw counts, shape (n_groups, n_features)."""
        event_lists = list(event_lists)
        lengths = np.fromiter((len(e) for e in event_lists), dtype=np.int64, count=len(event_lists))
        flat = np.concatenate(event_lists).astype(np.int64)
        idx = np.minimum(np.searchsorted(self.events, flat), len(self.events) - 1)
        col = np.where(self.events[idx] == flat, idx, len(self.events))  # unknown -> "other"
        row = np.repeat(np.arange(len(event_lists)), lengths)
        n_col = self.n_features
        flat_index = row * n_col + col
        return np.bincount(flat_index, minlength=len(event_lists) * n_col).reshape(-1, n_col)

    def transform(self, event_lists) -> np.ndarray:
        """log(1 + count): tames very skewed counts (1 to 42,272 per BGL window)."""
        return np.log1p(self.counts(event_lists)).astype(np.float32)
