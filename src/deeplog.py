"""
DeepLog (Du, Li, Zheng & Srikumar, ACM CCS 2017) - the "improved method".

Idea: a normal system writes its logs in a predictable ORDER, like the grammar of
a language. An LSTM (a neural network that reads a sequence step by step and
keeps a memory of what it has read) is trained ONLY on normal groups to answer:

    "given the last h events, which event comes next?"

At test time we slide along each group. At every position the model ranks all
known events by how likely they are to come next. If the event that really
comes next is not among the g most likely ones (the "top-g"), that position is
surprising, and a group with at least one surprising position is flagged.

    window (h = 10)                     -> real next event   rank   top-g (g = 6)?
    [PAD .. PAD, E5]                    -> E22               0      yes, normal
    [PAD .. E5, E22, E11, E9, E11, E9]  -> E26               14     NO  -> alert

Additions to the paper, both motivated by the EDA (docs/02_eda.md, section 11):
  * END token: an end-of-sequence event is appended to every group, so the model
    also learns WHEN a group normally ends. 36.8% of HDFS anomalies are blocks that
    stop too early - without END they would never produce a surprising position.
  * UNK token: an event the model never saw in normal training cannot be
    predicted, so it always counts as surprising (as in the original paper).
  * PAD tokens fill the history at the start, so the first events are checked too.

Efficiency trick: HDFS has 8.5 million training windows but only ~34,000
DIFFERENT ones. Each distinct window is stored once with its count, and its
loss is multiplied by that count: mathematically the same objective as training
on all 8.5 million windows, at a fraction of the cost.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import RANDOM_SEED

PAD, END, UNK = 0, 1, 2  # special tokens; real events are numbered from 3
N_SPECIAL = 3


@dataclass
class DeepLogConfig:
    window: int = 10        # h: how many previous events the model looks at (paper: 10)
    embed_dim: int = 32     # each event becomes a vector of 32 learned numbers
    hidden: int = 64        # LSTM memory size (paper: 64)
    layers: int = 2         # stacked LSTM layers (paper: 2)
    epochs: int = 20        # passes over the training windows (40 gave no gain on validation)
    batch_size: int = 1024
    lr: float = 1e-3        # Adam learning rate
    seed: int = RANDOM_SEED


class EventVocab:
    """Maps event IDs seen in NORMAL training groups to tokens 3, 4, 5, ...; anything else -> UNK."""

    def __init__(self, events):
        self.events = np.unique(np.asarray(events, dtype=np.int64))

    @property
    def size(self) -> int:
        return len(self.events) + N_SPECIAL

    def encode(self, event_ids) -> np.ndarray:
        event_ids = np.asarray(event_ids, dtype=np.int64)
        idx = np.minimum(np.searchsorted(self.events, event_ids), len(self.events) - 1)
        return np.where(self.events[idx] == event_ids, idx + N_SPECIAL, UNK)


def make_windows(encoded_groups, h: int):
    """All (history of h tokens -> next token) windows of all groups, de-duplicated.

    Each group [e1 .. en] becomes PAD*h + [e1 .. en] + END, giving n + 1 windows.
    Returns
      unique  : (n_unique, h + 1) array - the distinct windows (last column = next token)
      inverse : for every window in reading order, the row of `unique` it equals
      counts  : how often each distinct window occurs
      lengths : number of windows per group (= group length + 1)
    """
    pad = np.full(h, PAD, dtype=np.int16)
    end = np.array([END], dtype=np.int16)
    parts = [np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, g.astype(np.int16), end]), h + 1)
             for g in encoded_groups]
    lengths = np.fromiter((len(p) for p in parts), dtype=np.int64, count=len(parts))
    windows = np.ascontiguousarray(np.concatenate(parts))
    as_bytes = windows.view(np.dtype((np.void, windows.dtype.itemsize * (h + 1)))).ravel()
    uniq, inverse, counts = np.unique(as_bytes, return_inverse=True, return_counts=True)
    unique = uniq.view(np.int16).reshape(-1, h + 1).astype(np.int64)
    return unique, inverse.ravel(), counts, lengths


class DeepLogLSTM(nn.Module):
    """event tokens -> embedding -> 2-layer LSTM -> one score per possible next token."""

    def __init__(self, vocab_size: int, cfg: DeepLogConfig):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.embed_dim)
        self.lstm = nn.LSTM(cfg.embed_dim, cfg.hidden, cfg.layers, batch_first=True)
        self.out = nn.Linear(cfg.hidden, vocab_size)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        states, _ = self.lstm(self.embed(history))
        return self.out(states[:, -1])  # read the memory after the last event of the window


class DeepLog:
    def __init__(self, cfg: DeepLogConfig | None = None):
        self.cfg = cfg or DeepLogConfig()
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------ training
    def fit(self, normal_groups, log=print) -> "DeepLog":
        """Train on NORMAL groups only (no anomalies are needed to learn 'normal')."""
        cfg = self.cfg
        normal_groups = list(normal_groups)
        self.vocab = EventVocab(np.concatenate(normal_groups))
        unique, _, counts, _ = make_windows([self.vocab.encode(g) for g in normal_groups], cfg.window)
        log(f"    {counts.sum():,} training windows, {len(unique):,} distinct, {self.vocab.size} tokens")

        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        self.model = DeepLogLSTM(self.vocab.size, cfg)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        history = torch.from_numpy(unique[:, :-1])
        target = torch.from_numpy(unique[:, -1])
        weight = torch.from_numpy((counts / counts.mean()).astype(np.float32))  # mean weight = 1

        self.model.train()
        for epoch in range(cfg.epochs):
            order = torch.from_numpy(rng.permutation(len(unique)))
            total = 0.0
            for start in range(0, len(unique), cfg.batch_size):
                b = order[start:start + cfg.batch_size]
                loss = (F.cross_entropy(self.model(history[b]), target[b], reduction="none") * weight[b]).mean()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += loss.item() * len(b)
            self.loss_history.append(total / len(unique))  # = average loss per training window
            if (epoch + 1) % 5 == 0:
                log(f"    epoch {epoch + 1:>2}/{cfg.epochs}  loss {self.loss_history[-1]:.4f}")
        return self

    # ------------------------------------------------------------------- scoring
    @torch.no_grad()
    def window_ranks(self, unique: np.ndarray) -> np.ndarray:
        """Rank of the real next token among all predictions (0 = the model's top guess)."""
        self.model.eval()
        ranks = np.empty(len(unique), dtype=np.int64)
        for start in range(0, len(unique), 8192):
            chunk = torch.from_numpy(unique[start:start + 8192])
            logits = self.model(chunk[:, :-1])
            logits[:, PAD] = -torch.inf  # PAD is never a real next event
            real = logits.gather(1, chunk[:, -1:])
            ranks[start:start + len(chunk)] = (logits > real).sum(1).numpy()
        return ranks

    def score(self, groups, ignore_unseen: bool = False) -> np.ndarray:
        """Anomaly score per group = the WORST rank in the group.

        A group is flagged at top-g exactly when its score >= g, so the score both
        ranks groups (for PR-AUC) and turns into DeepLog's rule with one threshold.
        ignore_unseen=True is a diagnostic only: UNK positions are treated as normal.
        """
        unique, inverse, _, lengths = make_windows([self.vocab.encode(g) for g in groups], self.cfg.window)
        ranks = self.window_ranks(unique)
        ranks[unique[:, -1] == UNK] = 0 if ignore_unseen else self.vocab.size  # unseen: always surprising
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        return np.maximum.reduceat(ranks[inverse], starts)

    # --------------------------------------------------------------- save / load
    def save(self, path, threshold: float) -> None:
        torch.save({"config": asdict(self.cfg), "events": self.vocab.events,
                    "state_dict": self.model.state_dict(), "threshold": threshold,
                    "loss_history": self.loss_history}, path)

    @classmethod
    def load(cls, path) -> tuple["DeepLog", float]:
        saved = torch.load(path, weights_only=False)
        obj = cls(DeepLogConfig(**saved["config"]))
        obj.vocab = EventVocab(saved["events"])
        obj.model = DeepLogLSTM(obj.vocab.size, obj.cfg)
        obj.model.load_state_dict(saved["state_dict"])
        obj.loss_history = saved["loss_history"]
        return obj, saved["threshold"]
