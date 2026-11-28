"""
Scoring engine (step 4): raw log lines in -> one scored result per group out.

    raw lines -> read the header -> mask + FROZEN parser -> group -> detector -> score 0-1, label, reasons
                 (hdfs.py, bgl.py)   (log_parser.py)               (models from step 3)

Nothing is re-implemented: the engine reuses the line readers, the masking, the
saved parser, the features, the trained models and the thresholds of the offline
pipeline. So a line is processed online exactly as it was when the models were
evaluated (no "training-serving skew"); scripts/08_check_engine.py proves this.

The engine is stateless: it scores whatever lines it is given. Deciding WHEN a group
is complete (e.g. a 5-minute BGL window has closed) is the job of the API (next step).
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.baselines import anomaly_scores
from src.bgl import parse_bgl_line
from src.config import BGL_WINDOW_MINUTES, PROCESSED_DIR
from src.deeplog import DeepLog
from src.experiment import MODELS_DIR
from src.hdfs import BLOCK_ID, parse_hdfs_line
from src.log_parser import UNSEEN_EVENT_ID, DrainParser
from src.rules import FATAL_LEVELS

# Served by default: the best detector on the TIME split (docs/03_models.md, section 9)
DEFAULT_DETECTOR = {"hdfs": "logreg", "bgl": "rule_fatal"}
DETECTORS = {"hdfs": ("logreg", "iforest", "deeplog"), "bgl": ("rule_fatal", "logreg", "iforest", "deeplog")}
WINDOW_S = BGL_WINDOW_MINUTES * 60
MAX_EXAMPLES = 3  # how many example messages / reasons to keep per group


# ----------------------------------------------------------------------------- reading lines
def read_hdfs(line: str):
    """-> (unix_seconds, level, content, group IDs) or None. A line can mention several blocks."""
    parsed = parse_hdfs_line(line)
    if parsed is None:
        return None
    ts, level, _component, content = parsed
    return ts, level, content, sorted(set(BLOCK_ID.findall(content)))


def read_bgl(line: str):
    """-> (unix_seconds, level, content, [window ID]) or None.

    The first field of a BGL line is the alert LABEL. It is read but never used:
    in real use labels do not exist, and the models never saw them as input.
    """
    parsed = parse_bgl_line(line)
    if parsed is None:
        return None
    _label, ts, _node, _component, level, content = parsed
    return ts, level, content, [str(ts // WINDOW_S)]


READERS = {"hdfs": read_hdfs, "bgl": read_bgl}


@dataclass
class Group:
    """All events of one HDFS block or one BGL 5-minute window, in arrival order."""
    group_id: str
    first_time: int
    last_time: int
    events: list[int] = field(default_factory=list)
    n_fatal: int = 0                                         # lines with level FATAL / FAILURE
    fatal_examples: list[int] = field(default_factory=list)  # their event IDs (first few)
    unseen_examples: list[str] = field(default_factory=list)  # raw text of never-seen messages

    def add(self, ts: int, event_id: int, level: str, content: str) -> None:
        self.events.append(event_id)
        self.first_time, self.last_time = min(self.first_time, ts), max(self.last_time, ts)
        if level in FATAL_LEVELS:
            self.n_fatal += 1
            if len(self.fatal_examples) < MAX_EXAMPLES:
                self.fatal_examples.append(event_id)
        if event_id == UNSEEN_EVENT_ID and len(self.unseen_examples) < MAX_EXAMPLES:
            self.unseen_examples.append(content)


# ----------------------------------------------------------------------------- detectors
class CountModelDetector:
    """Logistic Regression or Isolation Forest on event counts (saved by scripts/04_baselines.py)."""

    def __init__(self, name: str, path):
        saved = joblib.load(path)
        self.name, self.features, self.model, self.threshold = name, saved["features"], saved["model"], saved["threshold"]
        self.normal_mean, self.normal_mean_counts = saved["normal_mean"], saved["normal_mean_counts"]
        self.max_score = 1.0  # LR gives a probability; Isolation Forest's score is also between 0 and 1

    def raw_scores(self, groups: list[Group]) -> np.ndarray:
        return anomaly_scores(self.model, self.features.transform([np.asarray(g.events) for g in groups]))

    def reasons(self, group: Group, describe) -> list[str]:
        if not isinstance(self.model, LogisticRegression):
            return ["Isolation Forest gives no per-event reasons: the group as a whole was unusual"]
        # LR's score is a weighted sum, so compared with the AVERAGE NORMAL group it splits exactly into
        # one part per event: weight x (this group's value - normal value). Positive parts raised the score.
        counts = self.features.counts([np.asarray(group.events)])[0]
        contribution = self.model.coef_[0] * (np.log1p(counts) - self.normal_mean)
        out = []
        for i in np.argsort(-contribution)[:MAX_EXAMPLES]:
            if contribution[i] <= 0:
                break
            is_other = i == len(self.features.events)
            what = "messages never seen in training" if is_other else describe(self.features.events[i])
            change = "more" if counts[i] > self.normal_mean_counts[i] else "fewer"
            out.append(f"{change} {what} than a normal group ({counts[i]} vs about "
                       f"{self.normal_mean_counts[i]:.2g}), +{contribution[i]:.2f}")
        return out


class DeepLogDetector:
    def __init__(self, path):
        self.name = "deeplog"
        self.model, self.threshold = DeepLog.load(path)  # threshold = g
        self.max_score = float(self.model.vocab.size)    # worst possible rank

    def raw_scores(self, groups: list[Group]) -> np.ndarray:
        return self.model.score([np.asarray(g.events) for g in groups]).astype(float)

    def reasons(self, group: Group, describe) -> list[str]:
        out = []
        for s in self.model.surprises(group.events, self.threshold)[:MAX_EXAMPLES]:
            expected = "; ".join(describe(e) for e in s["expected"])
            out.append(f"step {s['position']}: got {describe(s['actual'])}, expected one of: {expected}")
        return out


class FatalRuleDetector:
    """BGL: alert when a window contains a FATAL or FAILURE line. Needs no saved model."""

    def __init__(self):
        self.name, self.threshold, self.max_score = "rule_fatal", 1.0, 1.0

    def raw_scores(self, groups: list[Group]) -> np.ndarray:
        return np.array([1.0 if g.n_fatal else 0.0 for g in groups])

    def reasons(self, group: Group, describe) -> list[str]:
        return [f"{group.n_fatal} line(s) with level FATAL/FAILURE"] + \
               [f"e.g. {describe(e)}" for e in dict.fromkeys(group.fatal_examples)]


def load_detector(dataset: str, name: str, split: str = "time"):
    """The live system uses the TIME-split models: trained on the past, scoring the future."""
    if name == "rule_fatal":
        if dataset != "bgl":
            raise ValueError("the FATAL rule needs BGL's severity levels")
        return FatalRuleDetector()
    if name in ("logreg", "iforest"):
        return CountModelDetector(name, MODELS_DIR / f"{dataset}_{split}_{name}.joblib")
    if name == "deeplog":
        return DeepLogDetector(MODELS_DIR / f"{dataset}_{split}_deeplog.pt")
    raise ValueError(f"unknown detector {name!r}; choose from {DETECTORS[dataset]}")


def to_unit_score(raw, threshold: float, max_score: float) -> np.ndarray:
    """Map any detector's raw score to 0-1 so that THE ALERT THRESHOLD IS ALWAYS 0.5.

    Below the threshold: 0 -> 0.0 ... threshold -> 0.5 (straight line)
    Above the threshold: threshold -> 0.5 ... max_score -> 1.0
    So 'score >= 0.5' means 'alert' for every detector, and the order of groups is kept.
    """
    raw = np.clip(np.asarray(raw, dtype=float), 0.0, max_score)
    below = 0.5 * raw / threshold if threshold > 0 else np.ones_like(raw)
    above = 0.5 + 0.5 * (raw - threshold) / (max_score - threshold) if max_score > threshold else np.ones_like(raw)
    return np.where(raw < threshold, below, above)


def _time(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------------- engine
class ScoringEngine:
    def __init__(self, dataset: str, parser: DrainParser, templates: dict[int, str], detector):
        self.dataset, self.parser, self.templates, self.detector = dataset, parser, templates, detector
        self.read = READERS[dataset]
        self.last_stats: dict = {}

    @classmethod
    def load(cls, dataset: str, detector: str | None = None, split: str = "time") -> "ScoringEngine":
        """Load the saved parser, templates and a trained detector from disk."""
        tpl = pd.read_csv(PROCESSED_DIR / dataset / "templates.csv")
        return cls(dataset, DrainParser.load(PROCESSED_DIR / dataset / "parser.pkl"),
                   dict(zip(tpl["event_id"], tpl["template"])),
                   load_detector(dataset, detector or DEFAULT_DETECTOR[dataset], split))

    def describe(self, event) -> str:
        """Readable name of an event for explanations."""
        if event == "END":
            return "END (no more events)"
        if event == UNSEEN_EVENT_ID:
            return "a never-seen message"
        return f'E{event} "{self.templates.get(int(event), "?")[:80]}"'

    @staticmethod
    def new_stats() -> dict:
        return {"lines": 0, "malformed": 0, "without_group": 0, "unseen_messages": 0}

    def add_line(self, groups: dict[str, Group], line: str, stats: dict):
        """Read and parse ONE raw line (frozen parser, never learns) and add it to its group(s).

        Returns (timestamp, group IDs) or None for a malformed line. Used both for a whole
        batch (group_lines) and for live streams (src/stream.py), so both behave identically.
        """
        stats["lines"] += 1
        parsed = self.read(line)
        if parsed is None:
            stats["malformed"] += 1
            return None
        ts, level, content, group_ids = parsed
        event_id = self.parser.match(content)
        stats["unseen_messages"] += event_id == UNSEEN_EVENT_ID
        if not group_ids:
            stats["without_group"] += 1
        for gid in group_ids:
            group = groups.get(gid)
            if group is None:
                group = groups[gid] = Group(gid, ts, ts)
            group.add(ts, event_id, level, content)
        return ts, group_ids

    def group_lines(self, lines) -> list[Group]:
        """Treat the given lines as complete: parse and group them, keeping arrival order."""
        groups: dict[str, Group] = {}
        stats = self.new_stats()
        for line in lines:
            self.add_line(groups, line, stats)
        self.last_stats = stats
        return list(groups.values())

    def score_groups(self, groups: list[Group]) -> list[dict]:
        if not groups:
            return []
        d = self.detector
        raw = d.raw_scores(groups)
        unit = to_unit_score(raw, d.threshold, d.max_score)
        results = []
        for g, r, u in zip(groups, raw, unit):
            alert = bool(r >= d.threshold)  # exactly the rule used in evaluation
            results.append({
                "dataset": self.dataset,
                "group_id": g.group_id,
                "detector": d.name,
                "first_seen": _time(g.first_time),
                "last_seen": _time(g.last_time),
                "n_events": len(g.events),
                "n_unseen_events": int(sum(e == UNSEEN_EVENT_ID for e in g.events)),
                "score": round(float(u), 4),
                "is_anomaly": alert,
                "raw_score": float(r),
                "threshold": float(d.threshold),
                "reasons": d.reasons(g, self.describe) if alert else [],
                "unseen_examples": g.unseen_examples,
            })
        return results

    def score_lines(self, lines) -> list[dict]:
        return self.score_groups(self.group_lines(lines))
