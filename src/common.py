"""Small helpers shared by the HDFS and BGL preprocessing code."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.log_parser import DrainParser


def count_lines(path: Path) -> int:
    """Fast line count (reads raw bytes)."""
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            n += chunk.count(b"\n")
    return n


class CategoryEncoder:
    """Maps strings (e.g. 'INFO', 'WARN') to small integer codes to save memory."""

    def __init__(self):
        self.codes: dict[str, int] = {}

    def encode(self, value: str) -> int:
        code = self.codes.get(value)
        if code is None:
            code = self.codes[value] = len(self.codes)
        return code

    def to_categorical(self, codes: np.ndarray) -> pd.Categorical:
        categories = sorted(self.codes, key=self.codes.get)
        return pd.Categorical.from_codes(codes, categories=categories)


def templates_table(parser: DrainParser, event_ids: np.ndarray, learned_mask: np.ndarray) -> pd.DataFrame:
    """One row per template with how often it occurs overall and in the learning period."""
    templates = parser.templates()
    size = max(templates) + 1
    valid = event_ids >= 0  # -1 marks malformed lines
    total = np.bincount(event_ids[valid], minlength=size)
    learned = np.bincount(event_ids[valid & learned_mask], minlength=size)
    df = pd.DataFrame({
        "event_id": list(templates),
        "template": list(templates.values()),
        "occurrences": total[list(templates)],
        "occurrences_learning_period": learned[list(templates)],
    })
    return df.sort_values("occurrences", ascending=False).reset_index(drop=True)


class Timer:
    def __init__(self):
        self.start = time.time()

    def seconds(self) -> float:
        return round(time.time() - self.start, 1)


def write_summary(path: Path, summary: dict) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        raise TypeError(type(o))

    path.write_text(json.dumps(summary, indent=2, default=default))
