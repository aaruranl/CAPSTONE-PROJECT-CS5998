"""
Milestone 3, step 2 - when is a live HDFS block "finished"? (the quiet time)

An HDFS block has no "end" message, so the live system declares it finished after
Q hours of silence. Too short: a normal block is cut in two, and the pieces look
abnormal (false alarms). Too long: alerts arrive late.

Uses only TRAINING + VALIDATION data (never test):
  1. how long are the silences inside normal blocks?
  2. simulation: cut every block wherever it is silent for more than Q, score the
     pieces with the time-split Logistic Regression, count false alarms and delay.
The chosen Q is the smallest value that (almost) never cuts a normal block in two.

Usage:
    python scripts/09_choose_quiet_time.py   # ~2 min -> reports/results/quiet_time.csv
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATASETS, HDFS_QUIET_HOURS, PROCESSED_DIR  # noqa: E402
from src.engine import read_hdfs  # noqa: E402
from src.experiment import MODELS_DIR, RESULTS_DIR  # noqa: E402

GRID_HOURS = [0.5, 1, 2, 4, 6, 8, 10, 12, 13, 14, 16]

# 1. timestamps of every event of every block, in file order (same order as sequences.parquet)
times: dict[str, list[int]] = {}
with open(DATASETS["HDFS"]["log_file"], encoding="utf-8", errors="replace") as f:
    for line in f:
        parsed = read_hdfs(line)
        if parsed is not None:
            for block in parsed[3]:
                times.setdefault(block, []).append(parsed[0])

seqs = pd.read_parquet(PROCESSED_DIR / "hdfs" / "sequences.parquet", columns=["block_id", "event_ids", "label", "split_time"])
seqs = seqs[seqs["split_time"] != "test"].reset_index(drop=True)  # NEVER look at the test period
seqs["ts"] = seqs["block_id"].map(times)
silence_h = seqs["ts"].map(lambda t: np.diff(t).max() / 3600 if len(t) > 1 else 0.0)
normal = seqs["label"] == 0
pct = np.percentile(silence_h[normal], [50, 90, 99, 99.9, 100])
print("Longest silence inside NORMAL blocks (train + validation), hours:")
print("  median %.2f | 90%% %.2f | 99%% %.2f | 99.9%% %.2f | max %.2f" % tuple(pct))

templates = pd.read_csv(PROCESSED_DIR / "hdfs" / "templates.csv").set_index("event_id")["template"]
after = seqs[normal].apply(lambda r: int(r.event_ids[np.argmax(np.diff(r.ts)) + 1]) if len(r.ts) > 1 else -1, axis=1)
print("Event that comes right after the longest silence (normal blocks):")
for event, n in after.value_counts().head(3).items():
    print(f"  E{event} {templates.get(event, '')[:60]!r}: {100 * n / normal.sum():.1f}%")

# 2. simulate the live rule for every Q
saved = joblib.load(MODELS_DIR / "hdfs_time_logreg.joblib")
features, model, threshold = saved["features"], saved["model"], saved["threshold"]
y = seqs["label"].to_numpy().astype(bool)
rows = []
for period, mask in [("train+validation", np.ones(len(seqs), bool)), ("validation only", (seqs["split_time"] == "val").to_numpy())]:
    part = seqs[mask].reset_index(drop=True)
    y_part = y[mask]
    for q in GRID_HOURS:
        pieces, owner, alert_time = [], [], []
        for i, (events, ts) in enumerate(zip(part["event_ids"], part["ts"])):
            ts = np.asarray(ts)
            cuts = np.flatnonzero(np.diff(ts) > q * 3600) + 1
            for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(ts)]):
                pieces.append(events[a:b])
                owner.append(i)
                alert_time.append(ts[b - 1] + q * 3600 - ts[0])  # piece finished -> scored
        owner = np.array(owner)
        flagged_piece = model.predict_proba(features.transform(pieces))[:, 1] >= threshold
        flagged = np.zeros(len(part), bool)
        np.logical_or.at(flagged, owner, flagged_piece)
        tp, fp, fn = (flagged & y_part).sum(), (flagged & ~y_part).sum(), (~flagged & y_part).sum()
        first_alert = pd.Series(np.array(alert_time)[flagged_piece]).groupby(owner[flagged_piece]).min()
        delay = first_alert[first_alert.index.isin(np.flatnonzero(y_part))] / 3600
        rows.append({"period": period, "quiet_hours": q,
                     "normal_blocks_cut_%": round(100 * (np.bincount(owner)[~y_part] > 1).mean(), 3),
                     "false_alarms": int(fp), "f1": round(2 * tp / (2 * tp + fp + fn), 4),
                     "median_hours_to_alert": round(float(delay.median()), 1)})

table = pd.DataFrame(rows)
print("\n" + table.to_string(index=False))
main = table[table["period"] == "train+validation"]
chosen = main.loc[main["normal_blocks_cut_%"] < 0.01, "quiet_hours"].min()
print(f"\nChosen quiet time: {chosen} h (config.HDFS_QUIET_HOURS = {HDFS_QUIET_HOURS})")
if chosen != HDFS_QUIET_HOURS:
    print("WARNING: config.HDFS_QUIET_HOURS differs from the data-driven choice")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
table.to_csv(RESULTS_DIR / "quiet_time.csv", index=False)
print(f"Saved {RESULTS_DIR / 'quiet_time.csv'}")
