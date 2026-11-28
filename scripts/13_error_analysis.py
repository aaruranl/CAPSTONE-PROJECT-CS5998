"""
Step 9 (part 2) - error analysis of the two detectors the live system serves
(time split, test set): what exactly are their false alarms and missed anomalies?

  HDFS, Logistic Regression: each error's event-count PATTERN (what the model sees) is looked
       up in the training data: never seen, seen only as normal, only as abnormal, or both.
  BGL, FATAL rule: which FATAL/FAILURE message types make normal windows look abnormal, and
       were they ever part of a real alert in training?
  Then a follow-up suggested by the BGL errors: the FATAL rule WITHOUT the message types that
       were never alerts in training (list built from training data only).

Usage:
    python scripts/13_error_analysis.py   # ~30 s -> reports/results/error_analysis_hdfs.csv,
                                          #   error_analysis_bgl.csv, fatal_rule_refined.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import BGL_WINDOW_MINUTES, PROCESSED_DIR  # noqa: E402
from src.experiment import PRED_DIR, RESULTS_DIR, SPLITS  # noqa: E402
from src.metrics import f1_score  # noqa: E402
from src.operations import pattern, split_days  # noqa: E402
from src.rules import FATAL_LEVELS, fatal_window_ids, fatal_window_ids_except, harmless_fatal_events  # noqa: E402

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def templates(dataset: str) -> pd.Series:
    return pd.read_csv(PROCESSED_DIR / dataset / "templates.csv").set_index("event_id")["template"]


# ----------------------------------------------------------------------------- HDFS, Logistic Regression
hdfs = pd.read_parquet(PROCESSED_DIR / "hdfs" / "sequences.parquet", columns=["event_ids", "label", "split_time"])
hdfs["pattern"] = hdfs["event_ids"].map(pattern)
train = hdfs[hdfs["split_time"] == "train"].groupby("pattern")["label"].agg(["size", "sum"])
test = hdfs[hdfs["split_time"] == "test"].copy()
pred = pd.read_parquet(PRED_DIR / "hdfs_time_logreg.parquet").set_index("row").loc[test.index]
test["score"], test["alert"] = pred["score"].to_numpy(), pred["pred"].to_numpy()


def in_training(p: str) -> str:
    if p not in train.index:
        return "never seen in training"
    n, abnormal = train.loc[p]
    return "seen in training only as abnormal" if abnormal == n else (
        "seen in training only as normal" if abnormal == 0 else "seen in training with both labels")


rows = []
for kind, mask in (("false alarm", test["alert"] & (test["label"] == 0)), ("missed anomaly", ~test["alert"] & (test["label"] == 1))):
    for p, g in test[mask].groupby("pattern"):
        n, abnormal = train.loc[p] if p in train.index else (0, 0)
        rows.append({"error": kind, "pattern": p, "blocks": len(g), "in_training": in_training(p),
                     "training_blocks": int(n), "training_abnormal": int(abnormal), "model_score": round(float(g["score"].median()), 4)})
hdfs_errors = pd.DataFrame(rows).sort_values(["error", "blocks"], ascending=[True, False])
hdfs_errors.to_csv(RESULTS_DIR / "error_analysis_hdfs.csv", index=False)
mixed = test.groupby("pattern")["label"].agg(["size", "sum"])
mixed = mixed[(mixed["sum"] > 0) & (mixed["sum"] < mixed["size"])]
print("HDFS, Logistic Regression (time split, test):")
print(hdfs_errors.groupby(["error", "in_training"])["blocks"].sum().to_string())
print(f"  test patterns that carry BOTH labels (errors no count-based model can avoid): {len(mixed)}")
tpl = templates("hdfs")
top = hdfs_errors[hdfs_errors["error"] == "false alarm"].iloc[0]
print(f"  largest false-alarm pattern ({top.blocks} blocks): {top.pattern}")
print("    " + " | ".join(f"{e.split('x')[0]}: {tpl.get(int(e[1:].split('x')[0]), '?')[:45]}" for e in top.pattern.split()))
reps = hdfs[(hdfs["split_time"] == "train") & hdfs["pattern"].str.contains(r"\bE1x4\b") & hdfs["pattern"].str.contains(r"\bE5x4\b")]
print(f"  training blocks with 4 receives and 4 stores (an extra copy): {len(reps):,}, of which abnormal: {reps['label'].mean():.1%}")

# ----------------------------------------------------------------------------- BGL, FATAL rule
lines = pd.read_parquet(PROCESSED_DIR / "bgl" / "lines.parquet", columns=["timestamp", "event_id", "is_alert", "level"])
lines["window_id"] = lines["timestamp"].astype("datetime64[s]").astype(np.int64) // (BGL_WINDOW_MINUTES * 60)
bgl = pd.read_parquet(PROCESSED_DIR / "bgl" / "sequences.parquet",
                      columns=["window_id", "start_time", "label", "split_random", "split_time"])
label = dict(zip(bgl["window_id"], bgl["label"]))
lines["part"] = lines["window_id"].map(dict(zip(bgl["window_id"], bgl["split_time"])))
fatal = lines[lines["level"].isin(FATAL_LEVELS)]
test_fatal = fatal[fatal["part"] == "test"]
fp_windows = {w for w in test_fatal["window_id"].unique() if label[w] == 0}
in_fp = test_fatal[test_fatal["window_id"].isin(fp_windows)]
tr = fatal[fatal["part"] == "train"].groupby("event_id")["is_alert"].agg(train_fatal_lines="size", train_alert_share="mean")
te = test_fatal.groupby("event_id")["is_alert"].agg(test_fatal_lines="size", test_alert_share="mean")
bgl_errors = (in_fp.groupby("event_id").agg(false_alarm_windows=("window_id", "nunique"))
              .join(te).join(tr).sort_values("false_alarm_windows", ascending=False))
tpl = templates("bgl")
bgl_errors["message"] = [tpl.get(e, "?") if e != 0 else "(never seen in training)" for e in bgl_errors.index]
bgl_errors["in_training"] = np.where(bgl_errors.index == 0, "never seen in training",
                                     np.where(bgl_errors["train_alert_share"] == 0, "FATAL in training, never an alert",
                                              np.where(bgl_errors["train_fatal_lines"].isna(), "never FATAL in training",
                                                       "FATAL in training, sometimes an alert")))
bgl_errors.round(3).to_csv(RESULTS_DIR / "error_analysis_bgl.csv")
print(f"\nBGL, FATAL rule (time split, test): {len(fp_windows)} false alarms; their FATAL lines: {len(in_fp):,}, "
      f"of which real alerts: {int(in_fp['is_alert'].sum())}")
print(bgl_errors[["false_alarm_windows", "train_fatal_lines", "train_alert_share", "in_training", "message"]]
      .head(8).to_string(max_colwidth=60))

# ----------------------------------------------------------------------------- follow-up: refined FATAL rule
rows = []
days_of = {}
for split, col in SPLITS.items():
    part_of = dict(zip(bgl["window_id"], bgl[col]))
    lines["part"] = lines["window_id"].map(part_of)
    for min_lines in (None, 5, 20, 100):  # None = the original rule
        harmless = set() if min_lines is None else harmless_fatal_events(lines[lines["part"] == "train"], min_lines)
        for part in ("val", "test"):
            windows = bgl[bgl[col] == part]
            sub = lines[lines["part"] == part]
            flagged = fatal_window_ids(sub) if min_lines is None else fatal_window_ids_except(sub, harmless)
            alert = windows["window_id"].isin(flagged).to_numpy()
            y = windows["label"].to_numpy() == 1
            tp, fp, fn = int((alert & y).sum()), int((alert & ~y).sum()), int((~alert & y).sum())
            days = split_days(bgl, col, part, pd.Timedelta(minutes=BGL_WINDOW_MINUTES))
            rows.append({"split": split, "part": part,
                         "rule": "FATAL rule" if min_lines is None else f"FATAL rule minus never-alert messages (>= {min_lines} lines)",
                         "ignored_message_types": len(harmless), "tp": tp, "false_alarms": fp, "missed": fn,
                         "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0, "recall": round(tp / (tp + fn), 3),
                         "f1": round(float(f1_score(tp, fp, fn)), 3), "false_alarms_per_day": round(fp / days, 2)})
refined = pd.DataFrame(rows)
refined.to_csv(RESULTS_DIR / "fatal_rule_refined.csv", index=False)
print("\nFollow-up: FATAL rule without the message types that were never alerts in TRAINING")
print(refined.to_string(index=False))
print(f"\nSaved error_analysis_hdfs.csv, error_analysis_bgl.csv, fatal_rule_refined.csv in {RESULTS_DIR}")
