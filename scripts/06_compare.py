"""
Step 3c - Compare all models: random split vs time split.

Reads the results of 04_baselines.py and 05_deeplog.py and produces
  * reports/results/results.csv         - every metric of every run
  * reports/results/drift_breakdown.csv - WHERE the time-split errors come from
  * reports/figures/08_f1_random_vs_time.png

Usage:
    python scripts/06_compare.py        # a few seconds
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import plot_style as ps  # noqa: E402
from src.config import PROCESSED_DIR  # noqa: E402
from src.experiment import PRED_DIR, RESULTS_DIR, load_sequences, split_parts  # noqa: E402
from src.features import EventCountFeatures  # noqa: E402

NAMES = {  # display order = top to bottom in the figure
    "logreg": "Logistic Regression",
    "iforest": "Isolation Forest",
    "deeplog": "DeepLog (LSTM)",
    "rule_anomaly_only": "Rule: anomaly-only events",
    "rule_fatal": "Rule: FATAL/FAILURE line",
}
DIAGNOSTIC = {"deeplog_ignore_unseen": "DeepLog, unseen events ignored"}

results = pd.concat([pd.read_csv(RESULTS_DIR / f) for f in ("baselines.csv", "deeplog.csv")], ignore_index=True)
results.to_csv(RESULTS_DIR / "results.csv", index=False, float_format="%.4f")


# ---------------------------------------------------------------- main table
def f1_table(models: dict) -> pd.DataFrame:
    rows = []
    for key, name in models.items():
        row = {"model": name}
        for ds in ("hdfs", "bgl"):
            r = results[(results.model == key) & (results.dataset == ds)].set_index("split")
            if r.empty:
                continue
            row[f"{ds}_random"] = r.loc["random", "f1"]
            row[f"{ds}_time"] = r.loc["time", "f1"]
            row[f"{ds}_change"] = r.loc["time", "f1"] - r.loc["random", "f1"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


print("Test F1 (threshold chosen on validation):")
print(f1_table(NAMES).round(3).to_string(na_rep="-"))
print("\nDiagnostic:")
print(f1_table(DIAGNOSTIC).round(3).to_string(na_rep="-"))
print("\nAll metrics:")
cols = ["dataset", "split", "model", "precision", "recall", "f1", "f1_ci_low", "f1_ci_high", "pr_auc", "oracle_f1", "val_f1"]
print(results[cols].round(3).to_string(index=False))


# ---------------------------------------------------- where do errors come from?
# BGL: does the window contain an event never seen in training?   (new templates / parser event 0)
# HDFS: is it a block without a delete step?  (a workload that is rare in train, absent in val,
#       but 60% of normal test blocks - found while checking Isolation Forest, docs/03_models.md)
def subgroup_flags(dataset: str, test: pd.DataFrame, train: pd.DataFrame) -> tuple[str, np.ndarray]:
    if dataset == "bgl":
        features = EventCountFeatures().fit(train["event_ids"])
        return "window has an unseen event", features.counts(test["event_ids"])[:, -1] > 0
    templates = pd.read_csv(PROCESSED_DIR / "hdfs" / "templates.csv")
    delete = set(templates.loc[templates.template.str.contains("Deleting block|NameSystem.delete"), "event_id"])
    return "block has no delete step", ~test["event_ids"].map(lambda e: not delete.isdisjoint(e.tolist())).to_numpy()


breakdown = []
for dataset in ("hdfs", "bgl"):
    parts = split_parts(load_sequences(dataset), "time")
    name, flag = subgroup_flags(dataset, parts["test"], parts["train"])
    y = parts["test"]["label"].to_numpy().astype(bool)
    for key in [*NAMES, *DIAGNOSTIC]:
        path = PRED_DIR / f"{dataset}_time_{key}.parquet"
        if not path.exists():
            continue
        pred = pd.read_parquet(path)["pred"].to_numpy()
        breakdown.append({
            "dataset": dataset, "subgroup": name, "model": key,
            "test_groups_in_subgroup_%": 100 * flag.mean(),
            "normal_in_subgroup_%": 100 * flag[~y].mean(),
            "false_alarms": int((pred & ~y).sum()),
            "false_alarms_in_subgroup_%": 100 * (pred & ~y & flag).sum() / max((pred & ~y).sum(), 1),
            "false_alarm_rate_in_subgroup_%": 100 * (pred & ~y & flag).sum() / max((~y & flag).sum(), 1),
            "false_alarm_rate_outside_%": 100 * (pred & ~y & ~flag).sum() / max((~y & ~flag).sum(), 1),
            "abnormal_in_subgroup_%": 100 * flag[y].mean(),
            "recall_in_subgroup_%": 100 * (pred & y & flag).sum() / max((y & flag).sum(), 1),
            "recall_outside_%": 100 * (pred & y & ~flag).sum() / max((y & ~flag).sum(), 1),
        })
breakdown = pd.DataFrame(breakdown)
breakdown.to_csv(RESULTS_DIR / "drift_breakdown.csv", index=False, float_format="%.2f")
print("\nTime split, test set - where do the false alarms come from?")
print(breakdown.round(1).to_string(index=False))


# ------------------------------------------------------------------- figure
ps.apply_style()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
order = list(NAMES)
for ax, ds in zip(axes, ("hdfs", "bgl")):
    for y_pos, key in enumerate(order):
        r = results[(results.model == key) & (results.dataset == ds)].set_index("split")
        if r.empty:
            ax.text(0.02, y_pos, "not applicable (HDFS has no FATAL level)", va="center", fontsize=8.5, color=ps.MUTED)
            continue
        rnd, tim = r.loc["random", "f1"], r.loc["time", "f1"]
        ax.plot([rnd, tim], [y_pos, y_pos], color=ps.AXIS, linewidth=2, zorder=2)
        ax.plot(rnd, y_pos, "o", color=ps.RANDOM_SPLIT, markersize=9, markeredgecolor=ps.SURFACE, markeredgewidth=1.5, zorder=3)
        ax.plot(tim, y_pos, "D", color=ps.TIME_SPLIT, markersize=8, markeredgecolor=ps.SURFACE, markeredgewidth=1.5, zorder=3)
        lo, hi = sorted([rnd, tim])  # smaller value labelled on the left, larger on the right
        ax.text(lo - 0.025, y_pos, f"{lo:.2f}", ha="right", va="center", fontsize=8.5, color=ps.TEXT_2)
        ax.text(hi + 0.025, y_pos, f"{hi:.2f}", ha="left", va="center", fontsize=8.5, color=ps.TEXT_2)
    ax.set_title(ds.upper())
    ax.set_xlim(-0.12, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("F1 on the test set (higher is better)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(range(len(order)), [NAMES[k] for k in order])
axes[0].invert_yaxis()
handles = [plt.Line2D([], [], marker="o", linestyle="", color=ps.RANDOM_SPLIT, markersize=8, label="random split"),
           plt.Line2D([], [], marker="D", linestyle="", color=ps.TIME_SPLIT, markersize=7,
                      label="time split (train on the past, test on the future)")]
fig.suptitle("Test F1 with a random split vs a time split", x=0.01, y=1.07, ha="left",
             fontsize=12, fontweight="bold", color=ps.TEXT)
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.3)
fig.tight_layout(rect=(0, 0, 1, 0.97))
print("\nSaved", ps.save(fig, "08_f1_random_vs_time"))
