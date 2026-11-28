"""
Step 9 (part 1) - false alarms per day, and alert budgets.

A. False alarms per day for every model, dataset and split (test set). For the time split
   also: the notifications an operator would get if alerts were bundled per 10 minutes.
B. Alert budgets. For the three scoring models, the threshold is chosen on VALIDATION so
   that at most B false alarms per day are raised there; then the test period shows how
   many there really are. Does a promise made on the past hold in the future?

Usage:
    python scripts/12_false_alarms.py   # ~30 s -> reports/results/false_alarms_per_day.csv,
                                        #   reports/results/alert_budget.csv,
                                        #   reports/figures/13_alert_budget.png
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import plot_style as ps  # noqa: E402
from src.baselines import anomaly_scores  # noqa: E402
from src.config import BGL_WINDOW_MINUTES, PROCESSED_DIR  # noqa: E402
from src.dashboard_data import MODEL_NAMES  # noqa: E402
from src.deeplog import DeepLog  # noqa: E402
from src.experiment import MODELS_DIR, PRED_DIR, RESULTS_DIR, SPLITS  # noqa: E402
from src.operations import budget_threshold, notifications, split_days  # noqa: E402

PAD = {"hdfs": pd.Timedelta(0), "bgl": pd.Timedelta(minutes=BGL_WINDOW_MINUTES)}  # a BGL window lasts 5 min
BUDGETS = {"hdfs": [100, 1_000, 10_000], "bgl": [1, 3, 10]}  # false alarms per day (BGL has ~50 windows a day)
SCORING_MODELS = ["logreg", "iforest", "deeplog"]

results = pd.read_csv(RESULTS_DIR / "results.csv")
seqs = {ds: pd.read_parquet(PROCESSED_DIR / ds / "sequences.parquet",
                            columns=["event_ids", "start_time", "end_time", "label", "split_random", "split_time"])
        for ds in ("hdfs", "bgl")}

# ----------------------------------------------------------------------------- A. false alarms per day
rows = []
for r in results.itertuples():
    s = seqs[r.dataset]
    col = SPLITS[r.split]
    days = split_days(s, col, "test", PAD[r.dataset])
    row = {"dataset": r.dataset, "split": r.split, "model": r.model, "test_days": round(days, 3),
           "test_groups_per_day": round(r.n_test / days), "false_alarms": int(r.fp),
           "false_alarms_per_day": round(r.fp / days, 1),
           "false_alarms_per_10k_normal_groups": round(1e4 * r.fp / (r.fp + r.tn), 1),
           "alerts_per_day": round((r.tp + r.fp) / days, 1),
           "alerts_per_real_anomaly_caught": round((r.tp + r.fp) / r.tp, 2) if r.tp else None,
           "precision": r.precision, "recall": r.recall, "f1": r.f1}
    if r.split == "time":  # bundling only makes sense for one continuous period
        test = s[s[col] == "test"]
        pred = pd.read_parquet(PRED_DIR / f"{r.dataset}_time_{r.model}.parquet").set_index("row").loc[test.index]
        n = notifications(test["end_time"], pred["pred"].to_numpy(), test["label"].to_numpy() == 1)
        row["notifications_per_day_10min"] = round(n["notifications"] / days, 1)
        row["false_notifications_per_day_10min"] = round(n["false_notifications"] / days, 1)
    rows.append(row)
per_day = pd.DataFrame(rows)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
per_day.to_csv(RESULTS_DIR / "false_alarms_per_day.csv", index=False)
show = ["dataset", "split", "model", "false_alarms", "test_days", "false_alarms_per_day", "f1",
        "false_notifications_per_day_10min"]
print("A. False alarms per day (test set)\n" + per_day[show].to_string(index=False))


# ----------------------------------------------------------------------------- B. alert budgets
def scores(dataset: str, split: str, model: str, part: pd.DataFrame) -> np.ndarray:
    events = [np.asarray(e) for e in part["event_ids"]]
    if model == "deeplog":
        net, _ = DeepLog.load(MODELS_DIR / f"{dataset}_{split}_deeplog.pt")
        return net.score(events).astype(float)
    saved = joblib.load(MODELS_DIR / f"{dataset}_{split}_{model}.joblib")
    return anomaly_scores(saved["model"], saved["features"].transform(events))


rows = []
for ds, s in seqs.items():
    for split, col in SPLITS.items():
        val, test = s[s[col] == "val"], s[s[col] == "test"]
        val_days, test_days = split_days(s, col, "val", PAD[ds]), split_days(s, col, "test", PAD[ds])
        yv, yt = val["label"].to_numpy() == 1, test["label"].to_numpy() == 1
        for model in SCORING_MODELS:
            sv = scores(ds, split, model, val).astype(np.float64)  # float64: see budget_threshold
            st = pd.read_parquet(PRED_DIR / f"{ds}_{split}_{model}.parquet").set_index("row").loc[test.index, "score"].to_numpy(np.float64)
            for budget in BUDGETS[ds]:
                allowed = int(budget * val_days)  # false alarms the validation period may have
                t = budget_threshold(sv[~yv], allowed)
                val_fa, test_fa = (sv[~yv] >= t).sum() / val_days, (st[~yt] >= t).sum() / test_days
                rows.append({"dataset": ds, "split": split, "model": model, "budget_per_day": budget,
                             "validation_days": round(val_days, 3), "allowed_on_validation": allowed,
                             "validation_false_alarms_per_day": round(val_fa, 1),
                             "test_false_alarms_per_day": round(test_fa, 1),
                             "times_the_budget": round(test_fa / budget, 2),
                             "validation_recall": round(float((sv[yv] >= t).mean()), 3),
                             "test_recall": round(float((st[yt] >= t).mean()), 3)})
budgets = pd.DataFrame(rows)
budgets.to_csv(RESULTS_DIR / "alert_budget.csv", index=False)
print("\nB. Alert budgets: chosen on validation, measured on test\n" + budgets[
    ["dataset", "split", "model", "budget_per_day", "validation_false_alarms_per_day", "test_false_alarms_per_day",
     "times_the_budget", "test_recall"]].to_string(index=False))

# ----------------------------------------------------------------------------- figure
ps.apply_style()
plt = ps.plt
plain = ps.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharey="row")
for i, ds in enumerate(("hdfs", "bgl")):
    for j, model in enumerate(SCORING_MODELS):
        ax = axes[i, j]
        b = budgets[(budgets["dataset"] == ds) & (budgets["model"] == model)]
        x = np.array(BUDGETS[ds], dtype=float)
        ax.plot(x, x, color=ps.MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1, label="budget kept exactly")
        for split, color, marker in (("random", ps.RANDOM_SPLIT, "o"), ("time", ps.TIME_SPLIT, "s")):
            y = b[b["split"] == split].sort_values("budget_per_day")["test_false_alarms_per_day"].to_numpy()
            ax.plot(x, y, color=color, marker=marker, markersize=6, linewidth=2, label=f"{split} split", zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(x, [f"{v:,.0f}" for v in x])
        ax.minorticks_off()
        ax.yaxis.set_major_formatter(plain)
        ax.set_title(f"{ds.upper()} · {MODEL_NAMES[model]}", fontsize=10)
        if j == 0:
            ax.set_ylabel("false alarms per day on test")
        if i == 1:
            ax.set_xlabel("budget set on validation (per day)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=3, frameon=False)
fig.suptitle("A false-alarm budget chosen on validation: kept on test?", x=0.06, ha="left", y=1.04,
             fontsize=12, fontweight="bold", color=ps.TEXT)
fig.tight_layout()
path = ps.save(fig, "13_alert_budget")
print(f"\nSaved {RESULTS_DIR / 'false_alarms_per_day.csv'}, {RESULTS_DIR / 'alert_budget.csv'}, {path}")
