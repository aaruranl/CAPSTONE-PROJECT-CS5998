"""
Step 3a - Baselines: two simple rules, Logistic Regression and Isolation Forest.

Every model is trained and tested twice per dataset:
  * random split - groups shuffled before cutting (what most papers do)
  * time split   - train on the past, test on the future (the realistic setting)

For each run: fit on train -> choose the alert threshold on validation -> score test once.

Usage:
    python scripts/04_baselines.py      # ~1 min -> reports/results/baselines.csv, models/*.joblib
"""
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import anomaly_scores, make_isolation_forest, make_logistic_regression  # noqa: E402
from src.config import PROCESSED_DIR  # noqa: E402
from src.experiment import DATASETS, MODELS_DIR, RESULTS_DIR, SPLITS, load_sequences, record, split_parts  # noqa: E402
from src.features import EventCountFeatures  # noqa: E402
from src.rules import anomaly_only_event_set, contains_any, fatal_window_ids  # noqa: E402


def show(row: dict) -> None:
    print(f"  {row['model']:<18} test P {row['precision']:.3f}  R {row['recall']:.3f}  "
          f"F1 {row['f1']:.3f} [{row['f1_ci_low']:.3f}-{row['f1_ci_high']:.3f}]   "
          f"(val F1 {row['val_f1']:.3f}, threshold {row['threshold']:.4g})")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    rows = []
    for dataset in DATASETS:
        seqs = load_sequences(dataset)
        fatal_ids = None
        if dataset == "bgl":
            lines = pd.read_parquet(PROCESSED_DIR / "bgl" / "lines.parquet", columns=["timestamp", "level"])
            fatal_ids = fatal_window_ids(lines)
            del lines

        for split in SPLITS:
            parts = split_parts(seqs, split)
            print(f"\n{dataset.upper()} - {split} split  "
                  + ", ".join(f"{p} {len(d):,} ({d['label'].sum():,} abnormal)" for p, d in parts.items()))

            # Rule 1 (BGL only): flag a window that contains a FATAL/FAILURE line. No training at all.
            if fatal_ids is not None:
                flag = {p: parts[p]["window_id"].isin(fatal_ids) for p in ("val", "test")}
                rows.append(record(dataset, split, "rule_fatal", "rule (no training)", parts,
                                   flag["val"], flag["test"], 0.0, is_rule=True))
                show(rows[-1])

            # Rule 2: flag a group with any event that only ever appeared in ABNORMAL training groups.
            start = time.time()
            events = anomaly_only_event_set(parts["train"])
            flag = {p: contains_any(parts[p]["event_ids"], events) for p in ("val", "test")}
            rows.append(record(dataset, split, "rule_anomaly_only", "rule (from training labels)", parts,
                               flag["val"], flag["test"], round(time.time() - start, 1), is_rule=True))
            show(rows[-1])

            # Count features; the vocabulary comes from the training part only (no peeking at test).
            features = EventCountFeatures().fit(parts["train"]["event_ids"])
            X = {p: features.transform(parts[p]["event_ids"]) for p in parts}
            y_train = parts["train"]["label"].to_numpy()

            for name, kind, make in [("logreg", "supervised", make_logistic_regression),
                                     ("iforest", "unsupervised", make_isolation_forest)]:
                start = time.time()
                model = make()
                if kind == "supervised":
                    model.fit(X["train"], y_train)
                else:
                    model.fit(X["train"])  # Isolation Forest never sees the labels
                seconds = round(time.time() - start, 1)
                rows.append(record(dataset, split, name, kind, parts,
                                   anomaly_scores(model, X["val"]), anomaly_scores(model, X["test"]), seconds))
                show(rows[-1])
                # the average NORMAL training group: the reference point for explaining alerts (src/engine.py)
                normal = y_train == 0
                joblib.dump({"features": features, "model": model, "threshold": rows[-1]["threshold"],
                             "normal_mean": X["train"][normal].mean(axis=0),
                             "normal_mean_counts": features.counts(parts["train"]["event_ids"][normal]).mean(axis=0)},
                            MODELS_DIR / f"{dataset}_{split}_{name}.joblib")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "baselines.csv", index=False, float_format="%.4f")
    print(f"\nSaved {RESULTS_DIR / 'baselines.csv'}")


if __name__ == "__main__":
    main()
