"""
Step 3b - Improved method: DeepLog (LSTM next-event prediction).

Trained on NORMAL training groups only. The value g ("flag if the real next
event is not in the model's top-g guesses") is the alert threshold and is
chosen on the validation set, exactly like the thresholds of the baselines.

Also records one DIAGNOSTIC run per experiment ("deeplog_ignore_unseen") where
events never seen in normal training are NOT counted as surprising. It is not a
competing model; it answers "how much of DeepLog's result comes from new events?".

Usage:
    python scripts/05_deeplog.py        # ~4 min on CPU -> reports/results/deeplog.csv, models/*.pt
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deeplog import DeepLog  # noqa: E402
from src.experiment import DATASETS, MODELS_DIR, RESULTS_DIR, SPLITS, load_sequences, record, split_parts  # noqa: E402


def show(row: dict) -> None:
    print(f"  {row['model']:<22} test P {row['precision']:.3f}  R {row['recall']:.3f}  "
          f"F1 {row['f1']:.3f} [{row['f1_ci_low']:.3f}-{row['f1_ci_high']:.3f}]   "
          f"(val F1 {row['val_f1']:.3f}, g = {row['threshold']:.0f})")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    rows, losses = [], []
    for dataset in DATASETS:
        seqs = load_sequences(dataset)
        for split in SPLITS:
            parts = split_parts(seqs, split)
            print(f"\n{dataset.upper()} - {split} split")
            train = parts["train"]
            start = time.time()
            model = DeepLog().fit(train.loc[train["label"] == 0, "event_ids"])
            seconds = round(time.time() - start, 1)
            losses += [{"dataset": dataset, "split": split, "epoch": i + 1, "loss": v}
                       for i, v in enumerate(model.loss_history)]

            val, test = parts["val"]["event_ids"], parts["test"]["event_ids"]
            rows.append(record(dataset, split, "deeplog", "semi-supervised (normal groups only)", parts,
                               model.score(val), model.score(test), seconds))
            show(rows[-1])
            model.save(MODELS_DIR / f"{dataset}_{split}_deeplog.pt", rows[-1]["threshold"])

            rows.append(record(dataset, split, "deeplog_ignore_unseen", "diagnostic", parts,
                               model.score(val, ignore_unseen=True), model.score(test, ignore_unseen=True), seconds))
            show(rows[-1])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "deeplog.csv", index=False, float_format="%.4f")
    pd.DataFrame(losses).to_csv(RESULTS_DIR / "deeplog_training_loss.csv", index=False, float_format="%.5f")
    print(f"\nSaved {RESULTS_DIR / 'deeplog.csv'}")


if __name__ == "__main__":
    main()
