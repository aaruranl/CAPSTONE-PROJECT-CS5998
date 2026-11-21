"""
Step 1 - Preprocess the raw logs into model-ready sequences.

    load -> parse (Drain) -> group (HDFS: block, BGL: 5-min window) -> label -> split

Outputs, per dataset, in data/processed/<dataset>/:
    sequences.parquet  one row per group: event ID sequence, label, both splits
    lines.parquet      one row per log line: time, event ID, level ... (for EDA)
    templates.csv      the event templates Drain found and how often they occur
    summary.json       counts, parser statistics and split statistics

Usage:
    python scripts/02_preprocess.py                      # both datasets (~2 min)
    python scripts/02_preprocess.py --dataset BGL
    python scripts/02_preprocess.py --dataset HDFS --max-lines 200000   # quick test run
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bgl import preprocess_bgl  # noqa: E402
from src.hdfs import preprocess_hdfs  # noqa: E402

RUNNERS = {"HDFS": preprocess_hdfs, "BGL": preprocess_bgl}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*RUNNERS, "all"], default="all")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="only read the first N lines (for quick testing)")
    args = parser.parse_args()

    names = list(RUNNERS) if args.dataset == "all" else [args.dataset]
    for name in names:
        print(f"\n=== {name} ===")
        summary = RUNNERS[name](max_lines=args.max_lines)
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
