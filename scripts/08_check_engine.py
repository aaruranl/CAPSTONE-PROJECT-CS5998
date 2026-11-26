"""
Milestone 3, step 1 - check the scoring engine against the offline results.

Takes the RAW log lines of the whole time-split test period, runs them through the
scoring engine (src/engine.py) and checks that it reproduces, for every test group:
  * the same event sequence as preprocessing (data/processed/*/sequences.parquet), and
  * the same score and alert decision as in step 3 (reports/results/predictions/).
Any difference would mean "training-serving skew": the live system would not be the
system we evaluated. It also prints example alerts and measures the speed.

Usage:
    python scripts/08_check_engine.py   # ~2 min -> reports/results/engine_check.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATASETS, PROCESSED_DIR  # noqa: E402
from src.engine import DETECTORS, WINDOW_S, ScoringEngine, load_detector  # noqa: E402
from src.experiment import PRED_DIR, RESULTS_DIR  # noqa: E402


def test_period_lines(dataset: str, seqs: pd.DataFrame) -> list[str]:
    """All raw lines that can belong to a test group (they all come after the test period starts)."""
    test = seqs[seqs["split_time"] == "test"]
    path = DATASETS[dataset.upper()]["log_file"]
    with open(path, encoding="utf-8", errors="replace") as f:
        if dataset == "hdfs":  # groups are blocks; every line of a test block is at/after its first line
            start = int(test["first_line"].min())
            return [line for i, line in enumerate(f) if i >= start]
        # BGL: keep lines whose 5-minute window is a test window
        windows = set(test["window_id"].astype(str))
        return [line for line in f if (p := line.split(maxsplit=2))[1:2] and p[1].isdigit()
                and str(int(p[1]) // WINDOW_S) in windows]


report = {}
for dataset in ("hdfs", "bgl"):
    seqs = pd.read_parquet(PROCESSED_DIR / dataset / "sequences.parquet")
    id_col = "block_id" if dataset == "hdfs" else "window_id"
    seqs["gid"] = seqs[id_col].astype(str)
    test = seqs[seqs["split_time"] == "test"]

    start = time.time()
    lines = test_period_lines(dataset, seqs)
    engine = ScoringEngine.load(dataset)
    t0 = time.time()
    groups = {g.group_id: g for g in engine.group_lines(lines)}
    parse_seconds = time.time() - t0
    groups = [groups[gid] for gid in test["gid"]]  # in the same order as the offline test set

    # 1) same event sequences as preprocessing?
    seq_mismatch = sum(not np.array_equal(g.events, ev) for g, ev in zip(groups, test["event_ids"]))
    out = {"test_groups": len(test), "raw_lines_read": len(lines), **engine.last_stats,
           "lines_per_second_parse_and_group": round(len(lines) / parse_seconds),
           "event_sequence_mismatches": seq_mismatch, "detectors": {}}
    print(f"\n{dataset.upper()}: {len(lines):,} raw lines -> {len(test):,} test groups, "
          f"{len(lines) / parse_seconds:,.0f} lines/s, sequence mismatches: {seq_mismatch}")

    # 2) same scores and decisions as step 3, for every detector?
    for name in DETECTORS[dataset]:
        engine.detector = load_detector(dataset, name)
        t0 = time.time()
        results = engine.score_groups(groups)
        seconds = time.time() - t0
        saved = pd.read_parquet(PRED_DIR / f"{dataset}_time_{name}.parquet").set_index("row").loc[test.index]
        raw = np.array([r["raw_score"] for r in results])
        alert = np.array([r["is_anomaly"] for r in results])
        unit_ok = all((r["score"] >= 0.5) == r["is_anomaly"] for r in results)
        out["detectors"][name] = {
            "max_score_difference": float(np.abs(raw - saved["score"].to_numpy()).max()),
            "decision_mismatches": int((alert != saved["pred"].to_numpy()).sum()),
            "alerts": int(alert.sum()),
            "score_0_1_threshold_is_0.5": unit_ok,
            "groups_per_second": round(len(groups) / seconds),
        }
        print(f"  {name:<11} alerts {alert.sum():>6,}  decision mismatches "
              f"{out['detectors'][name]['decision_mismatches']}  max score diff "
              f"{out['detectors'][name]['max_score_difference']:.2e}  ({len(groups) / seconds:,.0f} groups/s)")
        # one example alert (a true anomaly) per detector, to show what the API will return
        hit = next((r for r, lab in zip(results, test["label"]) if r["is_anomaly"] and lab == 1), None)
        if hit and name in ("logreg", "deeplog", "rule_fatal"):
            print("    example:", json.dumps({k: hit[k] for k in ("group_id", "score", "is_anomaly", "reasons")},
                                            indent=1)[:900])
    out["seconds_total"] = round(time.time() - start, 1)
    report[dataset] = out

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "engine_check.json").write_text(json.dumps(report, indent=2))
print(f"\nSaved {RESULTS_DIR / 'engine_check.json'}")
