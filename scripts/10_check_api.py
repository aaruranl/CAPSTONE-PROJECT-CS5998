"""
Milestone 3, step 2 - check the API in live mode against the offline results.

Replays the raw lines of the whole time-split test period through POST /ingest in
batches (as a live feed would), then POST /flush, using the real models and the
default detectors. For every test group it compares the live alert with step 3.
It runs the API in-process (FastAPI TestClient), so no server needs to be started.

Usage:
    python scripts/10_check_api.py   # ~1 min -> reports/results/api_check.json
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import app  # noqa: E402
from src.config import PROCESSED_DIR  # noqa: E402
from src.engine import DEFAULT_DETECTOR  # noqa: E402
from src.experiment import PRED_DIR, RESULTS_DIR, test_period_lines  # noqa: E402

BATCH = 10_000
client = TestClient(app)
report = {}
for dataset in ("hdfs", "bgl"):
    detector = DEFAULT_DETECTOR[dataset]
    seqs = pd.read_parquet(PROCESSED_DIR / dataset / "sequences.parquet")
    seqs["gid"] = seqs["block_id" if dataset == "hdfs" else "window_id"].astype(str)
    test = seqs[seqs["split_time"] == "test"]
    lines = test_period_lines(dataset, seqs)

    client.post("/reset", json={"dataset": dataset})
    start = time.time()
    live, before_flush = {}, 0
    for i in range(0, len(lines), BATCH):
        out = client.post("/ingest", json={"dataset": dataset, "lines": lines[i:i + BATCH]})
        out.raise_for_status()
        for r in out.json()["finished"]:
            live.setdefault(r["group_id"], []).append(r)
    before_flush = sum(map(len, live.values()))
    flushed = client.post("/flush", json={"dataset": dataset}).json()
    for r in flushed["finished"]:
        live.setdefault(r["group_id"], []).append(r)
    seconds = time.time() - start

    saved = pd.read_parquet(PRED_DIR / f"{dataset}_time_{detector}.parquet").set_index("row").loc[test.index]
    offline = dict(zip(test["gid"], saved["pred"]))
    is_test = [gid in offline for gid in live]
    live_alert = {gid: any(r["is_anomaly"] for r in rs) for gid, rs in live.items()}
    mismatches = sum(live_alert[gid] != bool(offline[gid]) for gid in offline)
    split_groups = sum(len(rs) > 1 for gid, rs in live.items() if gid in offline)
    # groups that began BEFORE the replay started: only their late events are seen (partial groups)
    partial = [gid for gid, t in zip(live, is_test) if not t]
    report[dataset] = {
        "detector": detector,
        "raw_lines": len(lines),
        "requests": -(-len(lines) // BATCH),
        "lines_per_second_through_api": round(len(lines) / seconds),
        "groups_finished_before_flush": before_flush,
        "test_groups": len(offline),
        "test_groups_scored_live": sum(is_test),
        "test_groups_cut_in_two": split_groups,
        "alert_mismatches_vs_offline": mismatches,
        "alerts": sum(live_alert[gid] for gid in offline),
        "partial_groups_from_before_the_replay": len(partial),
        "alerts_on_partial_groups": sum(live_alert[gid] for gid in partial),
        "final_status": flushed["status"],
    }
    print(f"\n{dataset.upper()} ({detector}): {len(lines):,} lines in {report[dataset]['requests']} requests, "
          f"{len(lines) / seconds:,.0f} lines/s")
    print(f"  finished before flush: {before_flush:,} | test groups scored: {sum(is_test):,} of {len(offline):,} | "
          f"cut in two: {split_groups} | alert mismatches vs offline: {mismatches}")
    print(f"  partial groups that started before the replay: {len(partial):,} "
          f"({report[dataset]['alerts_on_partial_groups']:,} of them flagged)")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "api_check.json").write_text(json.dumps(report, indent=2))
print(f"\nSaved {RESULTS_DIR / 'api_check.json'}")
