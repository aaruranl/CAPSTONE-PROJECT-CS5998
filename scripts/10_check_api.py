"""
Milestone 3, steps 2-3 - check the API in live mode against the offline results.

Replays raw log lines through POST /reset, /ingest (batches of 10,000, as a live feed
would) and /flush, with the real models and the default detectors, in two ways:

  A. "switched on at the test start, no cold-start rule": the stream begins exactly at
     the start of the test period and every finished group is judged (step 2).
     -> shows the COLD-START problem: groups whose beginning was never seen.
  B. "back-fill + cold-start rule": the stream begins 13 h earlier for HDFS (as the log
     feed src/replay.py does) and groups that may have started before watching are not
     judged (src/stream.py). -> the fix.

For each it checks: is every test group judged, with the same alert as the offline evaluation? Is any
judged group INCOMPLETE (fewer events than the full group in preprocessing), and why:
  * "started before watching": part of it came before the stream began (cold start);
  * "cut by the quiet time": the monitor saw all of it, but a silence of more than 13 h
    split it in two (the known price of the HDFS rule, scripts/09_choose_quiet_time.py).
It runs the API in-process (FastAPI TestClient): no server or database is needed.

Usage:
    python scripts/10_check_api.py   # ~2 min -> reports/results/api_check.json
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import create_app  # noqa: E402
from src.config import PROCESSED_DIR  # noqa: E402
from src.engine import DEFAULT_DETECTOR  # noqa: E402
from src.experiment import PRED_DIR, RESULTS_DIR, test_period_start  # noqa: E402
from src.replay import BACKFILL, log_lines  # noqa: E402

BATCH = 10_000
MODES = {"A_no_rule_from_test_start": dict(backfill=False, rule=False),
         "B_backfill_and_cold_start_rule": dict(backfill=True, rule=True)}


def run(client, dataset, watch_from, rule):
    """Replay from `watch_from` to the end of the log; return {group ID: [results]} and the last response."""
    client.post("/reset", json={"dataset": dataset, "watch_from": watch_from, "cold_start_rule": rule}).raise_for_status()
    live, batch, n, requests = {}, [], 0, 0

    def send(lines):
        out = client.post("/ingest", json={"dataset": dataset, "lines": lines}).raise_for_status().json()
        for r in out["finished"]:
            live.setdefault(r["group_id"], []).append(r)

    for _ts, line in log_lines(dataset, watch_from):
        batch.append(line)
        if len(batch) == BATCH:
            send(batch)
            n, requests, batch = n + len(batch), requests + 1, []
    if batch:
        send(batch)
        n, requests = n + len(batch), requests + 1
    before_flush = sum(map(len, live.values()))
    last = client.post("/flush", json={"dataset": dataset}).raise_for_status().json()
    for r in last["finished"]:
        live.setdefault(r["group_id"], []).append(r)
    return live, last, n, requests, before_flush


client = TestClient(create_app())  # no database: this check is about the scoring logic
report = {}
for dataset in ("hdfs", "bgl"):
    detector = DEFAULT_DETECTOR[dataset]
    id_col = "block_id" if dataset == "hdfs" else "window_id"
    seqs = pd.read_parquet(PROCESSED_DIR / dataset / "sequences.parquet", columns=[id_col, "n_events", "start_time", "split_time"])
    seqs["gid"] = seqs[id_col].astype(str)
    full_size = dict(zip(seqs["gid"], seqs["n_events"]))  # events of each complete group (preprocessing)
    began = dict(zip(seqs["gid"], (seqs["start_time"] - pd.Timestamp(0)) // pd.Timedelta(seconds=1)))
    test = seqs[seqs["split_time"] == "test"]
    saved = pd.read_parquet(PRED_DIR / f"{dataset}_time_{detector}.parquet").set_index("row").loc[test.index]
    offline = dict(zip(test["gid"], saved["pred"].astype(bool)))
    go_live = test_period_start(dataset)
    report[dataset] = {"detector": detector, "test_period_starts": str(pd.Timestamp(go_live, unit="s")),
                       "test_groups": len(offline)}

    for mode, cfg in MODES.items():
        watch_from = go_live - (BACKFILL[dataset] if cfg["backfill"] else 0)
        start = time.time()
        live, last, n_lines, requests, before_flush = run(client, dataset, watch_from, cfg["rule"])
        seconds = time.time() - start
        alert = {gid: any(r["is_anomaly"] for r in rs) for gid, rs in live.items()}
        incomplete = [gid for gid, rs in live.items() if sum(r["n_events"] for r in rs) < full_size[gid]]
        cold = [gid for gid in incomplete if began[gid] < watch_from]
        cut = [gid for gid in incomplete if began[gid] >= watch_from]
        st = last["status"]
        report[dataset][mode] = out = {
            "watching_from": st["watching_since"],
            "raw_lines": n_lines, "requests": requests,
            "lines_per_second_through_api": round(n_lines / seconds),
            "groups_judged": len(live),
            "groups_judged_before_flush": before_flush,
            "test_groups_judged": sum(gid in live for gid in offline),
            "test_groups_cut_in_two": sum(len(live[gid]) > 1 for gid in offline if gid in live),
            "test_alert_mismatches_vs_offline": sum(alert.get(gid, False) != a for gid, a in offline.items()),
            "judged_groups_not_in_test_set": sum(gid not in offline for gid in live),
            "judged_groups_incomplete_started_before_watching": len(cold),
            "alerts_on_them": sum(alert[gid] for gid in cold),
            "judged_groups_incomplete_cut_by_quiet_time": len(cut),
            "alerts_on_cut_pieces": sum(alert[gid] for gid in cut),
            "not_judged_started_before_watching": st["not_judged_started_before_watching"],
        }
        print(f"\n{dataset.upper()} ({detector}) {mode}: watching from {st['watching_since']}, "
              f"{n_lines:,} lines in {requests} requests, {n_lines / seconds:,.0f} lines/s")
        print(f"  judged {len(live):,} groups ({before_flush:,} before the flush) | test groups judged "
              f"{out['test_groups_judged']:,} of {len(offline):,} | mismatches vs offline "
              f"{out['test_alert_mismatches_vs_offline']} | cut in two {out['test_groups_cut_in_two']}")
        print(f"  judged INCOMPLETE groups - started before watching: {len(cold):,} ({out['alerts_on_them']:,} flagged)"
              f" | cut by the quiet time: {len(cut):,} ({out['alerts_on_cut_pieces']:,} flagged)")
        print(f"  not judged (may have started before watching): {out['not_judged_started_before_watching']:,}")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "api_check.json").write_text(json.dumps(report, indent=2))
print(f"\nSaved {RESULTS_DIR / 'api_check.json'}")
