"""
Milestone 3, step 3 - end-to-end check of the live system: log feed -> API -> PostgreSQL.

Unlike scripts/10 (everything inside one Python process), this runs the real pieces:
  1. the API as a real web server (uvicorn), with DATABASE_URL pointing to PostgreSQL;
  2. the log feed (src/replay.py) replaying the time-split test period, as fast as possible;
then reads the judged groups back FROM THE DATABASE and compares them with the
offline evaluation (Milestone 2).
So what the dashboard will show is exactly what was evaluated.

It also measures how long after a BGL window ends its alert arrives (log time):
  * in the fast replay, which sends no clock signal: a window waits for the next line;
  * in a short replay at real pace (speed 600), which sends the clock signal ("now")
    every second: a window is judged at most about one second (real time) after it ends.

Needs the PostgreSQL database named by DATABASE_URL (in .env, see .env.example).
Usage:
    python scripts/11_check_replay.py   # ~2.5 min -> reports/results/replay_check.json
    # against the API running in Docker (step 5) instead of a local server:
    python scripts/11_check_replay.py --api http://127.0.0.1:18000 --out replay_check_docker.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR  # noqa: E402
from src.engine import WINDOW_S  # noqa: E402
from src.engine import DEFAULT_DETECTOR  # noqa: E402
from src.experiment import PRED_DIR, RESULTS_DIR  # noqa: E402
from src.replay import replay  # noqa: E402
from src.store import ResultStore, database_url  # noqa: E402

DATABASE_URL = database_url() or sys.exit("DATABASE_URL is not set: copy .env.example to .env and fill it in")
ap = argparse.ArgumentParser(description="End-to-end check: log feed -> API -> database vs the offline evaluation")
ap.add_argument("--api", help="use this already running API (e.g. in Docker) instead of starting a local one")
ap.add_argument("--out", default="replay_check.json", help="results file in reports/results/")
args = ap.parse_args()
PORT = 8077
API = args.api or f"http://127.0.0.1:{PORT}"
PACED = dict(speed=600, hours=6, max_wait=3600)  # 6 log hours at 600x = 36 real seconds, no fast-forward


def bgl_delay_minutes(rows: pd.DataFrame, until: pd.Timestamp | None = None) -> pd.Series:
    """Minutes (log time) between the end of each BGL window and the moment it was judged."""
    window_end = pd.to_datetime(rows["group_id"].astype(int) * WINDOW_S + WINDOW_S, unit="s")
    delay = (rows["judged_at"] - window_end).dt.total_seconds() / 60
    return delay[window_end <= until] if until is not None else delay


def describe(delay: pd.Series) -> dict:
    return {"windows": int(len(delay)), "median_minutes": round(float(delay.median()), 1),
            "p90_minutes": round(float(delay.quantile(0.9)), 1), "max_minutes": round(float(delay.max()), 1)}

store = ResultStore(DATABASE_URL)
try:
    store.connect()
except Exception as e:  # noqa: BLE001 - tell the user what to do instead of a long traceback
    sys.exit(f"Cannot reach the database {store.describe()} ({e.__class__.__name__}): check DATABASE_URL in .env")

server = None if args.api else subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.api:app", "--port", str(PORT), "--log-level", "warning"],
    cwd=ROOT, env={**os.environ, "DATABASE_URL": DATABASE_URL})
try:
    for _ in range(60):  # wait until the API answers
        try:
            health = httpx.get(f"{API}/health").json()
            break
        except httpx.TransportError:
            time.sleep(0.5)
    else:
        sys.exit("The API did not start")
    if health.get("database") is None:
        sys.exit(f"The API at {API} stores nothing: it was started without DATABASE_URL")
    print(f"API running at {API} ({'already running' if args.api else 'started here'}), storing in the database")

    report = {"database": store.label(),  # no host or user name: this file is published
              "api": "already running (" + API + ")" if args.api else "local uvicorn server started by this script"}
    for dataset in ("bgl", "hdfs"):
        detector = DEFAULT_DETECTOR[dataset]
        start = time.time()
        summary = replay(dataset, api=API, speed=0, quiet=True)
        seconds = time.time() - start

        rows = pd.DataFrame(store.results(summary["run_id"]))      # read back from PostgreSQL
        (run,) = [r for r in store.runs(dataset) if r["run_id"] == summary["run_id"]]
        id_col = "block_id" if dataset == "hdfs" else "window_id"
        seqs = pd.read_parquet(PROCESSED_DIR / dataset / "sequences.parquet", columns=[id_col, "split_time"])
        test = seqs[seqs["split_time"] == "test"]
        saved = pd.read_parquet(PRED_DIR / f"{dataset}_time_{detector}.parquet").set_index("row").loc[test.index]
        offline = pd.DataFrame({"pred": saved["pred"].astype(bool).to_numpy(), "score": saved["score"].to_numpy()},
                               index=test[id_col].astype(str).to_numpy())

        stored_alert = rows.groupby("group_id")["is_anomaly"].any()
        in_test = rows[rows["group_id"].isin(offline.index)].set_index("group_id")
        report[dataset] = out = {
            "detector": detector,
            "run_id": summary["run_id"],
            "raw_lines_sent": summary["lines_sent"],
            "requests": summary["requests"],
            "seconds_end_to_end": round(seconds, 1),
            "lines_per_second_end_to_end": round(summary["lines_sent"] / seconds),
            "rows_in_database": len(rows),
            "rows_match_api_responses": len(rows) == summary["judged"],
            "alerts_in_database": int(rows["is_anomaly"].sum()),
            "test_groups": len(offline),
            "test_groups_in_database": int(offline.index.isin(rows["group_id"]).sum()),
            "test_alert_mismatches_vs_offline": int((stored_alert.reindex(offline.index, fill_value=False)
                                                     != offline["pred"]).sum()),
            "max_score_difference_vs_offline": float(np.abs(in_test["raw_score"] - offline.loc[in_test.index, "score"]).max()),
            "other_groups_in_database": int((~rows["group_id"].isin(offline.index)).sum()),
            "not_judged_started_before_watching": run["status"]["not_judged_started_before_watching"],
            "run_status_in_database": run["status"],
        }
        if dataset == "bgl":  # alert delay without a clock signal: windows wait for the next line
            out["alert_delay_without_clock_signal"] = {
                "all_windows": describe(bgl_delay_minutes(rows)),
                "alerts": describe(bgl_delay_minutes(rows[rows["is_anomaly"]]))}
        print(f"\n{dataset.upper()} ({detector}), run {summary['run_id']}: {summary['lines_sent']:,} lines in "
              f"{summary['requests']} requests, {seconds:.0f} s ({out['lines_per_second_end_to_end']:,} lines/s end to end)")
        print(f"  database rows {len(rows):,} (= API responses: {out['rows_match_api_responses']}) | "
              f"alerts {out['alerts_in_database']:,}")
        print(f"  test groups in database {out['test_groups_in_database']:,} of {len(offline):,} | alert mismatches "
              f"vs offline {out['test_alert_mismatches_vs_offline']} | max score difference "
              f"{out['max_score_difference_vs_offline']:.1e} | other groups {out['other_groups_in_database']}")
        if dataset == "bgl":
            print("  alert delay after the window ends, WITHOUT a clock signal:", out["alert_delay_without_clock_signal"]["alerts"])

    # a short replay at real pace, with the clock signal
    summary = replay("bgl", api=API, quiet=True, **PACED)
    rows = pd.DataFrame(store.results(summary["run_id"]))
    last_line = pd.Timestamp(summary["final_status"]["log_time"])
    delay = bgl_delay_minutes(rows, until=last_line)  # windows still running at the end were only flushed
    report["bgl_paced_with_clock_signal"] = {**PACED, "run_id": summary["run_id"], "requests": summary["requests"],
                                             "lines_sent": summary["lines_sent"], **describe(delay),
                                             "max_real_seconds": round(float(delay.max()) * 60 / PACED["speed"], 2)}
    print(f"\nBGL at {PACED['speed']}x for {PACED['hours']} log hours, WITH the clock signal: {summary['requests']} requests"
          f" | alert delay after the window ends: {describe(delay)} = at most "
          f"{report['bgl_paced_with_clock_signal']['max_real_seconds']} real seconds")
finally:
    if server is not None:
        server.terminate()
        server.wait(timeout=30)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / args.out).write_text(json.dumps(report, indent=2, default=str))
print(f"\nSaved {RESULTS_DIR / args.out}")
