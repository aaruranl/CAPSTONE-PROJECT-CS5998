"""
Live log feed (Milestone 3, step 3): replays a raw log file into the API, as if the
system were running right now. It plays the role of a log COLLECTOR: a small program
that reads new log lines and sends them to the detector.

    python -m src.replay --dataset bgl                  # the test period, 1 log hour per second
    python -m src.replay --dataset bgl --speed 300      # 5 log minutes (one window) per second
    python -m src.replay --dataset hdfs --speed 0       # as fast as possible
    python -m src.replay --dataset bgl --hours 48       # only the first 48 log hours

What it does:
  1. /reset: start a new stream, telling the API from which log time it is watching.
  2. Back-fill: read the recent history (HDFS: the last 13 h; BGL: nothing) as fast as
     possible. A real collector does the same on start-up. Together with the API's
     cold-start rule (src/stream.py), every group from the go-live moment on is then
     judged, and no half-seen group ever is.
  3. Go live: send the next lines at `speed` x real time (log seconds per real second),
     in about one request per second. During a silence, send the current (replayed) time
     every second, so that groups whose time is up are finished without waiting for the
     next line. A long silence is shortened to at most --max-wait seconds (demo).
  4. End of the file (or of --hours): /flush, which scores the groups still open.

Default start: the beginning of the time-split TEST period, the "future" that no model
was trained or tuned on. The log files are in time order (checked: 0 lines out of
order), so the start is found by binary search, like looking up a word in a dictionary.
"""
import argparse
import os
import time
from collections.abc import Callable, Iterable, Iterator

import httpx

from src.config import DATASETS as RAW_FILES
from src.config import HDFS_QUIET_HOURS
from src.engine import DEFAULT_DETECTOR, READERS, _time
from src.experiment import test_period_start

TICK = 1.0                                   # about one request per second when live
BACKFILL = {"hdfs": HDFS_QUIET_HOURS * 3600, "bgl": 0}


def line_time(dataset: str, raw: bytes) -> int | None:
    parsed = READERS[dataset](raw.decode("utf-8", errors="replace"))
    return parsed[0] if parsed else None


def seek_time(f, t: int, time_of: Callable[[bytes], int | None]) -> None:
    """Move the binary file `f` to the first line whose timestamp is >= t (binary search)."""
    def line_start(pos: int) -> int:     # start of the first line at or after byte `pos`
        if pos == 0:
            return 0
        f.seek(pos - 1)
        f.readline()
        return f.tell()

    def at_or_after(pos: int) -> bool:   # is the line starting there already late enough?
        f.seek(line_start(pos))
        for raw in f:                    # skip unreadable lines (there are none in HDFS / BGL)
            ts = time_of(raw)
            if ts is not None:
                return ts >= t
        return True                      # end of file

    lo, hi = 0, f.seek(0, os.SEEK_END)
    while lo < hi:
        mid = (lo + hi) // 2
        if at_or_after(mid):
            hi = mid
        else:
            lo = mid + 1
    f.seek(line_start(lo))


def log_lines(dataset: str, from_time: int, until: int | None = None) -> Iterator[tuple[int, str]]:
    """(log time, raw line) for every line of the raw log file from `from_time` (until `until`)."""
    read, ts = READERS[dataset], from_time
    with open(RAW_FILES[dataset.upper()]["log_file"], "rb") as f:
        seek_time(f, from_time, lambda raw: line_time(dataset, raw))
        for raw in f:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            parsed = read(line)
            ts = parsed[0] if parsed else ts  # unreadable lines are sent too (the API counts them)
            if until is not None and ts >= until:
                return
            yield ts, line


def batches(records: Iterable[tuple[int, str]], go_live: int, speed: float, batch_size: int,
            max_wait: float = 2.0, clock=time.monotonic, sleep=time.sleep) -> Iterator[tuple[list[str], int | None]]:
    """(log time, line) pairs in time order -> (lines, now) requests to send.

    Lines before `go_live` (the back-fill) go out as fast as possible. After it, a line with
    log time ts is due at real time  t0 + (ts - go_live) / speed  (speed 0 = no waiting);
    lines due within the next TICK seconds travel together in one request. While waiting
    for the next line, an empty request is sent every TICK with `now`, the replayed clock,
    so the API can finish groups whose time is up during the silence. `now` is None when it
    would add nothing (back-fill, speed 0, full batches).
    """
    batch: list[str] = []
    t0 = None

    def replayed_clock(next_ts: int) -> int:  # the log time "now", never past the next line
        return min(int(go_live + (clock() - t0) * speed), next_ts - 1)

    for ts, line in records:
        if speed > 0 and ts >= go_live:
            if t0 is None:                       # the go-live moment
                if batch:
                    yield batch, None
                    batch = []
                t0 = clock()
            due = t0 + (ts - go_live) / speed
            if due - clock() > TICK:             # this line belongs to a later request
                if batch:
                    yield batch, replayed_clock(ts)
                    batch = []
                if due - clock() > max_wait:     # long silence in the logs: skip ahead
                    t0 -= due - clock() - max_wait
                    due = t0 + (ts - go_live) / speed
                while (wait := due - clock()) > 0:
                    sleep(min(TICK, wait))
                    if due - clock() > 0:        # heartbeat: time passes, no lines
                        yield [], replayed_clock(ts)
        batch.append(line)
        if len(batch) >= batch_size:
            yield batch, None
            batch = []
    if batch:
        yield batch, None


def replay(dataset: str, api: str = "http://127.0.0.1:8000", detector: str | None = None,
           start: int | None = None, speed: float = 3600, hours: float | None = None,
           batch_size: int = 10_000, max_wait: float = 2.0, quiet: bool = False) -> dict:
    """Run one replay; returns a summary (run ID, counts, final status)."""
    go_live = test_period_start(dataset) if start is None else start
    watch_from = go_live - BACKFILL[dataset]
    stop_at = go_live + hours * 3600 if hours else None
    say = (lambda *a: None) if quiet else print
    with httpx.Client(base_url=api, timeout=600) as client:
        status = client.post("/reset", json={"dataset": dataset, "detector": detector,
                                             "watch_from": watch_from}).raise_for_status().json()["status"]
        say(f"{dataset.upper()} replay, run {status['run_id']}: watching from {_time(watch_from)}, "
            f"live from {_time(go_live)}, speed {speed:g}x")

        sent = requests = judged = alerts = 0
        last_print = time.monotonic()
        for batch, now in batches(log_lines(dataset, watch_from, stop_at), go_live, speed, batch_size, max_wait):
            out = client.post("/ingest", json={"dataset": dataset, "detector": detector,
                                               "lines": batch, "now": now}).raise_for_status().json()
            sent, requests = sent + len(batch), requests + 1
            judged += len(out["finished"])
            alerts += sum(r["is_anomaly"] for r in out["finished"])
            if not quiet:
                for r in out["finished"]:
                    if r["is_anomaly"]:
                        print(f"  ALERT {r['group_id']} score {r['score']:.2f} ({r['last_seen']}): "
                              f"{' | '.join(r['reasons'][:2])}"[:160])
                if time.monotonic() - last_print > 5:
                    last_print = time.monotonic()
                    print(f"  ... log time {out['status']['log_time']} | {sent:,} lines | "
                          f"judged {judged:,} | alerts {alerts:,}")
        out = client.post("/flush", json={"dataset": dataset, "detector": detector}).raise_for_status().json()
        judged += len(out["finished"])
        alerts += sum(r["is_anomaly"] for r in out["finished"])
        st = out["status"]
        say(f"done: {sent:,} lines in {requests} requests | judged {judged:,} groups, {alerts:,} alerts | "
            f"not judged (started before watching): {st['not_judged_started_before_watching']:,} | "
            f"stored in database: {out['stored']}")
        return {"run_id": st["run_id"], "lines_sent": sent, "requests": requests,
                "judged": judged, "alerts": alerts, "stored": out["stored"], "final_status": st}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay a raw log file into the scoring API, as a live feed.")
    ap.add_argument("--dataset", choices=["hdfs", "bgl"], required=True)
    ap.add_argument("--api", default=os.environ.get("API_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--detector", default=None, help=f"default: {DEFAULT_DETECTOR}")
    ap.add_argument("--start", type=int, default=None,
                    help="log time (unix seconds) to go live; default: start of the test period")
    ap.add_argument("--speed", type=float, default=3600, help="log seconds per real second (0 = as fast as possible)")
    ap.add_argument("--hours", type=float, default=None, help="stop after this many log hours")
    ap.add_argument("--batch-size", type=int, default=10_000)
    ap.add_argument("--max-wait", type=float, default=2.0, help="longest pause (real seconds) for a silence in the logs")
    a = ap.parse_args(argv)
    replay(a.dataset, a.api, a.detector, a.start, a.speed, a.hours, a.batch_size, a.max_wait)


if __name__ == "__main__":
    main()
