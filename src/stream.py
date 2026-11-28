"""
Live scoring (step 5): lines arrive a few at a time, and each group is
scored as soon as it is FINISHED.

  * BGL : a 5-minute window is finished when a line from a LATER window arrives.
          All 4.7M BGL lines are in time order, so this is exact.
  * HDFS: a block has no "end" message. It counts as finished after
          config.HDFS_QUIET_HOURS of silence (chosen from data in
          scripts/09_choose_quiet_time.py; normal blocks can be silent for ~13 h
          before their delete step).

"Time" means the timestamps written IN the logs (the latest one seen so far), not the
wall clock. The check for finished groups runs after EVERY line, so the results do not
depend on how the lines are cut into requests: replaying old logs fast or slow, in big or
small batches, gives exactly the same results. Each result records `finished_at`, the log
time at which its group was finished and judged.

A silence in the logs would otherwise stop the clock: a BGL window could only be closed
when the next line arrives, which takes minutes to hours (a crashed machine may never
write again). So a client may also send `now`, the current time, with or without lines
(a "watermark", as in stream-processing systems): groups whose time is up are then
finished even if no line arrives. The log feed (src/replay.py) sends it about once a second.

The open groups are kept in order of their last activity, so finding the finished ones
only looks at the oldest group instead of all of them.

The cold-start rule (step 6): a monitor switched on in the middle of a running
system sees groups without their beginning, and such half-groups look abnormal (99.4% of
them were flagged in scripts/10_check_api.py). So a group is only JUDGED if monitoring
began early enough that none of its events can have been missed:
  * BGL : the window started at or after `watch_from`;
  * HDFS: its first event came at least HDFS_QUIET_HOURS after `watch_from` (an event
          more than 13 h earlier would have belonged to a separate, already finished block).
Other groups are still followed (so their late events cannot start a "new" group), but
when they finish they are counted, not scored. The log feed (src/replay.py) back-fills
13 h of history on start-up, so every HDFS block from the go-live moment on is judged.
"""
from collections import OrderedDict
from datetime import datetime, timezone
from uuid import uuid4

from src.config import HDFS_QUIET_HOURS
from src.engine import WINDOW_S, Group, ScoringEngine, _time


def new_run_id(dataset: str) -> str:
    """A readable, unique name for one live stream, e.g. 'bgl-20261128-101500-3f2a'."""
    return f"{dataset}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:4]}"


class LiveScorer:
    def __init__(self, engine: ScoringEngine, quiet_seconds: int = HDFS_QUIET_HOURS * 3600,
                 watch_from: int | None = None, cold_start_rule: bool = True, run_id: str | None = None):
        self.engine = engine
        self.quiet_seconds = quiet_seconds
        self.watch_from = watch_from          # log time when monitoring began (default: the first line)
        self.cold_start_rule = cold_start_rule
        self.run_id = run_id or new_run_id(engine.dataset)
        self.open: OrderedDict[str, Group] = OrderedDict()  # oldest activity first
        self.log_time: int | None = None                     # latest timestamp seen in the logs
        self.stats = engine.new_stats()
        self.finished_ids: set[str] = set()
        self.n_judged = self.n_alerts = self.n_reopened = self.n_not_judged = 0

    def _is_finished(self, group: Group) -> bool:
        if self.engine.dataset == "bgl":
            return int(group.group_id) < self.log_time // WINDOW_S  # a later window has started
        return self.log_time - group.last_time >= self.quiet_seconds

    def seen_from_start(self, group: Group) -> bool:
        """Can we be sure that no event of this group came before monitoring began?"""
        if self.engine.dataset == "bgl":
            earliest = int(group.group_id) * WINDOW_S         # a window has no events before it starts
        else:
            earliest = group.first_time - self.quiet_seconds  # anything earlier = a separate block
        return earliest >= self.watch_from

    def _advance(self, ts: int) -> None:
        if self.watch_from is None:
            self.watch_from = ts
        self.log_time = ts if self.log_time is None else max(self.log_time, ts)

    def _take_finished(self, done: list) -> None:
        """Move every finished group (oldest first) to `done`, with the log time it finished at."""
        while self.open:
            gid = next(iter(self.open))
            if not self._is_finished(self.open[gid]):
                break  # the oldest open group is still active, so all newer ones are too
            done.append((self.open.pop(gid), self.log_time))
            self.finished_ids.add(gid)

    def ingest(self, lines, now: int | None = None) -> list[dict]:
        """Add new lines (and optionally the current log time `now`); return the scored
        results of every group that finished."""
        done = []
        for line in lines:
            added = self.engine.add_line(self.open, line, self.stats)
            if added is None:
                continue
            ts, group_ids = added
            self._advance(ts)
            for gid in group_ids:
                if gid in self.finished_ids and len(self.open[gid].events) == 1:
                    self.n_reopened += 1  # a "finished" group received more events later
                self.open.move_to_end(gid)
            self._take_finished(done)
        if now is not None:  # time has passed even without new lines
            self._advance(now)
            self._take_finished(done)
        return self._judge(done)

    def flush(self) -> list[dict]:
        """End of the stream: score every group that is still open."""
        done = [(group, self.log_time) for group in self.open.values()]
        self.finished_ids.update(self.open)
        self.open.clear()
        return self._judge(done)

    def _judge(self, done: list) -> list[dict]:
        if self.cold_start_rule:
            judged = [(g, t) for g, t in done if self.seen_from_start(g)]
            self.n_not_judged += len(done) - len(judged)
            done = judged
        results = self.engine.score_groups([g for g, _ in done])  # one call: fast
        for r, (_, t) in zip(results, done):
            r["finished_at"] = _time(t)
        self.n_judged += len(results)
        self.n_alerts += sum(r["is_anomaly"] for r in results)
        return results

    def status(self) -> dict:
        return {
            "run_id": self.run_id,
            "dataset": self.engine.dataset,
            "detector": self.engine.detector.name,
            "watching_since": _time(self.watch_from) if self.watch_from is not None else None,
            "log_time": _time(self.log_time) if self.log_time is not None else None,
            "cold_start_rule": self.cold_start_rule,
            "open_groups": len(self.open),
            "judged_groups": self.n_judged,
            "alerts": self.n_alerts,
            "not_judged_started_before_watching": self.n_not_judged,
            "groups_reopened": self.n_reopened,
            **self.stats,
        }
