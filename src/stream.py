"""
Live scoring (Milestone 3, step 2): lines arrive a few at a time, and each group is
scored as soon as it is FINISHED.

  * BGL : a 5-minute window is finished when a line from a LATER window arrives.
          All 4.7M BGL lines are in time order, so this is exact.
  * HDFS: a block has no "end" message. It counts as finished after
          config.HDFS_QUIET_HOURS of silence (chosen from data in
          scripts/09_choose_quiet_time.py; normal blocks can be silent for ~13 h
          before their delete step).

"Time" means the timestamps written IN the logs (the latest one seen so far), not the
wall clock. Replaying old logs fast or slow therefore gives exactly the same results.

The open groups are kept in order of their last activity, so finding the finished ones
only looks at the oldest few groups instead of all of them.
"""
from collections import OrderedDict

from src.config import HDFS_QUIET_HOURS
from src.engine import WINDOW_S, Group, ScoringEngine, _time


class LiveScorer:
    def __init__(self, engine: ScoringEngine, quiet_seconds: int = HDFS_QUIET_HOURS * 3600):
        self.engine = engine
        self.quiet_seconds = quiet_seconds
        self.open: OrderedDict[str, Group] = OrderedDict()  # oldest activity first
        self.log_time: int | None = None                     # latest timestamp seen in the logs
        self.stats = engine.new_stats()
        self.finished_ids: set[str] = set()
        self.n_finished = self.n_alerts = self.n_reopened = 0

    def _is_finished(self, group: Group) -> bool:
        if self.engine.dataset == "bgl":
            return int(group.group_id) < self.log_time // WINDOW_S  # a later window has started
        return self.log_time - group.last_time >= self.quiet_seconds

    def ingest(self, lines) -> list[dict]:
        """Add new lines; return the scored results of every group that is now finished."""
        for line in lines:
            added = self.engine.add_line(self.open, line, self.stats)
            if added is None:
                continue
            ts, group_ids = added
            self.log_time = ts if self.log_time is None else max(self.log_time, ts)
            for gid in group_ids:
                if gid in self.finished_ids and len(self.open[gid].events) == 1:
                    self.n_reopened += 1  # a "finished" group received more events later
                self.open.move_to_end(gid)
        ready = []
        for gid, group in self.open.items():  # oldest first; stop at the first one still active
            if not self._is_finished(group):
                break
            ready.append(gid)
        return self._score(ready)

    def flush(self) -> list[dict]:
        """End of the stream: score every group that is still open."""
        return self._score(list(self.open))

    def _score(self, ids: list[str]) -> list[dict]:
        groups = [self.open.pop(gid) for gid in ids]
        self.finished_ids.update(ids)
        results = self.engine.score_groups(groups)
        self.n_finished += len(results)
        self.n_alerts += sum(r["is_anomaly"] for r in results)
        return results

    def status(self) -> dict:
        return {
            "dataset": self.engine.dataset,
            "detector": self.engine.detector.name,
            "log_time": _time(self.log_time) if self.log_time is not None else None,
            "open_groups": len(self.open),
            "finished_groups": self.n_finished,
            "alerts": self.n_alerts,
            "groups_reopened": self.n_reopened,
            **self.stats,
        }
