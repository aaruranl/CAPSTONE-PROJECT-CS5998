"""Tests for live scoring (src/stream.py) and the API (src/api.py), with a tiny hand-made engine."""
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from src.api import MAX_LINES, create_app
from src.engine import FatalRuleDetector, ScoringEngine, read_hdfs
from src.log_parser import DrainParser
from src.store import ResultStore
from src.stream import LiveScorer

T = 1117838400  # start of a 5-minute window (unix seconds)


def toy_engine(dataset, detector=None):
    parser = DrainParser()
    for m in ["instruction cache parity error corrected", "data storage interrupt",
              "Receiving block blk_1 src: /1.2.3.4:5 dest: /1.2.3.4:6"]:
        parser.learn(m)
    parser.freeze()
    return ScoringEngine(dataset, parser, {}, FatalRuleDetector())


def bgl(ts, level="INFO", content="instruction cache parity error corrected"):
    node = "R02-M1-N0-C:J12-U11"
    return f"- {ts} 2005.06.03 {node} 2005-06-03-15.42.50.675872 {node} RAS KERNEL {level} {content}"


def hdfs(hhmmss, block):
    return f"081109 {hhmmss} 143 INFO dfs.DataNode$DataXceiver: Receiving block {block} src: /10.0.0.1:5 dest: /10.0.0.2:6"


def test_live_bgl_window_finishes_when_the_next_window_starts():
    client = TestClient(create_app(load_engine=toy_engine))
    client.post("/reset", json={"dataset": "bgl", "watch_from": T})                # watching from the window start
    first = client.post("/ingest", json={"dataset": "bgl", "lines": [bgl(T + 1), bgl(T + 2, "FATAL", "data storage interrupt")]})
    assert first.status_code == 200 and first.json()["finished"] == []          # window still open
    second = client.post("/ingest", json={"dataset": "bgl", "lines": [bgl(T + 301)]}).json()
    assert [r["group_id"] for r in second["finished"]] == [str(T // 300)]      # next window started
    assert second["finished"][0]["is_anomaly"] and second["status"]["open_groups"] == 1
    last = client.post("/flush", json={"dataset": "bgl"}).json()
    assert [r["group_id"] for r in last["finished"]] == [str(T // 300 + 1)]
    assert last["status"]["alerts"] == 1 and last["status"]["judged_groups"] == 2
    assert last["stored"] is False                                              # no database configured


def test_live_hdfs_block_finishes_after_the_quiet_time():
    live = LiveScorer(toy_engine("hdfs"), quiet_seconds=60, cold_start_rule=False)
    assert live.ingest([hdfs("203500", "blk_1"), hdfs("203530", "blk_2")]) == []
    done = live.ingest([hdfs("203610", "blk_3")])        # 70 s after blk_1, 40 s after blk_2
    assert [r["group_id"] for r in done] == ["blk_1"]
    live.ingest([hdfs("203620", "blk_1")])               # blk_1 was declared finished too early
    assert live.status()["groups_reopened"] == 1
    assert sorted(r["group_id"] for r in live.flush()) == ["blk_1", "blk_2", "blk_3"]


def test_results_do_not_depend_on_how_lines_are_batched():
    lines = [hdfs("203500", "blk_1"), hdfs("203610", "blk_2"), hdfs("203620", "blk_1"), hdfs("203900", "blk_3")]

    def run(batch_size):
        live = LiveScorer(toy_engine("hdfs"), quiet_seconds=60, cold_start_rule=False)
        out = [r for i in range(0, len(lines), batch_size) for r in live.ingest(lines[i:i + batch_size])]
        out += live.flush()
        return [(r["group_id"], r["n_events"], r["finished_at"]) for r in out], live.status()["groups_reopened"]
    # blk_1 is silent for 70 s > 60 s: finished at 20:36:10, then its event at 20:36:20 starts a new piece
    assert run(1) == run(4) == run(2)
    assert run(4)[0][0] == ("blk_1", 1, "2008-11-09 20:36:10") and run(4)[1] == 1


def test_watermark_finishes_a_window_during_a_silence():
    live = LiveScorer(toy_engine("bgl"), watch_from=T)
    assert live.ingest([bgl(T + 1, "FATAL", "data storage interrupt")]) == []
    assert live.ingest([], now=T + 299) == []                       # the window is not over yet
    (alert,) = live.ingest([], now=T + 300)                          # no new line, but its time is up
    assert alert["is_anomaly"] and alert["finished_at"] == "2005-06-03 22:45:00"
    client = TestClient(create_app(load_engine=toy_engine))         # the same through the API
    client.post("/reset", json={"dataset": "bgl", "watch_from": T})
    client.post("/ingest", json={"dataset": "bgl", "lines": [bgl(T + 1, "FATAL", "data storage interrupt")]})
    assert len(client.post("/ingest", json={"dataset": "bgl", "lines": [], "now": T + 300}).json()["finished"]) == 1


def test_cold_start_rule_skips_groups_that_may_have_started_before_watching():
    # HDFS: watching from the first line (20:35:00); quiet time 60 s. blk_10 may have begun before, blk_20 did not
    live = LiveScorer(toy_engine("hdfs"), quiet_seconds=60)
    live.ingest([hdfs("203500", "blk_10"), hdfs("203530", "blk_10")])
    live.ingest([hdfs("203700", "blk_20")])
    assert [r["group_id"] for r in live.flush()] == ["blk_20"]
    assert live.status()["not_judged_started_before_watching"] == 1
    assert live.watch_from == read_hdfs(hdfs("203500", "blk_30"))[0]

    # BGL: watching from T + 1 -> window T may have had earlier lines; window T + 300 started later
    bgl_live = LiveScorer(toy_engine("bgl"))
    bgl_live.ingest([bgl(T + 1, "FATAL", "data storage interrupt"), bgl(T + 301)])
    assert [r["group_id"] for r in bgl_live.flush()] == [str(T // 300 + 1)]
    assert bgl_live.status()["not_judged_started_before_watching"] == 1

    # switched off, everything is judged (only to show the problem)
    off = LiveScorer(toy_engine("bgl"), cold_start_rule=False)
    judged = off.ingest([bgl(T + 1), bgl(T + 301)]) + off.flush()
    assert [r["group_id"] for r in judged] == [str(T // 300), str(T // 300 + 1)]


def test_live_results_are_stored_in_the_database(tmp_path):
    store = ResultStore(f"sqlite:///{tmp_path / 'alerts.db'}")
    client = TestClient(create_app(load_engine=toy_engine, store=store))
    run_id = client.post("/reset", json={"dataset": "bgl", "watch_from": T}).json()["status"]["run_id"]
    client.post("/ingest", json={"dataset": "bgl", "lines": [bgl(T + 1, "FATAL", "data storage interrupt")]})
    out = client.post("/ingest", json={"dataset": "bgl", "lines": [bgl(T + 301)]}).json()
    assert out["stored"] is True
    client.post("/flush", json={"dataset": "bgl"})
    rows = store.results(run_id)
    assert [(r["group_id"], r["is_anomaly"]) for r in rows] == [(str(T // 300 + 1), False), (str(T // 300), True)]
    assert rows[1]["reasons"][0].startswith("1 line(s) with level FATAL")
    (run,) = store.runs()
    assert run["run_id"] == run_id and run["status"]["judged_groups"] == 2 and run["status"]["lines"] == 2
    assert client.get("/health").json()["database"].startswith("sqlite")


def test_detection_continues_when_the_database_fails():
    class BrokenStore:
        def save(self, *args):
            raise OperationalError("INSERT", {}, Exception("database is down"))

        def describe(self):
            return "postgresql://broken"
    client = TestClient(create_app(load_engine=toy_engine, store=BrokenStore()))
    client.post("/reset", json={"dataset": "bgl", "watch_from": T})
    client.post("/ingest", json={"dataset": "bgl", "lines": [bgl(T + 1, "FATAL", "data storage interrupt")]})
    out = client.post("/flush", json={"dataset": "bgl"})
    assert out.status_code == 200 and out.json()["stored"] is False
    assert out.json()["finished"][0]["is_anomaly"]


def test_score_endpoint_and_input_checks():
    client = TestClient(create_app(load_engine=toy_engine))
    out = client.post("/score", json={"dataset": "bgl", "lines": [bgl(T + 1, "FAILURE", "data storage interrupt")]})
    assert out.status_code == 200
    (result,) = out.json()["results"]
    assert result["is_anomaly"] and result["score"] == 1.0 and result["reasons"]
    assert client.post("/score", json={"dataset": "windows", "lines": []}).status_code == 422
    assert client.post("/score", json={"dataset": "bgl", "detector": "magic", "lines": []}).status_code == 400
    assert client.post("/score", json={"dataset": "bgl", "lines": ["x"] * (MAX_LINES + 1)}).status_code == 422


def test_health_detectors_and_missing_models():
    client = TestClient(create_app(load_engine=toy_engine))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/detectors").json()["bgl"]["default"] == "rule_fatal"

    def missing(dataset, detector):
        raise FileNotFoundError(2, "No such file", "models/hdfs_time_logreg.joblib")
    broken = TestClient(create_app(load_engine=missing))
    assert broken.post("/score", json={"dataset": "hdfs", "lines": []}).status_code == 503
