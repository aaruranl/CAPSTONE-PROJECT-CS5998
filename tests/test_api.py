"""Tests for live scoring (src/stream.py) and the API (src/api.py), with a tiny hand-made engine."""
from fastapi.testclient import TestClient

from src.api import MAX_LINES, create_app
from src.engine import FatalRuleDetector, ScoringEngine
from src.log_parser import DrainParser
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
    first = client.post("/ingest", json={"dataset": "bgl", "lines": [bgl(T + 1), bgl(T + 2, "FATAL", "data storage interrupt")]})
    assert first.status_code == 200 and first.json()["finished"] == []          # window still open
    second = client.post("/ingest", json={"dataset": "bgl", "lines": [bgl(T + 301)]}).json()
    assert [r["group_id"] for r in second["finished"]] == [str(T // 300)]      # next window started
    assert second["finished"][0]["is_anomaly"] and second["status"]["open_groups"] == 1
    last = client.post("/flush", json={"dataset": "bgl"}).json()
    assert [r["group_id"] for r in last["finished"]] == [str(T // 300 + 1)]
    assert last["status"]["alerts"] == 1 and last["status"]["finished_groups"] == 2


def test_live_hdfs_block_finishes_after_the_quiet_time():
    live = LiveScorer(toy_engine("hdfs"), quiet_seconds=60)
    assert live.ingest([hdfs("203500", "blk_1"), hdfs("203530", "blk_2")]) == []
    done = live.ingest([hdfs("203610", "blk_3")])        # 70 s after blk_1, 40 s after blk_2
    assert [r["group_id"] for r in done] == ["blk_1"]
    live.ingest([hdfs("203620", "blk_1")])               # blk_1 was declared finished too early
    assert live.status()["groups_reopened"] == 1
    assert sorted(r["group_id"] for r in live.flush()) == ["blk_1", "blk_2", "blk_3"]


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
