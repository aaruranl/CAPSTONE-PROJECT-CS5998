"""Tests for the scoring engine (Milestone 3, step 1) on tiny hand-made inputs."""
import numpy as np
import pytest

from src.engine import FatalRuleDetector, ScoringEngine, to_unit_score
from src.log_parser import UNSEEN_EVENT_ID, DrainParser


def bgl_line(label, ts, level, content):
    node = "R02-M1-N0-C:J12-U11"
    return f"{label} {ts} 2005.06.03 {node} 2005-06-03-15.42.50.675872 {node} RAS KERNEL {level} {content}"


def frozen_parser(messages):
    parser = DrainParser()
    for m in messages:
        parser.learn(m)
    parser.freeze()
    return parser


def test_unit_score_puts_every_threshold_at_one_half():
    raw = np.array([0.0, 0.05, 0.1, 0.55, 1.0])
    unit = to_unit_score(raw, threshold=0.1, max_score=1.0)
    assert unit.tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.all(np.diff(unit) > 0)                                  # order of groups is kept
    assert to_unit_score([0, 1], 1.0, 1.0).tolist() == [0.0, 1.0]       # a 0/1 rule
    assert to_unit_score([184], 184, 184).tolist() == [1.0]             # DeepLog with g = vocabulary size


def test_parser_save_and_load_give_the_same_matches(tmp_path):
    parser = frozen_parser(["Receiving block blk_1 src: /1.2.3.4:5", "Deleting block blk_2 file /a/b"])
    parser.save(tmp_path / "parser.pkl")
    loaded = DrainParser.load(tmp_path / "parser.pkl")
    for msg in ["Receiving block blk_9 src: /9.9.9.9:1", "Deleting block blk_7 file /c", "something new"]:
        assert loaded.match(msg) == parser.match(msg)
    assert loaded.match("something new") == UNSEEN_EVENT_ID
    with pytest.raises(RuntimeError):
        DrainParser().save(tmp_path / "unfrozen.pkl")


def test_hdfs_line_mentioning_two_blocks_goes_to_both_groups():
    parser = frozen_parser(["BLOCK* ask 1.2.3.4:5 to delete blk_1 blk_2"])
    engine = ScoringEngine("hdfs", parser, {}, FatalRuleDetector())
    groups = engine.group_lines(["081109 203518 143 INFO dfs.FSNamesystem: BLOCK* ask 10.0.0.1:50010 to delete blk_7 blk_-8"])
    assert sorted(g.group_id for g in groups) == ["blk_-8", "blk_7"]


def test_bgl_windows_fatal_rule_and_unseen_messages():
    parser = frozen_parser(["instruction cache parity error corrected", "data storage interrupt"])
    engine = ScoringEngine("bgl", parser, {1: "instruction cache parity error corrected", 2: "data storage interrupt"},
                           FatalRuleDetector())
    t = 1117838400  # start of a 5-minute window
    results = engine.score_lines([
        bgl_line("KERNDTLB", t + 1, "INFO", "instruction cache parity error corrected"),  # label field must be ignored
        bgl_line("-", t + 2, "INFO", "a message never seen before"),
        bgl_line("-", t + 400, "FATAL", "data storage interrupt"),                      # next window
        "not a log line",
    ])
    first, second = results
    assert first["group_id"] == str(t // 300) and not first["is_anomaly"] and first["score"] == 0.0
    assert first["n_unseen_events"] == 1 and first["unseen_examples"] == ["a message never seen before"]
    assert second["is_anomaly"] and second["score"] == 1.0
    assert second["reasons"][0] == "1 line(s) with level FATAL/FAILURE"
    assert engine.last_stats == {"lines": 4, "malformed": 1, "without_group": 0, "unseen_messages": 1}
