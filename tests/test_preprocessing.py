"""
Unit tests for the preprocessing building blocks.
Run with:  python -m pytest -q
"""
import numpy as np
import pytest

from src.bgl import EMPTY_CONTENT, parse_bgl_line
from src.hdfs import BLOCK_ID, parse_hdfs_line
from src.log_parser import UNSEEN_EVENT_ID, DrainParser, mask
from src.splits import chronological_split, random_split

HDFS_RAW = ("081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block "
            "blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010")
BGL_NORMAL = ("- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 "
              "R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected")
BGL_ALERT = ("APPREAD 1117869872 2005.06.04 R23-M1-N8-I:J18-U11 2005-06-04-00.24.32.398284 "
             "R23-M1-N8-I:J18-U11 RAS APP FATAL ciod: failed to read message prefix on control "
             "stream (CioStream socket to 172.16.96.116:33399")
BGL_EMPTY = ("- 1120866514 2005.07.08 R02-M1-N8-C:J13-U01 2005-07-08-16.48.34.004234 "
             "R02-M1-N8-C:J13-U01 RAS KERNEL FATAL")


# ---------------------------------------------------------------- line parsing
def test_parse_hdfs_line():
    ts, level, component, content = parse_hdfs_line(HDFS_RAW)
    assert ts == 1226262918                      # 2008-11-09 20:35:18 UTC
    assert level == "INFO"
    assert component == "dfs.DataNode$DataXceiver"
    assert content.startswith("Receiving block blk_-1608999687919862906")
    assert BLOCK_ID.findall(content) == ["blk_-1608999687919862906"]


def test_parse_hdfs_line_rejects_garbage():
    assert parse_hdfs_line("this is not a log line") is None


def test_parse_bgl_lines():
    label, ts, node, component, level, content = parse_bgl_line(BGL_NORMAL)
    assert (label, ts, node, component, level) == ("-", 1117838570, "R02-M1-N0-C:J12-U11", "KERNEL", "INFO")
    assert content == "instruction cache parity error corrected"

    label, *_ , content = parse_bgl_line(BGL_ALERT)
    assert label == "APPREAD"
    assert content.startswith("ciod: failed to read")

    *_, content = parse_bgl_line(BGL_EMPTY)
    assert content == EMPTY_CONTENT


# ---------------------------------------------------------------- masking + Drain
def test_mask_replaces_variables():
    content = parse_hdfs_line(HDFS_RAW)[3]
    assert mask(content) == "Receiving block <BLK> src: /<IP> dest: /<IP>"
    assert mask("generating core.2275") == "generating core.<NUM>"
    assert mask("CE sym 9, at 0x0b85eee0, mask 0x10") == "CE sym <NUM>, at <HEX>, mask <HEX>"


def test_drain_learns_then_freezes():
    p = DrainParser()
    a = p.learn("Received block blk_1 of size 100 from /10.0.0.1")
    b = p.learn("Received block blk_2 of size 200 from /10.0.0.2")
    assert a == b                                 # same template, different values
    p.freeze()
    assert p.match("Received block blk_3 of size 300 from /10.0.0.3") == a
    assert p.match("completely new message never seen before") == UNSEEN_EVENT_ID
    with pytest.raises(RuntimeError):
        p.learn("learning after freeze is not allowed")


# ---------------------------------------------------------------- splits
def test_random_split_is_stratified_and_reproducible():
    labels = np.array([0] * 970 + [1] * 30)
    s1, s2 = random_split(labels, seed=42), random_split(labels, seed=42)
    assert (s1 == s2).all()                       # same seed -> same split
    assert (s1 == "train").sum() == 700 and (s1 == "val").sum() == 100 and (s1 == "test").sum() == 200
    for part in ["train", "val", "test"]:         # each part keeps the 3% anomaly rate
        assert labels[s1 == part].mean() == pytest.approx(0.03, abs=0.002)


def test_chronological_split_respects_time():
    times = np.random.default_rng(0).permutation(1000)   # shuffled start times
    s = chronological_split(times)
    assert times[s == "train"].max() < times[s == "val"].min()
    assert times[s == "val"].max() < times[s == "test"].min()
    assert (s == "train").sum() == 700 and (s == "test").sum() == 200
