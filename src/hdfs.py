"""
HDFS preprocessing: raw log lines -> one labelled event sequence per block.

A raw HDFS line looks like:

    081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
    |date | time |pid|level|        component        |                          content

Each file in HDFS is stored as "blocks". Every operation on a block (allocate,
receive, replicate, delete...) writes log lines that mention its ID, so all the
lines for one block form a natural "session". The Loghub labels say whether
each BLOCK was normal or abnormal, so the block is our unit of detection.

Pipeline (two passes over the 1.5 GB file):
  Pass 1  find the first line where each block appears -> order blocks in time
          -> decide the splits -> find the cut-off line for the parser.
  Pass 2  parse every line with Drain (learning only before the cut-off) and
          append each event ID to the sequence of every block it mentions.
"""
import re
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.common import CategoryEncoder, Timer, count_lines, templates_table, write_summary
from src.log_parser import DrainParser
from src.splits import chronological_split, random_split, split_summary

HDFS_LINE = re.compile(r"^(\d{6}) (\d{6}) (\d+) (\w+) (\S+?): (.*)$")
BLOCK_ID = re.compile(r"blk_-?\d+")


@lru_cache(maxsize=None)
def _day_epoch(yymmdd: str) -> int:
    day = datetime(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6]), tzinfo=timezone.utc)
    return int(day.timestamp())


def parse_hdfs_line(line: str):
    """Split a raw line into (unix_seconds, level, component, content), or None if malformed."""
    m = HDFS_LINE.match(line.rstrip("\n"))
    if m is None:
        return None
    date, hhmmss, _pid, level, component, content = m.groups()
    ts = _day_epoch(date) + int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:])
    return ts, level, component, content


def load_labels(path) -> dict[str, int]:
    df = pd.read_csv(path)
    return dict(zip(df["BlockId"], (df["Label"] == "Anomaly").astype(int)))


def preprocess_hdfs(max_lines: int | None = None) -> dict:
    info = config.DATASETS["HDFS"]
    log_path, out_dir = info["log_file"], info["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = Timer()

    n_lines = count_lines(log_path) if max_lines is None else max_lines

    # ---------------------------------------------------------------- pass 1
    first_line: dict[str, int] = {}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(tqdm(f, total=n_lines, desc="HDFS pass 1/2 (find blocks)")):
            if i >= n_lines:
                break
            for blk in BLOCK_ID.findall(line):
                first_line.setdefault(blk, i)

    labels = load_labels(info["label_file"])
    blocks = pd.DataFrame({"block_id": list(first_line), "first_line": list(first_line.values())})
    missing_label = (~blocks["block_id"].isin(labels.keys())).sum()
    blocks = blocks[blocks["block_id"].isin(labels.keys())].reset_index(drop=True)
    blocks["label"] = blocks["block_id"].map(labels).astype("int8")
    blocks["split_random"] = random_split(blocks["label"].to_numpy())
    blocks["split_time"] = chronological_split(blocks["first_line"].to_numpy())

    # The parser may only learn from lines written before the first validation block appears.
    cutoff_line = int(blocks.loc[blocks["split_time"] != "train", "first_line"].min())

    # ---------------------------------------------------------------- pass 2
    parser = DrainParser()
    ts_arr = np.full(n_lines, -1, dtype=np.int64)
    eid_arr = np.full(n_lines, -1, dtype=np.int32)
    level_arr = np.full(n_lines, -1, dtype=np.int16)
    comp_arr = np.full(n_lines, -1, dtype=np.int16)
    levels, components = CategoryEncoder(), CategoryEncoder()
    sequences: dict[str, list[int]] = {b: [] for b in blocks["block_id"]}
    start_ts: dict[str, int] = {}
    end_ts: dict[str, int] = {}
    n_malformed = n_no_block = 0

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(tqdm(f, total=n_lines, desc="HDFS pass 2/2 (parse + group)")):
            if i >= n_lines:
                break
            parsed = parse_hdfs_line(line)
            if parsed is None:
                n_malformed += 1
                continue
            ts, level, component, content = parsed

            learn = i < cutoff_line
            if not learn and not parser.frozen:
                parser.freeze()
            event_id = parser.parse(content, learn=learn)

            ts_arr[i], eid_arr[i] = ts, event_id
            level_arr[i], comp_arr[i] = levels.encode(level), components.encode(component)

            block_ids = set(BLOCK_ID.findall(content))
            if not block_ids:
                n_no_block += 1
            for blk in block_ids:
                seq = sequences.get(blk)
                if seq is None:          # block without a label
                    continue
                seq.append(event_id)
                start_ts.setdefault(blk, ts)
                end_ts[blk] = ts

    # ---------------------------------------------------------------- outputs
    blocks["event_ids"] = blocks["block_id"].map(sequences)
    blocks["n_events"] = blocks["event_ids"].map(len).astype("int32")
    blocks["start_time"] = pd.to_datetime(blocks["block_id"].map(start_ts), unit="s")
    blocks["end_time"] = pd.to_datetime(blocks["block_id"].map(end_ts), unit="s")
    blocks["duration_s"] = (blocks["end_time"] - blocks["start_time"]).dt.total_seconds().astype("int32")
    blocks = blocks.sort_values("first_line").reset_index(drop=True)
    blocks = blocks[["block_id", "first_line", "start_time", "end_time", "duration_s",
                     "n_events", "event_ids", "label", "split_random", "split_time"]]
    blocks["split_random"] = blocks["split_random"].astype("category")
    blocks["split_time"] = blocks["split_time"].astype("category")
    blocks.to_parquet(out_dir / "sequences.parquet", index=False)

    ok = eid_arr >= 0
    lines = pd.DataFrame({
        "line_no": np.flatnonzero(ok).astype(np.int32),
        "timestamp": pd.to_datetime(ts_arr[ok], unit="s"),
        "event_id": eid_arr[ok],
        "level": levels.to_categorical(level_arr[ok]),
        "component": components.to_categorical(comp_arr[ok]),
    })
    lines.to_parquet(out_dir / "lines.parquet", index=False)

    learned_mask = np.arange(n_lines) < cutoff_line
    templates = templates_table(parser, eid_arr, learned_mask)
    templates.to_csv(out_dir / "templates.csv", index=False)

    summary = {
        "dataset": "HDFS",
        "group_unit": "block",
        "lines_total": n_lines,
        "lines_malformed": n_malformed,
        "lines_without_block_id": n_no_block,
        "blocks": len(blocks),
        "blocks_found_in_log_without_label": int(missing_label),
        "anomalous_blocks": int(blocks["label"].sum()),
        "anomaly_rate_%": round(blocks["label"].mean() * 100, 3),
        "time_range": [str(lines["timestamp"].min()), str(lines["timestamp"].max())],
        "parser": {
            "learning_cutoff_line": cutoff_line,
            "learning_cutoff_time": str(pd.to_datetime(ts_arr[cutoff_line], unit="s")),
            "lines_learned_from": parser.n_learned,
            "lines_matched_frozen": parser.n_matched,
            "lines_unseen_template": parser.n_unseen,
            "templates": len(templates) - 1,  # minus the <UNSEEN> bucket
        },
        "events_per_block": blocks["n_events"].describe().round(2).to_dict(),
        "split_random": split_summary(blocks, "split_random").to_dict("index"),
        "split_time": split_summary(blocks, "split_time").to_dict("index"),
        "runtime_seconds": timer.seconds(),
    }
    write_summary(out_dir / "summary.json", summary)
    return summary
