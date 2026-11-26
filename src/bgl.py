"""
BGL preprocessing: raw log lines -> one labelled event sequence per 5-minute window.

A raw BGL line looks like:

    - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
    |label|unix time|  date  |      node         |       full time          |    node (again)   |type|comp.|level|            content

  * label = "-" for a normal line, otherwise an alert category (e.g. APPREAD, KERNDTLB).
    The label is stored separately and NEVER given to the parser or the models as input.
  * ~34k lines have no content at all (header only). We keep them and give them
    the message "<EMPTY>" so they still count as an event.
  * 324 lines have no node ("-") and are missing the "node (again)" field, so every
    later field shifts one place left (found during EDA). We detect and repair them:

    - 1119415930 2005.06.21 - 2005-06-21-21.52.10.214285 RAS KERNEL FATAL Kill job 20251 timed out. Block freed.

BGL has no natural session ID, so we cut the timeline into fixed, non-overlapping
5-minute windows (aligned to the clock: 00:00-00:05, 00:05-00:10 ...).
A window is abnormal (label 1) if it contains at least one alert line.
Windows with no log lines at all are not created.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.common import CategoryEncoder, Timer, count_lines, templates_table, write_summary
from src.log_parser import DrainParser
from src.splits import chronological_split, random_split, split_summary

EMPTY_CONTENT = "<EMPTY>"


def parse_bgl_line(line: str):
    """Return (alert_label, unix_seconds, node, component, level, content), or None if malformed."""
    line = line.rstrip("\n")
    parts = line.split(maxsplit=9)
    if len(parts) >= 4 and parts[3] == "-":  # node missing -> one header field fewer
        parts = line.split(maxsplit=8)
        parts.insert(5, "-")                  # put the missing "node (again)" back
    if len(parts) < 9:
        return None
    try:
        ts = int(parts[1])
    except ValueError:
        return None
    content = parts[9] if len(parts) == 10 else EMPTY_CONTENT
    return parts[0], ts, parts[3], parts[7], parts[8], content


def preprocess_bgl(max_lines: int | None = None, window_minutes: int = config.BGL_WINDOW_MINUTES) -> dict:
    info = config.DATASETS["BGL"]
    log_path, out_dir = info["log_file"], info["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    window_s = window_minutes * 60

    n_lines = count_lines(log_path) if max_lines is None else max_lines

    # ---------------------------------------------------------------- pass 1
    # Which windows exist, and does each contain an alert? (needed to decide splits)
    window_alert: dict[int, bool] = {}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(tqdm(f, total=n_lines, desc="BGL pass 1/2 (find windows)")):
            if i >= n_lines:
                break
            parsed = parse_bgl_line(line)
            if parsed is None:
                continue
            w = parsed[1] // window_s
            window_alert[w] = window_alert.get(w, False) or parsed[0] != "-"

    windows = pd.DataFrame({"window": list(window_alert),
                            "label": np.fromiter(window_alert.values(), dtype=np.int8)})
    windows = windows.sort_values("window").reset_index(drop=True)
    windows["split_random"] = random_split(windows["label"].to_numpy())
    windows["split_time"] = chronological_split(windows["window"].to_numpy())

    # The parser may only learn from lines before the first validation window starts.
    cutoff_ts = int(windows.loc[windows["split_time"] != "train", "window"].min()) * window_s

    # ---------------------------------------------------------------- pass 2
    parser = DrainParser()
    ts_arr = np.full(n_lines, -1, dtype=np.int64)
    eid_arr = np.full(n_lines, -1, dtype=np.int32)
    alert_arr = np.zeros(n_lines, dtype=np.int8)
    codes = {k: np.full(n_lines, -1, dtype=np.int32) for k in ["alert_type", "level", "component", "node"]}
    enc = {k: CategoryEncoder() for k in codes}
    sequences: dict[int, list[int]] = {w: [] for w in windows["window"]}
    alert_lines: dict[int, int] = dict.fromkeys(windows["window"], 0)
    n_malformed = n_empty = n_no_node = 0

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(tqdm(f, total=n_lines, desc="BGL pass 2/2 (parse + group)")):
            if i >= n_lines:
                break
            parsed = parse_bgl_line(line)
            if parsed is None:
                n_malformed += 1
                continue
            alert_type, ts, node, component, level, content = parsed
            n_empty += content == EMPTY_CONTENT
            n_no_node += node == "-"

            learn = ts < cutoff_ts
            if not learn and not parser.frozen:
                parser.freeze()
            event_id = parser.parse(content, learn=learn)

            is_alert = alert_type != "-"
            ts_arr[i], eid_arr[i], alert_arr[i] = ts, event_id, is_alert
            for key, value in (("alert_type", alert_type), ("level", level),
                               ("component", component), ("node", node)):
                codes[key][i] = enc[key].encode(value)

            w = ts // window_s
            sequences[w].append(event_id)
            alert_lines[w] += is_alert

    # ---------------------------------------------------------------- outputs
    windows["start_time"] = pd.to_datetime(windows["window"] * window_s, unit="s")
    windows["end_time"] = windows["start_time"] + pd.Timedelta(seconds=window_s)
    windows["event_ids"] = windows["window"].map(sequences)
    windows["n_events"] = windows["event_ids"].map(len).astype("int32")
    windows["n_alert_lines"] = windows["window"].map(alert_lines).astype("int32")
    windows = windows.rename(columns={"window": "window_id"})
    windows = windows[["window_id", "start_time", "end_time", "n_events", "n_alert_lines",
                       "event_ids", "label", "split_random", "split_time"]]
    windows["split_random"] = windows["split_random"].astype("category")
    windows["split_time"] = windows["split_time"].astype("category")
    windows.to_parquet(out_dir / "sequences.parquet", index=False)

    ok = eid_arr >= 0
    lines = pd.DataFrame({
        "line_no": np.flatnonzero(ok).astype(np.int32),
        "timestamp": pd.to_datetime(ts_arr[ok], unit="s"),
        "event_id": eid_arr[ok],
        "is_alert": alert_arr[ok],
        **{k: enc[k].to_categorical(codes[k][ok]) for k in codes},
    })
    lines.to_parquet(out_dir / "lines.parquet", index=False)

    templates = templates_table(parser, eid_arr, ts_arr < cutoff_ts)
    templates.to_csv(out_dir / "templates.csv", index=False)
    parser.freeze()
    parser.save(out_dir / "parser.pkl")  # used by the scoring engine (src/engine.py)

    summary = {
        "dataset": "BGL",
        "group_unit": f"{window_minutes}-minute window",
        "lines_total": n_lines,
        "lines_malformed": n_malformed,
        "lines_empty_content": n_empty,
        "lines_missing_node_repaired": n_no_node,
        "alert_lines": int(alert_arr.sum()),
        "alert_line_rate_%": round(alert_arr[ok].mean() * 100, 3),
        "windows": len(windows),
        "anomalous_windows": int(windows["label"].sum()),
        "anomaly_rate_%": round(windows["label"].mean() * 100, 3),
        "time_range": [str(lines["timestamp"].min()), str(lines["timestamp"].max())],
        "parser": {
            "learning_cutoff_time": str(pd.to_datetime(cutoff_ts, unit="s")),
            "lines_learned_from": parser.n_learned,
            "lines_matched_frozen": parser.n_matched,
            "lines_unseen_template": parser.n_unseen,
            "templates": len(templates) - 1,  # minus the <UNSEEN> bucket
        },
        "events_per_window": windows["n_events"].describe().round(2).to_dict(),
        "split_random": split_summary(windows, "split_random").to_dict("index"),
        "split_time": split_summary(windows, "split_time").to_dict("index"),
        "runtime_seconds": timer.seconds(),
    }
    write_summary(out_dir / "summary.json", summary)
    return summary
