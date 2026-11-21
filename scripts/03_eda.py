"""
Step 2 - Exploratory Data Analysis (EDA).

Looks at the preprocessed data BEFORE any model is trained, to understand it and
to make modelling decisions with evidence. Each "# %%" block is one question.

Usage:
    python scripts/03_eda.py        # ~1 min -> reports/figures/*.png + reports/eda_stats.json

In VS Code each "# %%" block can also be run on its own ("Run Cell"), like a notebook.
"""
# %% Setup ---------------------------------------------------------------------------
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # running cell-by-cell in VS Code / Jupyter
    ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from src import plot_style as ps  # noqa: E402
from src.eda import anomaly_only_events, event_group_counts, label_conflicts, pattern_overlap  # noqa: E402

ps.apply_style()
pd.set_option("display.width", 160)
P = ROOT / "data" / "processed"
stats: dict = {}

hdfs = pd.read_parquet(P / "hdfs" / "sequences.parquet")
hdfs_lines = pd.read_parquet(P / "hdfs" / "lines.parquet", columns=["timestamp", "event_id", "level"])
hdfs_tpl = pd.read_csv(P / "hdfs" / "templates.csv").set_index("event_id")
bgl = pd.read_parquet(P / "bgl" / "sequences.parquet")
bgl_lines = pd.read_parquet(P / "bgl" / "lines.parquet")
bgl_tpl = pd.read_csv(P / "bgl" / "templates.csv").set_index("event_id")


def time_split_starts(seqs: pd.DataFrame) -> dict:
    """When do the validation and test periods of the time-based split begin?"""
    first = seqs.groupby("split_time", observed=True)["start_time"].min()
    return {"val": first["val"], "test": first["test"]}


def pct(x: float) -> float:
    return round(float(x) * 100, 2)


# %% 1. HDFS - log volume and anomaly rate over time -----------------------------------
# Question: is the 38.7-hour HDFS log steady, and is the anomaly rate the same all the time?
per_10min = hdfs_lines.set_index("timestamp").resample("10min").size()
hourly = hdfs.groupby(hdfs["start_time"].dt.floor("h")).agg(blocks=("label", "size"), rate=("label", "mean"))
hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h"))
hourly.loc[~(hourly["blocks"] >= 200), "rate"] = np.nan   # hours with < 200 blocks: too few to be reliable -> gap
bounds = time_split_starts(hdfs)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"hspace": 0.35})
ax1.fill_between(per_10min.index, per_10min.values, color=ps.SERIES_1, alpha=0.10, linewidth=0)
ax1.plot(per_10min.index, per_10min.values, color=ps.SERIES_1)
ax1.set_title("HDFS log lines per 10 minutes")
ax1.set_ylabel("lines")
ax1.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
ax2.plot(hourly.index, hourly["rate"] * 100, color=ps.ABNORMAL, marker="o", markersize=3.5,
         markeredgecolor=ps.SURFACE, markeredgewidth=1)
overall = hdfs["label"].mean() * 100
ax2.axhline(overall, color=ps.MUTED, linewidth=0.9)
ax2.text(hourly.index[0], overall, f"overall {overall:.2f}%", va="bottom", fontsize=8.5, color=ps.TEXT_2)
ax2.set_title("Share of blocks (by start hour) that are abnormal  ·  gaps = hours with < 200 blocks")
ax2.set_ylabel("abnormal blocks (%)")
ax2.set_ylim(bottom=0)
for ax in (ax1, ax2):
    ps.mark_boundaries(ax, bounds)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
print("saved", ps.save(fig, "01_hdfs_timeline"))

stats["hdfs_timeline"] = {
    "lines_per_10min_median": int(per_10min.median()),
    "lines_per_10min_max": int(per_10min.max()),
    "hourly_abnormal_rate_min_%": round(hourly["rate"].min() * 100, 2),
    "hourly_abnormal_rate_max_%": round(hourly["rate"].max() * 100, 2),
    "level_counts": hdfs_lines["level"].value_counts().to_dict(),
}
print(stats["hdfs_timeline"])

# %% 2. HDFS - which events occur, and which ones signal anomalies? --------------------
# Question: how unequal are event frequencies, and are some events (almost) only seen in abnormal blocks?
ev = event_group_counts(hdfs).join(hdfs_tpl[["template", "occurrences"]])
ev = ev.sort_values("occurrences", ascending=True)       # largest at the top of a horizontal bar chart
labels = [f"{i:>2} · {t[:62] + ('…' if len(t) > 62 else '')}" for i, t in zip(ev.index, ev["template"])]

fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 11), sharey=True, gridspec_kw={"wspace": 0.05,
                                                                                  "width_ratios": [1.1, 1]})
y = np.arange(len(ev))
axl.barh(y, ev["occurrences"], height=0.62, color=ps.SERIES_1)
axl.set_xscale("log")
axl.set_xlim(left=0.8)
axl.set_yticks(y, labels, fontsize=8)
axl.set_title("How often each event occurs (log scale)")
axl.set_xlabel("lines (log scale)")
axr.barh(y, ev["abnormal_share_%"], height=0.62, color=ps.ABNORMAL)
axr.axvline(hdfs["label"].mean() * 100, color=ps.MUTED, linewidth=0.9)
axr.set_xlim(0, 100)
axr.set_title("Of blocks containing the event, % abnormal")
axr.set_xlabel("abnormal blocks (%)   ·   grey line = overall rate 2.93%")
axr.tick_params(axis="y", length=0)
axl.grid(axis="y", visible=False)
axr.grid(axis="y", visible=False)
print("saved", ps.save(fig, "02_hdfs_events"))

top10_share = hdfs_tpl.drop(index=0)["occurrences"].nlargest(10).sum() / hdfs_tpl["occurrences"].sum()
stats["hdfs_events"] = {
    "templates": int(len(ev.drop(index=0, errors="ignore"))),
    "top10_templates_share_of_lines_%": pct(top10_share),
    "events_only_in_abnormal_blocks": ev.index[ev["groups_normal"] == 0].tolist(),
    "events_over_90pct_abnormal": ev.index[ev["abnormal_share_%"] > 90].tolist(),
}
print(stats["hdfs_events"])
print(ev.sort_values("abnormal_share_%", ascending=False).head(15)[
    ["groups_normal", "groups_abnormal", "abnormal_share_%", "template"]].to_string())

# %% 3. BGL - log volume and alerts over 7 months ---------------------------------------
# Question: how bursty is BGL, and when do alerts happen?
daily = bgl_lines.set_index("timestamp").resample("D").agg({"event_id": "size", "is_alert": "sum"})
daily.columns = ["lines", "alert_lines"]
bounds_bgl = time_split_starts(bgl)

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(daily.index, daily["lines"].where(daily["lines"] > 0), color=ps.NORMAL, label="all lines")
ax.plot(daily.index, daily["alert_lines"].where(daily["alert_lines"] > 0), color=ps.ABNORMAL,
        label="alert lines")
ax.set_yscale("log")
ax.set_title("BGL log lines per day (log scale)")
ax.set_ylabel("lines per day")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
ps.mark_boundaries(ax, bounds_bgl)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
print("saved", ps.save(fig, "03_bgl_timeline"))

stats["bgl_timeline"] = {
    "days_with_logs": int((daily["lines"] > 0).sum()),
    "days_total": int(len(daily)),
    "days_with_alerts": int((daily["alert_lines"] > 0).sum()),
    "busiest_day": str(daily["lines"].idxmax().date()),
    "busiest_day_lines": int(daily["lines"].max()),
    "median_lines_per_active_day": int(daily.loc[daily["lines"] > 0, "lines"].median()),
}
print(stats["bgl_timeline"])

# %% 4. BGL - what kinds of alerts are there, and do they change over time? ----------------
# Each alert line is assigned to the period of the time-based split it falls in.
alert_lines = bgl_lines.loc[bgl_lines["is_alert"] == 1, ["timestamp", "alert_type"]].copy()
alert_lines["alert_type"] = alert_lines["alert_type"].astype(str)
alert_lines["period"] = np.select(
    [alert_lines["timestamp"] < bounds_bgl["val"], alert_lines["timestamp"] < bounds_bgl["test"]],
    ["train", "val"], default="test")
alerts = alert_lines["alert_type"].value_counts()
mix = pd.crosstab(alert_lines["alert_type"], alert_lines["period"], normalize="columns") * 100
mix = mix[["train", "val", "test"]]
shown = list(alerts.index[:10])
mix_top = pd.concat([mix.loc[shown], mix.drop(index=shown).sum().to_frame(f"other ({len(alerts) - 10} types)").T])

fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True, gridspec_kw={"wspace": 0.08})
period_names = {"train": "Training period", "val": "Validation period", "test": "Test period"}
counts = alert_lines["period"].value_counts()
for ax, period in zip(axes, ["train", "val", "test"]):
    values = mix_top[period][::-1]
    ax.barh(values.index, values.values, height=0.62, color=ps.ABNORMAL)
    ax.set_xlim(0, 70)
    ax.set_title(f"{period_names[period]}\n{counts.get(period, 0):,} alert lines", fontsize=10)
    ax.set_xlabel("% of the period's alert lines")
    ax.grid(axis="y", visible=False)
axes[1].tick_params(axis="y", length=0)
axes[2].tick_params(axis="y", length=0)
fig.suptitle("BGL alert types by period of the time-based split", x=0.125, ha="left",
             fontweight="bold", fontsize=11, y=1.04)
print("saved", ps.save(fig, "04_bgl_alert_types"))

stats["bgl_alert_types"] = {
    "n_types": int(len(alerts)),
    "share_all_%": {k: round(v, 2) for k, v in (alerts.head(10) / alerts.sum() * 100).items()},
    "share_by_period_%": mix_top.round(1).to_dict(),
    "types_in_test_never_in_train": sorted(
        set(alert_lines.loc[alert_lines["period"] == "test", "alert_type"])
        - set(alert_lines.loc[alert_lines["period"] == "train", "alert_type"])),
}
print(json.dumps(stats["bgl_alert_types"], indent=1))

# %% 5. BGL - is the "level" field (INFO / FATAL ...) the same as the label? ---------------
# Question: could we simply flag every FATAL line? (a no-ML rule to compare models against later)
cross = pd.crosstab(bgl_lines["level"].astype(str), bgl_lines["is_alert"].map({0: "normal", 1: "alert"}))
cross = cross.reindex(columns=["normal", "alert"], fill_value=0).sort_values("alert", ascending=False)
print(cross.to_string())
fatal = bgl_lines["level"].isin(["FATAL", "FAILURE"])
alert = bgl_lines["is_alert"] == 1
# the same rule on 5-minute windows (our detection unit): flag a window if it has any FATAL/FAILURE line
fatal_windows = set(bgl_lines.loc[fatal, "timestamp"].dt.floor("5min"))
flagged = bgl["start_time"].isin(fatal_windows)
abnormal = bgl["label"] == 1
test = bgl["split_time"] == "test"
stats["bgl_level_rule"] = {
    "crosstab": cross.to_dict("index"),
    "rule": "flag if level is FATAL or FAILURE",
    "lines_alerts_caught_%": pct((fatal & alert).sum() / alert.sum()),
    "lines_flags_that_are_alerts_%": pct((fatal & alert).sum() / fatal.sum()),
    "windows_abnormal_caught_%": pct((flagged & abnormal).sum() / abnormal.sum()),
    "windows_flags_that_are_abnormal_%": pct((flagged & abnormal).sum() / flagged.sum()),
    "time_test_windows_abnormal_caught_%": pct((flagged & abnormal & test).sum() / (abnormal & test).sum()),
    "time_test_windows_flags_that_are_abnormal_%": pct((flagged & abnormal & test).sum() / (flagged & test).sum()),
}
print({k: v for k, v in stats["bgl_level_rule"].items() if k != "crosstab"})

# Data quality check: every line should now have a real log level. (The first EDA run found 324
# lines with shifted header fields; preprocessing was fixed to repair them - see docs/02_eda.md.)
real_levels = {"INFO", "WARNING", "ERROR", "SEVERE", "FATAL", "FAILURE"}
odd = bgl_lines.loc[~bgl_lines["level"].isin(real_levels), "line_no"]
bgl_summary = json.loads((P / "bgl" / "summary.json").read_text())
stats["bgl_data_quality"] = {
    "lines_with_unknown_level": int(len(odd)),
    "lines_missing_node_repaired": bgl_summary.get("lines_missing_node_repaired"),
    "lines_empty_content": bgl_summary.get("lines_empty_content"),
}
print(stats["bgl_data_quality"])

# %% 6. BGL - do log messages change over time? (concept drift) --------------------------
# (a) how many templates had been discovered by each date (learning period only)
# (b) after the parser cut-off: daily % of lines that match NO learned template
cutoff = bounds_bgl["val"]
first_seen = (bgl_lines[bgl_lines["event_id"] > 0].groupby("event_id")["timestamp"].min().sort_values())
first_seen = first_seen[first_seen < cutoff]
after = bgl_lines[bgl_lines["timestamp"] >= cutoff]
unseen_daily = after.set_index("timestamp")["event_id"].eq(0).resample("D").mean() * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True, gridspec_kw={"hspace": 0.35})
ax1.step(first_seen.values, np.arange(1, len(first_seen) + 1), where="post", color=ps.SERIES_1)
ax1.set_title("Templates discovered so far (parser learning period)")
ax1.set_ylabel("templates")
ax1.set_ylim(bottom=0)
ax2.bar(unseen_daily.index, unseen_daily.values, width=0.8, color=ps.SERIES_1)
ax2.set_title("After the cut-off: % of each day's lines with an unseen template")
ax2.set_ylabel("unseen lines (%)")
ax2.set_ylim(0, 100)
for ax in (ax1, ax2):
    ps.mark_boundaries(ax, {"parser frozen": cutoff})
ax2.set_xlim(bgl_lines["timestamp"].min(), bgl_lines["timestamp"].max())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
print("saved", ps.save(fig, "05_bgl_drift"))

window_has_unseen = bgl["event_ids"].map(lambda s: bool((np.asarray(s) == 0).any()))
months = first_seen.dt.to_period("M").value_counts().sort_index()
stats["bgl_drift"] = {
    "new_templates_per_month_learning_period": {str(k): int(v) for k, v in months.items()},
    "unseen_line_share_after_cutoff_%": pct(after["event_id"].eq(0).mean()),
    "days_after_cutoff_with_over_50pct_unseen": int((unseen_daily > 50).sum()),
    "windows_with_unseen_event_%_by_time_split": {
        s: pct(window_has_unseen[bgl["split_time"] == s].mean()) for s in ["train", "val", "test"]},
    "hdfs_unseen_lines_after_cutoff": int((hdfs_lines["event_id"] == 0).sum()),
}
print(stats["bgl_drift"])

# %% 7. Group sizes: are abnormal groups longer or shorter? -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, seqs, name, unit in [(axes[0], hdfs, "HDFS", "events per block"),
                             (axes[1], bgl, "BGL", "events per 5-min window")]:
    bins = np.logspace(0, np.log10(seqs["n_events"].max() + 1), 40)
    for lab, color, text in [(0, ps.NORMAL, "normal"), (1, ps.ABNORMAL, "abnormal")]:
        values = seqs.loc[seqs["label"] == lab, "n_events"]
        ax.hist(values, bins=bins, density=False, weights=np.full(len(values), 100 / len(values)),
                histtype="step", linewidth=1.6, color=color)
        ax.plot([], [], color=color, label=f"{text} (median {values.median():.0f})")  # line-style legend
    ax.set_xscale("log")
    ax.set_title(f"{name}: group size, normal vs abnormal")
    ax.set_xlabel(f"{unit} (log scale)")
    ax.set_ylabel("% of groups")
    ax.legend(loc="upper right")
print("saved", ps.save(fig, "06_group_sizes"))

stats["group_sizes"] = {
    name: seqs.groupby("label")["n_events"].describe(percentiles=[0.5, 0.95]).round(1)
    .rename(index={0: "normal", 1: "abnormal"}).to_dict("index")
    for name, seqs in [("HDFS", hdfs), ("BGL", bgl)]
}
# HDFS: very short blocks. The shortest normal block has 13 events - how many abnormal blocks are shorter?
min_normal = int(hdfs.loc[hdfs["label"] == 0, "n_events"].min())
short = hdfs[(hdfs["label"] == 1) & (hdfs["n_events"] < min_normal)]
common = short["event_ids"].map(tuple).value_counts().head(3)
stats["hdfs_short_blocks"] = {
    "shortest_normal_block": min_normal,
    "abnormal_blocks_shorter_than_that": int(len(short)),
    "share_of_abnormal_%": pct(len(short) / hdfs["label"].sum()),
    "most_common": [{"blocks": int(n), "events": [hdfs_tpl.loc[e, "template"] for e in seq]}
                    for seq, n in common.items()],
}
print(json.dumps(stats["group_sizes"], indent=1))
print(json.dumps(stats["hdfs_short_blocks"], indent=1))

# %% 8. Random vs time split: how many test groups were already "seen" in training? --------
# A group's count pattern = how many times each event occurs in it (what count-based models see).
overlap = {f"{name} {split}": pattern_overlap(seqs, f"split_{split}")
           for name, seqs in [("HDFS", hdfs), ("BGL", bgl)] for split in ["random", "time"]}
stats["pattern_overlap"] = overlap
print(json.dumps(overlap, indent=1))
# Patterns that occur with BOTH labels: a count-based model can never get all of these right.
stats["label_conflicts"] = {"HDFS": label_conflicts(hdfs), "BGL": label_conflicts(bgl)}
print(json.dumps(stats["label_conflicts"], indent=1))

fig, ax = plt.subplots(figsize=(8.5, 4.2))
x = np.arange(len(overlap))
w = 0.36
normal_v = [v["test_normal_seen_in_train_%"] for v in overlap.values()]
abn_v = [v["test_abnormal_seen_in_train_%"] for v in overlap.values()]
ax.bar(x - w / 2 - 0.01, normal_v, width=w, color=ps.NORMAL, label="normal test groups")
bars = ax.bar(x + w / 2 + 0.01, abn_v, width=w, color=ps.ABNORMAL, label="abnormal test groups")
ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=8.5, color=ps.TEXT_2)
ax.set_xticks(x, [k.replace(" ", "\n") + " split" for k in overlap])
ax.set_ylim(0, 105)
ax.set_ylabel("% with a pattern seen in training")
ax.set_title("Test groups whose exact event-count pattern also appears in training")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)
ax.grid(axis="x", visible=False)
print("saved", ps.save(fig, "07_pattern_overlap"))

# %% 9. How far does a trivial rule get? ---------------------------------------------------
# Rule: flag a group if it contains an event that never appeared in any NORMAL training group.
rule = {f"{name} {split}": anomaly_only_events(seqs, f"split_{split}")
        for name, seqs in [("HDFS", hdfs), ("BGL", bgl)] for split in ["random", "time"]}
stats["anomaly_only_event_rule"] = rule
print(pd.DataFrame(rule).T.drop(columns="anomaly_only_events").to_string())

# %% Save all numbers ----------------------------------------------------------------------
out = ROOT / "reports" / "eda_stats.json"
out.write_text(json.dumps(stats, indent=2, default=str))
print("saved", out)
