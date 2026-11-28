"""
Milestone 3, step 4 - the live dashboard (Streamlit).

    streamlit run dashboard/app.py        # then open http://localhost:8501

It reads the alert store (the PostgreSQL database named in .env), where the scoring API
saves every group the live system judges. Start a stream to watch it fill up:
    uvicorn src.api:app                   # terminal 1: the scoring API
    python -m src.replay --dataset bgl    # terminal 2: the live log feed

Tabs
    Live monitor          what the system decided: alerts, scores, reasons (no labels used)
    Check with labels     grades the live decisions against the true labels (evaluation only)
    Drift                 how many log messages were never seen in training
    Models                random vs time split: the research question
    Score your own lines  paste raw log lines and send them to the API
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import httpx
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import dashboard_data as dd  # noqa: E402
from src.engine import DEFAULT_DETECTOR  # noqa: E402
from src.plot_style import (ABNORMAL, AXIS, GRID, MUTED, NORMAL, RANDOM_SPLIT, SURFACE, TEXT,  # noqa: E402
                            TEXT_2, TIME_SPLIT)
from src.store import ResultStore  # noqa: E402

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
REFRESH_SECONDS = 5
SYSTEMS = {"bgl": "BGL (supercomputer)", "hdfs": "HDFS (file system)"}
EXAMPLE_LINES = {
    "bgl": ("- 1131054600 2005.11.03 R02-M1-N0-C:J12-U11 2005-11-03-13.50.00.000000 R02-M1-N0-C:J12-U11 "
            "RAS KERNEL FATAL data storage interrupt\n"
            "- 1131054620 2005.11.03 R02-M1-N0-C:J12-U11 2005-11-03-13.50.20.000000 R02-M1-N0-C:J12-U11 "
            "RAS KERNEL INFO instruction cache parity error corrected"),
    "hdfs": ("081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 "
             "src: /10.250.19.102:54106 dest: /10.250.19.102:50010\n"
             "081109 203518 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: "
             "/mnt/hadoop/mapred/system/job_200811092030_0001/job.jar. blk_-1608999687919862906"),
}

st.set_page_config(page_title="Log anomaly monitor", layout="wide")


# ----------------------------------------------------------------------------- data access
@st.cache_resource
def get_store():
    return ResultStore.from_env()


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_runs(dataset: str) -> list[dict]:
    return get_store().runs(dataset)


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_frame(run_id: str) -> pd.DataFrame:
    return get_store().frame(run_id)


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_results(run_id: str, only_alerts: bool, limit: int | None) -> list[dict]:
    return get_store().results(run_id, only_alerts=only_alerts, limit=limit)


@st.cache_data(ttl=60, show_spinner=False)
def store_label() -> str:
    return get_store().label()


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def api_running() -> bool:
    try:
        return httpx.get(f"{API_URL}/health", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


# ----------------------------------------------------------------------------- charts
def styled(chart: alt.Chart) -> alt.Chart:
    """The project's chart look (same colours and recessive axes as the report figures)."""
    return (chart.configure(background=SURFACE)
            .configure_axis(gridColor=GRID, domainColor=AXIS, tickColor=AXIS, labelColor=TEXT_2,
                            titleColor=TEXT_2, titleFontWeight="normal", labelFontSize=11, titleFontSize=11)
            .configure_view(strokeWidth=0)
            .configure_legend(orient="top", title=None, labelColor=TEXT_2, labelFontSize=12, symbolType="square"))


def time_axis(step: str) -> str:
    """Date format for an axis in hours or in days."""
    return "%d %b %H:%M" if step == "h" else "%d %b %Y"


def alerts_chart(data: pd.DataFrame, step: str, span, by_kind: bool = False) -> alt.Chart:
    """Alerts per hour/day as bars over the whole stream (log time)."""
    fmt = time_axis(step)
    per = "hour" if step == "h" else "day"
    data = data.assign(end=data["time"] + pd.Timedelta(1, step))
    x = alt.X("time:T", title=None, scale=alt.Scale(domain=[t.isoformat() for t in span]),
              axis=alt.Axis(format=fmt, labelAngle=0, labelOverlap="greedy", labelSeparation=12, grid=False))
    tooltip = [alt.Tooltip("time:T", title=per, format=fmt), alt.Tooltip("alerts:Q", title="alerts")]
    if by_kind:
        kinds = ["real anomaly", "false alarm"]
        bars = alt.Chart(dd.stacked(data, kinds)).mark_bar(stroke=SURFACE, strokeWidth=1.5).encode(
            x=x, x2="end:T", y=alt.Y("bottom:Q", title=f"alerts per {per}", axis=alt.Axis(tickMinStep=1)), y2="top:Q",
            color=alt.Color("kind:N", scale=alt.Scale(domain=kinds, range=[ABNORMAL, NORMAL])),
            tooltip=[*tooltip, alt.Tooltip("kind:N", title="kind")])
    else:
        bars = alt.Chart(data).mark_bar(color=ABNORMAL, stroke=SURFACE, strokeWidth=1.5).encode(
            x=x, x2="end:T", y=alt.Y("alerts:Q", title=f"alerts per {per}", axis=alt.Axis(tickMinStep=1)),
            y2=alt.datum(0), tooltip=tooltip)
    return styled(bars.properties(height=240))


def drift_chart(data: pd.DataFrame, step: str, span) -> alt.Chart:
    fmt = time_axis(step)
    hover = alt.selection_point(fields=["time"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    base = alt.Chart(data).encode(x=alt.X("time:T", title=None, scale=alt.Scale(domain=[t.isoformat() for t in span]),
                                          axis=alt.Axis(format=fmt, labelAngle=0, labelOverlap="greedy", labelSeparation=12, grid=False)))
    line = base.mark_line(color=NORMAL, strokeWidth=2).encode(
        y=alt.Y("unseen_share:Q", title="never seen in training", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])))
    points = base.mark_point(color=NORMAL, filled=True, size=70).encode(
        y="unseen_share:Q", opacity=alt.condition(hover, alt.value(1), alt.value(0))).add_params(hover)
    rule = base.mark_rule(color=MUTED).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("time:T", title="hour" if step == "h" else "day", format=fmt),
                 alt.Tooltip("unseen_share:Q", title="never seen", format=".1%")])
    return styled(alt.layer(line, points, rule).properties(height=240))


def models_chart(data: pd.DataFrame) -> alt.Chart:
    order = list(data[data["split"] == "time"].sort_values("f1", ascending=False)["model_name"])  # best on the time split first
    base = alt.Chart(data).encode(
        y=alt.Y("model_name:N", title=None, sort=order, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("split_name:N", sort=["random split", "time split"]),
        x=alt.X("f1:Q", title="F1 on the test set", scale=alt.Scale(domain=[0, 1.08]),
                axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1.0])),
        tooltip=[alt.Tooltip("model_name:N", title="model"), alt.Tooltip("split_name:N", title="split"),
                 alt.Tooltip("f1:Q", format=".3f", title="F1"),
                 alt.Tooltip("f1_ci_low:Q", format=".3f", title="95% CI from"),
                 alt.Tooltip("f1_ci_high:Q", format=".3f", title="95% CI to"),
                 alt.Tooltip("precision:Q", format=".3f"), alt.Tooltip("recall:Q", format=".3f")])
    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.Color("split_name:N", scale=alt.Scale(domain=["random split", "time split"],
                                                        range=[RANDOM_SPLIT, TIME_SPLIT])))
    labels = base.mark_text(align="left", dx=4, color=TEXT_2, fontSize=11).encode(text=alt.Text("f1:Q", format=".2f"))
    return styled(alt.layer(bars, labels).properties(height=alt.Step(18)))


# ----------------------------------------------------------------------------- formatting
def duration(td) -> str:
    if td is None or pd.isna(td):
        return "-"
    minutes = td / pd.Timedelta(minutes=1)
    return f"{minutes:.0f} min" if minutes < 90 else f"{minutes / 60:.1f} h"


def is_live(run: dict) -> bool:
    updated = run["updated_at"].replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - updated).total_seconds() < 30


def find_run(dataset: str, run_id: str) -> dict | None:
    return next((r for r in load_runs(dataset) if r["run_id"] == run_id), None)


# ----------------------------------------------------------------------------- sidebar
store = get_store()
if store is None:
    st.error("No alert store configured. Copy `.env.example` to `.env` and set `DATABASE_URL`.")
    st.stop()

with st.sidebar:
    st.title("Log anomaly monitor")
    dataset = st.radio("System", list(SYSTEMS), format_func=SYSTEMS.get)
    runs = load_runs(dataset)
    run = None
    if runs:
        run = st.selectbox("Stream", runs, format_func=lambda r: ("● " if is_live(r) else "")
                           + f"{r['started_at']:%d %b %H:%M} UTC · {r['status'].get('lines', 0):,} lines")
    auto = st.toggle(f"Refresh every {REFRESH_SECONDS} s", value=True)
    st.caption(f"Alert store: {store_label()}")
    st.caption(f"Scoring API: {'running' if api_running() else 'not reachable'} at `{API_URL}`")
    st.caption("● = a stream that is still receiving lines")

st.header(SYSTEMS[dataset])
if run is None:
    st.info(f"No stream of {dataset.upper()} stored yet. Start one: `uvicorn src.api:app`, then "
            f"`python -m src.replay --dataset {dataset}`.")
else:
    s = run["status"]
    st.caption(f"Detector **{s.get('detector', DEFAULT_DETECTOR[dataset])}** · watching since {s.get('watching_since')} "
               f"(log time) · stream {run['run_id']}")

tabs = st.tabs(["Live monitor", "Check with labels", "Drift", "Models", "Score your own lines"])
every = f"{REFRESH_SECONDS}s" if auto else None


# ----------------------------------------------------------------------------- tab 1: live monitor
@st.fragment(run_every=every)
def live_monitor(run_id: str):
    run = find_run(dataset, run_id)
    frame = load_frame(run_id)
    h = dd.headline(frame, run["status"])
    a, b, c, d = st.columns(4)
    date, _, clock = (h["log_time"] or "- -").partition(" ")
    a.metric(f"Log time now ({date})", clock, help="The latest timestamp the system has read in the logs")
    b.metric("Lines read", f"{h['lines']:,}")
    c.metric("Groups judged", f"{h['judged']:,}", help="Blocks (HDFS) or 5-minute windows (BGL) that are finished and scored")
    d.metric("Alerts", f"{h['alerts']:,}", f"{h['alert_share']:.1%} of judged groups", delta_color="off",
             delta_arrow="off")
    a, b, c, d = st.columns(4)
    a.metric("Never-seen messages", f"{h['unseen_share']:.1%}", help="Log messages the parser never saw in training (drift)")
    b.metric("Groups still open", f"{h['open_groups']:,}", help="Waiting to be finished (HDFS: 13 h of silence)")
    c.metric("Not judged", f"{h['not_judged']:,}", help="May have started before monitoring began (cold-start rule)")
    d.metric("Median time to alert", duration(h["median_delay"]),
             help="Log time from a group's last event to its alert. A replay sends about one request per second, "
                  "so at 3600x this includes up to an hour of log time per request.")

    if frame.empty:
        st.info("Nothing judged yet in this stream.")
        return
    step = dd.time_step(frame)
    st.subheader("Alerts over time")
    st.caption("When the flagged groups' activity ended (log time), over the whole stream. Hover a bar for its count.")
    data = dd.alerts_over_time(frame, step)
    if data.empty:
        st.write("No alerts yet.")
    else:
        span = dd.time_range(frame, run["status"], step)
        st.altair_chart(alerts_chart(data, step, span), theme=None, width="stretch")

    st.subheader("Latest alerts")
    recent = load_results(run_id, True, 50)
    if not recent:
        st.write("No alerts yet.")
        return
    table = pd.DataFrame([{"judged at (log time)": r["judged_at"], "group": r["group_id"], "score": r["score"],
                           "events": r["n_events"], "never-seen": r["n_unseen_events"],
                           "main reason": r["reasons"][0] if r["reasons"] else ""} for r in recent])
    st.dataframe(table, hide_index=True, width="stretch", height=260,
                 column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=1, format="%.2f")})

    pick = st.selectbox("Why was it flagged?", recent,
                        format_func=lambda r: f"{r['group_id']} · score {r['score']:.2f} · ended {r['last_seen']}")
    left, right = st.columns([1, 2])
    with left:
        st.metric("Score (alert line 0.5)", f"{pick['score']:.2f}")
        st.write(f"First event: {pick['first_seen']}  \nLast event: {pick['last_seen']}  \n"
                 f"Judged at: {pick['judged_at']}  \nEvents: {pick['n_events']:,} "
                 f"({pick['n_unseen_events']:,} never seen)")
    with right:
        st.markdown("**Reasons given by the detector**")
        st.markdown("\n".join(f"- {reason}" for reason in pick["reasons"]))
        if pick["unseen_examples"]:
            with st.expander("Messages never seen in training"):
                st.markdown("\n".join(f"- `{m}`" for m in pick["unseen_examples"]))


# ----------------------------------------------------------------------------- tab 2: check with labels
@st.fragment(run_every=every)
def check_with_labels(run_id: str):
    st.caption("Evaluation only: the true labels from the dataset are joined here to grade the live "
               "decisions. The live system never sees them, as in real use.")
    frame = load_frame(run_id)
    if frame.empty:
        st.info("Nothing judged yet in this stream.")
        return
    labels = dd.true_labels(dataset)
    g = dd.grade(frame, labels)
    status = find_run(dataset, run_id)["status"]
    a, b, c, d = st.columns(4)
    a.metric("Real anomalies caught", f"{g['caught']:,} of {g['abnormal']:,}")
    b.metric("False alarms", f"{g['false_alarms']:,}", help="Normal groups that were flagged")
    c.metric("Missed anomalies", f"{g['missed']:,}")
    d.metric("False alarms per day", f"{g['false_alarms_per_day']:,.1f}",
             help=f"False alarms / log days covered ({g['days']:.2f} days)")
    a, b, c, d = st.columns(4)
    a.metric("Precision", f"{g['precision']:.3f}", help="Of the alerts, the share that were real anomalies")
    b.metric("Recall", f"{g['recall']:.3f}", help="Of the real anomalies, the share that were caught")
    c.metric("F1", f"{g['f1']:.3f}")
    d.metric("Groups graded", f"{g['groups']:,}")

    step = dd.time_step(frame)
    data = dd.alerts_over_time(frame, step, labels)
    if not data.empty:
        st.subheader("Alerts over time: real anomalies and false alarms")
        span = dd.time_range(frame, status, step)
        st.altair_chart(alerts_chart(data, step, span, by_kind=True), theme=None, width="stretch")
        alerts = load_results(run_id, True, None)
        false = [r for r in alerts if labels.get(r["group_id"], 0) == 0][:20]
        if false:
            st.subheader("Latest false alarms and their reasons")
            st.dataframe(pd.DataFrame([{"group": r["group_id"], "score": r["score"], "events": r["n_events"],
                                        "ended": r["last_seen"], "reasons": " | ".join(r["reasons"])} for r in false]),
                         hide_index=True, width="stretch")


# ----------------------------------------------------------------------------- tab 3: drift
@st.fragment(run_every=every)
def drift(run_id: str):
    frame = load_frame(run_id)
    if frame.empty:
        st.info("Nothing judged yet in this stream.")
        return
    share = frame["n_unseen_events"].sum() / max(frame["n_events"].sum(), 1)
    st.metric("Log messages never seen in training", f"{share:.1%}")
    st.caption("The parser learned its message templates from the training period only and was then frozen. "
               "A message it cannot match is 'never seen' (event 0): new software versions or reworded "
               "messages make this share grow. BGL drifts strongly; HDFS hardly at all.")
    step = dd.time_step(frame)
    span = dd.time_range(frame, find_run(dataset, run_id)["status"], step)
    st.altair_chart(drift_chart(dd.drift_over_time(frame, step), step, span), theme=None, width="stretch")
    examples = dd.unseen_examples(load_results(run_id, False, 500))
    if examples:
        st.subheader("Examples of never-seen messages")
        st.markdown("\n".join(f"- `{m}`" for m in examples))


# ----------------------------------------------------------------------------- tab 4: models
def models():
    data = dd.model_results(dataset)
    served = DEFAULT_DETECTOR[dataset]
    data.loc[data["model"] == served, "model_name"] += "  (served live)"
    st.caption("Test F1 of every model when the data is split at random (as many papers do) and when it is "
               "split by time (train on the past, test on the future). The live system serves the best "
               "model on the time split.")
    st.altair_chart(models_chart(data), theme=None, width="stretch")
    with st.expander("Table"):
        st.dataframe(data.drop(columns=["model", "split"]).rename(columns={"model_name": "model", "split_name": "split"}),
                     hide_index=True, width="stretch")


# ----------------------------------------------------------------------------- tab 5: score your own lines
def score_lines():
    st.caption(f"Paste raw {dataset.upper()} log lines; they are sent to the scoring API (`POST /score`), "
               "which treats them as complete and scores every group at once.")
    text = st.text_area("Log lines", EXAMPLE_LINES[dataset], height=140, key=f"lines_{dataset}")
    if st.button("Score these lines"):
        lines = [line for line in text.splitlines() if line.strip()]
        try:
            out = httpx.post(f"{API_URL}/score", json={"dataset": dataset, "lines": lines}, timeout=60)
            out.raise_for_status()
        except httpx.HTTPError as e:
            st.error(f"The scoring API did not answer ({e.__class__.__name__}). Start it with `uvicorn src.api:app`.")
            return
        body = out.json()
        st.write(f"{body['stats']['lines']} lines read, {len(body['results'])} group(s) scored.")
        for r in body["results"]:
            verdict = "ALERT" if r["is_anomaly"] else "normal"
            st.markdown(f"**{r['group_id']}**: {verdict}, score {r['score']:.2f}")
            if r["reasons"]:
                st.markdown("\n".join(f"- {reason}" for reason in r["reasons"]))


if run is None:
    for tab in tabs[:3]:
        tab.info("No stream yet.")
else:
    with tabs[0]:
        live_monitor(run["run_id"])
    with tabs[1]:
        check_with_labels(run["run_id"])
    with tabs[2]:
        drift(run["run_id"])
with tabs[3]:
    models()
with tabs[4]:
    score_lines()
