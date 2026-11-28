"""
Step 5 - the scoring API: the scoring engine behind a web address.

Run it:
    uvicorn src.api:app            # then open http://127.0.0.1:8000/docs to try every endpoint

Endpoints
    GET  /health      is the service up, and what is loaded?
    GET  /detectors   which detectors exist, the default, and their test F1 (time split)
    POST /score       BATCH: the lines sent are complete; score all their groups now
    POST /ingest      LIVE: lines arrive bit by bit; returns the groups that just finished
    POST /flush       LIVE: end of stream; score every group still open
    POST /reset       LIVE: start a new stream (new run ID), e.g. to restart a demo;
                      optionally says from which log time monitoring begins (watch_from)

Models and the parser are loaded on first use and then kept in memory. Every
(dataset, detector) pair has its own engine and lock, so simultaneous requests
cannot corrupt each other's state.

Alert store (step 6): if DATABASE_URL is set (in the environment or in the
.env file), every group judged in LIVE mode is saved in that database (src/store.py), with
the stream's status. The connection is checked at start-up and reported in the log.
Without it the API works the same, it just does not store anything. If the database
fails, the results are still returned (and "stored" is false): detection never stops
because of the database.
"""
import logging
from contextlib import asynccontextmanager
from threading import Lock
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from src.engine import DEFAULT_DETECTOR, DETECTORS, ScoringEngine
from src.experiment import RESULTS_DIR
from src.store import ResultStore
from src.stream import LiveScorer

log = logging.getLogger("uvicorn.error")

MAX_LINES = 50_000  # per request: protects the service from huge requests

Dataset = Literal["hdfs", "bgl"]


class LinesIn(BaseModel):
    dataset: Dataset = Field(description="which system the lines come from")
    detector: str | None = Field(None, description="leave empty for the default "
                                 "(hdfs: logreg, bgl: rule_fatal - the best on the time split)")
    lines: list[str] = Field(max_length=MAX_LINES, description="raw log lines, exactly as written",
                             examples=[["081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block "
                                        "blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"]])
    now: int | None = Field(None, description="LIVE only (/ingest): the current log time in unix seconds, sent even "
                            "with no lines, so that groups whose time is up finish during a silence (a 'watermark')")


class StreamIn(BaseModel):
    dataset: Dataset
    detector: str | None = None


class ResetIn(StreamIn):
    watch_from: int | None = Field(None, description="log time (unix seconds) at which monitoring begins; "
                                   "default: the timestamp of the first line received")
    cold_start_rule: bool = Field(True, description="do not judge groups that may have started before "
                                  "monitoring began (see src/stream.py); false only to show the problem")


class GroupResult(BaseModel):
    dataset: str
    group_id: str = Field(description="HDFS block ID, or BGL window number (unix seconds // 300)")
    detector: str
    first_seen: str
    last_seen: str
    n_events: int
    n_unseen_events: int = Field(description="messages the parser had never seen (event 0)")
    score: float = Field(description="0 to 1; 0.5 is the alert line for every detector")
    is_anomaly: bool
    raw_score: float
    threshold: float
    reasons: list[str] = Field(description="why the group was flagged (empty if normal)")
    unseen_examples: list[str]
    finished_at: str | None = Field(None, description="LIVE only: log time at which the group was finished and judged")


class ScoreOut(BaseModel):
    results: list[GroupResult]
    stats: dict


class IngestOut(BaseModel):
    finished: list[GroupResult] = Field(description="groups that finished (and were judged) with this batch")
    status: dict
    stored: bool = Field(description="true if the results were saved in the database")


def _test_f1() -> dict:
    """Time-split test F1 of every detector, from the offline evaluation (shown by /detectors)."""
    path = RESULTS_DIR / "results.csv"
    if not path.exists():
        return {}
    r = pd.read_csv(path)
    r = r[r["split"] == "time"]
    return {(d, m): round(float(f), 3) for d, m, f in zip(r["dataset"], r["model"], r["f1"])}


def create_app(load_engine=ScoringEngine.load, store: ResultStore | None = None) -> FastAPI:
    """Build the app. Tests pass their own `load_engine` (no real models needed) and `store`."""

    @asynccontextmanager
    async def lifespan(_app):  # runs when the server starts
        if store is not None:
            try:
                store.connect()
                log.info("alert store ready: %s", store.describe())
            except SQLAlchemyError as e:
                log.error("alert store NOT reachable (%s): %s. Live results will not be stored.",
                          store.describe(), e.__class__.__name__)
        yield

    app = FastAPI(lifespan=lifespan, title="Log anomaly detection API", version="1.0",
                  description="Scores raw HDFS / BGL log lines with the models of the CS5998 capstone project. "
                              "Every group gets a score from 0 to 1 (alert line 0.5), a label and the reasons.")
    engines: dict[tuple[str, str], tuple[ScoringEngine, Lock]] = {}
    live: dict[tuple[str, str], LiveScorer] = {}
    registry_lock = Lock()

    def get(dataset: str, detector: str | None):
        detector = detector or DEFAULT_DETECTOR[dataset]
        if detector not in DETECTORS[dataset]:
            raise HTTPException(400, f"unknown detector {detector!r} for {dataset}; choose from {list(DETECTORS[dataset])}")
        key = (dataset, detector)
        with registry_lock:
            if key not in engines:
                try:
                    engines[key] = (load_engine(dataset, detector), Lock())
                except FileNotFoundError as e:
                    raise HTTPException(503, f"model files missing ({e.filename}); run scripts 02, 04 and 05 first")
            if key not in live:
                live[key] = LiveScorer(engines[key][0])
        engine, lock = engines[key]
        return key, engine, lock

    def finish(scorer: LiveScorer, finished: list[dict]) -> dict:
        """Store the results (if a database is configured) and build the response."""
        status, stored = scorer.status(), False
        if store is not None:
            try:
                store.save(status, finished)
                stored = True
            except SQLAlchemyError as e:
                log.error("could not store results of %s: %s", scorer.run_id, e)
        return {"finished": finished, "status": status, "stored": stored}

    @app.get("/health")
    def health():
        return {"status": "ok", "loaded": [f"{d}/{m}" for d, m in engines],
                "database": store.describe() if store is not None else None,
                "live_streams": {f"{d}/{m}": s.status() for (d, m), s in live.items() if s.stats["lines"]}}

    @app.get("/detectors")
    def detectors():
        f1 = _test_f1()
        return {ds: {"default": DEFAULT_DETECTOR[ds],
                     "available": [{"name": m, "test_f1_time_split": f1.get((ds, m))} for m in names]}
                for ds, names in DETECTORS.items()}

    @app.post("/score", response_model=ScoreOut)
    def score(body: LinesIn):
        _, engine, lock = get(body.dataset, body.detector)
        with lock:
            results = engine.score_lines(body.lines)
            return {"results": results, "stats": engine.last_stats}

    @app.post("/ingest", response_model=IngestOut)
    def ingest(body: LinesIn):
        key, _, lock = get(body.dataset, body.detector)
        with lock:
            return finish(live[key], live[key].ingest(body.lines, body.now))

    @app.post("/flush", response_model=IngestOut)
    def flush(body: StreamIn):
        key, _, lock = get(body.dataset, body.detector)
        with lock:
            return finish(live[key], live[key].flush())

    @app.post("/reset")
    def reset(body: ResetIn):
        key, engine, lock = get(body.dataset, body.detector)
        with lock:
            live[key] = LiveScorer(engine, watch_from=body.watch_from, cold_start_rule=body.cold_start_rule)
            return {"status": live[key].status()}

    return app


app = create_app(store=ResultStore.from_env())
