"""
Milestone 3, step 2 - the scoring API: the scoring engine behind a web address.

Run it:
    uvicorn src.api:app            # then open http://127.0.0.1:8000/docs to try every endpoint

Endpoints
    GET  /health      is the service up, and what is loaded?
    GET  /detectors   which detectors exist, the default, and their test F1 (time split)
    POST /score       BATCH: the lines sent are complete; score all their groups now
    POST /ingest      LIVE: lines arrive bit by bit; returns the groups that just finished
    POST /flush       LIVE: end of stream; score every group still open
    POST /reset       LIVE: forget the open groups (e.g. to restart a demo)

Models and the parser are loaded on first use and then kept in memory. Every
(dataset, detector) pair has its own engine and lock, so simultaneous requests
cannot corrupt each other's state.
"""
from threading import Lock
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.engine import DEFAULT_DETECTOR, DETECTORS, ScoringEngine
from src.experiment import RESULTS_DIR
from src.stream import LiveScorer

MAX_LINES = 50_000  # per request: protects the service from huge requests

Dataset = Literal["hdfs", "bgl"]


class LinesIn(BaseModel):
    dataset: Dataset = Field(description="which system the lines come from")
    detector: str | None = Field(None, description="leave empty for the default "
                                 "(hdfs: logreg, bgl: rule_fatal - the best on the time split)")
    lines: list[str] = Field(max_length=MAX_LINES, description="raw log lines, exactly as written",
                             examples=[["081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block "
                                        "blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"]])


class StreamIn(BaseModel):
    dataset: Dataset
    detector: str | None = None


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


class ScoreOut(BaseModel):
    results: list[GroupResult]
    stats: dict


class IngestOut(BaseModel):
    finished: list[GroupResult] = Field(description="groups that finished with this batch")
    status: dict


def _test_f1() -> dict:
    """Time-split test F1 of every detector, from step 3 (shown by /detectors)."""
    path = RESULTS_DIR / "results.csv"
    if not path.exists():
        return {}
    r = pd.read_csv(path)
    r = r[r["split"] == "time"]
    return {(d, m): round(float(f), 3) for d, m, f in zip(r["dataset"], r["model"], r["f1"])}


def create_app(load_engine=ScoringEngine.load) -> FastAPI:
    """Build the app. Tests pass their own `load_engine` to avoid needing the real models."""
    app = FastAPI(title="Log anomaly detection API", version="1.0",
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

    @app.get("/health")
    def health():
        return {"status": "ok", "loaded": [f"{d}/{m}" for d, m in engines],
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
            finished = live[key].ingest(body.lines)
            return {"finished": finished, "status": live[key].status()}

    @app.post("/flush", response_model=IngestOut)
    def flush(body: StreamIn):
        key, _, lock = get(body.dataset, body.detector)
        with lock:
            finished = live[key].flush()
            return {"finished": finished, "status": live[key].status()}

    @app.post("/reset")
    def reset(body: StreamIn):
        key, engine, lock = get(body.dataset, body.detector)
        with lock:
            live[key] = LiveScorer(engine)
            return {"status": live[key].status()}

    return app


app = create_app()
