"""
Alert store (Milestone 3, step 3): every group the live system judges is saved in a
database, so that the dashboard (step 4) can show alerts, history and drift, and nothing
is lost when the API restarts.

    deployment: PostgreSQL, a hosted database named by DATABASE_URL in the .env file
    tests     : SQLite (a file, no server needed)

The same code serves both, through SQLAlchemy (a Python library that writes the SQL for
whichever database is used). All times are UTC.

The connection string contains a password, so it is never written in the code: it is
read from the environment variable DATABASE_URL or from the project's .env file, which
git ignores (.env.example shows the format). The connection is only opened when the
store is first used, so importing the API (e.g. in the tests) never touches the network.

Tables
    runs           one row per live stream (one replay / monitoring session) and its latest status
    group_results  one row per JUDGED group: score 0-1, label, reasons, times
"""
import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, Index, Integer, MetaData, String,
                        Table, create_engine, insert, select, update)

from src.config import PROJECT_ROOT

ENV_FILE = PROJECT_ROOT / ".env"

metadata = MetaData()

runs = Table(
    "runs", metadata,
    Column("run_id", String(64), primary_key=True),
    Column("dataset", String(8), nullable=False),
    Column("detector", String(16), nullable=False),
    Column("started_at", DateTime, nullable=False),   # wall clock, when the stream started
    Column("updated_at", DateTime, nullable=False),   # wall clock, last request
    Column("status", JSON, nullable=False),           # the live scorer's latest status
)

group_results = Table(
    "group_results", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False),
    Column("dataset", String(8), nullable=False),
    Column("detector", String(16), nullable=False),
    Column("group_id", String(64), nullable=False),     # HDFS block ID or BGL window number
    Column("first_seen", DateTime, nullable=False),     # log time of the group's first event
    Column("last_seen", DateTime, nullable=False),      # log time of its last event
    Column("judged_at", DateTime, nullable=False),      # log time when it was finished and scored
    Column("n_events", Integer, nullable=False),
    Column("n_unseen_events", Integer, nullable=False),
    Column("score", Float, nullable=False),             # 0-1, alert line 0.5
    Column("is_anomaly", Boolean, nullable=False),
    Column("raw_score", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("reasons", JSON, nullable=False),
    Column("unseen_examples", JSON, nullable=False),
)
Index("ix_group_results_run_time", group_results.c.run_id, group_results.c.last_seen)
Index("ix_group_results_run_alert", group_results.c.run_id, group_results.c.is_anomaly)

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"  # how the engine writes times


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def normalize_url(url: str) -> str:
    """Accept the forms hosting providers give (postgres://..., postgresql://...) and use the
    psycopg 3 driver. Options such as ?sslmode=require are kept."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def database_url() -> str | None:
    """DATABASE_URL from the environment, or else from the .env file; None if neither has it."""
    load_dotenv(ENV_FILE)  # never overrides a variable that is already set
    url = os.environ.get("DATABASE_URL", "").strip()
    return normalize_url(url) if url else None


class ResultStore:
    def __init__(self, url: str):
        # pool_pre_ping: test a pooled connection before use; pool_recycle: renew connections
        # after 5 min, because hosted databases close idle ones
        self.db = create_engine(normalize_url(url), pool_pre_ping=True, pool_recycle=300)
        self._ready = False

    @classmethod
    def from_env(cls) -> "ResultStore | None":
        """The store named by DATABASE_URL (environment or .env), or None if it is not set."""
        url = database_url()
        return cls(url) if url else None

    def connect(self) -> None:
        """Open the connection and create the tables the first time (later: does nothing)."""
        if not self._ready:
            metadata.create_all(self.db)
            self._ready = True

    def describe(self) -> str:
        """The connection string without the password (for logs on this computer)."""
        return self.db.url.render_as_string(hide_password=True)

    def label(self) -> str:
        """A description safe to publish, without host or user, e.g. 'postgresql 18.7, hosted (Neon)'."""
        self.connect()
        url = self.db.url
        with self.db.connect() as conn:
            version = conn.exec_driver_sql("select version()").scalar().split()[1] if url.get_backend_name() == "postgresql" else ""
        host = url.host or ""
        where = "local" if host in ("", "localhost", "127.0.0.1") else "hosted" + (" (Neon)" if host.endswith(".neon.tech") else "")
        return f"{url.get_backend_name()} {version}".strip() + f", {where}"

    def save(self, status: dict, results: list[dict]) -> None:
        """Store the live results that just finished and the stream's latest status, in one transaction."""
        self.connect()
        now = _now()
        rows = [{**{k: r[k] for k in ("dataset", "detector", "group_id", "n_events", "n_unseen_events", "score",
                                      "is_anomaly", "raw_score", "threshold", "reasons", "unseen_examples")},
                 "run_id": status["run_id"],
                 "first_seen": datetime.strptime(r["first_seen"], TIME_FORMAT),
                 "last_seen": datetime.strptime(r["last_seen"], TIME_FORMAT),
                 "judged_at": datetime.strptime(r["finished_at"], TIME_FORMAT)}
                for r in results]
        with self.db.begin() as conn:
            changed = conn.execute(update(runs).where(runs.c.run_id == status["run_id"])
                                   .values(updated_at=now, status=status)).rowcount
            if not changed:
                conn.execute(insert(runs).values(run_id=status["run_id"], dataset=status["dataset"],
                                                 detector=status["detector"], started_at=now,
                                                 updated_at=now, status=status))
            if rows:
                conn.execute(insert(group_results), rows)

    def runs(self, dataset: str | None = None) -> list[dict]:
        """All streams, newest first."""
        self.connect()
        query = select(runs).order_by(runs.c.started_at.desc(), runs.c.run_id.desc())
        if dataset:
            query = query.where(runs.c.dataset == dataset)
        with self.db.connect() as conn:
            return [dict(row._mapping) for row in conn.execute(query)]

    def frame(self, run_id: str) -> pd.DataFrame:
        """The small columns of every judged group of one stream (no reasons), for charts."""
        self.connect()
        c = group_results.c
        query = select(c.group_id, c.first_seen, c.last_seen, c.judged_at, c.n_events, c.n_unseen_events,
                       c.score, c.is_anomaly).where(c.run_id == run_id)
        with self.db.connect() as conn:
            return pd.read_sql(query, conn)

    def results(self, run_id: str, only_alerts: bool = False, limit: int | None = None) -> list[dict]:
        """The judged groups of one stream, most recently finished first."""
        self.connect()
        query = (select(group_results).where(group_results.c.run_id == run_id)
                 .order_by(group_results.c.judged_at.desc(), group_results.c.id.desc()))
        if only_alerts:
            query = query.where(group_results.c.is_anomaly)
        if limit:
            query = query.limit(limit)
        with self.db.connect() as conn:
            return [dict(row._mapping) for row in conn.execute(query)]
