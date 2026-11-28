"""Tests for the alert store (src/store.py), on SQLite (same code as PostgreSQL, no server needed)."""
from src.store import ResultStore


def result(gid, anomaly, last_seen="2005-11-09 04:44:59"):
    return {"dataset": "bgl", "detector": "rule_fatal", "group_id": gid, "first_seen": "2005-11-09 04:40:00",
            "last_seen": last_seen, "n_events": 3, "n_unseen_events": 1, "score": 1.0 if anomaly else 0.0,
            "is_anomaly": anomaly, "raw_score": float(anomaly), "threshold": 1.0,
            "reasons": ["1 line(s) with level FATAL/FAILURE"] if anomaly else [],
            "unseen_examples": ["some new message"], "finished_at": "2005-11-09 04:45:00"}


def status(run_id, judged):
    return {"run_id": run_id, "dataset": "bgl", "detector": "rule_fatal", "judged_groups": judged}


def test_save_and_read_back(tmp_path):
    store = ResultStore(f"sqlite:///{tmp_path / 'a.db'}")
    store.save(status("run-1", 0), [])                                 # status only: creates the run
    store.save(status("run-1", 2), [result("1", True), result("2", False)])
    store.save(status("run-2", 1), [result("9", True)])

    assert [r["run_id"] for r in store.runs()] == ["run-2", "run-1"]   # newest first
    assert store.runs("bgl")[1]["status"]["judged_groups"] == 2       # the run row was updated, not duplicated
    rows = store.results("run-1")
    assert {r["group_id"] for r in rows} == {"1", "2"}
    (alert,) = store.results("run-1", only_alerts=True)
    assert alert["group_id"] == "1" and alert["reasons"] == ["1 line(s) with level FATAL/FAILURE"]
    assert alert["unseen_examples"] == ["some new message"]
    assert str(alert["judged_at"]) == "2005-11-09 04:45:00" and str(alert["last_seen"]) == "2005-11-09 04:44:59"
    assert len(store.results("run-1", limit=1)) == 1


def test_tables_survive_a_restart(tmp_path):
    url = f"sqlite:///{tmp_path / 'a.db'}"
    ResultStore(url).save(status("run-1", 1), [result("1", True)])
    assert len(ResultStore(url).results("run-1")) == 1                 # a new process sees the old results


def test_connection_strings_from_hosting_providers_use_psycopg():
    from src.store import normalize_url
    assert normalize_url("postgres://u:p@host:5432/db") == "postgresql+psycopg://u:p@host:5432/db"
    assert (normalize_url("postgresql://u:p@host/db?sslmode=require")
            == "postgresql+psycopg://u:p@host/db?sslmode=require")
    assert normalize_url("postgresql+psycopg://u:p@host/db") == "postgresql+psycopg://u:p@host/db"
    assert normalize_url("sqlite:///a.db") == "sqlite:///a.db"


def test_creating_a_store_does_not_connect():
    store = ResultStore("postgresql://user:secret@db.example.invalid:5432/logs")  # unreachable on purpose
    assert store.describe() == "postgresql+psycopg://user:***@db.example.invalid:5432/logs"  # password hidden


def test_label_is_safe_to_publish(tmp_path):
    assert ResultStore(f"sqlite:///{tmp_path / 'a.db'}").label() == "sqlite, local"
