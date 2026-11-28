# Milestone 3, step 3 – Live log feed and alert store

**Goal of this step:** make the system run like a real monitor.

- A **log feed** sends log lines to the API as if they were being written right now.
- The API judges each group when it is finished.
- **Every result is saved in a PostgreSQL database**, hosted on Neon, where the dashboard (step 4) will read it.

Running it live exposed **three problems**, which were measured and fixed:

1. half-seen groups at start-up (§2);
2. results that depended on the batch size (§3);
3. alerts delayed by silences in the logs (§4).

Code:
- [`src/replay.py`](../src/replay.py): the log feed.
- [`src/store.py`](../src/store.py): the alert store.
- [`src/stream.py`](../src/stream.py): live scoring, now with the cold-start rule and the clock signal.
- [`.env.example`](../.env.example): where the database connection string goes (in `.env`, never committed).
- [`docker-compose.yml`](../docker-compose.yml): an optional local database.
- Checks: [`scripts/10_check_api.py`](../scripts/10_check_api.py) and [`scripts/11_check_replay.py`](../scripts/11_check_replay.py).
- Tests: [`tests/test_replay.py`](../tests/test_replay.py), [`tests/test_store.py`](../tests/test_store.py) and [`tests/test_api.py`](../tests/test_api.py).

```bash
cp .env.example .env                                 # once: put the database connection string in .env
uvicorn src.api:app                                  # the API (reads .env)
python -m src.replay --dataset bgl                   # second terminal: replay the BGL test period
python scripts/10_check_api.py                       # ~2.5 min -> reports/results/api_check.json
python scripts/11_check_replay.py                    # ~2.5 min -> reports/results/replay_check.json
```

---

## 1. The pieces

```
raw log file ──> log feed ──> scoring API ──> PostgreSQL (Neon) ──> dashboard (step 4)
               (replay.py)   (api.py, stream.py)     (store.py)
```

| Piece | What it does |
|---|---|
| **Log feed** (a "collector") | Reads the raw log file from a chosen moment and sends the lines to `POST /ingest`, about one request per second, at a chosen speed |
| **Scoring API** | Keeps the unfinished groups in memory, decides when each is finished, judges it with the trained model |
| **Alert store** | Saves every judged group (score 0–1, label, reasons, times) and the stream's status |

**A replay in plain words:** it's like watching a recording of a football match. The events come
in the same order and with the same gaps (sped up if you like), and the commentator (the API)
reacts as if it were live. The feed starts by default at the **beginning of the time-split test
period**, the "future" that no model was trained or tuned on.

**Speed** is in log seconds per real second:
- `--speed 3600` (the default) plays one log hour per second, so the 56-day BGL test period takes at most about 23 minutes (less, because long silences are fast-forwarded);
- `--speed 1` is real time;
- `--speed 0` is as fast as possible.

**Finding the start:** both log files are in perfect time order (checked: 0 of 15.9M lines out
of order). So the feed jumps straight to the start time by **binary search**, like looking up a
word in a dictionary: about 2 ms instead of reading 9 million lines.

---

## 2. Problem 1: the cold start (fixed)

**The problem (found in step 2):** a monitor switched on in the middle of a running system sees
groups **without their beginning**. An HDFS block whose first events came before the switch-on
arrives as a "delete-only" piece, and that looks broken.

**The fix has two parts:**

**a) The cold-start rule (in the API).** A group is only *judged* if we can be sure that none of
its events came before monitoring began (`watch_from`):
- **BGL:** the window started at or after `watch_from`. A window has no events before its own start.
- **HDFS:** the block's first event came at least **13 hours** after `watch_from`. Why is that
  enough? Any earlier event of the same block would have been within 13 h before it, so after
  `watch_from`, and we would have seen it. An event more than 13 h earlier would have ended the
  block anyway: by the live rule of step 2, that is a separate, finished piece.

Groups that fail the rule are still **followed**, so their late events cannot start a fake "new" group. When
they finish, they are **counted** (`not_judged_started_before_watching`), not scored.

**b) Back-fill (in the log feed).** On start-up, the feed first reads the **last 13 hours** of HDFS
history as fast as possible, then goes live. Real collectors do the same with the log files
already on disk. So every block that starts after the go-live moment is judged. For BGL no
back-fill is needed, because the test period starts exactly at a window boundary.

**Evidence** (HDFS, [`scripts/10_check_api.py`](../scripts/10_check_api.py)):

| | A: switched on at the test start, no rule | B: 13 h back-fill + cold-start rule |
|---|---|---|
| Raw lines sent | 2,163,461 | 7,022,481 (+4.86M back-fill, ~40 s) |
| Test blocks judged | 115,012 of 115,012 | 115,012 of 115,012 |
| Alert differences vs the offline evaluation | 0 | 0 |
| **Judged blocks with events missed before watching** | **65,425 (65,041 flagged = false alarms)** | **0** |
| Not judged (may have started before watching) | 0 | 348,189 (training/validation blocks of the back-fill) |
| Judged blocks cut by the 13 h quiet time | 0 | 10 (all flagged) |

**The 10 cut blocks are not a cold-start failure.** They are normal training blocks written at
about 20:40, whose delete step came **more than 13 hours** later. The monitor saw both parts, but the
13-hour rule split them, and the one-event piece looks abnormal. A monitor running from the very
first log line would cut them in exactly the same way. This is the known price of the quiet time:
[`scripts/09`](../scripts/09_choose_quiet_time.py) predicted that 0.003% of normal blocks would
be cut (docs/05 §3).

---

## 3. Problem 2: results depended on the batch size (fixed)

In step 2, the "is a group finished?" check ran **once per request**. Take an HDFS block that is
silent for more than 13 h and then gets one more event. If that event arrived in the **same request** as
the line that ended the 13 hours, it rejoined the block. If it arrived in a later request, it
started a new piece. So the result could depend on how the lines were cut into requests. One
block (silent for 13 h 0 min 1 s) really did behave like this: 9 cut pieces became 10.

**Fix:** the check now runs **after every line**. It only looks at the oldest open group, so it
costs almost nothing. Results are now the same for any batch size. This is tested in
`test_results_do_not_depend_on_how_lines_are_batched`.

Each live result now also records **`finished_at`**: the log time at which its group was finished
and judged.

---

## 4. Problem 3: a silence stops the clock (fixed with a clock signal)

The API's clock is the **log timestamps**, so it only moves forward when a line arrives. A BGL window
was therefore only closed when the **next line** arrived. Measured on the whole BGL test period
(fast replay without a clock signal, [`scripts/11_check_replay.py`](../scripts/11_check_replay.py)):

| Delay from the end of a window to its alert | Median | 90% | Worst |
|---|---|---|---|
| The 343 alert windows | 7.3 min | 81 min | 7.1 h |
| All 2,899 windows | 5.9 min | 64 min | 17.8 h |

That's a real weakness for a monitor. **A crashed machine goes silent**, so its alert would wait
for a line that may never come.

**Fix: a clock signal (a "watermark").** `POST /ingest` accepts `now`, the current time, with or
without lines. Groups whose time is up are then finished even if no line arrives. This is a
standard idea in stream processing. The log feed sends `now` with its requests and, during a
silence, an empty request with `now` **every second**.

One detail: `now` never goes past the time of the next line. Otherwise a window could be closed
just before one of its own lines arrived.

**Evidence:** BGL replayed at 600× for 6 log hours with the clock signal (32 requests):

| | Median | 90% | Worst |
|---|---|---|---|
| Delay from the end of a window to its alert | 5.7 log-min | 8.9 log-min | 9.3 log-min = **0.94 real seconds** |

So a window is judged **at most about one real second after it ends**, whatever the speed.
- At real speed (`--speed 1`), that's about 1 second of log time.
- At the demo speed of 3600×, one second is a whole log hour. So `judged_at` can be up to an hour
  after the window's end *in log time*, while it is still under a second in real time.
- The demo option `--max-wait` (default 2 s) fast-forwards long silences. During a fast-forward the
  clock jumps, so the delays measured in log time look longer. The measurement above therefore
  switched fast-forwarding off.

---

## 5. The alert store

**Why a database** and not, say, a CSV file:
- the dashboard can ask questions ("the last 50 alerts", "alerts per hour");
- several programs can read and write at the same time;
- each save is all-or-nothing;
- the results survive a restart of the API.

**Why PostgreSQL:** it was named in Milestone 1. It's a standard, free, production-grade
database.

**Why a hosted database (Neon):** Neon runs PostgreSQL in the cloud (here PostgreSQL 18.7, in a
Singapore data centre). So the database is not tied to this laptop:
- the dashboard or a second computer can reach the same alerts;
- the data survives if the laptop is wiped;
- nothing has to be installed.

The connection is encrypted (`sslmode=require`), and a round trip takes about 14 ms from here.
For working offline, the same code also runs on a local PostgreSQL in Docker
(`docker-compose up -d --wait db`, see `.env.example`).

**Why SQLAlchemy:** it's a Python library that writes the SQL for whichever database is used. The
same code runs on **SQLite** in the tests (a file, no server needed) and on **PostgreSQL** in the
deployment.

**Tables** (all times UTC):

| Table | One row per | Main columns |
|---|---|---|
| `runs` | live stream (one replay or monitoring session) | `run_id`, `dataset`, `detector`, `started_at`, `updated_at`, `status` (the latest live status as JSON: log time, lines, judged groups, alerts, unseen messages, ...) |
| `group_results` | judged group | `run_id`, `group_id`, `first_seen`, `last_seen`, `judged_at`, `n_events`, `n_unseen_events`, `score` (0–1), `is_anomaly`, `raw_score`, `threshold`, `reasons` and `unseen_examples` (JSON lists) |

**Design decisions:**

| Decision | Why |
|---|---|
| Only **live** results are stored; `/score` (batch) results are just returned | The database is the history of the monitor |
| The results and the status of one request are saved in **one transaction** | A request is never half-saved |
| If the database fails, the results are still returned, with `"stored": false` and an error in the log | Detection never stops because of the database (tested) |
| The connection string (it contains the password) lives in **`.env`**, which git ignores; the repository only has `.env.example` | Secrets never reach GitHub. Published results name the database only as "postgresql 18.7, hosted (Neon)", without host or user |
| Strings in the form providers give (`postgres://...`, `postgresql://...?sslmode=require`) are accepted | Paste and go; the code switches them to the psycopg 3 driver |
| The connection is opened at first use, not when the API is imported; checked and logged when the server starts | The tests never touch the network; a wrong `.env` shows up at once in the server log |
| Connections are tested before use and renewed every 5 minutes | Hosted databases close idle connections |
| Without `DATABASE_URL`, nothing is stored | The API still runs without a database (tests, `scripts/10`) |

**Look inside it yourself:** in the Neon console's SQL editor (database `capstone`), or from
this computer:

```bash
psql "$(grep ^DATABASE_URL= .env | cut -d= -f2-)" -c \
  "select group_id, first_seen, last_seen, judged_at, score, reasons->>0 as reason
   from group_results where is_anomaly order by judged_at desc limit 5;"
```

The test periods of both datasets (118,000 rows) take about 30 MB.

---

## 6. Proof: log feed → API server → PostgreSQL = the evaluated results

[`scripts/11_check_replay.py`](../scripts/11_check_replay.py) runs the **real pieces**:
- a real uvicorn server, writing to the hosted PostgreSQL on Neon;
- the log feed replaying the test periods as fast as possible.

It then reads the results back **from the database** and compares them with the offline evaluation:

| | BGL (FATAL rule) | HDFS (Logistic Regression) |
|---|---|---|
| Raw lines sent | 628,954 (63 requests) | 7,022,481 (703 requests) |
| Time, end to end (with the database on Neon) | 9 s (71,000 lines/s) | 93 s (75,000 lines/s) |
| Rows in the database (= API responses) | 2,899 | 115,028 |
| Test groups in the database | 2,899 of 2,899 | 115,012 of 115,012 |
| **Alert differences vs the offline evaluation** | **0** | **0** |
| Largest score difference | 0 | 2.4 × 10⁻⁷ (see below) |
| Other groups in the database | 0 | 16 = 10 quiet-time pieces (§2) + 6 complete validation blocks that started in the same second as the test period (5 normal, 1 anomaly: all correctly judged) |
| Alerts | 343 | 1,758 = 1,747 on test blocks (as offline) + 10 pieces + 1 validation anomaly |

Writing to the cloud costs some speed: with a local PostgreSQL the same check took 5 s and 61 s.
Each request now also makes a round trip over the internet. That's still far more than the logs
need (HDFS writes about 170 lines per second in its test period).

**Why the HDFS scores differ by 2.4 × 10⁻⁷:** the Logistic Regression computes in float32. Scoring the same
groups in a batch of a different size or order adds the numbers up in a slightly different order.
Tested directly: reversing the order of the test set changes scores by up to 1.3 × 10⁻⁷, and not a
single decision changes.

---

## 7. Running the demo

```bash
uvicorn src.api:app                                  # reads DATABASE_URL from .env
python -m src.replay --dataset bgl --hours 12        # in a second terminal
```

Real output (12 log hours in 13 seconds):

```
BGL replay, run bgl-20261128-024410-7e2d: watching from 2005-11-09 04:40:00, live from 2005-11-09 04:40:00, speed 3600x
  ALERT 3771752 score 1.00 (2005-11-09 08:40:15): 54 line(s) with level FATAL/FAILURE | e.g. E159 "data storage interrupt"
  ... log time 2005-11-09 10:25:58 | 96 lines | judged 11 | alerts 1
  ... log time 2005-11-09 16:21:39 | 117 lines | judged 16 | alerts 1
done: 149 lines in 15 requests | judged 17 groups, 1 alerts | not judged (started before watching): 0 | stored in database: True
```

Other options: `--speed`, `--detector`, `--start` (unix seconds), `--hours`, `--max-wait`. For HDFS
(`--dataset hdfs`), the 13-hour back-fill runs first (about 40 s). Because of the quiet time,
the alerts only arrive at the end (`/flush`).

---

## 8. Limitations

- **The back-fill needs history.** At start-up, the last 13 h of HDFS logs must be readable. After a
  restart of the API, the open groups are lost and the cold-start rule applies again. That's
  correct, but HDFS blocks active at the restart are not judged.
- **HDFS is still slow:** about 13 h from a block's last event to its alert (docs/05 §3).
- **The feed is a replay** of a file. A real deployment would tail live log files or receive
  syslog messages, but the API side would stay the same.
- **Security:** the database password stays in `.env` (never committed), and the connection is
  encrypted. But the API itself has no login, which is fine on one computer for a demo, not for a
  real deployment. The optional local database uses a demo password (in `docker-compose.yml`).
- **The free Neon plan has limits** (storage, compute hours). A full check run adds about 30 MB.
- **One API process** (more than 100,000 lines/s, enough for this data).

---

## 9. Questions a lecturer might ask

**What is the cold-start problem, and how did you fix it?** A monitor switched on mid-stream
sees groups without their beginning. 65,425 such HDFS blocks were judged and 65,041 flagged, all false
alarms. Fix: never judge a group that may have started before monitoring began, and back-fill
13 h of history so that every block after the go-live moment is judged. Result: 0 half-seen blocks
judged, and all 115,012 test blocks still judged exactly as offline.

**Don't you miss anomalies by not judging those groups?** Only groups that began before the
monitor existed, and the monitor says how many (`not_judged_started_before_watching`). The
choice is between not judging what it hasn't fully seen, and 65,000 false alarms.

**Why 13 hours of back-fill?** It's the same number as the quiet time. An earlier event of a
block more than 13 h back would have made it a separate, already finished piece. So 13 h of history is
exactly enough.

**What is a watermark, and why do you need one?** In a live system the clock is the log
timestamps, and with no new lines the clock stops. A watermark is a message saying "it is now time T".
Without it, BGL alerts waited a median of 7 minutes, and up to 7 hours, for the next line. With it,
they arrive within about one second of the window's end.

**Why can the watermark never pass the next line's time?** Then a window could be closed while one
of its own lines was still on its way. The line would then start a fake second window.

**Why a database, and why PostgreSQL?** The dashboard needs to query the history, several programs
use it at the same time, saves must be all-or-nothing, and results must survive restarts.
PostgreSQL was named in Milestone 1. SQLAlchemy lets the tests use SQLite with the same code.

**Why hosted, and how do you protect the password?** A hosted database (Neon) can be reached by the
dashboard from anywhere and doesn't depend on this laptop. The connection string lives only in
`.env`, which git ignores, and the connection is encrypted.

**What happens if the database is down?** Detection continues: results are returned with
`"stored": false` and the error is logged (tested in `test_detection_continues_when_the_database_fails`).

**How do you know the dashboard will show what you evaluated?** `scripts/11` replays both test
periods through the real server into PostgreSQL, reads them back from the database and finds
0 alert differences against the offline evaluation.

**Why are the HDFS scores not exactly identical?** Float32 arithmetic. The order in which numbers
are added depends on the batch, which changes scores by about 10⁻⁷ and never a decision.
