# Milestone 3, step 2 – Scoring API

**Goal of this step:** put the scoring engine (step 1) behind a **web address**, so that other
programs, such as the live log feed (step 3) and the dashboard (step 4), can send it log lines and
get alerts back. It also decides **when a group is finished** in a live stream. That turned out to
be the most interesting question of this step.

Code:
- [`src/api.py`](../src/api.py): the API.
- [`src/stream.py`](../src/stream.py): the live scoring.
- [`scripts/09_choose_quiet_time.py`](../scripts/09_choose_quiet_time.py): the data behind the HDFS rule.
- [`scripts/10_check_api.py`](../scripts/10_check_api.py): the proof that the API matches the evaluated results.
- Tests in [`tests/test_api.py`](../tests/test_api.py).

```bash
uvicorn src.api:app                    # start the API, then open http://127.0.0.1:8000/docs
python scripts/09_choose_quiet_time.py # ~1.5 min -> reports/results/quiet_time.csv
python scripts/10_check_api.py         # ~2.5 min -> reports/results/api_check.json
```

---

## 1. What is an API?

An **API** (Application Programming Interface) is a **service window** for programs:

- A **client** (here: the log feed or the dashboard) sends a **request** to an address such as
  `http://127.0.0.1:8000/score`.
- The **server** (our API) does the work and sends back a **response**.
- Both are written in **JSON**, a simple text format for data:
  `{"dataset": "bgl", "lines": ["..."]}`.
- Each address is an **endpoint**. `GET` endpoints only read information; `POST` endpoints send
  data to be processed.

**Why FastAPI** (named in Milestone 1):
- It's Python, so it uses the scoring engine directly.
- It **checks every request automatically** (a wrong dataset name or a missing field is rejected
  with a clear error).
- It creates an **interactive test page** at `/docs`, where every endpoint can be tried in the
  browser without writing code. That's useful for the demo video.

---

## 2. The endpoints

| Endpoint | Mode | What it does |
|---|---|---|
| `GET /health` | – | Is the service running? Which models are loaded? Live stream status |
| `GET /detectors` | – | Available detectors per dataset, the default, and each one's test F1 on the time split |
| `POST /score` | **batch** | The lines sent are complete (e.g. a whole log file). Scores all their groups now |
| `POST /ingest` | **live** | Lines arrive a few at a time. The API keeps unfinished groups in memory and returns those that **just finished**. It can also carry `now`, the current time (added in Milestone 3 step 3) |
| `POST /flush` | live | End of the stream: scores every group that is still open |
| `POST /reset` | live | Starts a new stream (e.g. to restart a demo), optionally saying from when it watches (added in Milestone 3 step 3) |

**A real request and response** (one BGL line with level FATAL):

```bash
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" \
  -d '{"dataset": "bgl", "lines": ["- 1131054600 2005.11.03 R02-M1-N0-C:J12-U11 2005-11-03-13.50.00.000000 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL data storage interrupt"]}'
```

```json
{
  "results": [{
    "dataset": "bgl", "group_id": "3770182", "detector": "rule_fatal",
    "first_seen": "2005-11-03 21:50:00", "last_seen": "2005-11-03 21:50:00",
    "n_events": 1, "n_unseen_events": 0,
    "score": 1.0, "is_anomaly": true, "raw_score": 1.0, "threshold": 1.0,
    "reasons": ["1 line(s) with level FATAL/FAILURE", "e.g. E159 \"data storage interrupt\""],
    "unseen_examples": []
  }],
  "stats": {"lines": 1, "malformed": 0, "without_group": 0, "unseen_messages": 0}
}
```

**Batch vs live, in plain words:**
- **Batch** is like handing over a finished exam paper to be marked.
- **Live** is like a teacher reading over a student's shoulder: the paper can only be marked
  once the student has *finished*. So the API must decide when that is.

---

## 3. The key design question: when is a group finished?

The models were trained and evaluated on **complete** groups, so a group must only be scored once it is
complete. Scoring a half-finished group would give the model an input unlike anything it was tested on.

"Time" here means the **timestamps written in the logs** (the latest one seen), not the computer's
clock. So replaying old logs fast or slow gives exactly the same results.

### BGL: a window is finished when the next window starts

A BGL group is a 5-minute window. It is finished as soon as a line from a **later** window arrives,
which is only safe if lines arrive in time order. **Checked:** all 4,747,963 BGL lines are in time
order (0 lines arrive after a later timestamp). So this rule is exact.

### HDFS: a block is finished after 13 hours of silence

An HDFS block has **no "end" message**. The only possible rule is: *finished after Q hours with no
new event*. Choosing Q is a trade-off:

- **Q too short:** a normal block is cut in two. The second piece (often just "delete" events)
  looks abnormal, which means false alarms.
- **Q too long:** alerts arrive late.

The choice was made **from data, using training and validation only**
([`scripts/09_choose_quiet_time.py`](../scripts/09_choose_quiet_time.py)).

**1. Normal blocks have long silences.** The longest silence inside a normal block has a median of
3.25 h, a 99th percentile of 12.2 h, a 99.9th percentile of 12.9 h, and a maximum of 14.9 h. In
**70%** of normal blocks the event after the long silence is **E32 "NameSystem.delete"**. HDFS
writes a block, then deletes it hours later when the job's files are cleaned up.

**2. Simulation:** every training and validation block was cut wherever it was silent for more than Q,
the pieces were scored with the deployed model (time-split Logistic Regression), and the results counted:

| Quiet time Q | Normal blocks cut in two | False alarms | F1 | Median time to alert |
|---|---|---|---|---|
| 1 h | 74.9% | 333,222 | 0.08 | 2.0 h |
| 4 h | 45.7% | 203,618 | 0.13 | 5.5 h |
| 8 h | 12.3% | 55,521 | 0.35 | 9.7 h |
| 12 h | 1.7% | 8,134 | 0.79 | 13.8 h |
| **13 h** | **0.003%** | **716** | **0.98** | **14.8 h** |
| 16 h | 0% | 703 | 0.98 | 17.8 h |

**Chosen: Q = 13 h**, the smallest value that (almost) never cuts a normal block in two. It is
set in [`src/config.py`](../src/config.py) as `HDFS_QUIET_HOURS`, and the script warns if the
config and the data ever disagree.

**A trap avoided:** using the **validation period alone**, Q = 4 h looked perfect (0 blocks cut,
F1 0.999). That's because the validation period happens to have short silences. Over training +
validation, 4 h cuts 46% of normal blocks. One short period can give a badly wrong answer, so the
choice used all non-test data.

**What this means (a finding for the report):** with block-level labels, HDFS cannot be monitored
in real time. A block can only be judged reliably about **13 hours after its last event**. This
is a property of how HDFS works, not of the model. Possible future work:
- use DeepLog to give an *early warning* the moment a surprising event appears;
- train a model on partial blocks.

---

## 4. Proof: the live API gives the evaluated results

All raw lines from the start of the time-split test period were sent to `POST /ingest` in batches of
10,000 lines (as a live feed would), followed by `POST /flush`. The real models were used, with the
default detectors. Every test group's live alert was then compared with the offline evaluation (Milestone 2)
(mode A of `scripts/10_check_api.py`; mode B, added in the next step, is in [docs/06](06_live_feed_and_store.md)):

| | HDFS (Logistic Regression) | BGL (FATAL rule) |
|---|---|---|
| Raw lines replayed | 2,163,461 (217 requests) | 628,954 (63 requests) |
| Speed through the API | ~106,000 lines/s | ~133,000 lines/s |
| Groups finished **during** the stream | **0** | **2,898 of 2,899** |
| Test groups scored | 115,012 of 115,012 | 2,899 of 2,899 |
| Test groups cut in two | 0 | 0 |
| **Alert differences vs the offline evaluation** | **0** | **0** |

- **BGL is live:** every window was scored as soon as a line from a later window arrived. Only the
  very last window had to wait for the flush. (The next step found that in quiet periods this "next line"
  can take hours, and added a clock signal; see [docs/06](06_live_feed_and_store.md) §4.)
- **HDFS finished nothing during the stream.** That's expected: the test period lasts only
  **3.6 hours**, which is less than the 13-hour quiet time. All its alerts arrived at the flush.
- **The chain is now proven end to end:** preprocessing → evaluated models → engine (step 1) →
  API (this step), with zero differences.

Numbers: [`reports/results/api_check.json`](../reports/results/api_check.json).

---

## 5. A problem found: the "cold start"

The HDFS replay started in the middle of the log, so the stream also judged **65,425 blocks that had
begun before it started**. It received only their late events (mostly the delete step), and
**65,041 of them (99.4%) were flagged**. These are false alarms, because a "delete-only" block looks
broken.

Any monitor switched on in the middle of a running system has this problem. **Fixed in the next step**
(a cold-start rule plus a 13-hour back-fill): 0 such blocks are judged any more, see
[docs/06](06_live_feed_and_store.md) §2.

---

## 6. Robustness decisions

| Decision | Why |
|---|---|
| At most 50,000 lines per request | A single huge request cannot exhaust the server's memory |
| Requests are validated (unknown dataset → error 422, unknown detector → error 400) | Clear errors instead of crashes |
| Missing model files → error 503 with "run scripts 02, 04 and 05 first" | Tells the user exactly what to do |
| Models are loaded on first use, then kept in memory | Fast start; each model is loaded only once |
| One engine and one **lock** per (dataset, detector) | Two requests arriving at the same moment cannot mix up the open groups |
| Live time = log timestamps | Replays are reproducible at any speed |

---

## 7. Limitations

- **Open groups live in memory.** If the API restarts, unfinished groups are lost. The next step adds a
  database for the *results*.
- **No login or authentication.** The API is meant to run locally for the demo; a real deployment
  would need access control.
- **HDFS alerts are slow** (section 3), and **cold start** causes false alarms (section 5).
- **One process.** Enough for this data (more than 100,000 lines/s), but not designed for many servers.

---

## 8. Questions a lecturer might ask

**What does the API add to the engine?** A standard way for other programs to use it over the
network, input checking, and the *live* logic: holding unfinished groups and deciding when each
is finished.

**How did you choose 13 hours?** From data, never the test set. Normal HDFS blocks can be silent for
up to ~13 hours before their delete step. A simulation on training and validation data showed
that shorter values cut normal blocks in two and cause hundreds of thousands of false alarms;
13 h is the smallest value that does not.

**Why not choose it on the validation set, as you did for thresholds?** We first did, and it said 4 h.
But the validation period is only a few hours long, so it cannot contain long silences. That is a
lesson about representativeness: a parameter about *time* must be chosen from a long enough period.

**So is your HDFS system real-time?** Not at block level, and the report says so. HDFS blocks can
only be judged when they finish, about 13 h after their last event. BGL is near real-time: with
the clock signal added in the next step, a window is judged about one second after it ends.

**Why use log timestamps instead of the clock?** So results are reproducible: replaying a day of
logs in a minute gives the same alerts as living through that day.

**How do you know the live API gives the same answers as your evaluation?** We replayed all
2.8 million test-period lines through it. There were zero differences for every test group.

**What is the cold-start problem?** A monitor switched on mid-stream sees groups without their
beginning. For HDFS these half-blocks look abnormal, and 99.4% were flagged. The fix belongs to
the replay design, and the next step fixed it.
