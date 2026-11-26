# Milestone 3, step 1 – Scoring engine

**Goal of this step:** build the part of the live system that does the actual data science.
It takes **raw log lines**, and for every group (HDFS block / BGL 5-minute window) returns:

- an **anomaly score from 0 to 1**,
- a **normal / abnormal label**,
- the **log events that caused the alert**.

These are the outputs promised in Milestone 1 ("Expected outputs: anomaly alerts"). The API and
dashboard built in the next steps only pass lines into this engine and display what comes out.

Code:
- [`src/engine.py`](../src/engine.py): the engine.
- [`scripts/08_check_engine.py`](../scripts/08_check_engine.py): proves it matches the evaluated results.
- [`tests/test_engine.py`](../tests/test_engine.py): unit tests.

```bash
python scripts/08_check_engine.py   # ~35 s -> reports/results/engine_check.json
```

---

## 1. What goes in, what comes out

**In:** raw lines, exactly as the system wrote them:

```
081111 074238 19 INFO dfs.FSNamesystem: BLOCK* ask 10.251.203.149:50010 to replicate blk_610605540713393724 to datanode(s) 10.251.193.224:50010
```

**Out:** one result per group. This is a real abnormal HDFS block from the test period (all 26 of its
lines given to the engine), scored by Logistic Regression:

```json
{
  "group_id": "blk_610605540713393724",
  "detector": "logreg",
  "first_seen": "2008-11-11 07:42:35",
  "last_seen": "2008-11-11 09:01:56",
  "n_events": 26,
  "n_unseen_events": 0,
  "score": 0.9978,
  "is_anomaly": true,
  "reasons": [
    "more E12 \"<IP> Starting thread to transfer block <BLK> to <IP>\" than a normal group (1 vs about 0.0051), +4.92",
    "more E11 \"BLOCK* ask <IP> to replicate <BLK> to datanode(s) <IP>\" than a normal group (1 vs about 0.0051), +4.92",
    "more E6 \"Received block <BLK> src: /<IP> dest: /<IP> of size <NUM>\" than a normal group (1 vs about 0.0054), +4.29"
  ]
}
```

In plain words: this block had to be **re-replicated** (copied again to another machine), which
almost never happens to a normal block. A replication problem is a plausible reading for an operator.

---

## 2. The five stages, and where each comes from

| Stage | What happens | Reused from |
|---|---|---|
| 1. Read | split the line into time, level and message; find the group (block ID / 5-minute window) | `src/hdfs.py`, `src/bgl.py` (step 1) |
| 2. Parse | mask IDs/IPs/numbers, then match the message to a template with the **frozen** Drain parser (never learns); unknown → event 0 | `src/log_parser.py` + saved `parser.pkl` (step 1) |
| 3. Group | append the event to its group, in arrival order | same rules as preprocessing |
| 4. Detect | the chosen detector gives a raw score; alert if score ≥ its threshold | models + thresholds saved in step 3 (**time split**) |
| 5. Report | score 0–1, label, reasons, examples of never-seen messages | new in this step |

**Nothing is re-implemented.** The engine imports the same functions and loads the same saved
files that produced the evaluated results.

The BGL label column (the first field of each line) is read but **never used**. Real systems have
no labels, and the models never saw them as input.

The live system uses the **time-split** models, which were trained on the past and evaluated on
the future. That is the situation the live system is in.

---

## 3. Proof that online = offline (no "training–serving skew")

**Training–serving skew** means the live system prepares data slightly differently from training,
for example with a different parser version or a missing masking step. The model then quietly gets
worse, and the evaluation results no longer describe the deployed system.

**The check:** all raw lines of the time-split **test period** were fed through the engine. For every
test group, the engine's output was compared with the step 1 sequences and the step 3 predictions:

| | HDFS | BGL |
|---|---|---|
| Raw lines processed | 2,163,435 | 628,954 |
| Test groups | 115,012 | 2,899 |
| Groups with a different event sequence | **0** | **0** |
| Detectors checked | Logistic Regression, Isolation Forest, DeepLog | FATAL rule, Logistic Regression, Isolation Forest, DeepLog |
| Groups with a different alert decision | **0** | **0** |
| Largest score difference | **0.0** | **0.0** |

So the scoring engine reproduces the evaluated system exactly. The numbers are in
[`reports/results/engine_check.json`](../reports/results/engine_check.json).

**Two small changes were needed in earlier steps:**

| Change | Proof that nothing else changed |
|---|---|
| Preprocessing now also saves the frozen parser (`parser.pkl`); before, it was lost when the script ended | Re-ran preprocessing: all data files are **byte-identical** (MD5 checksums) |
| The saved baseline models now also store the "average normal group", used for explanations (section 5) | Re-ran the baselines: all metrics **identical** |

---

## 4. One score scale: 0 to 1, alert line at 0.5

Each detector has its own raw score and threshold:

| Detector | Raw score | Threshold |
|---|---|---|
| Logistic Regression | a probability | e.g. 0.091 (HDFS) |
| DeepLog | a worst rank | e.g. g = 6 of 22 tokens |
| FATAL rule | 0 or 1 | 1 |

A raw "0.09" means nothing to an operator. The engine therefore maps every raw score onto **one
scale where the alert line is always 0.5**:

```
below the threshold:  0 ............ threshold  ->  0.0 ... 0.5   (straight line)
above the threshold:  threshold .... maximum    ->  0.5 ... 1.0   (straight line)
```

| Example (HDFS, time split) | Raw | Score 0–1 |
|---|---|---|
| Logistic Regression, threshold 0.091 | 0.045 | 0.25 |
| | 0.091 | **0.50** (alert line) |
| | 1.0 | 1.00 |
| DeepLog, g = 6, worst possible rank 22 | rank 13 | 0.72 |

- The **order of groups never changes**, so a higher score always means more suspicious.
- **"Score ≥ 0.5" gives exactly the same alerts** as the original thresholds. The check
  confirmed this for all 117,911 test groups.
- **It is not a probability.** It is a rescaled score: 0.75 means "halfway between the alert
  line and the most extreme value", not "75% chance of failure".

---

## 5. Explanations: why was this flagged?

| Detector | How the reason is found | Example (real test data) |
|---|---|---|
| **Logistic Regression** | The score is a weighted sum of event counts. Compared with the **average normal training group**, it splits exactly into one part per event: *weight × (this group's value − normal value)*. The largest positive parts are the reasons. They can be "more of X" or "fewer of Y" (a missing event) | HDFS block above: "more E12 *Starting thread to transfer block* than a normal group (1 vs about 0.0051)" |
| **DeepLog** | The steps where the real next event was not in the model's top-g guesses, with what it expected instead | same block: "step 4: got E11 *ask … to replicate*, expected one of: E3 *PacketResponder…*; E5 *addStoredBlock…*; E4 *Received block…*" |
| **FATAL rule** | the FATAL/FAILURE lines in the window | BGL window 3771752: "54 line(s) with level FATAL/FAILURE", e.g. E159 *data storage interrupt* |
| **Isolation Forest** | no per-event reason (it isolates the group as a whole) | – (a known limitation) |

**Why compare with the average normal group?** Without a reference point, the first version said
"E32 *delete* ×3 raised the score". But normal blocks also have E32 ×3, so that was misleading.
Comparing with a normal group answers the operator's real question: *what is different here?*

Results also include up to three **never-seen messages** in their raw form, because a template ID
means nothing for a message the parser has never seen.

---

## 6. Which detector is used

The default is the **best detector on the time split** (docs/03 §9). Any other can be chosen by name.

| Dataset | Default | Test F1 (time split) | Other choices |
|---|---|---|---|
| HDFS | Logistic Regression | 0.974 | Isolation Forest, DeepLog |
| BGL | FATAL/FAILURE rule | 0.792 | Logistic Regression, Isolation Forest, DeepLog |

The engine treats a rule like any model: it gives a score, has a threshold and explains itself. So
the API can switch between them without special cases.

---

## 7. Is it fast enough?

Measured on one CPU core of the laptop (pure Python):

| | HDFS | BGL |
|---|---|---|
| Reading + parsing + grouping | ~204,000 lines/s | ~161,000 lines/s |
| Scoring, Logistic Regression | ~130,000 groups/s | ~49,000 groups/s |
| Scoring, DeepLog | ~29,000 groups/s | ~365 windows/s (windows have up to 42,272 events) |
| **How fast the real system wrote logs** | ~80 lines/s on average (11.2M lines in 38.7 h) | ~0.26 lines/s on average (4.7M lines in 215 days) |

The engine is **more than 1,000× faster** than these systems produced logs, so speed is not a
concern for the live demo.

---

## 8. Drift, seen from the engine

In the BGL test period, **67.4% of all lines** (423,650 of 628,954) were messages the frozen parser
had never seen (event 0). The HDFS test period had none. The engine counts these for every batch
(`n_unseen_events` per group, `unseen_messages` per call). This is what the planned dashboard drift
monitor will show. A rising share is the signal that the models need retraining (docs/03 §10.4).

---

## 9. Limitations

- **The engine is stateless.** It scores whatever lines it is given. Deciding *when* a group is
  complete is the next step (the API). For BGL that is the end of a 5-minute window; for HDFS
  blocks, which can stay active for hours, it will be a quiet-time rule.
- **Isolation Forest has no explanation.**
- **Explanations show association, not cause.** "More replication events than normal" points
  the operator in a direction; it does not prove the root cause.
- **The parser is saved with Python's `pickle`.** Loading a pickle can run code, so the engine only
  loads files created by this project's own pipeline.

---

## 10. Questions a lecturer might ask

**What is training–serving skew, and how did you rule it out?** It is when the live system
prepares data differently from training, so the deployed model is no longer the one you evaluated.
We fed the raw lines of the entire test period through the engine. All 117,911 test groups got the
same events, scores and decisions as the offline evaluation.

**Why does the live system use the time-split models?** Because a live system is always
predicting the future from the past. That is what the time split simulates and what its results
describe.

**Why is the default for BGL a rule and not a machine-learning model?** It had the best F1 on the
realistic time split (0.79). Choosing by evidence rather than by complexity is the honest choice.
The ML models remain available.

**Is the 0–1 score a probability?** No. It is a rescaled score whose alert line is 0.5 for every
detector. It keeps the order of groups and gives exactly the same alerts as the original thresholds.

**How does the Logistic Regression explanation work?** The model's score is a weighted sum. If you
subtract the weighted sum of an average normal group, the difference splits into one term per
event. The biggest positive terms are what makes this group look abnormal.

**Why is the engine stateless?** A stateless component is easy to test: the same input always gives
the same output. Keeping track of open groups over time is a separate job (the API's).

**Isn't loading a pickle file unsafe?** Only if the file comes from someone else. The engine only
loads the parser this project's own preprocessing wrote.
