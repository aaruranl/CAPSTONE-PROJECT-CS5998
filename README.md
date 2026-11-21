# Detecting Abnormal Behaviour in Software Systems Using Log Data and Machine Learning

CS5998 Capstone Project, MSc Data Science & AI. Student: Aaruran Ladchumanan (268442D)

The system reads raw software logs, learns what normal behaviour looks like, and
flags groups of log lines that look abnormal. It also tests a question most
published work skips: **do the reported near-perfect results still hold when a
model is trained only on past logs and tested on future logs?**

| Scope item | Choice |
|---|---|
| Problem type | Detection (classify each group of logs as normal / abnormal) |
| Data type | Time series of log events, created from text log files |
| Technique category | Machine learning and deep learning |
| System context | End-to-end pipeline + comparative analysis, served as an API and dashboard |

## Datasets

| Dataset | Source | Size | Group unit | Labels |
|---|---|---|---|---|
| HDFS v1 | [Loghub, Zenodo 8196385](https://zenodo.org/records/8196385) | 11.2M lines, 38.7 h | file block (`blk_...`) | per block |
| BGL | [Loghub, Zenodo 8196385](https://zenodo.org/records/8196385) | 4.7M lines, 215 days | 5-minute time window | per line → per window |

## Project structure

```
├── scripts/
│   ├── 01_download_data.py   # download + checksum + unzip the raw datasets
│   └── 02_preprocess.py      # load → parse → group → label → split
├── src/
│   ├── config.py             # all paths, seeds, ratios and parser settings
│   ├── log_parser.py         # masking + Drain parser with a learn/freeze life-cycle
│   ├── hdfs.py               # HDFS-specific preprocessing (group by block)
│   ├── bgl.py                # BGL-specific preprocessing (group by time window)
│   ├── splits.py             # random vs chronological train/val/test splits
│   └── common.py             # shared helpers
├── tests/                    # unit tests (python -m pytest -q)
├── docs/                     # explanation of each project step
└── data/                     # raw/ and processed/ (generated, not committed)
```

## How to reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/01_download_data.py     # ~245 MB download, ~2.3 GB unzipped
python scripts/02_preprocess.py        # ~2 min on an Apple M2 laptop
python -m pytest -q                    # unit tests
```

## Progress

| Step | Status | Notes |
|---|---|---|
| Data preprocessing | Done | [docs/01_data_preprocessing.md](docs/01_data_preprocessing.md) |
| Exploratory data analysis | Planned | |
| Baselines (Logistic Regression, Isolation Forest) | Planned | |
| Improved model (DeepLog LSTM) | Planned | |
| Scoring API + dashboard | Planned | |
