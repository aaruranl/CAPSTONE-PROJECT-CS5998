"""
Central settings for the whole project.

Keeping every path, ratio and seed in one place means:
  * every script uses the same values (no silent mismatches), and
  * the reproducibility checklist can point to a single file.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # downloaded, never modified
PROCESSED_DIR = DATA_DIR / "processed"  # everything our pipeline creates

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# Train / validation / test proportions (from the project proposal)
# ----------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# ----------------------------------------------------------------------------
# Datasets (Loghub, Zenodo record 8196385)
# ----------------------------------------------------------------------------
ZENODO_RECORD = "8196385"
DATASETS = {
    "HDFS": {
        "zip_name": "HDFS_v1.zip",
        "md5": "76a24b4d9a6164d543fb275f89773260",
        "log_file": RAW_DIR / "HDFS" / "HDFS.log",
        "label_file": RAW_DIR / "HDFS" / "preprocessed" / "anomaly_label.csv",
        "out_dir": PROCESSED_DIR / "hdfs",
    },
    "BGL": {
        "zip_name": "BGL.zip",
        "md5": "4452953c470f2d95fcb32d5f6e733f7a",
        "log_file": RAW_DIR / "BGL" / "BGL.log",
        "label_file": None,  # BGL labels are inside each log line
        "out_dir": PROCESSED_DIR / "bgl",
    },
}

# ----------------------------------------------------------------------------
# Drain log parser settings. Values follow the Loghub parser benchmark
# (Zhu et al., ICSE-SEIP 2019), which used sim_th=0.5, depth=4 for HDFS and BGL.
# ----------------------------------------------------------------------------
DRAIN_SIM_TH = 0.5       # how similar a message must be to join a template
DRAIN_DEPTH = 4          # depth of Drain's parse tree
DRAIN_MAX_CHILDREN = 100

# ----------------------------------------------------------------------------
# BGL grouping: BGL has no natural "session" like HDFS blocks, so we cut the
# timeline into fixed, non-overlapping time windows.
# ----------------------------------------------------------------------------
BGL_WINDOW_MINUTES = 5

# ----------------------------------------------------------------------------
# Live scoring (src/stream.py): when is a group FINISHED and ready to score?
# BGL: a 5-minute window is finished when a line from a later window arrives
#      (all 4.7M BGL lines are in time order, so this is exact).
# HDFS: a block has no "end" message; it is finished after this much silence.
#      Chosen in scripts/09_choose_quiet_time.py on training + validation data:
#      the smallest value that (almost) never cuts a normal block in two.
# ----------------------------------------------------------------------------
HDFS_QUIET_HOURS = 13
