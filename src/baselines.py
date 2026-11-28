"""
The two baseline models promised in the project proposal. Both read event-count vectors
(src/features.py) and ignore the order of events.

* Logistic Regression - SUPERVISED: learns a weight per event from labelled
  training groups; the score is the predicted probability of "abnormal".
* Isolation Forest   - UNSUPERVISED: never sees a label. It builds random trees
  that split the data at random; unusual groups get isolated after only a few
  splits, so a short average path = a high anomaly score.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_SEED


def make_logistic_regression(seed: int = RANDOM_SEED) -> LogisticRegression:
    # class_weight="balanced": an abnormal group counts ~33x (HDFS) / ~13x (BGL) more
    # than a normal one in the loss, so the rare class is not ignored
    return LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed)


def make_isolation_forest(seed: int = RANDOM_SEED) -> IsolationForest:
    # library defaults (100 trees, 256 samples per tree), as in the original paper (Liu et al., 2008)
    return IsolationForest(n_estimators=100, max_samples=256, random_state=seed, n_jobs=-1)


def anomaly_scores(model, X: np.ndarray) -> np.ndarray:
    """One score per group, higher = more abnormal, for either model."""
    if isinstance(model, LogisticRegression):
        return model.predict_proba(X)[:, 1]
    return -model.score_samples(X)  # sklearn: lower score_samples = more abnormal, so flip the sign
