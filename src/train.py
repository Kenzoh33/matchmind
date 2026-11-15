"""Train baselines, logistic regression and XGBoost with a time-based split.

Run from the repo root:  python src/train.py
"""
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
DATA_PATH = Path("data/processed/features.csv")
MODEL_DIR = Path("models")
TRAIN_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
VAL_SEASONS = ["2024-25"]
TEST_SEASONS = ["2025-26"]
STATS = ["gf", "ga", "pts", "sot_for", "sot_against"]
FEATURES = [f"{p}_{s}" for p in ("home", "away", "diff") for s in STATS] + ["home_elo", "away_elo", "diff_elo"]
CLASSES = [0, 1, 2]  # H, D, A
XGB_GRID = {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1], "min_child_weight": [1, 5]}


def evaluate(y, P):
    """Accuracy, log loss and multiclass Brier score (sum over 3 outcomes, mean over matches)."""
    onehot = np.eye(3)[y]
    return {
        "accuracy": accuracy_score(y, P.argmax(axis=1)),
        "log_loss": log_loss(y, P, labels=CLASSES),
        "brier": np.mean(np.sum((P - onehot) ** 2, axis=1)),
    }


def home_baseline_probs(freqs, n):
    """Class frequencies for every match; argmax is always H (the most common outcome)."""
    return np.tile(freqs, (n, 1))


def bookmaker_probs(df):
    """Implied probabilities from market-average odds, normalised to remove the margin."""
    inv = 1.0 / df[["AvgH", "AvgD", "AvgA"]].to_numpy()
    return inv / inv.sum(axis=1, keepdims=True)


def class_freqs(y):
    return np.bincount(y, minlength=3) / len(y)


def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def tune_xgb(X_tr, y_tr, X_val, y_val):
    """Grid search; tree count chosen by early stopping on validation log loss."""
    best = None
    for depth, lr, mcw in itertools.product(*XGB_GRID.values()):
        model = XGBClassifier(objective="multi:softprob", n_estimators=1000, max_depth=depth,
                              learning_rate=lr, min_child_weight=mcw, eval_metric="mlogloss",
                              early_stopping_rounds=50, random_state=SEED)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        score = log_loss(y_val, model.predict_proba(X_val), labels=CLASSES)
        if best is None or score < best[0]:
            params = {"max_depth": depth, "learning_rate": lr, "min_child_weight": mcw,
                      "n_estimators": model.best_iteration + 1}
            best = (score, params, model)
    return best


def main():
    df = pd.read_csv(DATA_PATH)
    train = df[df["season"].isin(TRAIN_SEASONS)]
    val = df[df["season"].isin(VAL_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)]
    assert len(train) and len(val) and len(test), "empty split"
    assert not (set(TRAIN_SEASONS) & set(VAL_SEASONS) | set(TRAIN_SEASONS) & set(TEST_SEASONS)
                | set(VAL_SEASONS) & set(TEST_SEASONS)), "overlapping splits"
    print(f"train {len(train)}  val {len(val)}  test {len(test)}")

    X_tr, y_tr = train[FEATURES], train["y"].to_numpy()
    X_val, y_val = val[FEATURES], val["y"].to_numpy()
    X_te, y_te = test[FEATURES], test["y"].to_numpy()

    # Validation scores: everything fitted on train only.
    val_res = {
        "home_baseline": evaluate(y_val, home_baseline_probs(class_freqs(y_tr), len(val))),
        "bookmaker": evaluate(y_val, bookmaker_probs(val)),
        "logreg": evaluate(y_val, make_logreg().fit(X_tr, y_tr).predict_proba(X_val)),
    }
    xgb_val_loss, xgb_params, xgb_val_model = tune_xgb(X_tr, y_tr, X_val, y_val)
    val_res["xgboost"] = evaluate(y_val, xgb_val_model.predict_proba(X_val))
    print(f"XGB best params (val log loss {xgb_val_loss:.4f}): {xgb_params}")

    # Refit on train+val with chosen settings, then score test once.
    trval = pd.concat([train, val])
    X_tv, y_tv = trval[FEATURES], trval["y"].to_numpy()
    freqs = class_freqs(y_tv)
    logreg = make_logreg().fit(X_tv, y_tv)
    xgb = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                        random_state=SEED, **xgb_params).fit(X_tv, y_tv)

    test_res = {
        "home_baseline": evaluate(y_te, home_baseline_probs(freqs, len(test))),
        "bookmaker": evaluate(y_te, bookmaker_probs(test)),
        "logreg": evaluate(y_te, logreg.predict_proba(X_te)),
        "xgboost": evaluate(y_te, xgb.predict_proba(X_te)),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"class_freqs": freqs}, MODEL_DIR / "home_baseline.joblib")
    joblib.dump(logreg, MODEL_DIR / "logreg.joblib")
    joblib.dump(xgb, MODEL_DIR / "xgb.joblib")
    (MODEL_DIR / "xgb_params.json").write_text(json.dumps(xgb_params, indent=2))

    pd.set_option("display.float_format", "{:.4f}".format)
    print("\nValidation 2024-25 (fitted on 2019-20..2023-24):")
    print(pd.DataFrame(val_res).T.to_string())
    print("\nTest 2025-26 (fitted on 2019-20..2024-25):")
    print(pd.DataFrame(test_res).T.to_string())
    print(f"\nModels saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
