"""Refit on all completed seasons, score the in-progress season, and predict upcoming fixtures.

Run from the repo root (after download.py and features.py):  python src/predict.py
"""
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier

from download import LEAGUES, RAW_DIR
from features import STATS, add_form, add_spells, to_long
from train import (FEATURES, MODEL_DIR, SEED, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS,
                   bookmaker_probs, class_freqs, evaluate, home_baseline_probs, make_logreg)

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
FORWARD_SEASON = "2026-27"
FULL_SEASONS = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS  # 2019-20 .. 2025-26
MATCHES_PATH = Path("data/processed/matches.csv")
FEATURES_PATH = Path("data/processed/features.csv")
OUT_PATH = Path("reports/predictions_2026-27.csv")
OUTCOMES = np.array(["H", "D", "A"])


def fit_full(feat):
    """Refit LR and XGB on every completed season, reusing the hyperparameters tuned in train.py."""
    full = feat[feat["season"].isin(FULL_SEASONS)]
    X, y = full[FEATURES], full["y"].to_numpy()
    xgb_params = json.loads((MODEL_DIR / "xgb_params.json").read_text())
    logreg = make_logreg().fit(X, y)
    xgb = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                        random_state=SEED, **xgb_params).fit(X, y)
    freqs = class_freqs(y)
    joblib.dump(logreg, MODEL_DIR / "logreg_full.joblib")
    joblib.dump(xgb, MODEL_DIR / "xgb_full.joblib")
    return freqs, logreg, xgb, len(full)


def forward_check(feat, freqs, logreg, xgb):
    """Score played matches of the forward season; it was never used for fitting or tuning."""
    fwd = feat[feat["season"] == FORWARD_SEASON]
    y = fwd["y"].to_numpy()
    res = {
        "home_baseline": evaluate(y, home_baseline_probs(freqs, len(fwd))),
        "bookmaker": evaluate(y, bookmaker_probs(fwd)),
        "logreg": evaluate(y, logreg.predict_proba(fwd[FEATURES])),
        "xgboost": evaluate(y, xgb.predict_proba(fwd[FEATURES])),
    }
    return len(fwd), pd.DataFrame(res).T


def load_fixtures(matches):
    """Download fixtures.csv and keep EPL/La Liga matches not already played."""
    resp = requests.get(FIXTURES_URL, timeout=30)
    resp.raise_for_status()
    (RAW_DIR / "fixtures.csv").write_bytes(resp.content)
    fx = pd.read_csv(io.BytesIO(resp.content), encoding="utf-8-sig")  # file starts with a BOM
    fx = fx[fx["Div"].isin(LEAGUES)].copy()
    fx["league"] = fx["Div"].map(LEAGUES)
    fx["season"] = FORWARD_SEASON
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, format="mixed")
    fx = fx[["league", "season", "Date", "HomeTeam", "AwayTeam", "AvgH", "AvgD", "AvgA"]]
    played = matches[["Date", "HomeTeam", "AwayTeam"]].assign(_played=True)
    fx = fx.merge(played, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    return fx[fx["_played"].isna()].drop(columns="_played").reset_index(drop=True)


def fixture_features(matches, fx):
    """Form for each fixture from that team's last 5 completed matches (same code as features.py).

    Fixtures are appended with no results; shift(1) in add_form keeps each row's own
    (empty) result out of its window, so only earlier completed matches are used.
    """
    allm = pd.concat([matches, fx.assign(is_fixture=True)], ignore_index=True)
    allm["is_fixture"] = allm["is_fixture"].fillna(False).astype(bool)
    allm["match_id"] = allm.index
    long = add_form(add_spells(to_long(allm)))

    form_cols = [f"form_{s}" for s in STATS]
    side = long[["match_id", "team"] + form_cols]
    home = side.rename(columns={"team": "HomeTeam", **{c: c.replace("form_", "home_") for c in form_cols}})
    away = side.rename(columns={"team": "AwayTeam", **{c: c.replace("form_", "away_") for c in form_cols}})
    out = allm[allm["is_fixture"]].merge(home, on=["match_id", "HomeTeam"]).merge(away, on=["match_id", "AwayTeam"])
    for s in STATS:
        out[f"diff_{s}"] = out[f"home_{s}"] - out[f"away_{s}"]
    # A team's 2nd listed fixture has an unplayed match in its window -> NaN -> not predictable yet.
    ok = out[FEATURES].notna().all(axis=1)
    return out[ok], int((~ok).sum())


def predict_table(fx_feat, logreg, xgb):
    cols = ["league", "Date", "HomeTeam", "AwayTeam"]
    out = fx_feat[cols].copy()
    for name, P in [("lr", logreg.predict_proba(fx_feat[FEATURES])),
                    ("xgb", xgb.predict_proba(fx_feat[FEATURES])),
                    ("book", bookmaker_probs(fx_feat))]:
        out[[f"{name}_pH", f"{name}_pD", f"{name}_pA"]] = P.round(3)
        out[f"{name}_pick"] = OUTCOMES[P.argmax(axis=1)]
    return out.sort_values(["Date", "league", "HomeTeam"])


def main():
    feat = pd.read_csv(FEATURES_PATH)
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["Date"])

    freqs, logreg, xgb, n_full = fit_full(feat)
    print(f"Refit LR + XGB on {FULL_SEASONS[0]}..{FULL_SEASONS[-1]} ({n_full} matches) "
          f"-> {MODEL_DIR}/logreg_full.joblib, xgb_full.joblib")

    n_fwd, table = forward_check(feat, freqs, logreg, xgb)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(f"\nForward check {FORWARD_SEASON}: {n_fwd} played matches")
    print(table.to_string())

    fx = load_fixtures(matches)
    if fx.empty:
        print("\nNo upcoming EPL/La Liga fixtures in fixtures.csv yet (all listed matches are played).")
        return
    fx_feat, n_skipped = fixture_features(matches, fx)
    if n_skipped:
        print(f"\nSkipped {n_skipped} fixture(s): a team has an earlier unplayed fixture.")
    preds = predict_table(fx_feat, logreg, xgb)
    OUT_PATH.parent.mkdir(exist_ok=True)
    preds.to_csv(OUT_PATH, index=False)
    print(f"\nPredictions for {len(preds)} upcoming fixtures -> {OUT_PATH}")
    print(preds[["Date", "HomeTeam", "AwayTeam", "lr_pH", "lr_pD", "lr_pA", "lr_pick", "book_pick"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
