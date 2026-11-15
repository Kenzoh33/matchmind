"""Scoreline model: Poisson regressions for home and away goals, optional Dixon-Coles correction.

Each match gets a 0-6 x 0-6 grid of scoreline probabilities. From the grid we read the most
likely score and derive H/D/A probabilities, scored next to the train.py baselines.

Run from the repo root (after features.py and train.py):
    python src/scoreline.py          # tune and score on validation 2024-25 only
    python src/scoreline.py --test   # also refit on train+val and score test 2025-26 once
"""
import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from train import (DATA_PATH, FEATURES, MODEL_DIR, SEED, TEST_SEASONS, TRAIN_SEASONS,
                   VAL_SEASONS, bookmaker_probs, class_freqs, evaluate, home_baseline_probs,
                   make_logreg)

REPORT_DIR = Path("reports")
MAX_GOALS = 6  # grid is 0..MAX_GOALS per side; the last cell holds P(>= MAX_GOALS)
LEAGUE_COL = "is_laliga"
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]  # L2 strength for PoissonRegressor
NOT_INPUTS = {"FTHG", "FTAG", "AvgH", "AvgD", "AvgA", "y", "FTR"}  # targets and odds


def design(df, use_league):
    """Model inputs: FEATURES, plus a 0/1 league flag if chosen (La Liga scores fewer goals)."""
    X = df[FEATURES].copy()
    if use_league:
        X[LEAGUE_COL] = (df["league"] == "LaLiga").astype(float)
    assert not (set(X.columns) & NOT_INPUTS), "target or odds column used as an input"
    return X


def fit_rates(df, alpha, use_league):
    """One Poisson GLM (log link) per side. Scaling lets one alpha penalise Elo and form fairly."""
    X = design(df, use_league)
    models = {}
    for side, target in [("home", "FTHG"), ("away", "FTAG")]:
        m = make_pipeline(StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=1000))
        models[side] = m.fit(X, df[target])
    return models


def predict_rates(models, df, use_league):
    X = design(df, use_league)
    return models["home"].predict(X), models["away"].predict(X)


def marginals(rate, lump=True):
    """P(goals = 0..MAX_GOALS) per match.

    lump=True puts P(goals >= MAX_GOALS) in the last cell, so each row sums to 1; used for
    H/D/A and log loss. lump=False keeps the exact P(goals = MAX_GOALS); used only to rank
    scores, because "6 or more" is not a single score and can otherwise win for a strong
    favourite (rate ~4: P(>= 6) 0.21 > P(= 3) 0.20).
    """
    k = np.arange(MAX_GOALS + 1 if not lump else MAX_GOALS)
    pmf = poisson.pmf(k[None, :], rate[:, None])
    if not lump:
        return pmf
    tail = poisson.sf(MAX_GOALS - 1, rate)[:, None]
    return np.hstack([pmf, tail])


def dc_tau(lam, mu, rho):
    """Dixon-Coles factors for the 0-0, 0-1, 1-0, 1-1 cells (home goals first), shape (n, 2, 2)."""
    tau = np.empty((len(lam), 2, 2))
    tau[:, 0, 0] = 1 - lam * mu * rho
    tau[:, 0, 1] = 1 + lam * rho
    tau[:, 1, 0] = 1 + mu * rho
    tau[:, 1, 1] = 1 - rho
    return tau


def rho_bounds(lam, mu):
    """Range of rho that keeps every tau factor >= 0 for these rates."""
    lo = max(-1 / lam.max(), -1 / mu.max(), -1.0)
    hi = min(1 / (lam * mu).max(), 1.0)
    return lo + 1e-6, hi - 1e-6


def fit_rho(lam, mu, home_goals, away_goals):
    """Maximum-likelihood rho with the rates held fixed.

    The correction leaves the grid's total unchanged, so only matches ending 0-0, 0-1, 1-0
    or 1-1 change the likelihood, through log tau.
    """
    low = (home_goals <= 1) & (away_goals <= 1)
    lam_l, mu_l, h, a = lam[low], mu[low], home_goals[low], away_goals[low]
    idx = np.arange(len(h))

    def nll(rho):
        tau = dc_tau(lam_l, mu_l, rho)[idx, h, a]
        return -np.sum(np.log(tau))

    res = minimize_scalar(nll, bounds=rho_bounds(lam, mu), method="bounded")
    return float(res.x)


def score_grid(lam, mu, rho=0.0, lump=True):
    """Joint P(home = i, away = j), shape (n, MAX_GOALS+1, MAX_GOALS+1).

    lump=False gives the exact (unlumped) grid for ranking scores; it sums to slightly under 1.
    """
    P = marginals(lam, lump)[:, :, None] * marginals(mu, lump)[:, None, :]
    if rho != 0.0:
        # Rates beyond the fitted range could push a factor below 0; clip, then renormalise.
        total = P.sum(axis=(1, 2), keepdims=True)
        P[:, :2, :2] *= np.clip(dc_tau(lam, mu, rho), 0, None)
        P *= total / P.sum(axis=(1, 2), keepdims=True)
    if lump:
        assert np.allclose(P.sum(axis=(1, 2)), 1.0), "scoreline grid does not sum to 1"
    return P


def hda_from_grid(P):
    """H = home goals > away goals (below the diagonal), D = diagonal, A = above it."""
    n = P.shape[1]
    lower = np.tril(np.ones((n, n)), k=-1).astype(bool)
    return np.column_stack([P[:, lower].sum(axis=1),
                            np.trace(P, axis1=1, axis2=2),
                            P[:, lower.T].sum(axis=1)])


def top_scores(P, k=3):
    """The k most likely (home, away) scores per match, most likely first.

    Pass the exact grid (score_grid(..., lump=False)) so the returned probabilities are
    P(exactly that score).
    """
    n = P.shape[1]
    flat = P.reshape(len(P), -1)
    order = np.argsort(-flat, axis=1, kind="stable")[:, :k]
    return order // n, order % n, np.take_along_axis(flat, order, axis=1)


def clipped_goals(df):
    """Actual goals capped at MAX_GOALS, to index the grid's lumped last cell."""
    return (np.minimum(df["FTHG"].to_numpy(), MAX_GOALS),
            np.minimum(df["FTAG"].to_numpy(), MAX_GOALS))


def scoreline_metrics(df, pred_h, pred_a, P=None, xg=None):
    """Exact-score hit rate, MAE of the point prediction, and scoreline log loss if P is given.

    pred_h/pred_a: the predicted score (most likely score for the model, a fixed score for
    baselines). xg: optional expected goals (lam, mu), whose MAE is reported separately.
    """
    h, a = df["FTHG"].to_numpy(), df["FTAG"].to_numpy()
    hc, ac = clipped_goals(df)
    out = {
        "exact_hit": np.mean((pred_h == h) & (pred_a == a)),  # real goals: a 6 pick misses a 7
        "score_mae": np.mean(np.abs(pred_h - h) + np.abs(pred_a - a)) / 2,
        "xg_mae": np.nan,
        "score_log_loss": np.nan,
    }
    if xg is not None:
        out["xg_mae"] = np.mean(np.abs(xg[0] - h) + np.abs(xg[1] - a)) / 2
    if P is not None:
        p = P[np.arange(len(df)), hc, ac]
        out["score_log_loss"] = -np.mean(np.log(p))
    return out


def train_score_freqs(train):
    """Empirical scoreline frequencies on the fitting seasons, with +1 smoothing so no cell is 0."""
    n = MAX_GOALS + 1
    hc, ac = clipped_goals(train)
    counts = np.ones((n, n))
    np.add.at(counts, (hc, ac), 1)
    return counts / counts.sum()


def scoreline_baselines(fit_df, eval_df):
    """Always 1-1, the most common score in the fitting seasons, and the empirical score grid."""
    freqs = train_score_freqs(fit_df)
    common = fit_df.groupby(["FTHG", "FTAG"]).size().idxmax()
    n = len(eval_df)
    grid = np.broadcast_to(freqs, (n,) + freqs.shape)
    gh, ga = np.unravel_index(freqs.argmax(), freqs.shape)
    return common, {
        "always_1-1": scoreline_metrics(eval_df, np.ones(n), np.ones(n)),
        f"most_common_{common[0]}-{common[1]}": scoreline_metrics(
            eval_df, np.full(n, common[0]), np.full(n, common[1])),
        "train_score_freqs": scoreline_metrics(eval_df, np.full(n, gh), np.full(n, ga), P=grid),
    }


def fit_scoreline(df, alpha, use_league, use_dc):
    """Poisson rates, plus rho fitted on the same rows (in-sample rates) if Dixon-Coles is on."""
    models = fit_rates(df, alpha, use_league)
    rho = 0.0
    if use_dc:
        lam, mu = predict_rates(models, df, use_league)
        rho = fit_rho(lam, mu, *clipped_goals(df))
    return {"models": models, "rho": rho, "alpha": alpha, "use_league": use_league,
            "use_dc": use_dc}


def predict_scoreline(fit, df):
    """Rates, the lumped grid (H/D/A, log loss) and the exact grid (ranking scores)."""
    lam, mu = predict_rates(fit["models"], df, fit["use_league"])
    return lam, mu, score_grid(lam, mu, fit["rho"]), score_grid(lam, mu, fit["rho"], lump=False)


def tune(train, val):
    """Choose alpha, league flag and Poisson vs Dixon-Coles by validation H/D/A log loss.

    Everything (rates and rho) is fitted on the training seasons only; validation only ranks.
    """
    rows, best = [], None
    y_val = val["y"].to_numpy()
    for alpha, use_league in itertools.product(ALPHAS, [False, True]):
        base = fit_scoreline(train, alpha, use_league, use_dc=False)
        lam_tr, mu_tr = predict_rates(base["models"], train, use_league)
        rho = fit_rho(lam_tr, mu_tr, *clipped_goals(train))
        for use_dc in [False, True]:
            fit = {**base, "use_dc": use_dc, "rho": rho if use_dc else 0.0}
            lam, mu, P, P_exact = predict_scoreline(fit, val)
            hda = evaluate(y_val, hda_from_grid(P))
            sc = scoreline_metrics(val, *top_scores(P_exact, 1)[:2], P=P, xg=(lam, mu))
            row = {"alpha": alpha, "league_flag": use_league, "dixon_coles": use_dc,
                   "rho": fit["rho"], "hda_log_loss": hda["log_loss"],
                   "score_log_loss": sc["score_log_loss"]}
            rows.append(row)
            if best is None or row["hda_log_loss"] < best[0]["hda_log_loss"]:
                best = (row, fit)
    return pd.DataFrame(rows), best[0], best[1]


def report(name, df, fit, hda_rows, fit_df):
    """Build the H/D/A and scoreline tables for one evaluation split; return them + predictions."""
    lam, mu, P, P_exact = predict_scoreline(fit, df)
    hda_rows["scoreline"] = evaluate(df["y"].to_numpy(), hda_from_grid(P))
    th, ta, tp = top_scores(P_exact, 3)
    common, sc_rows = scoreline_baselines(fit_df, df)
    sc_rows["scoreline"] = scoreline_metrics(df, th[:, 0], ta[:, 0], P=P, xg=(lam, mu))

    hda_tab, sc_tab = pd.DataFrame(hda_rows).T, pd.DataFrame(sc_rows).T
    print(f"\n{name}: H/D/A")
    print(hda_tab.to_string())
    print(f"\n{name}: scorelines (most common score in fitting seasons: {common[0]}-{common[1]})")
    print(sc_tab.to_string())

    pred = df[["league", "season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
    pred["xg_home"], pred["xg_away"] = lam.round(3), mu.round(3)
    pred[["pH", "pD", "pA"]] = hda_from_grid(P).round(4)
    for i in range(3):
        pred[f"score_{i + 1}"] = [f"{h}-{a}" for h, a in zip(th[:, i], ta[:, i])]
        pred[f"p_score_{i + 1}"] = tp[:, i].round(4)
    return hda_tab, sc_tab, pred


def save_tables(prefix, hda_tab, sc_tab):
    out = pd.concat({"hda": hda_tab, "scoreline": sc_tab}, names=["table", "model"])
    path = REPORT_DIR / f"scoreline_{prefix}_metrics.csv"
    out.to_csv(path, float_format="%.4f")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test", action="store_true",
                        help="refit on train+val and score the 2025-26 test season once")
    args = parser.parse_args()
    np.random.seed(SEED)  # nothing here is random; fixed anyway for reproducibility

    df = pd.read_csv(DATA_PATH)
    train = df[df["season"].isin(TRAIN_SEASONS)]
    val = df[df["season"].isin(VAL_SEASONS)]
    assert len(train) and len(val), "empty split"
    print(f"train {len(train)}  val {len(val)}")
    REPORT_DIR.mkdir(exist_ok=True)
    pd.set_option("display.float_format", "{:.4f}".format)

    tuning, chosen, val_fit = tune(train, val)
    tuning.to_csv(REPORT_DIR / "scoreline_tuning.csv", index=False, float_format="%.5f")
    print("\nTuning on validation 2024-25 (fitted on 2019-20..2023-24), best 5 by H/D/A log loss:")
    print(tuning.sort_values("hda_log_loss").head(5).to_string(index=False))
    print(f"\nChosen: alpha={chosen['alpha']}, league flag={'on' if chosen['league_flag'] else 'off'}, "
          f"{'Dixon-Coles' if chosen['dixon_coles'] else 'plain Poisson'} (rho={chosen['rho']:.4f})")

    # Validation baselines, all fitted on train only. XGB reuses train.py's tuned settings;
    # its tree count came from early stopping on this validation set, as in train.py.
    X_tr, y_tr, X_val = train[FEATURES], train["y"].to_numpy(), val[FEATURES]
    xgb_params = json.loads((MODEL_DIR / "xgb_params.json").read_text())
    xgb_val = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                            random_state=SEED, **xgb_params).fit(X_tr, y_tr)
    hda_val = {
        "home_baseline": evaluate(val["y"].to_numpy(), home_baseline_probs(class_freqs(y_tr), len(val))),
        "bookmaker": evaluate(val["y"].to_numpy(), bookmaker_probs(val)),
        "logreg": evaluate(val["y"].to_numpy(), make_logreg().fit(X_tr, y_tr).predict_proba(X_val)),
        "xgboost": evaluate(val["y"].to_numpy(), xgb_val.predict_proba(X_val)),
    }
    hda_tab, sc_tab, _ = report("Validation 2024-25 (fitted on 2019-20..2023-24)",
                                val, val_fit, hda_val, train)
    print(f"\n-> {save_tables('val', hda_tab, sc_tab)}, {REPORT_DIR}/scoreline_tuning.csv")

    if not args.test:
        print("\nTest season not scored. Run with --test once tuning is final.")
        return

    # Refit the chosen configuration (rates and rho) on train+val, then score test once.
    test = df[df["season"].isin(TEST_SEASONS)]
    assert len(test), "empty test split"
    trval = pd.concat([train, val])
    final = fit_scoreline(trval, chosen["alpha"], chosen["league_flag"], chosen["dixon_coles"])
    y_tv, y_te = trval["y"].to_numpy(), test["y"].to_numpy()
    hda_test = {
        "home_baseline": evaluate(y_te, home_baseline_probs(class_freqs(y_tv), len(test))),
        "bookmaker": evaluate(y_te, bookmaker_probs(test)),
        "logreg": evaluate(y_te, joblib.load(MODEL_DIR / "logreg.joblib").predict_proba(test[FEATURES])),
        "xgboost": evaluate(y_te, joblib.load(MODEL_DIR / "xgb.joblib").predict_proba(test[FEATURES])),
    }
    hda_tab, sc_tab, pred = report("Test 2025-26 (fitted on 2019-20..2024-25)",
                                   test, final, hda_test, trval)
    pred.to_csv(REPORT_DIR / "scoreline_test_predictions.csv", index=False)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_DIR / "scoreline.joblib")
    print(f"\n-> {save_tables('test', hda_tab, sc_tab)}, {REPORT_DIR}/scoreline_test_predictions.csv, "
          f"{MODEL_DIR}/scoreline.joblib (rho={final['rho']:.4f})")


if __name__ == "__main__":
    main()
