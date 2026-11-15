"""Calibration curves, confusion matrices and a per-league table for all five H/D/A models.

Models are refitted with the settings already chosen (no re-tuning): on the training seasons
for validation, on train+val for test. Overall metrics must match the published results.

Run from the repo root (after train.py and scoreline.py):
    python src/evaluate.py          # validation 2024-25 only
    python src/evaluate.py --test   # also test 2025-26
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")  # write PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

from scoreline import REPORT_DIR, fit_scoreline, hda_from_grid, predict_scoreline
from train import (DATA_PATH, FEATURES, MODEL_DIR, SEED, TEST_SEASONS, TRAIN_SEASONS,
                   VAL_SEASONS, bookmaker_probs, class_freqs, evaluate, home_baseline_probs,
                   make_logreg)

MODELS = ["home_baseline", "bookmaker", "logreg", "xgboost", "scoreline"]  # names used in reports/
LABELS = {"home_baseline": "Always home", "bookmaker": "Bookmaker", "logreg": "Logistic regression",
          "xgboost": "XGBoost", "scoreline": "Scoreline"}
OUTCOMES = ["H", "D", "A"]
OUTCOME_NAMES = {"H": "Home win", "D": "Draw", "A": "Away win"}
N_BINS = 10
EXPECTED_SCORELINE = {"alpha": 0.1, "league_flag": True, "dixon_coles": True}

# Bookmaker is the reference, so it gets a neutral dashed line; the three fitted models take the
# first three categorical slots; always-home is a single point. Marker shapes double-encode
# identity, so color is never the only cue.
STYLE = {
    "bookmaker": dict(color="#52514e", linestyle="--", marker="s"),
    "logreg": dict(color="#2a78d6", linestyle="-", marker="o"),
    "xgboost": dict(color="#eb6834", linestyle="-", marker="^"),
    "scoreline": dict(color="#1baf7a", linestyle="-", marker="D"),
    "home_baseline": dict(color="#0b0b0b", linestyle="none", marker="X"),
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def scoreline_settings():
    """Best row of the committed tuning table (lowest validation H/D/A log loss)."""
    tuning = pd.read_csv(REPORT_DIR / "scoreline_tuning.csv")
    best = tuning.loc[tuning["hda_log_loss"].idxmin()]
    got = {"alpha": float(best["alpha"]), "league_flag": bool(best["league_flag"]),
           "dixon_coles": bool(best["dixon_coles"])}
    assert got == EXPECTED_SCORELINE, f"unexpected scoreline settings {got}"
    return got


def hda_probs(fit_df, eval_df, sc):
    """H/D/A probabilities from each model, fitted on fit_df with settings chosen earlier.

    Own copy of the code in train.py / scoreline.py; check_matches_reports guards against drift.
    """
    X_fit, y_fit, X_ev = fit_df[FEATURES], fit_df["y"].to_numpy(), eval_df[FEATURES]
    xgb_params = json.loads((MODEL_DIR / "xgb_params.json").read_text())
    xgb = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                        random_state=SEED, **xgb_params).fit(X_fit, y_fit)
    sc_fit = fit_scoreline(fit_df, sc["alpha"], sc["league_flag"], sc["dixon_coles"])
    return {
        "home_baseline": home_baseline_probs(class_freqs(y_fit), len(eval_df)),
        "bookmaker": bookmaker_probs(eval_df),
        "logreg": make_logreg().fit(X_fit, y_fit).predict_proba(X_ev),
        "xgboost": xgb.predict_proba(X_ev),
        "scoreline": hda_from_grid(predict_scoreline(sc_fit, eval_df)[2]),
    }


def check_matches_reports(split, y, probs):
    """Overall metrics must equal the published ones (CSV is rounded to 4 dp)."""
    pub = pd.read_csv(REPORT_DIR / f"scoreline_{split}_metrics.csv")
    pub = pub[pub["table"] == "hda"].set_index("model")
    for m in MODELS:
        res = evaluate(y, probs[m])
        for k in ["accuracy", "log_loss", "brier"]:
            assert abs(res[k] - pub.loc[m, k]) <= 1e-4, (split, m, k, res[k], pub.loc[m, k])


def per_league(split, df, probs):
    rows = []
    for league, mask in [("EPL", df["league"] == "EPL"), ("LaLiga", df["league"] == "LaLiga"),
                         ("all", np.ones(len(df), bool))]:
        mask = np.asarray(mask)
        y = df["y"].to_numpy()[mask]
        for m in MODELS:
            rows.append({"split": split, "model": m, "league": league, "n": int(mask.sum()),
                         **evaluate(y, probs[m][mask])})
    return pd.DataFrame(rows)


def confusion(split, y, probs):
    rows = []
    for m in MODELS:
        pred = probs[m].argmax(axis=1)
        for a, actual in enumerate(OUTCOMES):
            for p, predicted in enumerate(OUTCOMES):
                rows.append({"split": split, "model": m, "actual": actual, "predicted": predicted,
                             "count": int(np.sum((y == a) & (pred == p)))})
    return pd.DataFrame(rows)


def quantile_bins(y_true, p):
    """Equal-count bins as sklearn's calibration_curve(strategy="quantile") builds them, with counts.

    Quantile bins keep roughly 73 matches per bin; equal-width bins would leave the extremes
    with a handful of matches and jumpy points.
    """
    edges = np.quantile(p, np.linspace(0, 1, N_BINS + 1))
    ids = np.searchsorted(edges[1:-1], p)
    count = np.bincount(ids, minlength=N_BINS)
    keep = count > 0
    obs = np.bincount(ids, weights=y_true, minlength=N_BINS)[keep] / count[keep]
    pred = np.bincount(ids, weights=p, minlength=N_BINS)[keep] / count[keep]
    ref_obs, ref_pred = calibration_curve(y_true, p, n_bins=N_BINS, strategy="quantile")
    assert np.allclose(obs, ref_obs) and np.allclose(pred, ref_pred), "binning differs from sklearn"
    return pred, obs, count[keep]


def calibration(split, y, probs):
    """Reliability points per model and outcome (one-vs-rest), plus expected calibration error.

    Always-home predicts the same probabilities for every match, so quantile bins collapse;
    it is one point per outcome: (its constant probability, observed frequency).
    """
    rows, err = [], []
    for m in MODELS:
        for k, out in enumerate(OUTCOMES):
            y_k, p_k = (y == k).astype(float), probs[m][:, k]
            if m == "home_baseline":
                pred, obs, cnt = np.array([p_k[0]]), np.array([y_k.mean()]), np.array([len(y)])
            else:
                pred, obs, cnt = quantile_bins(y_k, p_k)
            for b in range(len(pred)):
                rows.append({"split": split, "model": m, "outcome": out, "bin": b,
                             "mean_predicted": pred[b], "observed": obs[b], "count": int(cnt[b])})
            err.append({"split": split, "model": m, "outcome": out,
                        "ece": float(np.sum(cnt * np.abs(obs - pred)) / cnt.sum())})
    return pd.DataFrame(rows), pd.DataFrame(err)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_calibration(split, title, cal, probs, path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for k, out in enumerate(OUTCOMES):
        ax, hx = axes[0, k], axes[1, k]
        sub = cal[cal["outcome"] == out]
        lo = min(sub["mean_predicted"].min(), sub["observed"].min())
        hi = max(sub["mean_predicted"].max(), sub["observed"].max())
        pad = 0.05 * (hi - lo)
        lim = (max(0, lo - pad), min(1, hi + pad))
        style_axes(ax)
        ax.plot(lim, lim, color=MUTED, linewidth=1, linestyle=":", label="Perfect calibration")
        for m in MODELS:
            s = sub[sub["model"] == m]
            st = STYLE[m]
            ax.plot(s["mean_predicted"], s["observed"], color=st["color"], linestyle=st["linestyle"],
                    marker=st["marker"], markersize=9 if m == "home_baseline" else 6,
                    linewidth=2, label=LABELS[m],
                    markeredgecolor=SURFACE if m != "home_baseline" else st["color"])
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(OUTCOME_NAMES[out], color=INK, fontsize=12, loc="left")
        ax.set_ylabel("Observed frequency" if k == 0 else "", color=MUTED)

        style_axes(hx)
        bins = np.linspace(*lim, 30)
        for m in MODELS[1:]:
            hx.hist(probs[m][:, k], bins=bins, histtype="step", linewidth=1.5,
                    color=STYLE[m]["color"], linestyle=STYLE[m]["linestyle"])
        hx.set_xlabel("Mean predicted probability", color=MUTED)
        hx.set_ylabel("Matches" if k == 0 else "", color=MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.suptitle(f"Calibration, {title} ({N_BINS} equal-count bins; always home is one point)",
                 color=INK, fontsize=13, x=0.01, y=0.985, ha="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=6, frameon=False, fontsize=10,
               bbox_to_anchor=(0.005, 0.95), labelcolor=INK)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(split, title, conf, path):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(17, 4), facecolor=SURFACE)
    cmap = plt.get_cmap("Blues")
    for ax, m in zip(axes, MODELS):
        s = conf[conf["model"] == m]
        counts = s.pivot(index="actual", columns="predicted", values="count").loc[OUTCOMES, OUTCOMES]
        counts = counts.to_numpy()
        row_pct = 100 * counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                ink = SURFACE if row_pct[i, j] > 55 else INK
                ax.text(j, i, f"{counts[i, j]}\n{row_pct[i, j]:.0f}%", ha="center", va="center",
                        color=ink, fontsize=10)
        ax.set_xticks(range(3), OUTCOMES)
        ax.set_yticks(range(3), OUTCOMES)
        ax.tick_params(colors=MUTED, length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(LABELS[m], color=INK, fontsize=11)
        ax.set_xlabel("Predicted", color=MUTED)
        ax.set_ylabel("Actual" if m == MODELS[0] else "", color=MUTED)
    fig.suptitle(f"Confusion matrices, {title} (count and % of actual row)", color=INK,
                 fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def run_split(split, title, fit_df, eval_df, sc):
    y = eval_df["y"].to_numpy()
    probs = hda_probs(fit_df, eval_df, sc)
    check_matches_reports(split, y, probs)
    print(f"\n{title}: overall metrics match reports/scoreline_{split}_metrics.csv (to 1e-4).")

    league = per_league(split, eval_df, probs)
    conf = confusion(split, y, probs)
    cal, ece = calibration(split, y, probs)
    prefix = REPORT_DIR / f"eval_{split}"
    league.to_csv(f"{prefix}_per_league.csv", index=False, float_format="%.4f")
    conf.to_csv(f"{prefix}_confusion.csv", index=False)
    cal.to_csv(f"{prefix}_calibration.csv", index=False, float_format="%.4f")
    ece.to_csv(f"{prefix}_calibration_error.csv", index=False, float_format="%.4f")
    plot_calibration(split, title, cal, probs, f"{prefix}_calibration.png")
    plot_confusion(split, title, conf, f"{prefix}_confusion.png")

    pd.set_option("display.float_format", "{:.4f}".format)
    print("\nPer league:")
    print(league.pivot(index="model", columns="league", values=["accuracy", "log_loss", "brier"])
          .loc[MODELS].to_string())
    print(f"(about {int(league['n'].iloc[0])} matches per league: differences of ~0.01 are noise)")
    print("\nPredicted outcome counts (argmax), actual counts:",
          dict(zip(OUTCOMES, np.bincount(y, minlength=3).tolist())))
    print(conf.groupby(["model", "predicted"])["count"].sum().unstack()[OUTCOMES]
          .loc[MODELS].to_string())
    print("\nExpected calibration error (lower is better):")
    print(ece.pivot(index="model", columns="outcome", values="ece").loc[MODELS, OUTCOMES].to_string())
    print(f"\n-> {prefix}_{{per_league,confusion,calibration,calibration_error}}.csv, "
          f"{prefix}_{{calibration,confusion}}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test", action="store_true", help="also evaluate the 2025-26 test season")
    args = parser.parse_args()
    np.random.seed(SEED)

    df = pd.read_csv(DATA_PATH)
    train = df[df["season"].isin(TRAIN_SEASONS)]
    val = df[df["season"].isin(VAL_SEASONS)]
    assert len(train) and len(val), "empty split"
    sc = scoreline_settings()
    print(f"Scoreline settings from reports/scoreline_tuning.csv: {sc}")
    REPORT_DIR.mkdir(exist_ok=True)

    run_split("val", "validation 2024-25", train, val, sc)
    if not args.test:
        print("\nTest season not evaluated. Run with --test to add 2025-26.")
        return
    test = df[df["season"].isin(TEST_SEASONS)]
    assert len(test), "empty test split"
    run_split("test", "test 2025-26", pd.concat([train, val]), test, sc)


if __name__ == "__main__":
    main()
