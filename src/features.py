"""Build pre-match rolling-form features from data/processed/matches.csv.

Run from the repo root:  python src/features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

IN_PATH = Path("data/processed/matches.csv")
OUT_PATH = Path("data/processed/features.csv")
WINDOW = 5
STATS = ["gf", "ga", "pts", "sot_for", "sot_against"]
TARGET_MAP = {"H": 0, "D": 1, "A": 2}
ID_COLS = ["league", "season", "Date", "HomeTeam", "AwayTeam"]
EXTRA_COLS = ["FTHG", "FTAG", "AvgH", "AvgD", "AvgA"]  # kept for later; NOT features
SEED = 0


def to_long(df):
    """One row per (match, team), so a team's home and away games share one history."""
    pts_home = df["FTR"].map({"H": 3, "D": 1, "A": 0})
    pts_away = df["FTR"].map({"H": 0, "D": 1, "A": 3})
    base = df[["match_id", "league", "season", "Date"]]
    home = base.assign(team=df["HomeTeam"], gf=df["FTHG"], ga=df["FTAG"], pts=pts_home,
                       sot_for=df["HST"], sot_against=df["AST"])
    away = base.assign(team=df["AwayTeam"], gf=df["FTAG"], ga=df["FTHG"], pts=pts_away,
                       sot_for=df["AST"], sot_against=df["HST"])
    return pd.concat([home, away], ignore_index=True)


def add_spells(long):
    """Spell id increments when a team returns after missing at least one top-flight season."""
    start = long["season"].str[:4].astype(int)
    seasons = (pd.DataFrame({"league": long["league"], "team": long["team"], "start": start})
               .drop_duplicates().sort_values(["league", "team", "start"]))
    gap = seasons.groupby(["league", "team"])["start"].diff().ne(1)  # first season or a gap
    seasons["spell"] = gap.groupby([seasons["league"], seasons["team"]]).cumsum()
    long = long.assign(start=start).merge(seasons, on=["league", "team", "start"])
    return long.drop(columns="start")


def add_form(long):
    """Mean of the previous WINDOW matches. shift(1) excludes the current match."""
    long = long.sort_values(["league", "team", "spell", "Date"]).reset_index(drop=True)
    g = long.groupby(["league", "team", "spell"])
    for s in STATS:
        long[f"form_{s}"] = g[s].transform(
            lambda x: x.shift(1).rolling(WINDOW, min_periods=WINDOW).mean())
    return long


def check_no_leakage(long, n=50):
    """Brute-force recompute form for random rows using only earlier-dated matches."""
    rng = np.random.default_rng(SEED)
    rows = long.dropna(subset=["form_gf"])
    for i in rng.choice(len(rows), size=min(n, len(rows)), replace=False):
        r = rows.iloc[i]
        hist = long[(long["league"] == r["league"]) & (long["team"] == r["team"])
                    & (long["spell"] == r["spell"]) & (long["Date"] < r["Date"])]
        last = hist.sort_values("Date").tail(WINDOW)
        assert len(last) == WINDOW
        for s in STATS:
            assert np.isclose(last[s].mean(), r[f"form_{s}"]), (r["team"], r["Date"], s)


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["Date"])
    df["match_id"] = df.index

    long = to_long(df)
    assert not long.duplicated(["team", "Date"]).any(), "a team plays twice on one date"
    long = add_spells(long)
    long = add_form(long)
    check_no_leakage(long)

    form_cols = [f"form_{s}" for s in STATS]
    side = long[["match_id", "team"] + form_cols]
    home = side.rename(columns={"team": "HomeTeam", **{c: c.replace("form_", "home_") for c in form_cols}})
    away = side.rename(columns={"team": "AwayTeam", **{c: c.replace("form_", "away_") for c in form_cols}})
    out = df.merge(home, on=["match_id", "HomeTeam"]).merge(away, on=["match_id", "AwayTeam"])
    for s in STATS:
        out[f"diff_{s}"] = out[f"home_{s}"] - out[f"away_{s}"]

    feat_cols = [f"{p}_{s}" for p in ("home", "away", "diff") for s in STATS]
    n_in = len(out)
    out = out.dropna(subset=feat_cols)
    out["y"] = out["FTR"].map(TARGET_MAP)
    assert out["y"].notna().all(), "unexpected FTR value"
    out["y"] = out["y"].astype(int)

    out = out.sort_values(["Date", "league", "HomeTeam"])
    out[ID_COLS + feat_cols + ["y"] + EXTRA_COLS].to_csv(OUT_PATH, index=False)

    print(f"Matches in: {n_in}, kept: {len(out)}, "
          f"dropped (team with < {WINDOW} prior matches in spell): {n_in - len(out)}")
    print("Leakage check passed (brute-force recompute of 50 random rows).")
    print(out.groupby(["league", "season"]).size().to_string())
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
