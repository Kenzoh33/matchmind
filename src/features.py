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
ELO_START = 1500.0  # neutral rating for every team in the first season
ELO_K = 20.0        # how far one result moves a rating
ELO_HOME = 60.0     # home-advantage bonus, used only in the expected score
ELO_CARRY = 0.8     # share of last season's rating kept over the summer
ELO_COLS = ["home_elo", "away_elo", "diff_elo"]


def add_elo(df):
    """Pre-match Elo ratings, computed per league in date order.

    Each match gets both teams' ratings BEFORE it is played, then the ratings update from
    its result. Matches with no result (FTR missing, e.g. fixtures) get ratings but do not
    update them. A team new to the league (promoted, or back after a gap) starts at the mean
    rating of the teams that left the league that summer.
    """
    df = df.copy()
    df["home_elo"] = np.nan
    df["away_elo"] = np.nan
    for _, lg in df.groupby("league"):
        ratings, prev_teams = {}, None
        for _, s in lg.groupby("season", sort=True):  # "2019-20" < "2020-21" < ...
            teams = set(s["HomeTeam"]) | set(s["AwayTeam"])
            if prev_teams is None:
                start = {t: ELO_START for t in teams}
            else:
                left = prev_teams - teams
                new_r = np.mean([ratings[t] for t in left]) if left else ELO_START
                start = {t: ratings[t] if t in prev_teams else new_r for t in teams}
            ratings = {t: ELO_CARRY * r + (1 - ELO_CARRY) * ELO_START for t, r in start.items()}
            prev_teams = teams
            for row in s.sort_values("Date").itertuples():
                rh, ra = ratings[row.HomeTeam], ratings[row.AwayTeam]
                df.at[row.Index, "home_elo"] = rh
                df.at[row.Index, "away_elo"] = ra
                if pd.isna(row.FTR):
                    continue
                expected_home = 1 / (1 + 10 ** ((ra - (rh + ELO_HOME)) / 400))
                actual_home = {"H": 1.0, "D": 0.5, "A": 0.0}[row.FTR]
                delta = ELO_K * (actual_home - expected_home)
                ratings[row.HomeTeam] = rh + delta
                ratings[row.AwayTeam] = ra - delta
    df["diff_elo"] = df["home_elo"] - df["away_elo"]
    return df


def check_elo_no_leakage(df, n=50):
    """Hide every result on or after a match's date; its pre-match Elo must not change."""
    rng = np.random.default_rng(SEED)
    for i in rng.choice(len(df), size=min(n, len(df)), replace=False):
        r = df.iloc[i]
        hidden = df.drop(columns=ELO_COLS).copy()
        hidden.loc[hidden["Date"] >= r["Date"], "FTR"] = np.nan
        re = add_elo(hidden).loc[r.name]
        for c in ELO_COLS:
            assert np.isclose(re[c], r[c]), (r["HomeTeam"], r["Date"], c)


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
    df = add_elo(df)
    check_elo_no_leakage(df)

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

    feat_cols = [f"{p}_{s}" for p in ("home", "away", "diff") for s in STATS] + ELO_COLS
    n_in = len(out)
    out = out.dropna(subset=feat_cols)
    out["y"] = out["FTR"].map(TARGET_MAP)
    assert out["y"].notna().all(), "unexpected FTR value"
    out["y"] = out["y"].astype(int)

    out = out.sort_values(["Date", "league", "HomeTeam"])
    out[ID_COLS + feat_cols + ["y"] + EXTRA_COLS].to_csv(OUT_PATH, index=False)

    print(f"Matches in: {n_in}, kept: {len(out)}, "
          f"dropped (team with < {WINDOW} prior matches in spell): {n_in - len(out)}")
    print("Leakage check passed (brute-force recompute of 50 random rows, form and Elo).")
    print(out.groupby(["league", "season"]).size().to_string())
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
