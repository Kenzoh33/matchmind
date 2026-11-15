"""Download Football-Data.co.uk season CSVs and combine them into one table.

Run from the repo root:  python src/download.py [--force]
"""
import argparse
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
LEAGUES = {"E0": "EPL", "SP1": "LaLiga"}
# Season codes like "1920" for 2019-20, through "2627" for 2026-27.
SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2019, 2027)]
CURRENT_SEASON = SEASONS[-1]
KEEP_COLS = [
    "league", "season", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST",
    "AvgH", "AvgD", "AvgA",
]

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def download(league, season, force=False):
    """Fetch one season file. Finished seasons are cached; the current one is always refreshed."""
    path = RAW_DIR / f"{league}_{season}.csv"
    if path.exists() and not force and season != CURRENT_SEASON:
        return path
    resp = requests.get(URL.format(season=season, league=league), timeout=30)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def load(path, league, season):
    # latin-1 decodes any byte sequence; some files have non-UTF-8 team names.
    df = pd.read_csv(path, encoding="latin-1")
    df = df.dropna(how="all")  # trailing blank lines of bare commas
    df["league"] = LEAGUES[league]
    df["season"] = f"20{season[:2]}-{season[2:]}"
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = df[KEEP_COLS].copy()
    # Files mix dd/mm/yy and dd/mm/yyyy; dayfirst stops 03/04 being read as 4 March.
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    if df["Date"].isna().any():
        raise ValueError(f"{path} has unparseable dates")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download all files")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for league in LEAGUES:
        for season in SEASONS:
            path = download(league, season, force=args.force)
            frames.append(load(path, league, season))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["Date", "league", "HomeTeam"]).reset_index(drop=True)
    out = PROCESSED_DIR / "matches.csv"
    df.to_csv(out, index=False)

    print(df.groupby(["league", "season"]).size().to_string())
    print(f"\nTotal: {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
