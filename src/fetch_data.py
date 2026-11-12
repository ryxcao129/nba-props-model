"""Download player and team game logs from stats.nba.com via nba_api.

One LeagueGameLog call returns every player-game (or team-game) in a season,
so a full pull is only ~2 requests per season.

Usage:
    python src/fetch_data.py --start 2018 --end 2025
    (2025 means the 2025-26 season; re-run during the season to refresh it)
"""
import argparse
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from common import DATA_RAW


def season_str(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def pull(season: str, kind: str) -> pd.DataFrame:
    # kind: "P" = players, "T" = teams
    for attempt in range(3):
        try:
            res = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation=kind,
                timeout=60,
            )
            return res.get_data_frames()[0]
        except Exception as e:  # stats.nba.com is flaky; back off and retry
            print(f"  retry {attempt + 1} for {season} {kind}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Could not download {season} {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()

    players, teams = [], []
    for y in range(args.start, args.end + 1):
        s = season_str(y)
        print(f"Downloading {s} ...")
        p = pull(s, "P")
        p["SEASON"] = s
        players.append(p)
        time.sleep(1.5)
        t = pull(s, "T")
        t["SEASON"] = s
        teams.append(t)
        time.sleep(1.5)
        print(f"  {len(p):,} player-games, {len(t):,} team-games")

    for name, new in (("player_logs.csv", pd.concat(players)), ("team_logs.csv", pd.concat(teams))):
        path = DATA_RAW / name
        if path.exists():  # keep seasons you didn't re-download, replace the ones you did
            old = pd.read_csv(path)
            new = pd.concat([old[~old["SEASON"].isin(new["SEASON"].unique())], new])
        new.to_csv(path, index=False)
    print(f"Saved to {DATA_RAW}")


if __name__ == "__main__":
    main()
