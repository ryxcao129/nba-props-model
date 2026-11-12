"""Generate fake game logs in the nba_api format, to test the pipeline offline.

    python src/make_synthetic.py && python src/train.py
"""
import numpy as np
import pandas as pd

from common import DATA_RAW

rng = np.random.default_rng(0)
TEAMS = [f"T{i:02d}" for i in range(30)]


def main(seasons=(2021, 2022, 2023, 2024), games_per_team=40, players_per_team=9):
    prow, trow = [], []
    talent = {(t, k): rng.gamma(2.0, 6.0) for t in TEAMS for k in range(players_per_team)}
    defence = {t: rng.normal(1.0, 0.07) for t in TEAMS}
    gid = 0
    for y in seasons:
        s = f"{y}-{str(y + 1)[-2:]}"
        start = pd.Timestamp(f"{y}-10-20")
        for d in range(games_per_team):
            order = rng.permutation(TEAMS)
            date = start + pd.Timedelta(days=2 * d + int(rng.integers(0, 2)))
            for a, h in zip(order[::2], order[1::2]):
                gid += 1
                for team, opp, home in ((a, h, 0), (h, a, 1)):
                    tot = {"PTS": 0, "REB": 0, "AST": 0, "FG3M": 0, "FGA": 0}
                    for k in range(players_per_team):
                        base = talent[(team, k)] * (1.03 if home else 1.0) * defence[opp]
                        mins = float(np.clip(rng.normal(12 + base, 5), 0, 44))
                        pts = rng.poisson(max(base * mins / 30, 0.1))
                        reb, ast, fg3 = rng.poisson(mins / 8), rng.poisson(mins / 10), rng.poisson(pts / 9 + 0.1)
                        fga = rng.poisson(pts / 1.1 + 1)
                        prow.append(dict(SEASON_ID=f"2{y}", PLAYER_ID=hash((team, k)) % 10**7,
                                         PLAYER_NAME=f"Player {team}-{k}", TEAM_ID=TEAMS.index(team),
                                         TEAM_ABBREVIATION=team, GAME_ID=gid, GAME_DATE=date.date(),
                                         MATCHUP=f"{team} {'vs.' if home else '@'} {opp}", MIN=round(mins),
                                         PTS=pts, REB=reb, AST=ast, FG3M=fg3, FGA=fga, FG3A=fg3 * 3,
                                         FTA=rng.poisson(2), TOV=rng.poisson(1.5), SEASON=s))
                        for c, v in (("PTS", pts), ("REB", reb), ("AST", ast), ("FG3M", fg3), ("FGA", fga)):
                            tot[c] += v
                    trow.append(dict(TEAM_ABBREVIATION=team, GAME_ID=gid, GAME_DATE=date.date(),
                                     MATCHUP=f"{team} {'vs.' if home else '@'} {opp}", SEASON=s, **tot))
    pd.DataFrame(prow).to_csv(DATA_RAW / "player_logs.csv", index=False)
    pd.DataFrame(trow).to_csv(DATA_RAW / "team_logs.csv", index=False)
    print(f"{len(prow):,} synthetic player-games written to {DATA_RAW}")


if __name__ == "__main__":
    main()
