"""Leak-free feature engineering.

Every feature for game t is computed only from games strictly before t:
rolling stats are computed per player and then shifted one game, and
opponent defence is joined "as of" the day before tip-off.
"""
import numpy as np
import pandas as pd

from common import FORM_COLS, OPP_COLS, TARGETS, WINDOWS


def parse_matchup(df: pd.DataFrame) -> pd.DataFrame:
    # MATCHUP looks like "LAL vs. BOS" (home) or "LAL @ BOS" (away)
    df = df.copy()
    df["HOME"] = df["MATCHUP"].str.contains("vs.").astype(int)
    df["OPP"] = df["MATCHUP"].str.split().str[-1]
    return df


def opponent_defence(team_logs: pd.DataFrame) -> pd.DataFrame:
    """For each team and date: rolling averages of what it ALLOWED, through that date."""
    t = team_logs[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE"] + OPP_COLS].copy()
    t["GAME_DATE"] = pd.to_datetime(t["GAME_DATE"])
    # pair each team with the other team in the same game -> stats allowed
    opp = t.rename(columns={c: f"ALLOW_{c}" for c in OPP_COLS})
    opp = opp.rename(columns={"TEAM_ABBREVIATION": "OPP_TEAM"})
    m = t.merge(opp[["GAME_ID", "OPP_TEAM"] + [f"ALLOW_{c}" for c in OPP_COLS]], on="GAME_ID")
    m = m[m["TEAM_ABBREVIATION"] != m["OPP_TEAM"]]
    # here ALLOW_x = what the *other* team scored, i.e. what TEAM_ABBREVIATION allowed
    m = m.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    g = m.groupby("TEAM_ABBREVIATION")
    out = m[["TEAM_ABBREVIATION", "GAME_DATE"]].copy()
    for c in OPP_COLS:
        out[f"OPP_ALLOW_{c}_10"] = g[f"ALLOW_{c}"].transform(lambda s: s.rolling(10, min_periods=3).mean())
    return out.rename(columns={"TEAM_ABBREVIATION": "OPP"})


def build_features(player_logs: pd.DataFrame, team_logs: pd.DataFrame) -> pd.DataFrame:
    df = parse_matchup(player_logs)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    g = df.groupby("PLAYER_ID")

    feats = {}
    # rolling form, shifted so game t only sees games < t
    for c in FORM_COLS:
        prev = g[c].shift(1)
        for w in WINDOWS:
            feats[f"{c}_r{w}"] = prev.groupby(df["PLAYER_ID"]).transform(
                lambda s: s.rolling(w, min_periods=1).mean())
    for c in TARGETS:
        prev = g[c].shift(1)
        feats[f"{c}_sd10"] = prev.groupby(df["PLAYER_ID"]).transform(
            lambda s: s.rolling(10, min_periods=3).std())
        # season-to-date average (resets each season)
        feats[f"{c}_std"] = prev.groupby([df["PLAYER_ID"], df["SEASON"]]).transform(
            lambda s: s.expanding().mean())
    # trend: last 5 vs last 20
    for c in ["PTS", "MIN", "FGA"]:
        feats[f"{c}_trend"] = feats[f"{c}_r5"] - feats[f"{c}_r20"]

    F = pd.DataFrame(feats)
    df = pd.concat([df, F], axis=1)

    df["PRIOR_GAMES"] = g.cumcount()
    rest = g["GAME_DATE"].diff().dt.days
    df["REST_DAYS"] = rest.clip(upper=7).fillna(7)
    df["B2B"] = (rest == 1).astype(int)

    # opponent defence as of the day before the game (strictly earlier dates)
    od = opponent_defence(team_logs)
    for d in (df, od):  # align dtypes (pandas string vs object, datetime units)
        d["OPP"] = d["OPP"].astype(object)
        d["GAME_DATE"] = d["GAME_DATE"].astype("datetime64[ns]")
    od = od.sort_values("GAME_DATE")
    df = df.sort_values("GAME_DATE")
    df = pd.merge_asof(df, od, on="GAME_DATE", by="OPP",
                       allow_exact_matches=False, direction="backward")
    return df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list:
    import re
    pat = re.compile(r"_(r\d+|sd10|std|trend)$")
    keep = [c for c in df.columns if pat.search(c) or c.startswith("OPP_ALLOW_")]
    return keep + ["HOME", "REST_DAYS", "B2B", "PRIOR_GAMES"]


def append_pending_games(player_logs: pd.DataFrame, pending: pd.DataFrame) -> pd.DataFrame:
    """Add rows for upcoming games (no box score yet) so features can be built for them.

    pending needs: PLAYER_NAME, GAME_DATE, OPP, HOME
    """
    last = (player_logs.sort_values("GAME_DATE")
            .groupby("PLAYER_NAME").tail(1)
            .set_index("PLAYER_NAME"))
    rows = []
    for _, r in pending.iterrows():
        if r["PLAYER_NAME"] not in last.index:
            print(f"  ! no history for {r['PLAYER_NAME']}, skipping")
            continue
        base = last.loc[r["PLAYER_NAME"]]
        team = base["TEAM_ABBREVIATION"]
        sep = "vs." if int(r["HOME"]) == 1 else "@"
        rows.append({
            "PLAYER_ID": base["PLAYER_ID"], "PLAYER_NAME": r["PLAYER_NAME"],
            "TEAM_ABBREVIATION": team, "SEASON": base["SEASON"],
            "GAME_ID": f"PENDING_{r['PLAYER_NAME']}", "GAME_DATE": r["GAME_DATE"],
            "MATCHUP": f"{team} {sep} {r['OPP']}", "PENDING": 1,
        })
    pend = pd.DataFrame(rows)
    out = pd.concat([player_logs.assign(PENDING=0), pend], ignore_index=True)
    for c in FORM_COLS:
        out.loc[out["PENDING"] == 1, c] = np.nan
    return out
