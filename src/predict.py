"""Project upcoming games and compare against sportsbook lines.

Fill in data/lines.csv (see lines_template.csv), one row per prop:
    date,player,opponent,home,stat,line,over_odds,under_odds
    2026-11-06,Jayson Tatum,MIA,1,PTS,27.5,-115,-105

Leave stat/line/odds blank for a row to just get a projection (handy for fantasy).

Usage:
    python src/predict.py                    # reads data/lines.csv
    python src/predict.py --min-edge 0.04    # only flag edges of 4+ points of probability
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import torch

from common import DATA_RAW, MODELS, REPORTS, ROOT, TARGETS
from features import append_pending_games, build_features
from models import (DEVICE, GaussianNet, american_to_decimal, american_to_prob,
                    kelly_fraction, prob_over)


def load_models():
    import lightgbm as lgb
    meta = pickle.load(open(MODELS / "meta.pkl", "rb"))
    lgbms = {t: lgb.Booster(model_file=str(MODELS / f"lgbm_{t}.txt")) for t in TARGETS}
    net = GaussianNet(meta["n_in"], meta["n_out"]).to(DEVICE)
    net.load_state_dict(torch.load(MODELS / "gaussian_net.pt", map_location=DEVICE))
    net.eval()
    return meta, lgbms, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default=str(ROOT / "data" / "lines.csv"))
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--kelly-mult", type=float, default=0.25, help="fraction of full Kelly to suggest")
    args = ap.parse_args()

    lines = pd.read_csv(args.lines)
    lines["date"] = pd.to_datetime(lines["date"])
    pend = (lines[["player", "date", "opponent", "home"]].drop_duplicates()
            .rename(columns={"player": "PLAYER_NAME", "date": "GAME_DATE", "opponent": "OPP", "home": "HOME"}))

    plogs = pd.read_csv(DATA_RAW / "player_logs.csv")
    plogs["GAME_DATE"] = pd.to_datetime(plogs["GAME_DATE"])
    tlogs = pd.read_csv(DATA_RAW / "team_logs.csv")
    df = build_features(append_pending_games(plogs, pend), tlogs)
    up = df[df["PENDING"] == 1].copy()

    meta, lgbms, net = load_models()
    X = up[meta["features"]].to_numpy(float)
    lg = np.column_stack([lgbms[t].predict(X) for t in TARGETS])
    with torch.no_grad():
        mu, lv = net(torch.tensor(meta["sx"].transform(X), dtype=torch.float32, device=DEVICE))
    mu_nn = mu.cpu().numpy() * meta["ysd"] + meta["ymu"]
    sd = np.exp(0.5 * lv.cpu().numpy()) * meta["ysd"]
    ens = 0.5 * lg + 0.5 * mu_nn

    proj = up[["PLAYER_NAME", "GAME_DATE", "OPP", "HOME"]].reset_index(drop=True)
    for j, t in enumerate(TARGETS):
        proj[f"{t}_proj"] = ens[:, j].round(1)
        proj[f"{t}_sd"] = sd[:, j].round(1)
    proj.to_csv(REPORTS / "projections.csv", index=False)
    print("\nProjections\n" + proj.drop(columns=[c for c in proj if c.endswith("_sd")]).to_string(index=False))

    props = lines.dropna(subset=["stat", "line"]).merge(
        proj, left_on=["player", "date"], right_on=["PLAYER_NAME", "GAME_DATE"])
    if props.empty:
        return
    out = []
    for _, r in props.iterrows():
        t = r["stat"].upper()
        m, s = r[f"{t}_proj"], r[f"{t}_sd"]
        p_over = float(prob_over(m, s, r["line"]))
        # remove the bookmaker's margin so the two implied probabilities sum to 1
        po, pu = american_to_prob(r["over_odds"]), american_to_prob(r["under_odds"])
        fair_over = float(po / (po + pu))
        side = "OVER" if p_over > fair_over else "UNDER"
        p_side = p_over if side == "OVER" else 1 - p_over
        odds = r["over_odds"] if side == "OVER" else r["under_odds"]
        edge = p_side - (fair_over if side == "OVER" else 1 - fair_over)
        k = float(kelly_fraction(p_side, american_to_decimal(odds))) * args.kelly_mult
        out.append({"player": r["player"], "stat": t, "line": r["line"], "proj": m, "sd": s,
                    "model_p_over": round(p_over, 3), "fair_p_over": round(fair_over, 3),
                    "side": side, "edge": round(edge, 3), "stake_%bankroll": round(100 * k, 2),
                    "bet": "YES" if edge >= args.min_edge else ""})
    res = pd.DataFrame(out).sort_values("edge", ascending=False)
    res.to_csv(REPORTS / "edges.csv", index=False)
    print("\nProp edges (model vs no-vig market)\n" + res.to_string(index=False))


if __name__ == "__main__":
    main()
