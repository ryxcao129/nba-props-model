"""Walk-forward evaluation, then fit and save the final models.

For each test season S we train only on seasons before S (the last of them is
used for early stopping) and score every game in S. This mimics how the model
would actually have been used, with no peeking at the future.

Usage:
    python src/train.py                 # evaluate on the last 2 seasons, then fit final models
    python src/train.py --test-seasons 3 --lgbm-gpu
"""
import argparse
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import DATA_PROC, DATA_RAW, MIN_MINUTES, MIN_PRIOR_GAMES, MODELS, REPORTS, TARGETS
from features import build_features, feature_columns
from models import DEVICE, fit_gaussian_net, fit_lgbm, predict_gaussian, prob_over


def load():
    p = pd.read_csv(DATA_RAW / "player_logs.csv")
    t = pd.read_csv(DATA_RAW / "team_logs.csv")
    df = build_features(p, t)
    df.to_parquet(DATA_PROC / "features.parquet") if _has_parquet() else df.to_csv(DATA_PROC / "features.csv", index=False)
    return df


def _has_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def trainable(df):
    m = (pd.to_numeric(df["MIN"], errors="coerce") >= MIN_MINUTES) & (df["PRIOR_GAMES"] >= MIN_PRIOR_GAMES)
    return df[m].copy()


def fit_all(train, val, feats, lgbm_gpu):
    Xtr, Xva = train[feats].to_numpy(float), val[feats].to_numpy(float)
    lgbms = {t: fit_lgbm(Xtr, train[t].to_numpy(float), Xva, val[t].to_numpy(float), lgbm_gpu) for t in TARGETS}
    nn_bundle = fit_gaussian_net(Xtr, train[TARGETS].to_numpy(float), Xva, val[TARGETS].to_numpy(float))
    return lgbms, nn_bundle


def predict_all(lgbms, nn_bundle, X):
    lg = np.column_stack([lgbms[t].predict(X, num_iteration=lgbms[t].best_iteration) for t in TARGETS])
    mu_nn, sd_nn = predict_gaussian(nn_bundle, X)
    ens = 0.5 * lg + 0.5 * mu_nn
    return lg, mu_nn, sd_nn, ens


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 30:
            rows.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return pd.DataFrame(rows, columns=["pred", "actual", "n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-seasons", type=int, default=2)
    ap.add_argument("--lgbm-gpu", action="store_true", help="LightGBM on GPU (needs OpenCL build)")
    args = ap.parse_args()

    print(f"Neural net device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    df = trainable(load())
    feats = feature_columns(df)
    seasons = sorted(df["SEASON"].unique())
    tests = seasons[-args.test_seasons:]
    print(f"{len(df):,} player-games, {len(feats)} features, seasons {seasons[0]}..{seasons[-1]}; testing on {tests}")

    rows, calib = [], []
    for S in tests:
        prior = [s for s in seasons if s < S]
        if len(prior) < 2:
            print(f"Skipping {S}: need 2+ earlier seasons")
            continue
        train = df[df["SEASON"].isin(prior[:-1])]
        val = df[df["SEASON"] == prior[-1]]
        test = df[df["SEASON"] == S]
        print(f"\n== Test {S}: train {len(train):,}, val {len(val):,}, test {len(test):,}")
        lgbms, nnb = fit_all(train, val, feats, args.lgbm_gpu)
        X = test[feats].to_numpy(float)
        lg, mu_nn, sd_nn, ens = predict_all(lgbms, nnb, X)
        Y = test[TARGETS].to_numpy(float)

        for j, t in enumerate(TARGETS):
            base = test[f"{t}_r10"].to_numpy(float)
            ok = ~np.isnan(base)
            # proxy betting line: player's season-to-date average, rounded to x.5
            line = np.floor(test[f"{t}_std"].fillna(test[f"{t}_r10"]).to_numpy(float)) + 0.5
            p = prob_over(ens[:, j], sd_nn[:, j], line)
            hit = (Y[:, j] > line).astype(float)
            rows.append({
                "season": S, "stat": t,
                "MAE_baseline_r10": np.mean(np.abs(base[ok] - Y[ok, j])),
                "MAE_lgbm": np.mean(np.abs(lg[:, j] - Y[:, j])),
                "MAE_nn": np.mean(np.abs(mu_nn[:, j] - Y[:, j])),
                "MAE_ensemble": np.mean(np.abs(ens[:, j] - Y[:, j])),
                "Brier_model": np.mean((p - hit) ** 2),
                "Brier_coinflip": np.mean((0.5 - hit) ** 2),
                "coverage_80": np.mean(np.abs(Y[:, j] - ens[:, j]) <= 1.2816 * sd_nn[:, j]),
            })
            calib.append(pd.DataFrame({"stat": t, "p": p, "y": hit}))
        print(pd.DataFrame(rows).query("season == @S").round(3).to_string(index=False))

    met = pd.DataFrame(rows)
    met.to_csv(REPORTS / "walk_forward_metrics.csv", index=False)
    summary = met.groupby("stat").mean(numeric_only=True)
    summary["MAE_improvement_vs_baseline_%"] = 100 * (1 - summary["MAE_ensemble"] / summary["MAE_baseline_r10"])
    print("\n== Average across test seasons\n" + summary.round(3).to_string())
    summary.round(4).to_csv(REPORTS / "summary.csv")

    # reliability diagram: when the model says 60%, does it hit ~60%?
    cal = pd.concat(calib)
    fig, ax = plt.subplots(figsize=(5, 5))
    for t in TARGETS:
        r = reliability(cal.loc[cal.stat == t, "p"].to_numpy(), cal.loc[cal.stat == t, "y"].to_numpy())
        ax.plot(r["pred"], r["actual"], marker="o", label=t)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="Predicted P(over line)", ylabel="Observed frequency", title="Calibration (walk-forward)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "calibration.png", dpi=150)

    # final models on all seasons (last season = early-stopping set)
    print("\nFitting final models on all data ...")
    lgbms, nnb = fit_all(df[df["SEASON"].isin(seasons[:-1])], df[df["SEASON"] == seasons[-1]], feats, args.lgbm_gpu)
    for t, m in lgbms.items():
        m.save_model(str(MODELS / f"lgbm_{t}.txt"), num_iteration=m.best_iteration)
    torch.save(nnb["net"].state_dict(), MODELS / "gaussian_net.pt")
    with open(MODELS / "meta.pkl", "wb") as f:
        pickle.dump({"features": feats, "sx": nnb["sx"], "ymu": nnb["ymu"], "ysd": nnb["ysd"],
                     "n_in": len(feats), "n_out": len(TARGETS)}, f)

    imp = pd.Series(lgbms["PTS"].feature_importance("gain"), index=feats).nlargest(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.plot.barh(ax=ax)
    ax.set_title("Top features for points (LightGBM gain)")
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_importance_pts.png", dpi=150)
    print(f"Saved models to {MODELS} and reports to {REPORTS}")


if __name__ == "__main__":
    main()
