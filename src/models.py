"""Models: a LightGBM point forecaster and a PyTorch heteroscedastic Gaussian network.

The network outputs a mean AND a variance for every stat, trained with the
Gaussian negative log-likelihood. The variance is what turns a projection
into a probability of going over a betting line.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- LightGBM
def lgbm_params(use_gpu: bool):
    p = dict(objective="regression", learning_rate=0.03, num_leaves=31,
             min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, verbose=-1)
    if use_gpu:
        # LightGBM's GPU build uses OpenCL; if it is not available we fall back
        p["device_type"] = "gpu"
    return p


def fit_lgbm(Xtr, ytr, Xva, yva, use_gpu=False):
    import lightgbm as lgb
    params = lgbm_params(use_gpu)
    dtr, dva = lgb.Dataset(Xtr, ytr), lgb.Dataset(Xva, yva)
    try:
        return lgb.train(params, dtr, 3000, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(100, verbose=False)])
    except Exception as e:
        if use_gpu:
            print(f"  LightGBM GPU unavailable ({e.__class__.__name__}), using CPU")
            return fit_lgbm(Xtr, ytr, Xva, yva, use_gpu=False)
        raise


# ---------------------------------------------------------------- Gaussian NN
class GaussianNet(nn.Module):
    def __init__(self, n_in, n_out, hidden=256, dropout=0.15):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
        )
        self.mu = nn.Linear(hidden // 2, n_out)
        self.logvar = nn.Linear(hidden // 2, n_out)

    def forward(self, x):
        h = self.body(x)
        return self.mu(h), self.logvar(h).clamp(-6, 6)


class Standardizer:
    def fit(self, X):
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0) + 1e-6
        return self

    def transform(self, X):
        Z = (X - self.mean) / self.std
        return np.nan_to_num(Z, nan=0.0)  # missing -> the (standardised) average


def fit_gaussian_net(Xtr, Ytr, Xva, Yva, epochs=200, batch=1024, lr=2e-3, patience=15, seed=0):
    torch.manual_seed(seed)
    sx = Standardizer().fit(Xtr)
    ymu, ysd = np.nanmean(Ytr, 0), np.nanstd(Ytr, 0) + 1e-6
    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    xtr, xva = to_t(sx.transform(Xtr)), to_t(sx.transform(Xva))
    ytr, yva = to_t((Ytr - ymu) / ysd), to_t((Yva - ymu) / ysd)

    net = GaussianNet(xtr.shape[1], ytr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    def nll(x, y):
        mu, lv = net(x)
        return (0.5 * (lv + (y - mu) ** 2 / lv.exp())).mean()

    best, best_state, bad = np.inf, None, 0
    n = len(xtr)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = nll(xtr[idx], ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = nll(xva, yva).item()
        sched.step(v)
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return {"net": net, "sx": sx, "ymu": ymu, "ysd": ysd, "val_nll": best, "epochs": ep + 1}


def predict_gaussian(bundle, X):
    net = bundle["net"].eval()
    x = torch.tensor(bundle["sx"].transform(X), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        mu, lv = net(x)
    mu = mu.cpu().numpy() * bundle["ysd"] + bundle["ymu"]
    sd = np.exp(0.5 * lv.cpu().numpy()) * bundle["ysd"]
    return mu, sd


# ---------------------------------------------------------------- probabilities
def prob_over(mu, sd, line):
    """P(stat > line). Lines are usually x.5, which acts as a continuity correction."""
    return 1.0 - norm.cdf((line - mu) / np.maximum(sd, 1e-6))


def american_to_prob(odds):
    odds = np.asarray(odds, dtype=float)
    return np.where(odds < 0, -odds / (-odds + 100), 100 / (odds + 100))


def american_to_decimal(odds):
    odds = np.asarray(odds, dtype=float)
    return np.where(odds < 0, 1 + 100 / -odds, 1 + odds / 100)


def kelly_fraction(p, decimal_odds):
    b = decimal_odds - 1
    return np.clip((b * p - (1 - p)) / b, 0, None)
