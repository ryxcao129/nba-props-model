# NBA Player Props Model

Predicts points, rebounds, assists and threes for upcoming NBA games, turns each projection into a
probability of going over a sportsbook line, and flags bets where the model disagrees with the
market by more than a threshold. Also gives plain projections for fantasy lineups.

## What's inside (the ML)

| Piece | What it does |
| --- | --- |
| Leak-free features (`features.py`) | Rolling 5/10/20-game form, season-to-date averages, volatility, trend, rest days, back-to-backs, home/away, and opponent defence as of the day before tip-off. Every feature for a game uses only earlier games. |
| LightGBM (`models.py`) | Gradient-boosted trees for the point forecast of each stat, with early stopping. |
| Heteroscedastic Gaussian network (`models.py`) | PyTorch MLP that outputs a **mean and a variance** for all four stats at once, trained with Gaussian negative log-likelihood. The variance is player- and game-specific, so a volatile scorer gets wider uncertainty than a steady one. Trains on your GPU. |
| Ensemble | Average of the two means; uncertainty from the network. |
| Walk-forward validation (`train.py`) | For each test season, train only on earlier seasons. Reports MAE against a rolling-average baseline, Brier score, 80% interval coverage, and a calibration plot. |
| Betting layer (`predict.py`) | Removes the bookmaker's margin to get fair probabilities, computes edge, and suggests a fractional-Kelly stake. |

## Setup (once, in Anaconda Prompt)

```bash
conda create -n nba python=3.11 -y
conda activate nba
cd %USERPROFILE%\Documents\ml_projects\nba_props
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line should print `True NVIDIA GeForce RTX 4050 Laptop GPU`. If it prints `False`, update
your NVIDIA driver and reinstall torch with the command from pytorch.org.

## Run it

```bash
python src/fetch_data.py --start 2018 --end 2025   # ~2-3 min, downloads 8 seasons
python src/train.py                                # walk-forward eval + final models (GPU)
```

Results land in `reports/`: `summary.csv`, `walk_forward_metrics.csv`, `calibration.png`,
`feature_importance_pts.png`.

Before each slate:

1. `python src/fetch_data.py --start 2025 --end 2025` refreshes the current season; older seasons
   already downloaded are kept. Re-run `train.py` every week or two so the models see recent games.
2. Copy `data/lines_template.csv` to `data/lines.csv` and fill in the props you're looking at.
   Player names must match NBA.com spelling. `home` is 1 for a home game, 0 for away.
3. `python src/predict.py` prints projections and edges and saves `reports/edges.csv`.

To test everything without downloading anything: `python src/make_synthetic.py` then the steps above.

## Reading the output

- `model_p_over` vs `fair_p_over`: the model's probability vs the market's, with the vig removed.
- `edge`: how much more likely the model thinks your side is. Above ~3-5% is worth a look.
- `stake_%bankroll`: quarter-Kelly. Treat it as an upper bound.
- Very large edges (15%+) usually mean the model is missing news the market has: an injury, a
  minutes restriction, or a role change. Check before betting.

## Known limits / ideas to extend

- No injury or lineup data: the model doesn't know if a teammate is out. Adding a "usage share
  when star X sits" feature is the biggest likely gain.
- Counting stats are modelled as Gaussian. A negative-binomial head for assists and threes would
  fit low counts better.
- Historical prop lines aren't free, so calibration is tested against a proxy line (the player's
  season-to-date average). Logging your own lines and results each night builds a real backtest.
