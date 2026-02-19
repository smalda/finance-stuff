# course/shared — reusable ML infrastructure for the 18-week course
#
# Import from specific modules:
#   from shared.metrics import pearson_ic, rank_ic, icir
#   from shared.temporal import walk_forward_splits, PurgedWalkForwardCV
#
# Section files need course/ on sys.path. Convention:
#   sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
#   (adjusting depth for lecture/seminar/hw files under code/)
#
# ── Module catalog ──────────────────────────────────────────────
#
# DATA (shared cross-week download cache):
#   data.py           — S&P 500 universe, prices, FF factors, fundamentals,
#                       FRED series. Cache at .data_cache/. Used by each
#                       week's data_setup.py to avoid redundant downloads.
#
# CORE (fully implemented, used across many weeks):
#   metrics.py        — IC, rank IC, ICIR, hit rate, R²_OOS, deflated SR, ECE,
#                       MinTRL, PBO (probability of backtest overfitting),
#                       compute_ic_series (per-period IC from pre-computed preds)
#   temporal.py       — walk-forward / expanding splits, PurgedWalkForwardCV,
#                       CombinatorialPurgedCV, PurgedKFold
#   evaluation.py     — rolling cross-sectional prediction harness
#   backtesting.py    — quantile portfolios, long-short, turnover, Sharpe,
#                       Sortino, Calmar, performance_summary, cumulative
#                       returns, drawdowns, VaR, ES, breakeven_cost
#   dl_training.py    — fit/predict for feedforward NNs, SequenceDataset,
#                       SklearnNNRegressor (sklearn-compatible NN wrapper)
#
# DOMAIN (fully implemented, week-specific but imported downstream):
#   portfolio.py      — Markowitz, risk parity, HRP, Black-Litterman, Kelly
#   derivatives.py    — Black-Scholes, Greeks, IV, Monte Carlo, binomial tree
#   microstructure.py — OFI, spreads, VWAP (+ Almgren-Chriss stub)
#   regime.py         — OU estimation, Engle-Granger cointegration (+ HMM stub)
#
# STUBS (interface contracts, impl deferred to week blueprints):
#   nlp.py            — text cleaning (impl) + FinBERT/embeddings (stubs)
#   causal.py         — DAG construction, ATE estimation (stubs → dowhy/econml)
#   rl_env.py         — TradingEnv, ExecutionEnv (stubs → gymnasium)
