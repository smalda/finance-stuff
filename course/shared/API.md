# Shared Library API Reference

> **Auto-generated** by `generate_api.py`. Do not edit manually.
> Regenerate: `python3 course/shared/generate_api.py`

This file is consumed by Step 4A agents to discover available shared
infrastructure. Import from specific modules:
```python
from shared.metrics import pearson_ic, rank_ic, deflated_sharpe_ratio
from shared.temporal import PurgedWalkForwardCV, PurgedKFold
from shared.backtesting import long_short_portfolio, sharpe_ratio
```

---

## Data Modules

### `shared/data` — Cross-week download cache and universe constants

**Constants:**
- *ALL*: `ALL_EQUITY_TICKERS`, `ALL_TICKERS`
- *FRED*: `FRED_TREASURY_YIELDS`, `FRED_TIPS_YIELDS`, `FRED_RATES`, `FRED_VOLATILITY`, `FRED_MACRO`, `FRED_CREDIT`, `FRED_HOUSING`, `FRED_LABOR`, `FRED_FINANCIAL_CONDITIONS`, `FRED_DOLLAR_FX`, `FRED_COMMODITIES_PRICES`, `FRED_INFLATION_EXPECTATIONS`, `FRED_ALL`
- *Other*: `SHARED_CACHE`, `SP500_TICKERS`, `DEMO_TICKERS`, `SP400_TICKERS`, `CORE_ETFS`, `SECTOR_ETFS`, `FACTOR_ETFS`, `FIXED_INCOME_ETFS`, `INTERNATIONAL_ETFS`, `COMMODITY_ETFS`, `VOLATILITY_ETFS`, `REIT_ETFS`, `THEMATIC_ETFS`, `ETF_TICKERS`, `CRYPTO_TICKERS`, `KF_PORTFOLIOS`

- **`load_sp500_prices`**`(start: str | None = None, end: str | None = None, tickers: list[str] | None = None, min_completeness: float = 0.5) -> pd.DataFrame` — Load daily adjusted close prices from the shared cache.
- **`load_sp500_ohlcv`**`(start: str | None = None, end: str | None = None, tickers: list[str] | None = None, fields: list[str] | None = None, min_completeness: float = 0.5) -> pd.DataFrame` — Load full OHLCV data from the shared cache.
- **`load_ff_factors`**`(model: str = '5', frequency: str = 'M') -> pd.DataFrame` — Load Fama-French factor data from the shared cache.
- **`load_carhart_factors`**`(frequency: str = 'M') -> pd.DataFrame` — Load Carhart 4-factor model data from the shared cache.
- **`load_ff_portfolios`**`(name: str = '25_size_bm', frequency: str = 'M') -> pd.DataFrame` — Load Ken French sorted portfolio returns from the shared cache.
- **`load_sp500_fundamentals`**`(tickers: list[str] | None = None, pit_lag_days: int = 0) -> dict` — Load fundamental data from the shared cache.
  - Returns dict: `'balance_sheet'`, `'income_stmt'`, `'cashflow'`, `'sector'`, `'industry'`, `'market_cap'`, `'shares'`, `'ratios'`
- **`load_fred_series`**`(series_ids: list[str], start: str = '2000-01-01', end: str = '2025-12-31') -> pd.DataFrame` — Load FRED economic data series from shared cache.
- **`load_crypto_prices`**`(start: str | None = None, end: str | None = None, tickers: list[str] | None = None) -> pd.DataFrame` — Load crypto/DeFi token daily close prices from the shared cache.
- **`load_options_chain`**`(tickers: list[str] | None = None, near_expiries: int = 4, max_age_days: int = 7) -> pd.DataFrame` — Load options chain data (current snapshot) from yfinance.
- **`load_sec_filings`**`(tickers: list[str] | None = None, filing_type: str = '10-K', max_filings: int = 5) -> dict[str, list[dict]]` — Download SEC filing metadata and full text from EDGAR.
- **`generate_synthetic_lob`**`(n_levels: int = 10, n_snapshots: int = 10000, mid_price: float = 100.0, tick_size: float = 0.01, avg_spread_ticks: float = 3.0, seed: int = 42) -> pd.DataFrame` — Generate synthetic limit order book snapshots.

---

## Core Modules

### `shared/metrics` — Signal quality, statistical tests, deflated Sharpe

- **`pearson_ic`**`(predicted: np.ndarray, actual: np.ndarray) -> float` — Pearson correlation between predicted and realized values.
- **`rank_ic`**`(predicted: np.ndarray, actual: np.ndarray) -> float` — Spearman rank correlation — robust IC variant.
- **`icir`**`(ic_series: np.ndarray) -> float` — Information Coefficient Information Ratio: mean(IC) / std(IC).
- **`hit_rate`**`(predicted: np.ndarray, actual: np.ndarray) -> float` — Fraction of correctly predicted return directions.
- **`r_squared_oos`**`(predicted: np.ndarray, actual: np.ndarray) -> float` — Out-of-sample R-squared (Campbell & Thompson, 2008).
- **`deflated_sharpe_ratio`**`(observed_sr: float, n_trials: int, n_obs: int, *, skew: float = 0.0, excess_kurt: float = 0.0) -> float` — Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
- **`calibration_error`**`(predicted_probs: np.ndarray, actual_outcomes: np.ndarray, n_bins: int = 10) -> float` — Expected Calibration Error for probabilistic predictions.
- **`ic_summary`**`(ic_series: np.ndarray) -> dict` — Compute standard IC statistics from a time series of monthly ICs.
  - Returns dict: `mean_ic`, `std_ic`, `icir`, `pct_positive`, `t_stat`, `p_value`, `significant_5pct`, `n`
- **`prediction_quality`**`(predicted: np.ndarray, actual: np.ndarray) -> dict` — Check whether model predictions are meaningfully differentiated.
  - Returns dict: `spread_ratio`, `unique_ratio`, `range_ratio`
- **`min_track_record_length`**`(observed_sr: float, n_trials: int = 1, *, benchmark_sr: float = 0.0, skew: float = 0.0, excess_kurt: float = 0.0, confidence: float = 0.95, max_T: int = 1200) -> float` — Minimum Track Record Length (Bailey & Lopez de Prado, 2014).
- **`probability_of_backtest_overfitting`**`(model_ic_dict: dict, n_splits: int = 6, n_test_splits: int = 2, purge_gap: int = 1) -> tuple` — Probability of Backtest Overfitting via Combinatorial Purged CV.
- **`vs_naive_baseline`**`(ic_model: np.ndarray, ic_naive: np.ndarray) -> dict` — Paired test: is the model's IC series significantly better than naive?
  - Returns dict: `mean_improvement`, `t_stat`, `p_value`, `significant_5pct`, `n`
- **`compute_ic_series`**`(predictions: pd.Series, actuals: pd.Series, method: str = 'pearson') -> pd.Series` — Compute per-period cross-sectional IC from pre-computed predictions.

---

### `shared/temporal` — Temporal CV splitters (walk-forward, purged, CPCV)

- **`walk_forward_splits`**`(dates: np.ndarray, train_window: int, purge_gap: int = 1) -> Iterator[tuple[np.ndarray, Any]]` — Yield (train_dates, pred_date) for rolling walk-forward evaluation.
- **`expanding_window_splits`**`(dates: np.ndarray, min_window: int, purge_gap: int = 1) -> Iterator[tuple[np.ndarray, Any]]` — Yield (train_dates, pred_date) for expanding-window evaluation.

- **`CombinatorialPurgedCV`**`(n_splits: int = 6, n_test_splits: int = 2, purge_gap: int = 1)` — Combinatorial Purged Cross-Validation (Lopez de Prado, 2018).
  - `.get_n_splits``(X=None, y=None, groups=None) -> int`
  - `.split``(X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]`
- **`PurgedKFold`**`(n_splits: int = 5, label_duration: int = 1, embargo: int = 1)` — K-Fold cross-validator with label-aware purging and embargo.
  - `.get_n_splits``(X=None, y=None, groups=None) -> int`
  - `.split``(X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]` — Generate purged k-fold train/test indices.
- **`PurgedWalkForwardCV`**`(n_splits: int = 3, purge_gap: int = 1)` — Walk-forward cross-validator with purge gap.
  - `.get_n_splits``(X=None, y=None, groups=None) -> int`
  - `.split``(X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]`

---

### `shared/evaluation` — Rolling cross-sectional prediction harness

- **`rolling_predict`**`(features: pd.DataFrame, target: pd.Series, predict_fn: Callable, train_window: int = 60, purge_gap: int = 1, expanding: bool = False, min_train_obs: int = 100, min_pred_obs: int = 30, verbose: bool = True) -> tuple[pd.DataFrame, pd.Series]` — Run walk-forward cross-sectional prediction, return IC series.

---

### `shared/backtesting` — Portfolio construction, performance, transaction costs

- **`quantile_portfolios`**`(predictions: pd.Series, returns: pd.Series, n_groups: int = 10) -> pd.DataFrame` — Sort stocks into quantile portfolios each period by predicted signal.
- **`long_short_returns`**`(predictions: pd.Series, returns: pd.Series, n_groups: int = 10) -> pd.Series` — Compute top-quantile minus bottom-quantile returns each period.
- **`portfolio_turnover`**`(predictions: pd.Series, n_groups: int = 10) -> pd.Series` — Monthly one-way turnover for an equal-weight long-short strategy.
- **`sharpe_ratio`**`(returns: pd.Series, periods_per_year: int = 12) -> float` — Annualized Sharpe ratio (excess returns assumed — rf already subtracted).
- **`cumulative_returns`**`(returns: pd.Series) -> pd.Series` — Cumulative wealth index from period returns.
- **`drawdown_series`**`(returns: pd.Series) -> pd.Series` — Running drawdown from peak (always <= 0).
- **`max_drawdown`**`(returns: pd.Series) -> float` — Maximum peak-to-trough drawdown (returned as a negative number).
- **`var_historical`**`(returns: pd.Series, alpha: float = 0.05) -> float` — Historical Value-at-Risk at confidence level (1 - alpha).
- **`expected_shortfall`**`(returns: pd.Series, alpha: float = 0.05) -> float` — Expected Shortfall (CVaR) — average loss beyond VaR.
- **`sortino_ratio`**`(returns: pd.Series, periods_per_year: int = 12) -> float` — Annualized Sortino ratio (penalizes downside volatility only).
- **`calmar_ratio`**`(returns: pd.Series, periods_per_year: int = 12) -> float` — Calmar ratio: annualized return / |max drawdown|.
- **`performance_summary`**`(returns: pd.Series, periods_per_year: int = 12, label: str = '') -> dict` — Comprehensive performance summary for a return series.
  - Returns dict: `label`, `sharpe`, `sortino`, `calmar`, `cagr`, `max_dd`, `skewness`, `excess_kurtosis`, `n_periods`
- **`net_returns`**`(gross_returns: pd.Series, turnover: pd.Series, cost_bps: float) -> pd.Series` — Subtract transaction costs from gross returns.
- **`breakeven_cost`**`(gross_returns: pd.Series, turnover: pd.Series, max_bps: int = 200, periods_per_year: int = 12) -> float` — Find the one-way transaction cost (in bps) at which net Sharpe = 0.

---

### `shared/dl_training` — Neural network fit/predict utilities

- **`fit_nn`**`(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, *, epochs: int = 50, lr: float = 0.001, batch_size: int = 256, weight_decay: float = 0.0001, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None, patience: int = 10, device: str | torch.device | None = None) -> dict` — Train a PyTorch model with optional early stopping.
  - Returns dict: `final_epoch`, `train_loss`, `val_loss`, `device`
- **`predict_nn`**`(model: nn.Module, x: np.ndarray, device: str | torch.device | None = None) -> np.ndarray` — Get predictions from a trained model.

- **`SequenceDataset`**`(features: np.ndarray, targets: np.ndarray, seq_len: int = 20)` — Sliding-window dataset for time-series DL (LSTM, TCN, Transformer).
- **`SklearnNNRegressor`**`(n_features: int, hidden: int = 32, dropout: float = 0.3, lr: float = 0.001, epochs: int = 50, batch_size: int = 256, patience: int = 10, weight_decay: float = 0.0001, val_frac: float = 0.2, device: str | None = None)` — Scikit-learn-compatible wrapper for PyTorch feedforward networks.
  - `.fit``(X, y, **kwargs)` — Train the network with early stopping.
  - `.predict``(X)` — Generate predictions.
  - `.get_params``(deep=True)` — Return estimator parameters (sklearn protocol).
  - `.set_params``(**params)` — Set estimator parameters (sklearn protocol).

---

## Domain Modules

### `shared/portfolio` — Portfolio optimization (Markowitz, HRP, Black-Litterman)

- **`shrink_covariance`**`(returns: pd.DataFrame, method: str = 'ledoit_wolf') -> np.ndarray` — Shrink sample covariance toward a structured target.
- **`mean_variance_weights`**`(mu: np.ndarray, cov: np.ndarray, *, target_vol: float | None = None, long_only: bool = True) -> np.ndarray` — Maximum-Sharpe (tangency) portfolio, optionally scaled to target vol.
- **`efficient_frontier`**`(mu: np.ndarray, cov: np.ndarray, n_points: int = 50, long_only: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — Compute the efficient frontier.
- **`risk_parity_weights`**`(cov: np.ndarray) -> np.ndarray` — Equal risk contribution portfolio weights.
- **`hierarchical_risk_parity`**`(returns: pd.DataFrame) -> pd.Series` — HRP portfolio weights via hierarchical clustering + inverse-variance bisection.
- **`black_litterman_posterior`**`(cov: np.ndarray, market_caps: np.ndarray, P: np.ndarray, Q: np.ndarray, *, omega: np.ndarray | None = None, tau: float = 0.05, risk_aversion: float = 2.5) -> np.ndarray` — Black-Litterman posterior expected returns.
- **`kelly_fraction`**`(mu: float, sigma_sq: float) -> float` — Full Kelly fraction for a single asset.
- **`half_kelly`**`(mu: float, sigma_sq: float) -> float` — Half-Kelly: common practitioner adjustment for estimation error.

---

### `shared/derivatives` — Options pricing, Greeks, implied volatility

- **`black_scholes_price`**`(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float` — Closed-form Black-Scholes European option price.
- **`bs_greeks`**`(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> dict` — Compute all standard Black-Scholes Greeks.
  - Returns dict: `delta`, `gamma`, `vega`, `theta`, `rho`
- **`implied_volatility`**`(market_price: float, S: float, K: float, T: float, r: float, option_type: str = 'call', *, tol: float = 1e-08, max_iter: int = 100) -> float` — Newton-Raphson implied volatility solver.
- **`monte_carlo_option`**`(payoff_fn, S0: float, r: float, sigma: float, T: float, *, n_paths: int = 50000, n_steps: int = 252, rng: np.random.Generator | None = None) -> dict` — Monte Carlo option pricing with geometric Brownian motion paths.
  - Returns dict: `price`, `std_error`, `paths`
- **`binomial_tree`**`(S: float, K: float, T: float, r: float, sigma: float, *, n_steps: int = 100, option_type: str = 'call', american: bool = False) -> float` — Cox-Ross-Rubinstein binomial tree option pricing.

---

### `shared/microstructure` — Order flow, spreads, execution models

- **`compute_ofi`**`(bid_price: np.ndarray, ask_price: np.ndarray, bid_size: np.ndarray, ask_size: np.ndarray) -> np.ndarray` — Compute Order Flow Imbalance from L1 quote snapshots.
- **`quoted_spread`**`(bid_price: np.ndarray, ask_price: np.ndarray) -> np.ndarray` — Quoted bid-ask spread in price units.
- **`relative_spread`**`(bid_price: np.ndarray, ask_price: np.ndarray) -> np.ndarray` — Relative spread: (ask - bid) / midpoint.
- **`effective_spread`**`(trade_price: np.ndarray, midpoint: np.ndarray, side: np.ndarray) -> np.ndarray` — Effective spread: 2 * side * (trade_price - midpoint).
- **`vwap`**`(prices: np.ndarray, volumes: np.ndarray) -> float` — Volume-weighted average price.
- **`almgren_chriss_trajectory`**`(total_shares: int, n_periods: int, sigma: float, eta: float, lam: float) -> np.ndarray` — Optimal execution trajectory (Almgren-Chriss, 2001).
- **`kyle_lambda`**`(prices: np.ndarray, order_flow: np.ndarray) -> float` — Estimate Kyle's lambda (price impact coefficient).

---

### `shared/regime` — Regime detection, cointegration, mean reversion

- **`ou_estimate`**`(spread: np.ndarray, dt: float = 1.0) -> dict` — Estimate Ornstein-Uhlenbeck parameters from a mean-reverting spread.
  - Returns dict: `theta`, `mu`, `sigma`, `half_life`
- **`engle_granger_cointegration`**`(y: np.ndarray, x: np.ndarray, significance: float = 0.05) -> dict` — Engle-Granger two-step cointegration test.
  - Returns dict: `intercept`, `hedge_ratio`, `spread`, `adf_statistic`, `pvalue`, `cointegrated`
- **`fit_gaussian_hmm`**`(returns: np.ndarray, n_states: int = 2, n_iter: int = 100, random_state: int = 42) -> dict` — Fit a Gaussian HMM to a return series for regime detection.
  - Returns dict: `model`, `states`, `means`, `variances`, `transition_matrix`
- **`detect_changepoints`**`(series: np.ndarray, method: str = 'pelt', penalty: float | None = None) -> list[int]` — Detect structural changepoints in a time series.

---

## Stub Modules

### `shared/nlp` — NLP text processing (partial impl + stubs)

- **`clean_financial_text`**`(text: str) -> str` — Clean financial document text for NLP processing.
- **`split_sentences`**`(text: str) -> list[str]` — Split financial text into sentences.
- **`extract_sentiment_finbert`**`(texts: list[str], model_name: str = 'ProsusAI/finbert', batch_size: int = 32) -> list[dict]` — Extract sentiment scores using FinBERT.
- **`embed_texts`**`(texts: list[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray` — Embed texts using sentence-transformers.

---

### `shared/causal` — Causal inference (stubs)

- **`build_causal_graph`**`(edges: list[tuple[str, str]]) -> object` — Build a causal DAG from an edge list.
- **`estimate_ate`**`(data: pd.DataFrame, treatment: str, outcome: str, confounders: list[str], *, method: str = 'doubly_robust') -> dict` — Estimate average treatment effect with confounding adjustment.
- **`detect_collider_bias`**`(data: pd.DataFrame, x: str, y: str, collider: str) -> dict` — Demonstrate collider bias by conditioning on a collider variable.
  - Returns dict: `unconditional_corr`, `conditional_corr`, `bias_magnitude`

---

### `shared/rl_env` — RL trading environments (stubs)

- **`TradingEnv`**`(**kwargs) -> None` — Financial trading environment for RL agents.
  - `.reset``(**kwargs) -> tuple[Any, dict]`
  - `.step``(action: Any) -> tuple[Any, float, bool, bool, dict]`
- **`ExecutionEnv`**`(**kwargs) -> None` — Optimal execution environment for RL agents.
  - `.reset``(**kwargs) -> tuple[Any, dict]`
  - `.step``(action: Any) -> tuple[Any, float, bool, bool, dict]`

---
