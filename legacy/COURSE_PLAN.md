# Machine Learning for Quantitative Finance
## A Lecture-Seminar-Homework Course (18 Weeks)

> **Design philosophy:** Important > Implementable > Realistic > Toy.
> Every homework must be completable on a MacBook Pro M4 (48 GB) or free-tier Google Colab.
> Python 3.10+ / PyTorch 2.x / scikit-learn throughout.

---

## Industry Stack Reference

Before diving in — here's what actual practitioners use, so you understand what you're learning relative to the real world.

| Role | Languages | Core ML Stack | Domain-Specific | Data |
|------|-----------|---------------|-----------------|------|
| **Quant Researcher** (hedge fund) | Python, C++, KDB+/q | XGBoost/LightGBM, PyTorch, scikit-learn | Qlib (MSFT), QuantLib, gs-quant | Bloomberg, Refinitiv, WRDS, internal tick DBs |
| **Algorithmic Trader** | Python, C++ | Gradient boosting, LSTM/GRU | Backtrader/LEAN/VectorBT, TA-Lib | Polygon.io, IB API, QuantConnect (400 TB) |
| **Financial ML Researcher** | Python, JAX | PyTorch, Transformers, FinRL | PyPortfolioOpt, ARCH, fracdiff | WRDS/CRSP, Kenneth French, SEC EDGAR |
| **Risk Analyst (ML)** | Python, R | PyMC, scikit-learn | QuantLib, Riskfolio-Lib | FRED, internal risk systems |
| **HFT / Market Maker** | C++, FPGA/Verilog, KDB+/q | Simple, fast models (linear, GBM) | Custom LOB engines, DPDK, Solarflare | NASDAQ ITCH, CME MDP, co-located feeds |
| **DeFi / Crypto Quant** | Python, Solidity | XGBoost, RL (SB3) | ccxt, web3.py, Dune Analytics | On-chain data, CEX APIs, Glassnode |

**What we'll use in this course:**
```
Core:        Python 3.10+ / PyTorch 2.x / scikit-learn / NumPy / Pandas
ML:          XGBoost, LightGBM, Hugging Face Transformers, Stable-Baselines3
GNNs:        PyTorch Geometric (torch-geometric)
LLMs:        FinGPT, OpenAI/Anthropic API (for embeddings), sentence-transformers
Finance:     yfinance, fredapi, fracdiff, ARCH, QuantLib-Python, Qlib
Backtesting: VectorBT (fast), QuantConnect/LEAN (production)
Analysis:    QuantStats, alphalens-reloaded, matplotlib, mplfinance
RL:          FinRL, Stable-Baselines3 (PyTorch backend), Gymnasium
TSFMs:       Kronos (finance-native), Chronos (Amazon), TimesFM (Google)
Crypto:      ccxt (exchange data), web3.py (on-chain), pycoingecko
Bayesian:    PyMC (optional), torch.nn.Dropout (MC Dropout)
```

### Where the $35B+ Goes: AI at Major Banks (2025-2026)

Understanding why banks invest helps you understand what the industry actually values vs. what academia focuses on.

| Institution | AI Budget | What They Built | Signal Generation via FM? |
|-------------|-----------|-----------------|--------------------------|
| **JPMorgan** | $2B/yr AI (of $18B tech) | LLM Suite (200K employees), DocLLM (document understanding, NeurIPS), IndexGPT (thematic investing), 450+ AI use cases | Experimental. LETS-C paper (ACL 2025) explores text embeddings for TS classification |
| **Goldman Sachs** | Undisclosed (46.5K employees on AI) | GS AI Assistant (GPT-4/Gemini/Claude), partnered w/ Anthropic for back-office agents, deployed Devin (autonomous coder) | AI trading system exists but relies on structured ML, not foundation models per se |
| **Morgan Stanley** | Undisclosed | OpenAI exclusive partner in wealth mgmt, 98% advisor adoption, AskResearchGPT (70K research reports) | No public evidence of FM-for-signals |
| **Two Sigma** | Undisclosed | Internal LLM "workbench" for parsing Fed speeches, research synthesis | **Yes — openly shifting from traditional ML to foundation models** for extracting predictive signals |
| **Citadel** | Undisclosed | AI assistant for scanning filings/research | CTO publicly skeptical: "not a source of enduring alpha" |
| **BNP Paribas** | Part of EUR 500M AI target | Mistral AI partnership, 750+ AI use cases, trade surveillance | Risk modeling pilots, not signal generation |

**The pattern:** ~90% of bank AI investment is **operational** (document processing, advisory, compliance, code generation). Trading signal generation via FMs is real but concentrated at quant hedge funds (Two Sigma, Man Group), not banks.

---

## Key Textbooks & References

| Priority | Book | Author | Year | Role in Course |
|----------|------|--------|------|----------------|
| **Primary** | *Advances in Financial Machine Learning* | Lopez de Prado | 2018 | Weeks 2-6, 18. Still the methodological foundation — no updated edition exists |
| **Primary** | *Machine Learning for Algorithmic Trading* (2nd ed.) | Stefan Jansen | 2020 | All weeks — implementation companion. No 3rd ed. yet; 2nd ed. still current |
| **Primary** | *Deep Learning in Quantitative Trading* | Zhang & Zohren | **2024** | Weeks 7-9, 14. **The most important new text.** Covers SOTA DL architectures for finance |
| **Primary** | *Causal Factor Investing* | Lopez de Prado | **2023** | Weeks 6, 18. Lopez de Prado's philosophical pivot — argues most factor research is associational, not causal |
| **Secondary** | *Empirical Asset Pricing via Machine Learning* (paper) | Gu, Kelly, Xiu | 2020 | Weeks 4-5, 7. Still THE benchmark. Extended by "Large (and Deep) Factor Models" (2024) |
| **Secondary** | *Expected Returns and Large Language Models* (paper) | Chen, Kelly, Xiu | **2023** | Week 10. Shows LLM embeddings significantly outperform older NLP for return prediction |
| **Secondary** | *Trading and Exchanges* | Larry Harris | 2003 | Weeks 1, 15. Dated but still the definitive market microstructure reference |
| **Secondary** | *High-Frequency Trading in a Limit Order Book* (paper) | Avellaneda & Stoikov | 2008 | Week 16. Foundational market making model |
| **Reference** | *Options, Futures, and Other Derivatives* | John Hull | 11th ed. | Week 14 |
| **Reference** | *Active Portfolio Management* | Grinold & Kahn | 1999 | Weeks 3, 18. Fundamental Law of Active Management |
| **Reference** | *Foundations of RL with Applications in Finance* (free PDF) | Ashwin Rao | 2022 | Week 13 |

### Key 2024-2026 Papers to Assign

| Paper | Authors | Year | Week |
|-------|---------|------|------|
| "Large (and Deep) Factor Models" | Kelly, Kuznetsov, Malamud, Xu | 2024 | 7 — shows NN depth matters; OOS performance increases up to ~100 layers |
| "Expected Returns and Large Language Models" | Chen, Kelly, Xiu | 2023 | 10 — LLM embeddings as SOTA for text-to-alpha |
| "Re(Visiting) Time Series Foundation Models in Finance" | Rahimikia, Ni, Wang | 2025 | 9 — generic TSFMs underperform; finance pre-training helps substantially |
| "Kronos: A Foundation Model for Financial Markets" | Various | AAAI 2026 | 9 — 93% RankIC improvement; finance-native K-line tokenization |
| "FinCast: Foundation Model for Financial Time-Series" | Various | CIKM 2025 | 9 — 1B param, 20-23% MSE reduction zero-shot on financial data |
| "Financial Fine-tuning a Large Time Series Model" | Fu, Hirano, Imajo | Dec 2024 | 9 — fine-tuned TimesFM achieves Sharpe 1.68 on S&P 500 |
| "LETS-C: Language Embeddings for Time Series Classification" | JPMorgan AI Research | ACL 2025 | 9/10 — text embeddings for time series classification |
| "Correcting the Factor Mirage" | Lopez de Prado, Zoonekynd | 2026 | 18 — causal protocol for factor investing |
| "A Survey of Large Language Models for Financial Applications" | Various | 2024 | 10 — comprehensive taxonomy |
| "Dropout as a Bayesian Approximation" | Gal & Ghahramani | 2016 | 11 — MC Dropout foundation |
| "Simple and Scalable Predictive Uncertainty" | Lakshminarayanan et al. | 2017 | 11 — Deep Ensembles |
| "Temporal Relational Ranking for Stock Prediction" | Feng et al. | 2019 | 12 — GNN+LSTM for stock ranking |
| "HIST: Mining Concept-Oriented Shared Information" | Xu et al. | 2021 | 12 — GNN SOTA on Qlib benchmarks |
| "MASTER: Market-Guided Stock Transformer" | Li et al. | AAAI 2024 | 12 — current GNN+Transformer SOTA |
| "DeepLOB: Deep CNNs for Limit Order Books" | Zhang, Zohren, Roberts | 2019 | 15 — reference DL model for LOB |
| "The Price Impact of Order Book Events" | Cont, Kukanov, Stoikov | 2014 | 15 — OFI as the key microstructure feature |
| "High-frequency trading in a limit order book" | Avellaneda & Stoikov | 2008 | 16 — foundational market making model |
| "RL to Improve Avellaneda-Stoikov" (Alpha-AS) | Falces et al. | PLOS ONE 2022 | 16 — RL vs analytical market making |

---

## Course Design Key

Each week is annotated with:

- **Course fit:** ★★★★★ = perfect fit for lecture-seminar-hw. ★★★☆☆ = needs adaptation.
- **Implementability:** ✅ fully on M4 | ⚡ benefits from Colab GPU | ☁️ requires Colab GPU
- **Importance:** 🔴 Critical | 🟡 Very Important | 🟢 Important
- **Format:** 📓 Full (lecture+seminar+homework) | 📖 Lecture+seminar only (no homework)

---

# MODULE 1: FINANCIAL FOUNDATIONS FOR ML ENGINEERS
*"You can't model what you don't understand."*

---

## Week 1 — Financial Markets, Data Structures & Microstructure

**Course fit:** ★★★★★ | **Implementability:** ✅ | **Importance:** 🔴 | **Format:** 📓

> This is the single most underestimated topic by ML people entering finance.
> Most ML-for-finance failures come from misunderstanding the data, not from bad models.

### What specialists use
Market data vendors (Bloomberg, Refinitiv), KDB+/q for tick databases, FIX protocol for order routing. We substitute: `yfinance`, `fredapi`, LOBSTER dataset samples.

### Lecture (90 min)
1. **How financial markets actually work** — order books, limit vs. market orders, bid-ask spread, market makers, exchanges vs. dark pools
2. **Financial data types and their quirks:**
   - Price data: OHLCV bars (time, tick, volume, dollar bars — Lopez de Prado Ch. 2)
   - Fundamental data: earnings, book value, point-in-time problem
   - Alternative data: news, satellite, social media
3. **Data pathologies ML people must know:**
   - Survivorship bias (only currently-listed stocks in Yahoo Finance)
   - Look-ahead bias (using data before it was actually available)
   - Non-stationarity of prices (why you can't feed raw prices into a model)
   - Fat tails and volatility clustering (returns are NOT Gaussian)
4. **Returns math:** simple vs. log returns, compounding, annualization

### Seminar (90 min)
- **Hands-on:** Download S&P 500 data with `yfinance`, compute returns, visualize distributional properties
- **Exercise:** Construct time bars, volume bars, and dollar bars from tick data (use simulated or LOBSTER sample data). Show that volume bars produce returns closer to Gaussian (replicate Lopez de Prado Fig. 2.4)
- **Exercise:** Demonstrate survivorship bias — download a historical list of S&P 500 constituents, compare backtesting results with vs. without delisted stocks
- **Discussion:** Why is the bid-ask spread a hidden cost? Calculate round-trip costs for various instruments

### Homework
**"Data quality audit"** — Given a dataset of daily stock prices for 200 US equities (2010–2024, via `yfinance`):
1. Compute simple and log returns. Show the distributional difference (QQ-plots, kurtosis, skewness)
2. Implement volume bars and dollar bars. Compare their return distributions to time bars (Jarque-Bera test)
3. Identify and document at least 3 specific data quality issues (missing data, stock splits not adjusted, survivorship)
4. Build a clean data pipeline that handles these issues. Output a pandas DataFrame ready for ML
5. **Deliverable:** Jupyter notebook with analysis + a reusable `DataLoader` class

---

## Week 2 — Time Series Properties & Stationarity

**Course fit:** ★★★★★ | **Implementability:** ✅ | **Importance:** 🔴 | **Format:** 📓

> The bridge between econometrics and ML. Without this, every model you build will silently overfit.

### What specialists use
`statsmodels` (ARIMA, ADF tests), `ARCH` library (GARCH family), `fracdiff` (Lopez de Prado's fractional differentiation). R's `rugarch` is the gold standard but we stay in Python.

### Lecture (90 min)
1. **Stationarity and why it matters for ML** — weak vs. strong stationarity, ADF/KPSS tests
2. **Classical time series models:** AR, MA, ARMA, ARIMA — not to use them as final models, but to understand what your neural net is competing against
3. **Volatility clustering and GARCH:** GARCH(1,1) as the baseline volatility model. Why variance is predictable even when returns aren't
4. **The integer differentiation problem** (Lopez de Prado Ch. 5): taking returns (d=1) throws away memory. Raw prices (d=0) are non-stationary. Fractional differentiation (0 < d < 1) is the sweet spot
5. **Autocorrelation, partial autocorrelation, and what they tell you about market efficiency**

### Seminar (90 min)
- **Exercise:** Fit ARIMA and GARCH(1,1) to S&P 500 returns using `statsmodels` and `arch`. Analyze residuals
- **Exercise:** Implement fractional differentiation (FFD method) from scratch in NumPy, then compare with the `fracdiff` library
- **Exercise:** Find the minimum d* for 10 different assets where ADF test rejects non-stationarity. Plot the ADF statistic and correlation with original series as a function of d
- **Discussion:** When do classical time series models beat ML? (spoiler: often, for univariate forecasting)

### Homework
**"Fractional differentiation study"**
1. For a universe of 50 US stocks, find the optimal d* per stock
2. Build a prediction model (Ridge regression) using three feature sets: (a) raw returns, (b) log prices, (c) fractionally differentiated prices at d*
3. Compare out-of-sample R² across all three. Show that (c) typically preserves more signal
4. Fit GARCH(1,1) to each stock. Forecast 1-day volatility. Evaluate with QLIKE loss
5. **Deliverable:** Notebook + a `FractionalDifferentiator` class compatible with sklearn's `TransformerMixin`

---

## Week 3 — Portfolio Theory, Factor Models & Risk

**Course fit:** ★★★★☆ | **Implementability:** ✅ | **Importance:** 🔴 | **Format:** 📓

> You need this to understand what "alpha" means, why Sharpe ratio is the metric,
> and what portfolio optimization ML models are actually trying to improve.
>
> **Note:** This is the authoritative week for risk metrics. We teach Sharpe, Sortino,
> Calmar, max drawdown, VaR, and CVaR here. Week 18 adds *advanced evaluation-only*
> metrics (profit factor, tail ratio, hit rate) specific to strategy evaluation.

### What specialists use
`PyPortfolioOpt` / `Riskfolio-Lib` (portfolio optimization), `alphalens` (factor analysis), Bloomberg PORT (commercial), Barra risk models (commercial). We use the open-source alternatives.

### Lecture (90 min)
1. **Mean-variance optimization** (Markowitz): the efficient frontier, what it gets right, and why it fails in practice (estimation error in covariance matrices)
2. **CAPM and its limitations:** beta, market risk premium, why CAPM underpredicts returns for small/value stocks
3. **Fama-French factor models:** market, size (SMB), value (HML), momentum, profitability, investment. The 5-factor model
4. **Risk metrics (foundations — used throughout the course):** Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, VaR, CVaR
5. **The Fundamental Law of Active Management:** IR = IC × √BR — why breadth matters more than accuracy
6. **Covariance estimation:** sample covariance is terrible for large portfolios. Shrinkage estimators (Ledoit-Wolf), random matrix theory (Marchenko-Pastur)
7. **Transaction cost modeling (intro):** simple constant-bps model (e.g., 10 bps round-trip). We'll build on this in Week 18 with realistic market impact

### Seminar (90 min)
- **Exercise:** Download Fama-French factors from Kenneth French Data Library. Run time-series regressions for 20 stocks. Interpret alphas and factor loadings
- **Exercise:** Build a mean-variance optimizer using `PyPortfolioOpt`. Show how the efficient frontier changes with Ledoit-Wolf shrinkage vs. sample covariance
- **Exercise:** Implement Hierarchical Risk Parity (HRP) — Lopez de Prado's tree-based portfolio allocation — and compare to mean-variance
- **Discussion:** Why do most "optimal" portfolios blow up out of sample? (hint: eigenvalue estimation error)

### Homework
**"Factor model & portfolio construction"**
1. Download 5 years of daily returns for 100 US stocks + Fama-French 5 factors
2. Run cross-sectional Fama-MacBeth regressions to estimate factor risk premia
3. Construct 3 portfolios: (a) equal-weight, (b) mean-variance optimized (Ledoit-Wolf), (c) HRP
4. Evaluate all three with: Sharpe, Sortino, max drawdown, Calmar ratio, turnover
5. Show how transaction costs (assume 10 bps round-trip) degrade each strategy differently
6. **Deliverable:** Notebook + tear sheet (use `QuantStats`) comparing the 3 portfolios

---

# MODULE 2: CLASSICAL ML FOR FINANCE
*"Trees and linear models are still the workhorses."*

---

## Week 4 — Cross-Sectional Return Prediction: Linear Models

**Course fit:** ★★★★★ | **Implementability:** ✅ | **Importance:** 🔴 | **Format:** 📓

> This is the bread-and-butter of quantitative asset management. Gu, Kelly, Xiu (2020)
> showed that even a simple neural net beats OLS — but regularized linear models are
> the baseline everything is compared against. You must understand this first.

### What specialists use
`scikit-learn` (Lasso, Ridge, Elastic Net), internal feature stores with 100+ firm characteristics, WRDS/CRSP databases. We use Kenneth French + `yfinance` + hand-crafted features.

### Lecture (90 min)
1. **The cross-sectional prediction problem:** given N stocks at time t with features X, predict returns at t+1
2. **Why this is different from time-series prediction:** you're ranking stocks against each other, not predicting absolute returns
3. **Feature engineering for stocks:** momentum (1mo, 3mo, 12mo-skip-1mo), reversal, volatility, volume, size, value (B/M), profitability
4. **Linear models:** OLS, Ridge, Lasso, Elastic Net. Bias-variance tradeoff in the context of noisy financial data
5. **The Gu-Kelly-Xiu framework:** 94 firm characteristics, 9 models, out-of-sample R² as the metric
6. **Information coefficient (IC):** rank correlation between predicted and realized returns. IC of 0.03 is useful (!)
7. **Expanding-window cross-validation** — the right way to evaluate financial ML models. Train on all data up to month t, predict month t+1. Never shuffle across time. This is the standard methodology we use for the rest of the course

### Seminar (90 min)
- **Exercise:** Build a cross-sectional feature matrix for 200+ stocks (momentum, volatility, size, value) using `yfinance` + `pandas`
- **Exercise:** Train Ridge and Lasso. Tune regularization with expanding-window CV. Compute out-of-sample IC
- **Exercise:** Sort stocks into quantile portfolios by prediction. Plot long-short portfolio returns
- **Exercise:** Use `alphalens` to analyze your best factor — IC, factor returns, turnover

### Homework
**"Cross-sectional alpha model v1 (linear)"**
1. Build a feature matrix: at least 20 features (momentum variants, volatility, volume, size, value, reversal) for 200+ US stocks, monthly frequency, 2005–2024
2. Implement expanding-window CV (train on all data up to month t, predict month t+1)
3. Train OLS, Ridge, Lasso, Elastic Net. Report out-of-sample IC and R² for each
4. Construct a long-short portfolio: long top decile predictions, short bottom decile
5. Report Sharpe ratio, max drawdown, and turnover. Include 10 bps transaction costs
6. **Deliverable:** Notebook + `alphalens` tear sheet for your best model

---

## Week 5 — Tree-Based Methods & Gradient Boosting

**Course fit:** ★★★★★ | **Implementability:** ✅ | **Importance:** 🔴 | **Format:** 📓

> Gradient boosted trees (XGBoost, LightGBM) are the single most used ML model class
> in production quant finance. They dominate Kaggle finance competitions. If you learn
> only one ML method for finance, learn this one.

### What specialists use
`XGBoost`, `LightGBM`, `CatBoost`. Feature importance via SHAP. Hyperparameter tuning via `Optuna`. Production systems retrain nightly or weekly.

### Lecture (90 min)
1. **Decision trees → Random Forests → Gradient Boosting** — the evolution
2. **Why trees dominate tabular financial data:** handle missing values, capture nonlinear interactions, no need for feature scaling, built-in feature importance
3. **XGBoost vs. LightGBM vs. CatBoost:** when to use which
4. **Feature importance methods:** gain-based, permutation, SHAP values. Why gain-based importance is misleading for correlated features
5. **Hyperparameter tuning for financial data:** max_depth, learning_rate, subsample, colsample, early stopping. Why aggressive regularization is needed
6. **From the Gu-Kelly-Xiu paper:** trees and neural nets both capture the momentum × volatility interaction that linear models miss

### Seminar (90 min)
- **Exercise:** Take the feature matrix from Week 4. Train Random Forest, XGBoost, LightGBM. Compare IC and R²
- **Exercise:** Tune XGBoost with `Optuna` using expanding-window CV (no shuffled CV!)
- **Exercise:** Compute SHAP values. Identify the top 5 features. Do they match known anomalies (momentum, value, etc.)?
- **Exercise:** Plot partial dependence plots for the top 3 features. Identify nonlinearities

### Homework
**"Cross-sectional alpha model v2 (trees)"**
1. Extend your Week 4 pipeline: add XGBoost and LightGBM
2. Tune hyperparameters with Optuna (at least 50 trials) using expanding-window CV
3. Compare all models (OLS, Ridge, Lasso, RF, XGBoost, LightGBM) on out-of-sample IC and portfolio Sharpe
4. Compute SHAP values for your best tree model. Create a feature importance ranking
5. Test for overfitting: plot in-sample vs. out-of-sample IC over time. Does the gap widen?
6. **Stretch goal:** Implement a simple model combination (average predictions of top 3 models)
7. **Deliverable:** Notebook + comparison table + SHAP analysis

---

## Week 6 — Financial ML Methodology: Labeling, CV, and Backtesting

**Course fit:** ★★★★★ | **Implementability:** ✅ | **Importance:** 🔴 | **Format:** 📓

> This is what separates people who "do ML" from people who do ML *correctly* in finance.
> Lopez de Prado's entire career is built on showing that methodology matters more than models.

### What specialists use
Custom implementations of triple-barrier labeling and purged CV. We implement from scratch + use `VectorBT`. (Note: MLFinLab by Hudson & Thames is the commercial reference implementation, but we don't need it — everything here is built from scratch.)

### Lecture (90 min)
1. **Why standard ML methodology fails in finance:**
   - K-fold CV leaks information through serial correlation and overlapping labels
   - Fixed-threshold labeling (up/down) ignores volatility regimes
   - Backtesting ≠ cross-validation (the "backtest overfitting" problem)
2. **Triple-barrier labeling** (Lopez de Prado Ch. 3): profit-take, stop-loss, and time barriers. Labels adapt to volatility
3. **Sample weighting and uniqueness** (Ch. 4): overlapping labels create redundancy. Average uniqueness and sample weights fix this
4. **Purged k-fold cross-validation** (Ch. 7): remove training samples whose labels overlap with test period. Include combinatorial purged CV (CPCV) for generating multiple backtest paths
5. **Meta-labeling** (Ch. 3): a secondary model that predicts whether the primary model's trade will be profitable. Enables adaptive position sizing. One of the most practical techniques from Lopez de Prado
6. **The multiple testing problem (preview):** running many backtests inflates the chance of finding fake alpha. The full solution (deflated Sharpe ratio) is covered in Week 18 — for now, understand *why* this is dangerous

### Seminar (90 min)
- **Exercise:** Implement triple-barrier labeling from scratch. Apply to S&P 500 with dynamic barriers (based on daily volatility × multiplier)
- **Exercise:** Compare labels from (a) fixed ±1% threshold, (b) sign of next-day return, (c) triple-barrier. Train the same model on all three. Compare stability
- **Exercise:** Implement purged k-fold CV. Show the information leakage: compare CV score with purging vs. without, on a model with a lagging feature that creates label overlap
- **Discussion:** How many backtests can you run before your "best" strategy is likely just noise?

### Homework
**"Meta-labeling pipeline"**
1. Implement a primary model: simple moving average crossover (50/200 day) as a long/short signal
2. Implement triple-barrier labeling with dynamic barriers (volatility-scaled)
3. Train a **meta-labeling** model (Random Forest or XGBoost) that predicts whether the primary model's next trade will be profitable
4. Implement purged k-fold CV (at least 5 folds, with embargo period = max label duration)
5. Show the full pipeline: primary signal → meta-label filter → position sizing → backtest
6. Compare: primary model alone vs. primary + meta-label. Report Sharpe, hit rate, profit factor
7. **Deliverable:** Notebook + a reusable `TripleBarrierLabeler` and `PurgedKFold` class

---

# MODULE 3: DEEP LEARNING FOR FINANCE
*"Neural networks add value when nonlinear structure and sequence matter."*

---

## Week 7 — Feedforward Neural Networks for Asset Pricing

**Course fit:** ★★★★★ | **Implementability:** ✅ | **Importance:** 🟡 | **Format:** 📓

> The transition from sklearn to PyTorch. We replicate the neural network results from
> Gu-Kelly-Xiu, showing that a simple feedforward net captures nonlinear factor interactions
> that trees and linear models partially miss.

### What specialists use
PyTorch (research), TensorFlow/Keras (some production), custom training loops with early stopping, batch normalization, dropout. Production models at firms like Two Sigma retrain weekly.

### Lecture (90 min)
1. **PyTorch fundamentals for finance:** tensors on MPS, `Dataset`/`DataLoader` for financial panels, custom loss functions
2. **Architecture choices for tabular financial data:** depth vs. width, batch normalization, dropout, skip connections
3. **The Gu-Kelly-Xiu neural net:** 3 hidden layers (32-16-8), ReLU, 1 output, huber loss. Why this specific architecture?
4. **Training pitfalls in finance:**
   - Don't shuffle across time (use `SequentialSampler` or custom logic)
   - Use early stopping on a validation set that's temporally after the training set
   - Financial loss functions: MSE is fine for regression, but consider IC-based losses or Sharpe-ratio-based losses
5. **Ensemble methods:** average over random seeds, hyperparameter combinations, or time periods

### Seminar (90 min)
- **Exercise:** Build the Gu-Kelly-Xiu 3-layer net in PyTorch. Train on the cross-sectional feature matrix from Weeks 4-5
- **Exercise:** Implement proper temporal train/val/test splitting in a PyTorch `DataLoader`
- **Exercise:** Compare: (a) MSE loss, (b) weighted MSE (more weight on large-cap stocks), (c) IC-maximizing loss
- **Exercise:** Train 5 models with different seeds. Compare individual vs. ensemble IC

### Homework
**"Deep cross-sectional model"**
1. Implement the Gu-Kelly-Xiu neural net architecture in PyTorch (3 layers: 32-16-8, ReLU, BN, dropout)
2. Train with expanding-window CV. Use MPS acceleration on your M4
3. Compare against your Week 5 best model (XGBoost). Report IC, R², and portfolio Sharpe
4. Implement an ensemble: average predictions from (a) your neural net, (b) XGBoost, (c) LightGBM. Does the ensemble beat any single model?
5. Analyze where the neural net wins vs. where trees win (e.g., by market regime, by market cap)
6. **Deliverable:** Notebook + model code + comparison analysis

---

## Week 8 — Sequence Models: LSTM/GRU for Volatility & Returns

**Course fit:** ★★★★★ | **Implementability:** ✅ (⚡ for larger models) | **Importance:** 🟡 | **Format:** 📓

> LSTMs remain the default deep learning approach for financial time series at most firms.
> We focus on volatility forecasting because (a) volatility IS predictable and (b) it's
> directly useful for risk management and options pricing.

### What specialists use
PyTorch LSTM/GRU, custom attention mechanisms, `ARCH` library as baseline. Production systems at options desks forecast 1-day, 5-day, 21-day realized vol.

### Lecture (90 min)
1. **Why sequence models for finance:** time dependencies, regime persistence, order-book dynamics
2. **RNN → LSTM → GRU:** vanishing gradients, gating mechanisms, when GRU suffices
3. **Volatility forecasting as a supervised learning problem:**
   - Target: realized volatility (sum of squared intraday returns, or Parkinson/Garman-Klass estimators)
   - Features: lagged RV, returns, volume, VIX, implied vol (if available)
   - Baseline: GARCH(1,1), HAR (Heterogeneous Autoregressive) model
4. **Practical LSTM architecture decisions:** number of layers, hidden size, bidirectional (no — it's time series), sequence length, teacher forcing
5. **Loss functions for volatility:** MSE, QLIKE (quasi-likelihood — better for heteroskedastic targets), MSE on log-vol

### Seminar (90 min)
- **Exercise:** Compute realized volatility from daily data using 3 estimators (close-to-close, Parkinson, Garman-Klass)
- **Exercise:** Implement GARCH(1,1) and HAR model as baselines using `arch` and `statsmodels`
- **Exercise:** Build an LSTM volatility forecaster in PyTorch. Compare 1-layer vs. 2-layer, hidden_size 32 vs. 64
- **Exercise:** Train, evaluate on held-out period. Compare LSTM vs. GARCH vs. HAR using MSE and QLIKE

### Homework
**"Volatility forecasting showdown"**
1. For 20 liquid US stocks + SPY, compute daily realized volatility (2010–2024)
2. Build baselines: GARCH(1,1), EGARCH, HAR model
3. Build an LSTM in PyTorch: input = [lagged RV (5 lags), lagged returns (5), lagged volume (5), VIX]. Sequence length = 20 trading days
4. Build a GRU variant. Compare
5. Evaluate all models on 2022–2024 out-of-sample: MSE, QLIKE, directional accuracy
6. **Key question:** Does the LSTM beat GARCH? By how much? For which stocks?
7. **Stretch goal:** Add attention mechanism to the LSTM. Does it help?
8. **Deliverable:** Notebook + model code + comparison table

---

## Week 9 — Foundation Models for Financial Time Series

**Course fit:** ★★★★★ | **Implementability:** ✅ (Kronos) / ⚡ (TFT, TimesFM) | **Importance:** 🔴 | **Format:** 📓

> **Heavily restructured after deep-diving the 2025-2026 literature.**
>
> The naive take is "foundation models don't work for finance." The reality is a
> three-layer story that's far more interesting — and far more important for
> your career to understand, given that major banks are investing billions here.

### What specialists use (2025-2026)
**General TSFMs:** TimesFM 2.5 (Google, 200M params), Chronos/Chronos-2 (Amazon, T5-based, 9M-710M), Moirai 2.0 (Salesforce, MoE, 11M-300M+), Lag-Llama (LLaMA-based, ~10M).
**Finance-native FMs:** Kronos (AAAI 2026, 4.1M-102M, trained on 12B K-line records from 45 exchanges), FinCast (CIKM 2025, 1B params, 20B+ financial time points).
**Custom transformers:** `pytorch-forecasting` (TFT, N-BEATS, DeepAR). TFT remains the best supervised transformer for multi-horizon finance forecasting.

### Lecture (90 min)

**Part A: Transformer Architectures for Finance (30 min)**
1. **Self-attention for time series:** long-range dependencies without recurrence
2. **Temporal Fusion Transformer (TFT):** variable selection networks, gated residual networks, multi-head attention, multiple quantile outputs
3. **Positional encoding for irregular time series:** markets have gaps (weekends, holidays)

**Part B: The Foundation Model Landscape (30 min)**
4. **The zoo of general-purpose TSFMs:** TimesFM, Chronos, Moirai 2.0, Lag-Llama
5. **Finance-native foundation models (the ones that actually work):**
   - **Kronos (AAAI 2026):** K-line tokenization, 93% RankIC improvement, MIT license
   - **FinCast (CIKM 2025):** 1B params, 20-23% MSE reduction zero-shot
6. **Why the gap?** Generic TSFMs were pre-trained on non-financial data

**Part C: The Nuanced Truth About "XGBoost Wins" (30 min)**
7. **The Re(Visiting) paper (2025)** — what it actually found
8. **The sample efficiency argument:** pre-trained TTM needs 3-10 fewer years of data
9. **The hybrid approach:** FM embeddings as features fed into XGBoost

### Seminar (90 min)
- **Exercise:** Load Kronos-mini (4.1M params) from HuggingFace. Run zero-shot inference on 5 stocks' OHLCV data
- **Exercise:** Load Chronos-small. Compare to Kronos (finance-native vs. generic)
- **Exercise:** Use `pytorch-forecasting` to train a TFT on the same data
- **Exercise:** Extract Kronos's internal representations. Feed into XGBoost. Does the hybrid approach beat either alone?
- **Discussion:** When should you use a foundation model vs. train from scratch?

### Homework
**"Foundation models for financial forecasting: the three-layer experiment"**
1. **Data:** Daily OHLCV for 50 US stocks, 2010–2024
2. **Layer 1 — Generic zero-shot:** Run Chronos-small + TimesFM 2.5 zero-shot
3. **Layer 2 — Finance-native zero-shot:** Run Kronos-base (102M, MIT license)
4. **Layer 3 — Hybrid approach:** Kronos features + Week 5 features → XGBoost
5. **Supervised baseline:** Train TFT
6. **Full comparison table:** Chronos / TimesFM / Kronos / Kronos+XGBoost / TFT / LSTM / XGBoost alone
7. **Deliverable:** Notebook + comparison table + analysis of when/where each approach wins

---

## Week 10 — NLP for Finance: From FinBERT to LLM Embeddings

**Course fit:** ★★★★★ | **Implementability:** ✅ (FinBERT) / ⚡ (LLM embeddings) | **Importance:** 🔴 | **Format:** 📓

> Text-based alpha is now the fastest-moving area in quant finance.
> Chen, Kelly, Xiu (2023) showed that LLM embeddings **significantly outperform**
> FinBERT and all older NLP methods for return prediction.

### What specialists use (2025-2026)
**Production tier:** Custom fine-tuned LLMs, Bloomberg internal NLP, RavenPack (commercial). **Open-source tier:** FinGPT (~$300 to fine-tune via LoRA), FinBERT (baseline), `sentence-transformers`. **Agentic tier:** Qlib RD-Agent (automated factor discovery), Man Group AlphaGPT.

### Lecture (90 min)
1. **The evolution of financial NLP (3 eras):** bag-of-words → FinBERT → LLM embeddings
2. **FinBERT as baseline:** lightweight, fast sentiment classification
3. **The LLM embedding revolution (Chen, Kelly, Xiu 2023):** don't use LLMs to *predict* — use them to *embed*
4. **Implementable approaches:** sentence-transformers (free, M4) / OpenAI API (cheap) / FinGPT LoRA ($300)
5. **From text to trading signal:** aggregation, PCA, cosine similarity to prototypes
6. **Agentic AI for quant research (2025):** AlphaGPT, RD-Agent — research automation

### Seminar (90 min)
- **Exercise:** Load FinBERT. Score 1000 financial news headlines (Financial PhraseBank)
- **Exercise:** Embed the same headlines with `sentence-transformers`. Compare IC: FinBERT sentiment vs. embeddings
- **Exercise:** Feed both feature sets to XGBoost from Week 5. Which helps more?
- **Discussion:** Why do embeddings outperform classification? (information bottleneck)

### Homework
**"Text alpha: FinBERT vs. LLM embeddings"**
1. Download financial news headlines (Financial PhraseBank + SEC EDGAR 8-K filings)
2. **Baseline:** Score with FinBERT. Aggregate to daily stock-level sentiment. Evaluate IC
3. **LLM embeddings:** Embed with `sentence-transformers`. PCA to 10-20 components
4. **Comparison:** Add each text feature set to your Week 5 XGBoost:
   - Model A: price/volume only | B: +FinBERT | C: +embeddings | D: +both
5. **Signal decay:** IC as a function of horizon (1-day to 20-day) for FinBERT vs. embeddings
6. **Deliverable:** Notebook + comparison table + signal decay plot

---

# MODULE 4: ADVANCED TOPICS
*"Where the frontier meets the practical."*

---

## Week 11 — Bayesian Deep Learning & Uncertainty Quantification

**Course fit:** ★★★★★ | **Implementability:** ✅ | **Importance:** 🟡 | **Format:** 📓

> Every ML model you've built so far outputs a point estimate. A prediction of "+2% return"
> without a confidence interval is useless for position sizing. This week adds the missing
> piece: **how uncertain is my model?**
>
> The key insight: you already have the tools. MC Dropout is just dropout left ON at
> inference time. Deep ensembles are just the seed-averaged models from Week 7.
> Three lines of code give you calibrated uncertainty estimates.

### What specialists use
MC Dropout (the default at quant funds — zero overhead), deep ensembles (5-10x training cost but best calibration), PyMC/Pyro for Bayesian factor models, `torch-uncertainty` (emerging). Full Bayesian NNs via VI/HMC remain research-only.

### Lecture (90 min)
1. **Two types of uncertainty:**
   - **Aleatoric** (data uncertainty): irreducible noise in financial returns. Cannot be reduced with more data
   - **Epistemic** (model uncertainty): what the model doesn't know. Reducible with more data or better models
   - Why the distinction matters: epistemic uncertainty should drive position sizing; aleatoric should drive risk management
2. **MC Dropout (Gal & Ghahramani 2016):**
   - Theoretical foundation: dropout at inference ≈ approximate variational inference over weights
   - Implementation: `model.train()` at inference, run T=50-100 forward passes, compute mean and variance
   - Cost: literally free if you already use dropout (and you should)
3. **Deep Ensembles (Lakshminarayanan et al. 2017):**
   - Train M=5-10 models with different random seeds
   - Mean of predictions = better point estimate. Variance = uncertainty estimate
   - Empirically better calibrated than MC Dropout, but M× more expensive to train
4. **Bayesian Linear Regression (conjugate priors):**
   - Exact closed-form posterior: no MCMC, no VI, instant computation
   - Perfect for factor models: put priors on factor loadings, get posterior distributions
   - Connection to Ridge regression: Ridge = Bayesian with Gaussian prior
5. **Applications in finance:**
   - **Kelly criterion with uncertainty:** f* = μ / (σ²_aleatoric + σ²_epistemic). Higher model uncertainty → smaller position
   - **Regime detection:** track universe-level average epistemic uncertainty. Spikes = out-of-distribution (regime change)
   - **Stock filtering:** only trade stocks where model uncertainty is below a threshold
   - **Prediction intervals:** "I predict +2% with a 90% interval of [-3%, +7%]"
6. **What DOESN'T work (honest assessment):**
   - Full Bayesian NNs (VI, HMC): prohibitively expensive for anything beyond tiny networks
   - Bayes by Backprop: double the parameters, unstable training, marginal benefit
   - **Practical sweet spot:** MC Dropout + Ensembles. Not full Bayesian inference

### Seminar (90 min)
- **Exercise:** Take your Week 7 neural net. Add MC Dropout: keep `model.train()` at inference, run 100 forward passes, compute mean and std of predictions per stock
- **Exercise:** Compare uncertainty methods: (a) single model, (b) MC Dropout (T=100), (c) 5-model ensemble from different seeds. Plot uncertainty vs. actual prediction error — do they correlate?
- **Exercise:** Implement Bayesian linear regression with conjugate priors for a factor model. Compare posterior distributions under weak vs. strong priors. Show connection to Ridge
- **Exercise:** Build an uncertainty-filtered strategy: only trade low-uncertainty stocks. Compare Sharpe vs. trading all stocks

### Homework
**"Uncertainty-aware stock prediction"**
1. Take your Week 7 Gu-Kelly-Xiu model. Ensure Dropout(p=0.5) after each hidden layer (matching the Week 7 architecture)
2. **MC Dropout:** At inference, run T=100 forward passes per stock. Compute mean prediction and prediction std
3. **Calibration analysis:** Do 90% prediction intervals actually contain 90% of realized returns? Plot reliability diagram
4. **Uncertainty over time:** Plot average epistemic uncertainty across the universe over time. Does it spike during known stress periods (COVID, rate hikes)?
5. **Uncertainty-filtered strategy:**
   - Strategy A: Long-short quintile portfolio using ALL stocks (your Week 7 model)
   - Strategy B: Same, but only trade stocks in the BOTTOM 50% of epistemic uncertainty
   - Strategy C: Same, but only trade stocks in the TOP 50% (high uncertainty)
   - Compare: Sharpe, IC, hit rate for A vs. B vs. C
6. **Kelly-criterion position sizing:** Weight positions by μ / (σ²_aleatoric + σ²_epistemic) instead of equal-weight. Does it improve Sharpe?
7. **Deliverable:** Notebook + calibration plot + uncertainty time series + strategy comparison table

**Why this works as HW:** MC Dropout adds 3 lines of code to an existing model. The calibration analysis is genuinely informative. The uncertainty-filtered strategy is a real technique used at quant funds. Everything runs on M4 in minutes (100 forward passes through a small net is trivial).

### Key Papers
| Paper | Year | Key Contribution |
|-------|------|------------------|
| Gal & Ghahramani, "Dropout as a Bayesian Approximation" | 2016 | MC Dropout = approximate VI |
| Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty" | 2017 | Deep ensembles |
| Kendall & Gal, "What Uncertainties Do We Need in Bayesian DL?" | 2017 | Epistemic vs. aleatoric decomposition |
| Liao, Ma, Neuhierl & Schilling, "Uncertainty of ML Predictions in Asset Pricing" | 2025 | Directly applicable to cross-sectional prediction |
| Blasco, Sanchez & Garcia, "Survey on UQ in DL for Financial TS" | 2024 | Comprehensive review |

---

## Week 12 — Graph Neural Networks for Finance

**Course fit:** ★★★★☆ | **Implementability:** ✅ | **Importance:** 🟡 | **Format:** 📓

> Every model we've built so far treats each stock independently. But stocks are connected:
> Apple and its suppliers move together. Banks co-move during crises. Sector rotations
> affect groups of stocks. **Graph Neural Networks encode these relationships explicitly.**
>
> The honest result: GNNs add the most value when you have simple features but rich
> relational data. With already-rich hand-crafted features, XGBoost can match or beat GNNs.
> The real power is in the *graph construction* — choosing the right edges matters more
> than the GNN architecture.

### What specialists use
PyTorch Geometric (`torch-geometric`), DGL (`dgl`), Microsoft Qlib (has built-in GNN models). Graph construction from: GICS sector/industry codes, rolling return correlations, supply chain data (FactSet Revere via WRDS), LLM-extracted relationships.

### Lecture (90 min)
1. **Why model stock relationships?** The relational view of markets
   - Stocks are not independent: sector co-movement, supply chain propagation, contagion
   - Information propagates through the cross-section (an earnings surprise at Apple affects TSMC)
   - Graph structure captures this propagation — each stock is a node, relationships are edges
2. **Graph construction methods (the most important design choice):**
   - **Correlation graphs:** rolling 60-day return correlation, threshold at 0.5-0.7 to create edges
   - **Sector/industry graphs:** connect stocks within the same GICS sector or sub-industry (free from Wikipedia/yfinance)
   - **Supply chain graphs:** customer-supplier edges from SEC 10-K filings or FactSet Revere
   - **Dynamic graphs:** re-estimate correlations monthly — the graph evolves over time
   - **LLM-extracted graphs** (frontier): use GPT-4 to extract relationships from news (FinDKG, ICAIF 2024)
3. **GNN architectures for finance:**
   - **GCN (Graph Convolutional Network):** spectral convolution, simplest baseline
   - **GAT (Graph Attention Network):** learns which neighbor stocks matter more. **Best default for finance**
   - **GraphSAGE:** samples and aggregates from neighbors. Good for inductive settings (IPOs, delistings)
   - **Temporal Graph Convolution (Feng et al. 2019):** LSTM for temporal + GNN for relational
4. **State of the art:**
   - **HIST (arXiv 2021):** concept-oriented graphs. IC=0.052 on CSI300 via Qlib, beating LightGBM (0.040) by 30%
   - **MASTER (AAAI 2024):** market-guided transformer+graph. IC=0.064 on CSI300, beating XGBoost (0.051) by 25%
   - **Key caveat:** These results are on Chinese A-shares. US equity results are less standardized
5. **The honest benchmark: when does GNN beat XGBoost?**
   - With simple features (Alpha360): GNN wins by 30%+ in IC
   - With rich hand-crafted features (Alpha158): XGBoost matches or beats GNN
   - GNNs are most powerful for **ranking tasks** (top-K stock selection) rather than point prediction
6. **PyTorch Geometric basics:** `Data` objects, `edge_index`, `GATConv`, batching graphs

### Seminar (90 min)
- **Exercise:** Build two stock graphs for S&P 100:
  - Correlation graph: compute 60-day rolling return correlation, threshold at 0.6
  - Sector graph: same GICS sector = edge (from Wikipedia table)
- **Exercise:** Implement a 2-layer GAT in PyTorch Geometric. Each node = stock features (momentum, vol, volume). Target = next 5-day return rank
- **Exercise:** Train: (a) GAT on correlation graph, (b) GAT on sector graph, (c) MLP (no graph), (d) XGBoost (no graph). Compare IC
- **Exercise:** Visualize GAT attention weights — which stock relationships does the model focus on?

### Homework
**"Relational stock ranking with Graph Attention Networks"**
1. **Data:** Download 2 years of daily OHLCV for S&P 100 stocks via `yfinance`
2. **Graphs:** Construct 3 graph types:
   - Correlation graph (threshold=0.6 on 60-day rolling correlation)
   - Sector graph (same GICS sector = edge, from Wikipedia)
   - Combined graph (union of both edge sets)
3. **Features:** Compute 8-10 features per stock per day: 5/10/20-day momentum, 20-day volatility, volume ratio, RSI, 5-day reversal
4. **Model:** 2-layer GATConv (hidden_dim=64, heads=4). Target: next 5-day return rank
5. **Baselines:** (a) MLP with same features (no graph), (b) XGBoost with same features
6. **Evaluation:** IC, RankIC, Precision@10, long-short decile portfolio Sharpe
7. **Ablation:** Which graph type helps most? Does the combined graph beat individual graphs?
8. **Attention analysis:** For 3 example stocks (e.g., AAPL, JPM, XOM), show which neighbors get the highest attention weights. Do they make economic sense?
9. **Deliverable:** Notebook + comparison table + attention visualization + analysis

**Why this works as HW:** S&P 100 = 100 nodes with ~500-2000 edges. Training takes seconds per epoch on M4. The graph construction + GNN pipeline is ~300 lines of Python. Students see empirically whether relationships help (usually: modest improvement over MLP, competitive with XGBoost).

### Key Papers
| Paper | Year | Key Contribution |
|-------|------|------------------|
| Feng et al., "Temporal Relational Ranking for Stock Prediction" | 2019 | First GCN+LSTM for stock ranking |
| Xu et al., "HIST: Mining Concept-Oriented Shared Information" | 2021 | Concept graphs; Qlib SOTA |
| Li et al., "MASTER: Market-Guided Stock Transformer" | AAAI 2024 | Current SOTA; IC=0.064 on CSI300 |
| FinDKG, "Dynamic Knowledge Graphs with LLMs" | ICAIF 2024 | LLM-constructed dynamic knowledge graphs |
| ACM Computing Surveys, "GNN-based Methods for Stock Forecasting" | 2024 | Comprehensive systematic review |

### Key Libraries
- `torch-geometric`: GATConv, GCNConv, SAGEConv, Data, DataLoader
- `networkx`: graph construction, visualization, centrality metrics
- `yfinance`: sector/industry info via `.info['sector']`

---

## Week 13 — Reinforcement Learning for Portfolio Management

**Course fit:** ★★★★☆ | **Implementability:** ⚡ | **Importance:** 🟡 | **Format:** 📓

> RL for finance is intellectually beautiful and practically difficult. It maps perfectly
> to the sequential decision-making nature of trading. But it's unstable, sample-inefficient,
> and hard to evaluate. We teach it because (a) it's important to understand and
> (b) FinRL makes it implementable. This also sets the stage for Week 16 (market making).

### What specialists use
Custom RL environments (Gymnasium-based), `Stable-Baselines3` (PPO, SAC, A2C), `FinRL` (wraps SB3 + financial environments), proprietary execution optimization systems. Stanford's CME 241 (Ashwin Rao) is the standard academic treatment.

### Lecture (90 min)
1. **Trading as a Markov Decision Process:**
   - State: portfolio weights, current prices, features
   - Action: target portfolio weights (continuous action space)
   - Reward: portfolio return, risk-adjusted return, or Sharpe-like objective
   - Transition: market dynamics (the hard part — non-stationary, partially observable)
2. **Key algorithms:** DQN (discrete actions), DDPG/TD3 (continuous, deterministic), PPO/A2C (continuous, stochastic), SAC (continuous, entropy-regularized)
3. **Why RL is hard in finance:** non-stationarity, partial observability, delayed rewards, distribution shift, training instability
4. **FinRL architecture:** data layer → environment → agent → backtest integration
5. **Realistic expectations:** RL learns something non-trivial (reduces drawdowns, adapts to volatility) but don't expect to "beat the market" in a homework
6. **Where RL actually works in industry:** optimal execution (how to execute a large order), **market making** (Week 16), options hedging (deep hedging)

### Seminar (90 min)
- **Exercise:** Set up a `FinRL` environment with Dow 30 stocks. Train a PPO agent for portfolio allocation
- **Exercise:** Implement a custom reward function: (a) simple return, (b) Sharpe-ratio-based, (c) return - λ × drawdown
- **Exercise:** Compare PPO, A2C, and DDPG on the same environment
- **Discussion:** How would you evaluate whether the RL agent has actually "learned" something vs. overfitting to the training period?

### Homework
**"RL portfolio manager"**
1. Use FinRL with 30 stocks (e.g., Dow 30), daily data 2010–2024
2. Train 3 agents: PPO, A2C, SAC using Stable-Baselines3
3. Benchmark against: (a) equal-weight buy-and-hold, (b) minimum-variance portfolio (from Week 3), (c) your best ML model from Weeks 5/7
4. Evaluate on 2022–2024 out-of-sample: Sharpe, max drawdown, turnover, transaction costs
5. Experiment with reward shaping: does adding a drawdown penalty change agent behavior?
6. **Key question:** Does the RL agent learn to reduce exposure during high-volatility periods?
7. **Deliverable:** Notebook + training curves + comparison table + analysis of agent behavior

---

## Week 14 — Derivatives Pricing with Neural Networks

**Course fit:** ★★★★☆ | **Implementability:** ✅ | **Importance:** 🟡 (🔴 if you want to work at an options desk) | **Format:** 📓

> Neural networks can learn the Black-Scholes pricing function — and then go beyond it.
> This is one of the cleanest applications of deep learning in finance because we have
> an analytical ground truth to compare against.

### What specialists use
`QuantLib` (the industry standard — C++ with Python bindings), `tf-quant-finance` (Google), custom Monte Carlo engines, neural network surrogates for fast pricing. Banks like JP Morgan and Goldman Sachs use neural nets to accelerate Greeks computation.

### Lecture (90 min)
1. **Options pricing crash course:** call/put payoffs, Black-Scholes formula, the Greeks (delta, gamma, theta, vega)
2. **Why neural nets for pricing?** BS assumes constant volatility (wrong). MC pricing is slow. Neural nets give gradients for free (autograd)
3. **Architecture for pricing:** input = (S, K, T, σ, r) → output = option price. How to enforce no-arbitrage constraints
4. **Deep Hedging** (Buehler et al. 2019): learn a hedging strategy end-to-end without specifying a pricing model
5. **Implied volatility surface learning:** input = (K/S, T) → output = implied vol. Capturing the smile/skew

### Seminar (90 min)
- **Exercise:** Implement Black-Scholes in PyTorch (vectorized, differentiable). Compute Greeks with `torch.autograd`
- **Exercise:** Generate 100K training samples from BS. Train a 4-layer neural net to learn the pricing function
- **Exercise:** Compute the neural net's Delta via autograd. Compare to analytical Delta
- **Exercise:** Download real options data (CBOE sample). Compare BS prices vs. market prices — where does BS fail?

### Homework
**"Neural network options pricer"**
1. Generate synthetic training data: 500K option prices from Black-Scholes
2. Train a neural net to learn the BS pricing function. Can it extrapolate?
3. Use `torch.autograd` to compute all Greeks. Compare to analytical values
4. **The interesting part:** Generate data from a Heston stochastic volatility model (no closed-form). Train a neural net on Heston prices
5. Learn the implied volatility surface: (moneyness, time-to-expiry) → implied vol
6. **Deliverable:** Notebook + trained models + Greeks comparison + IV surface visualization

---

## Week 15 — HFT & Market Microstructure ML

**Course fit:** ★★★★☆ | **Implementability:** ✅ (FI-2010 is small) | **Importance:** 🟡 | **Format:** 📖 Lecture + Seminar only (no homework — data access is the bottleneck)

> **This is a "mind-expanding" week, not a "build-and-deploy" week.**
>
> You will never build an HFT system in Python. The infrastructure costs $10M+/year
> and the latency competition is measured in nanoseconds. But understanding market
> microstructure makes you a **fundamentally better** ML practitioner in *every* area of finance.
>
> Think of it as three concentric rings:
> - **Inner ring (HFT execution, <10μs):** ML is irrelevant. This is hardware engineering. Your Python code literally cannot participate.
> - **Middle ring (execution quality, seconds to minutes):** ML adds significant value. Optimal execution, trade timing, impact estimation. Accessible to you.
> - **Outer ring (microstructure-informed longer-horizon models):** ML is transformative. Understanding microstructure makes every model better — from daily return prediction to risk management. This is where 99% of you will apply what you learn.

### What specialists use
**HFT firms (Citadel Securities, Virtu, Jane Street, Optiver):** C++, FPGA (Xilinx/AMD Alveo), co-located servers, DPDK/RDMA kernel bypass, KDB+/q for tick databases, proprietary ITCH/MDP parsers. **Researchers:** LOBSTER (NASDAQ LOB reconstruction), FI-2010 benchmark, LOBFrame (UCL), `tick` library (Hawkes processes).

### Lecture (90 min)
1. **What HFT firms actually do:**
   - Market making (the dominant HFT strategy — 50-60% of US equity volume)
   - Latency arbitrage (cross-exchange price discrepancies lasting microseconds)
   - Statistical arbitrage at microsecond timescales
   - News-based trading (NLP on press releases in milliseconds)
2. **The infrastructure stack ($10M+/year to compete):**
   - Co-location: servers in the exchange data center. CME 10G connection: $12K/month
   - FPGA order routing: $5K-50K per card, tick-to-trade in <1 microsecond
   - Kernel bypass (DPDK, RDMA): skip the OS network stack entirely
   - Microwave towers (NYC-Chicago): speed-of-light advantage over fiber
3. **Why Python can't compete:**
   - GIL (Global Interpreter Lock): single-threaded execution
   - Garbage collector: a 2ms pause is an eternity when competitors act in 100ns
   - Interpreter overhead: floor of ~10-20 microseconds per decision — 100x slower than FPGA
   - Industry practice: Python for research/backtesting, C++/FPGA for execution
4. **LOB data — what you're missing with daily OHLCV:**
   - A limit order book at every moment: bid/ask queues at every price level
   - Message rates: 100K+ events per second for liquid stocks
   - Daily OHLCV = a **40,000:1 compression** of actual market data
   - What's lost: intraday volatility patterns, order flow dynamics, queue position, spread time series
5. **What ML CAN do for microstructure (even without speed):**
   - **Order Flow Imbalance (Cont et al. 2014):** ΔP ≈ α + β × OFI. The single most important microstructure feature
   - **DeepLOB (Zhang et al. 2019):** CNN+LSTM on LOB features. F1 ~83% on FI-2010 for mid-price direction
   - **VPIN (Easley, Lopez de Prado, O'Hara):** real-time measure of order flow toxicity. Spiked before the 2010 Flash Crash
   - **Trade classification:** ML improves Lee-Ready accuracy from 85% to 90%+
   - **Market impact estimation:** predict how much your order will move the price
   - **Optimal execution (Almgren-Chriss 2000):** balance market impact vs. timing risk. RL extensions handle time-varying liquidity
6. **The three rings (summary):**
   - Inner (HFT, <10μs): hardware competition. ML irrelevant
   - Middle (execution, seconds-minutes): ML valuable. Almgren-Chriss + RL for optimal execution
   - Outer (microstructure-informed models): ML transformative. Understanding microstructure makes *every* model better

### Seminar (90 min)
- **Exercise:** Load the FI-2010 benchmark dataset. Visualize LOB snapshots at different time points. Compute basic features: bid-ask spread, order imbalance, mid-price
- **Exercise:** Implement a simplified DeepLOB-lite: CNN on the top 10 levels of the LOB. Train to predict mid-price direction at horizon k=10 events. Report F1 per class (up/down/stationary)
- **Exercise:** Compute Order Flow Imbalance from the FI-2010 data. Verify the linear relationship: ΔP ≈ β × OFI (Cont et al. 2014)
- **Discussion:** What microstructure knowledge improves your daily-frequency models? How does understanding LOB dynamics change how you think about transaction costs, slippage, and market impact?

**No homework** — The free FI-2010 dataset covers only 5 Finnish stocks over 10 days. Real LOBSTER data costs ~$500/stock/year. This week is about understanding and intellectual framework, not about building a deployable system.

### Key Papers
| Paper | Year | Key Contribution |
|-------|------|------------------|
| Cont, Kukanov, Stoikov, "Price Impact of Order Book Events" | 2014 | OFI → ΔP linear relationship |
| Zhang, Zohren, Roberts, "DeepLOB" | IEEE TSP 2019 | CNN+LSTM SOTA on FI-2010, F1 ~83% |
| Sirignano & Cont, "Universal Features of Price Formation" | QF 2019 | Cross-asset deep learning on billions of quotes |
| Berti et al., "TLOB: Transformer for LOB" | arXiv 2025 | Current SOTA, +3.7 F1 over DeepLOB |
| Briola et al., "Deep LOB Forecasting: A Microstructural Guide" | QF 2024 | LOBFrame benchmark on real NASDAQ data |
| Aquilina, Budish, O'Neill, "Quantifying the HFT Arms Race" | QJE 2022 | Latency arbitrage ≈ 0.5bps tax on trading |
| Almgren & Chriss, "Optimal Execution of Portfolio Transactions" | 2000 | Foundational optimal execution model |

### Data Sources
| Source | Cost | Notes |
|--------|------|-------|
| **FI-2010** | Free | 5 Finnish stocks, 10 days, pre-processed. Standard benchmark but limited |
| **LOBSTER** | ~$500/stock/year (academic) | NASDAQ LOB reconstruction from ITCH. Gold standard |
| **LOBFrame** (UCL GitHub) | Free (framework) | Needs LOBSTER data. End-to-end DL pipeline |
| **Binance order book snapshots** | Free (via ccxt) | Crypto, not equities. Good proxy for LOB exercises |

---

## Week 16 — Market Making with ML

**Course fit:** ★★★★★ | **Implementability:** ✅ | **Importance:** 🟡 | **Format:** 📓

> Now that you understand what HFT firms do (Week 15) and how RL works (Week 13),
> we combine both into the most classic HFT strategy: **market making.**
>
> Market making is the art of continuously quoting bid and ask prices, earning the spread
> while managing inventory risk. Avellaneda & Stoikov (2008) solved this analytically.
> We'll implement their model, then show that RL can marginally improve on it by adapting
> to changing market conditions.
>
> **Everything runs in simulation.** You cannot deploy to a real exchange in a homework,
> but the economic intuition transfers directly.

### What specialists use
Custom LOB simulators, `mbt_gym` (Gymnasium environments for market making), Stable-Baselines3 (PPO, SAC), `hummingbot` (production-grade open-source MM bot). Research: Hawkes process models (`tick` library), queue-reactive models.

### Lecture (90 min)
1. **Market making economics:**
   - Earn the bid-ask spread: buy at bid, sell at ask
   - The risk: **adverse selection** — informed traders pick off your stale quotes
   - The challenge: **inventory management** — getting stuck long or short
2. **Avellaneda-Stoikov (2008) — the three core formulas:**
   - **Reservation price:** r(t) = S(t) - q × γ × σ² × (T-t). When long (q>0), your "fair value" drops below mid-price (you're eager to sell)
   - **Optimal spread:** δ_a + δ_b = γσ²(T-t) + (2/γ)ln(1 + γ/κ). Wider when volatile, tighter when liquid
   - **Optimal quotes:** bid/ask symmetric around the reservation price, with spread as above
   - Parameters: γ (risk aversion), σ (volatility), κ (order book liquidity), T-t (time remaining)
3. **Simple LOB simulator:**
   - Mid-price follows Brownian motion: dS = σ dW
   - Order arrivals: Poisson with intensity λ(δ) = A × exp(-κδ) — fill probability decays with distance from mid
   - No actual order book matching engine needed — just fill probabilities
4. **RL enhancement — turning the simulator into a Gymnasium environment:**
   - State: [inventory, mid_price_return, time_remaining, spread, recent_volatility]
   - Action: [bid_distance, ask_distance] from mid-price (continuous → PPO or SAC)
   - Reward: ΔPnL - φ × inventory² (penalize large positions)
5. **Does RL beat A-S? The Alpha-AS result (PLOS ONE 2022):**
   - RL (Double DQN tuning A-S parameters) beats pure A-S on 24 of 30 days (Sharpe ratio)
   - RL dramatically improves PnL-to-max-adverse-position ratio (~20x)
   - But: RL introduces tail risk (worse max drawdown)
   - **When RL wins:** regime changes, mis-calibrated parameters, when additional signals (order flow) are available
   - **When A-S wins:** stable markets where its assumptions hold, risk management (predictable behavior)
6. **What the homework simplifies vs. real market making:**
   - No actual LOB (Poisson fill model instead)
   - No adverse selection, no queue position, no latency
   - Unit trade sizes, discrete time steps
   - We learn the **decision theory** (when to quote, how wide, how to manage inventory), not the **systems engineering**

### Seminar (90 min)
- **Exercise:** Implement the Avellaneda-Stoikov model from scratch (~100-150 lines Python). Run 1000 Monte Carlo simulations. Track PnL, inventory, and Sharpe ratio
- **Exercise:** Parameter sensitivity: vary γ (risk aversion) from 0.01 to 1.0. Show how it affects spread width and terminal inventory distribution
- **Exercise:** Wrap your simulator as a `gymnasium.Env`. Train a PPO agent using Stable-Baselines3 (100K-500K steps, ~5-10 minutes)
- **Exercise:** Compare three strategies on the same 100 simulated paths:
  1. Naive symmetric maker (fixed spread, no inventory skew)
  2. Avellaneda-Stoikov (analytical solution)
  3. RL agent (PPO)

### Homework
**"Market making: analytical vs. learned"**
1. **A-S implementation:** Implement the full Avellaneda-Stoikov model with Poisson fill probabilities. Parameters: S₀=100, σ=2, T=1 (normalized session), N=200 steps, γ=0.1, κ=1.5
2. **Monte Carlo analysis:** Run 10,000 simulations. Report: mean PnL, Sharpe ratio, max drawdown, mean |terminal inventory|, % of paths profitable
3. **Parameter sweep:** Create a heatmap of Sharpe ratio as a function of γ ∈ {0.01, 0.05, 0.1, 0.5, 1.0} and κ ∈ {0.5, 1.0, 1.5, 3.0, 5.0}. Identify the optimal parameter region
4. **Gymnasium environment:** Wrap your simulator as `MarketMakingEnv(gymnasium.Env)`. State = [inventory, mid_price_return, time_remaining]. Action = [bid_distance, ask_distance]
5. **RL agent:** Train PPO with Stable-Baselines3. Experiment with reward: (a) ΔPnL, (b) ΔPnL - 0.1 × inventory²
6. **The comparison:** Evaluate on 1000 fresh simulated paths:
   - Naive symmetric market maker
   - A-S with optimal parameters from step 3
   - RL agent (PPO)
   - Report: mean PnL, Sharpe, max drawdown, mean |terminal inventory|
7. **Regime test:** Run 500 paths with time-varying volatility (σ doubles halfway through the session). Which strategy adapts better?
8. **Deliverable:** Notebook + parameter heatmap + 3-strategy comparison + regime analysis

**Why this works as HW:** The A-S model is ~100 lines of Python. The Gymnasium wrapper is ~50 more. PPO training on a simple MLP takes 5-10 minutes. Students get hands-on experience with: analytical quant models, Monte Carlo simulation, RL environment design, and a genuine comparison of analytical vs. learned approaches. Everything runs on M4.

### Key Papers
| Paper | Year | Key Contribution |
|-------|------|------------------|
| Avellaneda & Stoikov, "HFT in a Limit Order Book" | 2008 | Foundational market making model |
| Spooner et al., "Market Making via RL" | AAMAS 2018 | SARSA(λ) with LOB features |
| Falces et al., "Alpha-AS: RL to Improve A-S" | PLOS ONE 2022 | RL vs. analytical comparison on BTC-USD |
| Gasperov & Kostanjcar, "Deep RL under Hawkes LOB" | IEEE CSL 2022 | DRL on Hawkes process simulator |
| Wang et al., "Market Making with Learned Beta Policies" | ICAIF 2024 | Current SOTA: RL-controlled beta distributions |
| Jerome et al., "mbt_gym" | ICAIF 2023 | Gymnasium environments for market making |

### Key Code References
- A-S implementation: [github.com/DYSIM/Avellaneda-Stoikov-Implementation](https://github.com/DYSIM/Avellaneda-Stoikov-Implementation)
- RL market making: [github.com/tspooner/rl_markets](https://github.com/tspooner/rl_markets)
- Gymnasium LOB environments: [github.com/JJJerome/mbt_gym](https://github.com/JJJerome/mbt_gym)
- Alpha-AS: [github.com/javifalces/HFTFramework](https://github.com/javifalces/HFTFramework)

---

# MODULE 5: ALTERNATIVE MARKETS & CAPSTONE
*"Beyond equities — and putting it all together."*

---

## Week 17 — Crypto & DeFi ML

**Course fit:** ★★★★☆ | **Implementability:** ✅ | **Importance:** 🟢 | **Format:** 📓

> Crypto is not "just another asset class." It has **fundamentally different data properties**
> that create both opportunities and challenges for ML:
> - 24/7 markets (no overnight gap, no weekends)
> - Public blockchain data (you can see *every* transaction, ever)
> - Decentralized exchanges with mathematically-defined pricing (AMMs)
> - Higher volatility, more regime changes, more structural breaks
> - Lower market efficiency (retail-dominated, less sophisticated participants)
>
> This week provides rich background on how DeFi works and where ML fits in.
> Many techniques from previous weeks (XGBoost, LSTM, RL) transfer directly —
> the key difference is the **data ecosystem** and the **unique features** blockchain provides.

### What specialists use
**Data:** `ccxt` (unified API for 100+ exchanges), Dune Analytics (SQL on blockchain data), Glassnode (800+ on-chain metrics), CoinGecko API, `web3.py` (Ethereum interaction), The Graph (indexed DeFi protocol data). **Trading:** Hummingbot (open-source MM for crypto), custom bots on CEX APIs. **DeFi-specific:** Uniswap V3 LP tools, MEV protection (Flashbots).

### Lecture (90 min)
1. **Crypto market structure — how it differs from TradFi:**
   - Centralized exchanges (CEX: Binance, Coinbase) vs. Decentralized exchanges (DEX: Uniswap, Curve)
   - CEX = traditional order book. DEX = Automated Market Maker (AMM) — a mathematical formula replaces the order book
   - 24/7 trading, global, largely unregulated, high retail participation
   - Stablecoins as the "risk-free" rate equivalent (USDC, USDT)
2. **On-chain data as a new feature set (you can't get this in TradFi):**
   - **Active addresses:** daily count of unique addresses transacting. Proxy for adoption/activity
   - **Exchange net flows:** coins moving to/from exchanges. Net inflow = selling pressure; net outflow = accumulation
   - **MVRV ratio (Market Value / Realized Value):** realized value = sum of each coin valued at its last movement price. MVRV > 3 → overheated; MVRV < 1 → undervalued
   - **SOPR (Spent Output Profit Ratio):** are transactors selling at profit or loss? SOPR < 1 = capitulation
   - **Hash rate (BTC) / Gas usage (ETH):** proxy for network security and demand
   - **Funding rates:** periodic payments between long/short futures traders. Positive = crowded longs; negative = crowded shorts. Strong mean-reversion signal
3. **DeFi mechanics — understanding the protocols:**
   - **AMMs and constant product formula:** x × y = k. Price is determined by pool ratios, not order matching
   - **Uniswap V3 concentrated liquidity:** LPs choose a price range [p_low, p_high] to provide liquidity. Earns more fees in-range but suffers **impermanent loss** when price moves out-of-range
   - **Impermanent loss:** the cost of providing liquidity. For a ±10% price move: ~0.5% IL. For 2x: ~5.7% IL. For 5x: ~25.5% IL
   - **Yield farming and liquidity mining:** protocols pay token rewards to attract liquidity. Often unsustainable
4. **Crypto-specific ML opportunities:**
   - **Funding rate mean-reversion:** when funding is extreme, short the crowded side
   - **On-chain feature engineering:** combine market data (OHLCV) with on-chain metrics for richer prediction models
   - **LP position optimization with RL:** dynamically adjust Uniswap V3 price ranges (Fan et al. 2023)
   - **Cross-exchange arbitrage:** price discrepancies between CEX and DEX
5. **MEV (Maximal Extractable Value) — the new front-running:**
   - Miners/validators can reorder transactions within a block for profit
   - Sandwich attacks: front-run and back-run a large DEX trade
   - Flashbots: infrastructure to mitigate MEV. Critical to understand if building DeFi strategies
6. **Key differences for ML practitioners:**
   - Much higher signal-to-noise ratio than equities (markets are less efficient)
   - But: faster alpha decay (more participants discover and exploit signals)
   - Extreme regime changes (Luna crash, FTX collapse, regulatory actions)
   - Data is abundant and free (blockchain is public), but noisy (wash trading, bot activity)

### Seminar (90 min)
- **Exercise:** Pull BTC and ETH daily OHLCV from Binance using `ccxt`. Compute returns, volatility, and basic features. Compare distributional properties to SPY (fatter tails, higher vol, no overnight gap)
- **Exercise:** Fetch on-chain metrics from Glassnode free tier (active addresses, exchange net flows, MVRV). Merge with price data. Compute IC between on-chain features and next-day returns
- **Exercise:** Simulate impermanent loss for a Uniswap V3 position: given a price range [0.9S, 1.1S], compute IL for a range of price movements. Plot IL vs. price change
- **Discussion:** What techniques from the course transfer directly to crypto? Which need adaptation? What's genuinely different?

### Homework
**"Crypto return prediction with on-chain features"**
1. **Market data:** Download 2 years of daily BTC and ETH OHLCV from Binance via `ccxt`
2. **On-chain features:** From Glassnode free tier or public API:
   - Active addresses (daily)
   - Exchange net flows
   - MVRV ratio (BTC)
   - Funding rate (Binance BTC perpetual futures)
3. **Feature engineering:** Compute 10-15 features combining market and on-chain data:
   - Price features: 5/10/20-day momentum, 20-day vol, volume ratio
   - On-chain: MVRV z-score, exchange flow ratio, address growth rate, funding rate z-score
4. **Model comparison:**
   - Model A: XGBoost on price/volume features only
   - Model B: XGBoost on price/volume + on-chain features
   - Model C: LSTM on raw price sequences (Week 8 style)
   - Model D: LSTM + on-chain features as static inputs
5. **Evaluation:** IC, directional accuracy, Sharpe of a simple long/flat strategy (long when predicted return > 0, flat otherwise)
6. **Uniswap V3 LP simulation:**
   - Using ETH/USDC historical prices, simulate a concentrated LP position with range width = 1.5× current ATR (average true range)
   - Compute: total fees earned, impermanent loss, net P&L vs. just holding ETH
   - Compare 3 range widths: narrow (0.5× ATR), medium (1.5× ATR), wide (3× ATR)
7. **Deliverable:** Notebook + on-chain feature analysis + model comparison + LP simulation

**Why this works as HW:** ccxt and Glassnode free tier provide all needed data with no cost. The on-chain features are genuinely novel — students see a data type that doesn't exist in traditional finance. The LP simulation teaches AMM mechanics through direct computation. Everything runs on M4.

### Key Papers
| Paper | Year | Key Contribution |
|-------|------|------------------|
| Fan et al., "Adaptive LP in Uniswap V3 with Deep RL" | arXiv 2023 | PPO agent for LP position management |
| "DeFi LP with DRL" | arXiv 2025 | MDP formulation for concentrated LP optimization |
| "Uniswap LP: An Online Learning Approach" | arXiv 2023 | Theoretical framework for LP as online learning |
| Survey of Deep Learning in Cryptocurrency | PMC 2023 | Comprehensive review of DL methods for crypto |

### Key Libraries
| Library | Purpose |
|---------|---------|
| `ccxt` | Unified API for 100+ crypto exchanges. OHLCV, order books, trades. No API key for public data |
| `web3.py` | Ethereum blockchain interaction. Read smart contract state, decode events |
| `dune-client` | Python client for Dune Analytics. SQL queries on blockchain data |
| `pycoingecko` | CoinGecko API wrapper for price/market data |
| `hummingbot` | Open-source market making bot for crypto. Supports Uniswap, Binance, etc. |

---

## Week 18 — Backtesting, Strategy Evaluation & Capstone Integration

**Course fit:** ★★★★★ | **Implementability:** ✅ | **Importance:** 🔴 | **Format:** 📓

> The final piece: putting it all together into a production-quality strategy.
> This is the week where methodology rigor meets implementation reality.
>
> **Note:** Basic risk metrics (Sharpe, Sortino, Calmar, max DD) were taught in Week 3.
> This week focuses on: (1) advanced evaluation metrics specific to strategy assessment,
> (2) the deflated Sharpe ratio and multiple testing correction, (3) realistic backtesting
> methodology, and (4) causal inference in factor investing.

### What specialists use
`QuantConnect/LEAN` (production backtesting), `VectorBT` (fast research), internal execution simulators with realistic market impact models. Sharpe ratio, deflated Sharpe ratio, probabilistic Sharpe ratio for evaluation.

### Lecture (90 min)
1. **The backtesting lie:** why most academic backtests don't work in practice
   - Transaction costs (bid-ask spread + market impact + commissions) — now you understand these from Week 15
   - Slippage and fill rates (market impact models: linear, square-root, Almgren-Chriss)
   - Capacity constraints (your strategy can't trade $1B)
   - Regime changes and alpha decay
2. **The deflated Sharpe ratio (Bailey & Lopez de Prado):**
   - The problem: if you test N strategies, the best Sharpe is inflated by selection bias
   - DSR formula: SR₀ = √(2 ln N) adjusted for skewness and kurtosis, then z-test against observed Sharpe
   - **Be honest about N** — count every model, parameter set, and feature combination you tried
   - Probability of backtest overfitting (PBO)
3. **Walk-forward optimization (formal treatment):**
   - Expanding-window CV (from Week 4) retrains the model but not hyperparameters
   - Walk-forward optimization: retrain model AND re-tune hyperparameters at each step
   - The tradeoff: more realistic but slower and risks overfitting the tuning process
4. **Causal inference in factor investing (Lopez de Prado 2023):**
   - The "factor mirage" — most published factors are associational, not causal
   - Why backtested factors fail out-of-sample: you found a correlation, not a mechanism
   - Lopez de Prado's research protocol: theory → causal graph → testable predictions → backtest
5. **Advanced strategy evaluation metrics (beyond Week 3 basics):**
   - Profit factor (gross profit / gross loss)
   - Tail ratio (95th percentile return / 5th percentile)
   - Hit rate and payoff ratio
   - Rolling Sharpe ratio (stability of performance over time)
6. **From signal to strategy (the full pipeline):**
   - Signal generation (your ML model)
   - Portfolio construction (signals → positions)
   - Execution (how to actually trade — connects to Week 15/16)
   - Risk management (when to cut positions)

### Seminar (90 min)
- **Exercise:** Take your best model from the course (Week 5 or 7). Build a full VectorBT backtest with realistic transaction costs (10 bps per side)
- **Exercise:** Implement walk-forward optimization: retrain your model every month, evaluate on the next month
- **Exercise:** Calculate the deflated Sharpe ratio. How many strategies did you try during this course? Is your best result still significant?
- **Discussion:** Build the same strategy with different random seeds and hyperparameters. What's the distribution of Sharpe ratios? This is the "garden of forking paths" — honest accounting matters

### Homework (Capstone)
**"End-to-end ML trading strategy"**
Build a complete, production-quality ML trading strategy that combines at least 2 techniques from the course:

1. **Data pipeline:** Clean data acquisition with proper handling of biases
2. **Feature engineering:** At least 15 features spanning momentum, volatility, volume, fundamental, and/or sentiment categories
3. **Model:** At least one ML model trained with proper financial CV (purged k-fold or expanding window)
4. **Labeling:** Triple-barrier labeling OR properly-justified alternatives
5. **Portfolio construction:** Signal → portfolio weights (can be simple rank-based or optimized)
6. **Backtest:** Walk-forward with realistic transaction costs (≥5 bps per side)
7. **Evaluation:** Full QuantStats tear sheet + deflated Sharpe ratio (be honest about n_trials!)
8. **Analysis:** Honest discussion of limitations, potential overfitting, capacity, and what you'd do differently
9. **Deliverable:** Full Jupyter notebook or Python package + a 5-page write-up

**Why this works as capstone:** It integrates everything. Students demonstrate that they can do financial ML correctly, not just run a model. The written analysis is as important as the code. This is the portfolio piece they'd show at a quant firm interview.

---

# SUMMARY TABLE

| Week | Topic | Fit | Impl. | Importance | Format | Key Technique | Key Library | Vintage |
|------|-------|-----|-------|------------|--------|---------------|-------------|---------|
| 1 | Markets & Data Structures | ★★★★★ | ✅ | 🔴 | 📓 | Volume/dollar bars | yfinance, pandas | Timeless |
| 2 | Time Series & Stationarity | ★★★★★ | ✅ | 🔴 | 📓 | Fractional differentiation | fracdiff, ARCH | 2018+ |
| 3 | Portfolio Theory & Risk | ★★★★☆ | ✅ | 🔴 | 📓 | HRP, factor models | PyPortfolioOpt, alphalens | Timeless |
| 4 | Linear Models for Returns | ★★★★★ | ✅ | 🔴 | 📓 | Lasso/Ridge + IC analysis | scikit-learn | 2020 (GKX) |
| 5 | Tree-Based Methods | ★★★★★ | ✅ | 🔴 | 📓 | XGBoost + SHAP | XGBoost, LightGBM, SHAP | **Still SOTA 2025** |
| 6 | Financial ML Methodology | ★★★★★ | ✅ | 🔴 | 📓 | Triple-barrier + meta-labeling | Custom + VectorBT | 2018+ |
| 7 | Feedforward Nets for Pricing | ★★★★★ | ✅ | 🟡 | 📓 | Gu-Kelly-Xiu replication | PyTorch | 2020, extended 2024 |
| 8 | LSTM/GRU for Volatility | ★★★★★ | ✅ | 🟡 | 📓 | LSTM vs GARCH | PyTorch, ARCH | Still default |
| 9 | Foundation Models for Finance | ★★★★★ | ✅/⚡ | 🔴 | 📓 | Kronos + hybrid FM→XGBoost | Kronos, Chronos, pytorch-forecasting | **Frontier 2025-2026** |
| 10 | NLP: FinBERT → LLM embeddings | ★★★★★ | ✅/⚡ | 🔴 | 📓 | LLM embeddings (Chen-Kelly-Xiu 2023) | sentence-transformers, FinGPT | **Updated 2023-2025** |
| 11 | **Bayesian DL & Uncertainty** | ★★★★★ | ✅ | 🟡 | 📓 | MC Dropout + Ensembles | PyTorch (torch.nn.Dropout) | 2016-2017 foundations |
| 12 | **Graph Neural Networks** | ★★★★☆ | ✅ | 🟡 | 📓 | GAT + stock relationship graphs | torch-geometric, networkx | **AAAI 2024 SOTA** |
| 13 | RL for Portfolios | ★★★★☆ | ⚡ | 🟡 | 📓 | PPO/SAC portfolio agent | FinRL, SB3 | Hybrid RL trend 2025 |
| 14 | Neural Options Pricing | ★★★★☆ | ✅ | 🟡 | 📓 | Autograd Greeks, deep hedging | PyTorch, QuantLib | 2019+ |
| 15 | **HFT & Microstructure ML** | ★★★★☆ | ✅ | 🟡 | 📖 | DeepLOB, OFI, VPIN | FI-2010, LOBFrame | 2014-2025 |
| 16 | **Market Making with ML** | ★★★★★ | ✅ | 🟡 | 📓 | Avellaneda-Stoikov + RL | Gymnasium, SB3 | 2008 + RL 2022 |
| 17 | **Crypto & DeFi ML** | ★★★★☆ | ✅ | 🟢 | 📓 | On-chain features, AMM/LP sim | ccxt, web3.py, Glassnode | **Active 2023-2025** |
| 18 | Backtesting + Causal Inference | ★★★★★ | ✅ | 🔴 | 📓 | Deflated Sharpe + causal factors | VectorBT, QuantConnect | **Updated 2023 (LdP)** |

---

# DEPENDENCY GRAPH

```
Week 1 (Data) ──→ Week 2 (Time Series) ──→ Week 3 (Portfolio/Risk)
                                                    │
                                                    ▼
Week 4 (Linear Models) ──→ Week 5 (Trees/XGBoost) ──→ Week 6 (Methodology)
         │                          │
         │                          ▼
         │              Week 7 (Neural Nets) ──→ Week 8 (LSTM/GRU)
         │                    │                        │
         │                    ├──→ Week 9 (Foundation Models)
         │                    │
         │                    ├──→ Week 10 (NLP/Embeddings)
         │                    │
         │                    ├──→ Week 11 (Bayesian DL) ←── builds on W7 dropout/ensembles
         │                    │
         │                    └──→ Week 12 (GNNs) ←── uses W4-5 cross-sectional framework
         │
         │              Week 13 (RL) ──→ Week 16 (Market Making)
         │                                      │
         │                              Week 15 (HFT/Microstructure) ←── provides context for W16
         │
         │              Week 14 (Neural Options) ←── uses W7 PyTorch skills
         │
         │              Week 17 (Crypto/DeFi) ←── relatively standalone, uses W5 XGBoost + W8 LSTM
         │
         └──────────→ Week 18 (Backtesting + Capstone) ←── integrates everything
```

---

# INSTALLATION

```bash
# Full course stack (run once)
pip install torch torchvision torchaudio
pip install numpy pandas scipy matplotlib seaborn jupyter
pip install scikit-learn xgboost lightgbm catboost optuna shap
pip install yfinance fredapi pandas-datareader
pip install arch statsmodels fracdiff
pip install pyportfolioopt riskfolio-lib
pip install vectorbt quantstats alphalens-reloaded pyfolio-reloaded empyrical-reloaded
pip install pytorch-forecasting pytorch-lightning
pip install transformers datasets sentence-transformers  # Hugging Face + embeddings (Week 10)
pip install chronos-forecasting  # Amazon Chronos (Week 9)
# Kronos: pip install from GitHub — see https://github.com/shiyu-coder/Kronos
# TimesFM: pip install timesfm (has Apple Silicon support)
pip install finrl stable-baselines3 gymnasium
pip install QuantLib-Python
pip install mplfinance plotly

# Week 11 (Bayesian DL) — optional extras
pip install pymc arviz  # for Bayesian linear regression with MCMC (optional, not required)

# Week 12 (GNNs)
pip install torch-geometric
pip install networkx

# Week 16 (Market Making) — no new installs needed (uses gymnasium + stable-baselines3)

# Week 17 (Crypto/DeFi)
pip install ccxt pycoingecko
pip install web3  # for on-chain data (optional)

# Optional: LLM embedding APIs (Week 10 stretch goals)
pip install openai anthropic  # For paid embedding APIs
pip install qlib  # Microsoft's AI-for-quant platform
```

---

# WHAT I DELIBERATELY LEFT OUT (AND WHY)

| Topic | Why excluded | Current status (2025-2026) |
|-------|-------------|---------------------------|
| **Quantum ML for finance** | Purely experimental. No practical implementations that outperform classical | Still experimental |
| **Agentic AI (full pipeline)** | Man Group's AlphaGPT is real, but building your own requires LLM API costs + significant infrastructure. Discussed conceptually in Week 10, not implemented as HW | **Biggest industry shift of 2024-2025** — discussed in lecture, not assigned |
| **Diffusion models for financial data** | Active research (FTS-Diffusion, ICLR 2024; DDPM+wavelets, Quant Finance 2025) but not yet production-standard | **Emerging but immature** |
| **Time series foundation models (full fine-tuning)** | Computationally expensive and results are mixed. Week 9 covers finance-native FMs (Kronos) and the hybrid approach instead | **Covered in Week 9 via Kronos/hybrid approach** |
| **Autoencoders / VAEs / Generative models** | Interesting for yield curve modeling and synthetic data, but provides less practical value than the 5 new topics added (Bayesian DL, GNNs, HFT, Market Making, Crypto). Can be explored as a capstone variant | Removed to make room for higher-priority topics |
| **Full Bayesian neural networks (VI, HMC)** | Prohibitively expensive beyond tiny networks. Week 11 covers the practical sweet spot (MC Dropout + Ensembles) instead | Research-only |

---

# SUGGESTED READING ORDER

1. **Before the course starts:** Lopez de Prado — *Advances in Financial Machine Learning*, Chapters 1–5
2. **Weeks 1–3:** Harris — *Trading and Exchanges*, Chapters 1–6 (skim); Grinold & Kahn — *Active Portfolio Management*, Chapters 1–3
3. **Weeks 4–6:** Gu, Kelly, Xiu — *Empirical Asset Pricing via Machine Learning* (paper); Lopez de Prado, Chapters 6–9
4. **Weeks 7–8:** Zhang & Zohren — *Deep Learning in Quantitative Trading* (2024), Chapters on feedforward/sequence models; Jansen — *Machine Learning for Algorithmic Trading*, relevant chapters
5. **Week 9:** Kronos paper (AAAI 2026); FinCast paper (CIKM 2025); "Re(Visiting) TSFMs in Finance" (2025)
6. **Week 10:** Chen, Kelly, Xiu — *Expected Returns and Large Language Models* (2023)
7. **Week 11:** Gal & Ghahramani (2016) — "Dropout as a Bayesian Approximation"; Lakshminarayanan et al. (2017) — "Simple and Scalable Predictive Uncertainty"
8. **Week 12:** Feng et al. (2019) — "Temporal Relational Ranking"; ACM Computing Surveys GNN review (2024); MASTER paper (AAAI 2024)
9. **Week 13:** Rao — *Foundations of RL* (free PDF), Chapters 1–5
10. **Week 14:** Hull — *Options, Futures*, Chapters 13–15; Buehler et al. (2019) — "Deep Hedging"
11. **Week 15:** Cont, Kukanov, Stoikov (2014) — "Price Impact of Order Book Events"; Zhang et al. (2019) — "DeepLOB"; Harris — *Trading and Exchanges*, Chapters 7–12
12. **Week 16:** Avellaneda & Stoikov (2008); Falces et al. (2022) — "Alpha-AS" (PLOS ONE)
13. **Week 17:** Fan et al. (2023) — "Adaptive LP in Uniswap V3 with Deep RL"; Survey of DL in Cryptocurrency (2023)
14. **Week 18:** Lopez de Prado — *Causal Factor Investing* (2023), Chapters 1–3; Bailey & Lopez de Prado — "Deflated Sharpe Ratio" and "Probability of Backtest Overfitting"
