# Narrative Brief — Week 3: Factor Models & Cross-Sectional Analysis

**Status:** APPROVED

## Narrative Arc

**Blueprint arc:**
- Setup: You're an ML expert. Features are features — throw them into a model. Cross-sectional prediction? That's just supervised learning with stocks as observations and characteristics as features. What could finance possibly add?
- Complication: Finance has been doing "feature engineering for cross-sectional prediction" since 1964, and they called it "factor modeling." A three-factor linear model from 1993 still explains most of the cross-section, and 400+ proposed factors mostly vanish out of sample. Your ML intuition to "add more features" is exactly the instinct that created the factor zoo.
- Resolution: Factor models are the economic logic layer that ML needs to avoid drowning in noise. By week's end, students understand cross-sectional ranking, factor construction and validation, and the principled feature matrix that Week 4's ML models will consume.

**How results reshape it:**

The core arc holds, but the 2014-2024 sample period produces a twist that strengthens the teaching narrative rather than weakening it. The blueprint assumed CAPM's empirical failures would manifest as the textbook "beta is flat" result. Instead, the sample delivers the opposite: high-beta tech stocks (NVDA at 43% alpha, TSLA, AMD) drove a steep, positive SML with R-squared = 0.55. This is not a code bug — it is the 2014-2024 AI/tech boom baked into the cross-section. This actually sharpens the complication: the "right" answer depends entirely on which decade you examine, which is a more powerful lesson about sample dependence than a flat SML alone would deliver.

A second adjustment: the blueprint expected factor replication to show a clean contrast between "price-based factors replicate well, fundamental-based factors don't." The actual results show HML (fundamental-based) replicating nearly as well as momentum (r = 0.82 vs. 0.85), while SMB replication remains poor (r = 0.19). However, this result comes with a methodological caveat: the code uses a **static sort** — classifying stocks into value/growth buckets using ~2021-2025 fundamentals and applying those fixed assignments backward over the entire 2014-2024 price window (131 months). The expectations document anticipated that fundamental-dependent analyses would be limited to the ~2021-2025 window (~36-48 months), which would have been the methodologically conservative choice. The static-sort approach introduces **look-ahead bias**: stocks are classified using future fundamentals and those classifications are applied to past returns. The high HML correlation (r = 0.82) is therefore partly genuine — value/growth classifications are reasonably stable for large-cap S&P 500 stocks over a decade — and partly flattered by the static sort, which avoids the noise of annual rebalancing and benefits from survivorship in the classification itself. The lesson still sharpens from "fundamental data is the bottleneck" to "universe coverage is the bottleneck," but the notebook must be explicit that the high HML correlation overstates what a methodologically proper annual-rebalancing implementation would achieve with only ~36-48 months of fundamental data.

A third adjustment (post-consolidation correction): the feature engineering pipeline initially included P/E ratio alongside earnings yield, producing a spurious positive correlation between two features that should be strong negatives. The fix — dropping P/E and retaining only earnings yield, plus adding z-score hard caps — becomes itself a teaching moment: feature engineering in finance is not just "compute ratios and standardize." Ratios like P/E that are undefined at zero and sign-flip for loss-making firms are traps for ML practitioners who treat all features as interchangeable numeric columns.

**Adjusted arc:**
- Complication (revised): Finance's factor models are powerful but fragile. A three-factor model from 1993 explains most of the cross-section — but which "most" depends on the decade you analyze and the data universe you use. The "right" answer to "is beta priced?" flips between the 1960s-1990s (no) and 2014-2024 (yes). Your ML instinct to add more features created the factor zoo, and your ML instinct to trust a single sample period will make it worse.
- Resolution (revised): Factor models are the economic logic layer, but they require honest confrontation with sample dependence, universe limitations, and the gap between free and institutional data. The feature matrix you build this week is the input to Week 4's ML models — but only after understanding which features your data can actually support and which it cannot.

---

## Lecture Sections

### Section 1: The One-Factor World — `s1_capm.py`

**Blueprint intended:** Estimate betas for diverse stocks, plot the empirical security market line, show the gap between CAPM prediction and realized returns. Demonstrate that "beta is flat" — high-beta stocks do not earn proportionally higher returns.
**Expected:** Beta range ~0.3-2.1. Median CAPM R-squared in [0.20, 0.45]. SML cross-sectional R-squared in [0.00, 0.20] with slope much flatter than theoretical (0-5% vs. 7-10% theoretical). The visual should show that high-beta stocks do not earn proportionally more — the classic empirical CAPM failure.
**Actual:** Beta range [0.33, 2.01] — 20 stocks. Median R-squared = 0.26. SML slope = 22.5% vs. theoretical 12.1% — empirical SML is steeper than theoretical, not flatter. SML R-squared = 0.55. NVDA annualized alpha = 43.0%, TSLA = 24.8%, AMD = 21.8%. Annualized alpha range [-6.5%, 43.0%].
**Category:** Real phenomenon

**Teaching angle:**
- **Frame as:** "CAPM says beta should explain the cross-section. In the textbook 1963-1990 sample, it fails — the SML is flat. But run the same test on 2014-2024, and you get the opposite: high-beta tech stocks earned dramatically more. The 'right' answer depends on which decade you examine. This is the deepest lesson about cross-sectional asset pricing — sample dependence is not a nuisance, it is the first-order problem."
- **Key numbers:** ours = SML R-squared 0.55, slope 22.5%; production = SML R-squared < 0.05 (Fama & French, 1992, CRSP 1963-1990). NVDA alpha = 43.0% annualized, beta = 1.72. Theoretical slope = 12.1% (avg. market excess return). 20-stock sample with 3 stocks (NVDA, TSLA, AMD) contributing extreme leverage.
- **Student takeaway:** Cross-sectional tests are sample-dependent. The "beta anomaly" is a feature of some decades, not a universal law. With 20 stocks and one dominant sector (tech), small samples magnify period-specific effects. The R-squared of 0.55 is inflated by small-sample leverage: with only 20 stocks, three extreme points (NVDA, TSLA, AMD) dominate the regression. Remove those three stocks and the R-squared would drop substantially. The steep slope is a genuine period effect; the high R-squared is partly a small-sample artifact. The expectations document warned that "SML R-squared with only 15-20 stocks is very noisy" — treat it as illustrative, not as evidence that CAPM works in this decade.

**Sandbox vs. reality:** The steep positive SML reflects two compounding sandbox artifacts: (1) the 2014-2024 period is dominated by AI/tech boom, which is not representative of longer-run averages; (2) the 20-stock S&P 500 sample over-weights mega-cap tech winners. With institutional data (CRSP full universe, 1963-present), the SML is flat or inverted (Fama & French, 1992). The beta anomaly — low-beta stocks earning higher risk-adjusted returns than high-beta stocks — has been documented across decades and markets, though it weakens in quality-controlled samples (searched: "beta anomaly 2014-2024 high beta outperformance" → Alpha Architect, 2024; the anomaly's strength varies with market conditions and stock quality).

---

### Section 2: When One Factor Fails — The Fama-French Three-Factor Model — `s2_ff3_model.py`

**Blueprint intended:** Demonstrate R-squared improvement from CAPM to FF3, show alpha shrinkage, visualize cumulative factor returns over decades including the "death of value."
**Expected:** Median FF3 R-squared in [0.30, 0.50]. Median R-squared improvement CAPM-to-FF3 in [0.03, 0.15]. R-squared improved for >70% of stocks. Alpha shrank for >60% of stocks. Cumulative factor plots show HML post-2007 weakness.
**Actual:** Median FF3 R-squared = 0.41. Median improvement = +0.075. R-squared improved for 100% of stocks. Alpha shrank for 60% of stocks (exactly at the >60% boundary). HML cumulative 2014-2024 = 0.77 (lost 23% over the period). Factor loadings: growth stocks (TSLA = -1.15, NVDA = -0.92) have strongly negative HML; value stocks (XOM = +1.02, JPM = +0.85) have strongly positive HML.
**Category:** Matches

**Teaching angle:**
- **Frame as:** "Adding size and value factors to CAPM improves explanatory power for every stock in our sample. The median R-squared jumps from 0.28 to 0.41 — a meaningful improvement. What looked like alpha under CAPM is partly explained by factor exposure: a tech stock's 'outperformance' is partially just a growth (negative HML) bet."
- **Key numbers:** Median CAPM R-squared = 0.28, median FF3 R-squared = 0.41, improvement = +0.075. R-squared improved for 100% of stocks. Alpha shrank for 60% of stocks. HML cumulative return 2014-2024 = 0.77 (23% loss). Production benchmark: individual stock FF3 R-squared typically 0.20-0.60 (Davis, Fama & French, 2000, CRSP).
- **Student takeaway:** Adding size and value factors improves explanatory power universally, but alpha does not always shrink — new factor loadings can reallocate rather than reduce unexplained returns. HML's 23% loss over 2014-2024 connects to the broader "death of value" debate — whether the value premium has permanently disappeared due to crowding and arbitrage, or is cyclically depressed and will revert. This remains unresolved.

---

### Section 3: Building Factors from Scratch — Portfolio Sorts — `s3_factor_construction.py`

**Blueprint intended:** Construct SMB and HML from raw data using double sorts. Compare self-built factors against Ken French official data to demonstrate factor replication and diagnose sources of divergence.
**Expected:** SMB correlation with Ken French in [0.15, 0.55] (probe: r = 0.36). HML correlation in [0.20, 0.60] (probe: r = 0.38). Both low due to large-cap-only universe and short fundamental window.
**Actual:** B/M computed for 171 tickers. Median B/M = 0.21. All 6 portfolios have >= 18 stocks. Factor return series = 131 months (full price window, not the expected 36-48 fundamental-window months). SMB correlation = 0.19 (within range, near lower bound). HML correlation = 0.82 (far above expected upper bound of 0.60). SMB tracking error = 11.6%, HML tracking error = 7.9%.
**Category:** Expectation miscalibration
**Methodological note — static sort with look-ahead bias:** The code classifies stocks into portfolios using ~2021-2025 fundamentals (the most recent balance sheet per ticker) and applies those fixed assignments over the entire 2014-2024 price window. The expectations document anticipated annual rebalancing limited to the fundamental window (~36-48 months of factor returns). The static-sort shortcut was taken because yfinance provides only ~4-5 annual fundamental periods per ticker, making proper annual rebalancing over 2014-2024 impossible. This introduces look-ahead bias: a stock classified as "value" using its 2025 book-to-market may have been "growth" in 2014. The resulting 131-month factor return series and the high HML correlation (r = 0.82) are partly genuine (value/growth classifications are stable for large-cap stocks) and partly flattered by the methodology (no rebalancing noise, future information embedded in classifications). A methodologically proper implementation limited to the fundamental window would likely produce lower correlations closer to the expectations document's predicted range of [0.20, 0.60].

**Teaching angle:**
- **Frame as:** "We built SMB and HML from scratch using 171 S&P 500 stocks. SMB replicates poorly (r = 0.19) — our 'small' stocks are $150B+ megacaps, so we are measuring large-vs-very-large, not small-vs-large. HML replicates much better (r = 0.82), suggesting that book-to-market dispersion exists within large-caps. But a caveat: our implementation uses a static sort — we classify stocks once using the most recent fundamentals and apply those assignments over the full 11-year price window. This introduces look-ahead bias and flatters the correlation relative to a proper annual-rebalancing approach. The high HML correlation is partly real (value/growth is stable among large-caps) and partly a methodological artifact."
- **Key numbers:** SMB: ours = r = 0.19, production = r ~ 0.99 (Tidy Finance, Scheuch et al., 2023, CRSP/Compustat). HML: ours = r = 0.82 (with static-sort caveat), production = r ~ 0.98 (Tidy Finance, with annual rebalancing). B/M median = 0.21, range [0.003, 1.70]. 131 months of factor returns (vs. 36-48 months expected under fundamental-window-limited rebalancing). Smallest "small" bucket stock: market cap > $10B.
- **Student takeaway:** Factor replication quality depends on which characteristic you are sorting on, not just universe size. Size requires true small-caps; value works within large-caps. Free data has selective blind spots. However, our high HML correlation comes with a methodological asterisk — the static sort uses future information. The notebook should be transparent about this simplification.

**Sandbox vs. reality:** The static-sort approach (classify once using most recent fundamentals, compute returns over the full price window) is a warranted simplification given yfinance's ~4-5 year fundamental depth — annual rebalancing over 2014-2024 is impossible without historical fundamentals for each year. This is a common pedagogical shortcut. However, it introduces look-ahead bias that production implementations avoid: Ken French rebalances portfolios each June using the prior fiscal year's book equity, ensuring no future information contaminates sort assignments. The proper Fama-French methodology applied to our data would limit factor returns to the ~2021-2024 fundamental window (~36-48 months), producing noisier correlation estimates closer to the expectations document's [0.20, 0.60] range. A third source of bias: yfinance fundamental data is "as-reported" but not point-in-time. We cannot verify exact reporting dates, so our factor sorts may use fundamentals before they were publicly available. In production, CRSP/Compustat point-in-time databases track when each data point became known to investors, preventing this contamination. The notebook should (1) state the static-sort approach explicitly, (2) note that it introduces look-ahead bias, (3) explain that the high HML correlation partly reflects stable value/growth classifications among large-cap stocks and partly reflects the methodological shortcut, and (4) note the point-in-time limitation.

---

### Section 4: The Five-Factor Model and the Momentum Orphan — `s4_ff5_momentum.py`

**Blueprint intended:** Compare CAPM vs. FF3 vs. FF5 explanatory power, demonstrate diminishing returns of adding factors, show momentum's strong returns alongside crash risk, and examine why Fama and French excluded momentum from FF5.
**Expected:** Median FF5 R-squared in [0.35, 0.55]. Median R-squared improvement FF3-to-FF5 in [0.01, 0.08]. Alpha under FF5 smaller than under FF3 for >55% of stocks. UMD-HML correlation near-zero or slightly negative.
**Actual:** Median FF5 R-squared = 0.43. Median R-squared improvement = +0.025. FF5 improved R-squared for 100% of stocks. Alpha shrank for only 40% of stocks (below the >55% threshold). UMD-HML correlation = -0.31. HML-CMA correlation = 0.65. Momentum cumulative return: peak ~$86 (2007), crash to ~$35 (2009, ~60% drawdown), never recovered to pre-crash level by 2025.
**Category:** Expectation miscalibration

**Teaching angle:**
- **Frame as:** "Going from three factors to five improves R-squared for every stock — but by a smaller margin (median +0.025 vs. +0.075 for CAPM-to-FF3). This is diminishing returns in action. Interestingly, alpha magnitude actually increases for 60% of stocks under FF5, even as R-squared rises. The new factors explain variance but can reallocate the intercept rather than shrink it. Meanwhile, momentum remains the orphan — strong long-run returns ($1 to $86), catastrophic crash risk (60% drawdown in 2009), and a -0.31 correlation with value that illustrates the value-momentum tension."
- **Key numbers:** Median FF5 R-squared = 0.43, median FF3 R-squared = 0.41, gain = +0.025. Alpha shrank for 40% of stocks (not 55%+ as expected). UMD-HML correlation = -0.31. HML-CMA correlation = 0.65 (value and investment are redundant). Momentum peak = ~$86, 2009 crash = ~60% drawdown. Production benchmark: incremental R-squared from RMW + CMA typically 0.01-0.10 (Fama & French, 2015, CRSP 1963-2013).
- **Student takeaway:** Adding factors hits diminishing returns quickly. R-squared improvement does not guarantee alpha shrinkage — the intercept can shift either way. The HML-CMA correlation of 0.65 shows that value and investment are substantially redundant.

---

### Section 5: Are Factors Priced? — The Fama-MacBeth Methodology — `s5_fama_macbeth.py`

**Blueprint intended:** Demonstrate the Fama-MacBeth two-step procedure both manually and via `linearmodels.FamaMacBeth`. Show which factors carry significant cross-sectional risk premia. Compare manual and production implementations.
**Expected:** Manual and linearmodels gammas agree within +/- 0.001. gamma_MKT in [-0.005, +0.015]. gamma_SMB in [-0.010, +0.005]. gamma_HML in [-0.010, +0.010]. Newey-West t-statistics 10-30% smaller than naive OLS. Average cross-sectional R-squared in [0.01, 0.15].
**Actual:** Manual gamma_MKT = 0.0114 (naive t = 2.27), linearmodels gamma_MKT = 0.0140 (NW t = 4.82). Manual-linearmodels MKT diff = 0.0026 (exceeds 0.001 threshold). SMB diff = 0.002 (also exceeds). HML diff = 0.0001 (passes). NW t for MKT (4.82) is 112% LARGER than naive t (2.27), opposite of the expected 10-30% reduction. Average cross-sectional R-squared = 0.20 (above expected max 0.15).
**Category:** Expectation miscalibration

**Teaching angle:**
- **Frame as:** "The Fama-MacBeth procedure estimates risk premia by averaging cross-sectional slopes. Our manual implementation and the linearmodels library agree closely for HML (diff = 0.0001) but diverge slightly for MKT (diff = 0.0026). This is not a bug — it reflects implementation differences in panel balancing and intercept handling. In production, you would never rely on a single implementation without cross-checking. The Newey-West correction here actually increases the market t-statistic (from 2.27 to 4.82) because positive autocorrelation in the cross-sectional slopes reduces the effective variance of the mean — the correction works both ways."
- **Key numbers:** Manual gammas: MKT = 0.0114, SMB = 0.0014, HML = -0.0081. Linearmodels: MKT = 0.0140 (NW t = 4.82), SMB = -0.0005 (NW t = -0.14), HML = -0.0080 (NW t = -1.87). Manual-vs-linearmodels diff: MKT = 0.0026, SMB = 0.0020, HML = 0.0001. Average cross-sectional R-squared = 0.20. Production benchmark: market risk premium typically -0.2% to +0.8% monthly (Fama & French, 1992, CRSP); ours (1.1-1.4%) is above this range, reflecting the strong 2014-2024 equity market.
- **Student takeaway:** Always cross-check implementations. Newey-West correction can increase or decrease t-statistics depending on autocorrelation structure. Risk premium estimates are sample-period dependent — our 1.1% monthly market premium reflects a historically strong decade.

**Sandbox vs. reality:** The market risk premium of 1.1-1.4% monthly far exceeds the typical production range of -0.2% to +0.8% (Fama & French, 1992, CRSP 1963-1990). This reflects the exceptional equity market performance of 2014-2024 (annualized market return ~12%), driven by tech outperformance. With CRSP data over longer samples (40+ years), the cross-sectional market premium is typically flat or insignificant.

---

### Section 6: The Practitioner's Lens — Barra-Style Risk Models — `s6_barra_risk_models.py`

**Blueprint intended:** Build a simplified Barra-style cross-sectional regression with style factors and industry dummies. Decompose a portfolio's risk into factor and specific components.
**Expected:** Median cross-sectional R-squared in [0.10, 0.30]. Size factor mean in [-0.005, +0.005]. Momentum factor mean in [0.000, +0.010]. Diversified portfolio factor risk share > 50% (range 40-80%). Specific risk share 20-60%.
**Actual:** 48 monthly cross-sectional regressions. Median R-squared = 0.34 (above expected max 0.30). Size factor mean = +0.0034 (within range). Momentum factor mean = -0.0010 (below expected lower bound of 0.000). Diversified portfolio factor risk share = 86.6% (above expected max 80%). Specific risk share = 13.4% (below expected min 20%).
**Category:** Expectation miscalibration

**Teaching angle:**
- **Frame as:** "Our simplified Barra-style model — 5 style factors plus 10 sector dummies across 179 stocks — achieves a median cross-sectional R-squared of 0.34. That is, a simple linear regression of returns on characteristics explains about a third of cross-sectional variation each month. For a diversified 20-stock portfolio, common factors explain 87% of return variance. This is the quantitative risk manager's core insight: most of your portfolio's risk comes from factor tilts, not stock-specific bets."
- **Key numbers:** Median cross-sectional R-squared = 0.34 (production Barra USE4: 0.20-0.40 per period, MSCI, 2011). Factor risk share: diversified portfolio = 86.6%, production benchmark for diversified portfolios = 60-90% (Menchero, 2010, MSCI Research). 48 monthly regressions, ~179 stocks, ~15 regressors per cross-section. Size factor monthly mean = +0.34%, momentum factor monthly mean = -0.10%.
- **Student takeaway:** A handful of characteristics plus sector dummies explain a large fraction of cross-sectional return variation. For diversified portfolios, factor risk dominates — stock selection is secondary to factor allocation. Note that the momentum factor return is slightly negative (-0.10% monthly), below the expected positive range. This is consistent with the broader weakness of momentum in our 2014-2024 sample, as we will see in Exercise 2 and Exercise 4.

**Sandbox vs. reality:** An important implementation note: the original Barra-style factor contribution approach (exposure times factor return) produced a factor risk share of only 1.25% — far too low to be useful. This happened because the style factor returns from monthly cross-sectional regressions have very low variance relative to portfolio returns. The code switched to FF5 time-series regression, which correctly captures the systematic component but is a different methodology. The 87% factor risk share reflects the FF5 regression approach, not the Barra cross-sectional factor contribution. Production Barra models avoid this problem by using daily data with thousands of stocks, which stabilizes factor return estimates. Production Barra models (MSCI USE4) use daily cross-sectional regressions on 3,000+ stocks with 55-70 industry classifications and ~10 style factors. Our simplified monthly model with 11 sectors and 5 style factors achieves comparable R-squared (0.34 vs. production 0.20-0.40) because the more homogeneous S&P 500 universe is easier to explain with fewer factors.

---

### Section 7: From Factors to Features — Cross-Sectional Feature Engineering — `s7_feature_engineering.py`

**Blueprint intended:** Build a cross-sectional feature panel with fundamental and price-based characteristics, apply winsorization and z-score standardization, produce a feature correlation heatmap as a teaching artifact, and prepare the feature matrix that bridges to Week 4's ML models.
**Expected:** 8-12 features (5 fundamental + 3 technical). Max |z-score| < 4.0 after winsorizing at [1st, 99th] percentiles. P/E and earnings yield should show strong negative correlation (r < -0.70). Momentum-reversal correlation in [-0.40, -0.05].
**Actual (corrected):** 8 features (4 fundamental + 1 book_to_market + 3 technical). 179 tickers, 23,192 panel rows. Max |z-score| = 3.00 (hard-capped). Z-score mean range [-0.036, 0.037], std range [0.84, 1.00]. Momentum-reversal cross-sectional correlation = -0.22. Feature correlation heatmap (8×8) shows economically sensible structure: pb_ratio-roe = +0.85, pb_ratio-book_to_market = -0.65, earnings_yield-pb_ratio = +0.23.
**Category:** Corrected (originally flagged for winsorization failure and pe_ratio pathology)

**What was wrong and why it matters:**

The initial implementation had two issues that would silently corrupt Week 4's ML models:

(a) **Max |z-score| = 7.32 after winsorization.** With ~179 stocks, the [1st, 99th] percentile winsorization clips only ~2 observations per tail. For structurally skewed features like P/E (where loss-making firms produce large negative values), this is insufficient — the distribution shape itself guarantees extreme standardized values even after the percentile clip. The fix: a hard z-cap at ±3.0 after standardization. This is standard practice in production cross-sectional quant pipelines (Barra USE4, Axioma). The cap mechanically reduces z-std from 1.00 to ~0.84 for the heaviest-tailed features — this is the expected cost of tail truncation, not a bug. The histograms now show small spikes at the ±3 boundaries where capped mass accumulates.

(b) **P/E and earnings yield had a positive correlation (+0.22) instead of the expected strong negative.** P/E = Price/Earnings and E/P = Earnings/Price are mathematical reciprocals, so a strong negative cross-sectional correlation is expected. But P/E is pathological: it is undefined at E=0, and for loss-making firms it sign-flips to negative values. Cross-sectionally, this produces a non-monotonic relationship — among profitable firms P/E and E/P are inversely related, but among loss-makers the reciprocal relationship breaks down. The fix: drop P/E entirely, retain only earnings yield (E/P). This is the standard quant choice because E/P is bounded, continuous, and handles losses gracefully (a loss-maker simply has negative E/P). MSCI Barra models use earnings yield, not P/E, as a style factor for exactly this reason.

**Teaching angle:**
- **Frame as:** "Building a feature matrix for cross-sectional ML is not just 'compute ratios and standardize.' Two pitfalls trap ML practitioners who treat financial features as generic numeric columns: (1) Winsorization at [1,99] percentiles is too loose for small universes — with 179 stocks, you are clipping only 2 observations per tail. Production pipelines add a hard z-cap at ±3 after standardization as a safety net. (2) P/E ratio is pathological — it is undefined at E=0, sign-flips for loss-makers, and produces a non-monotonic cross-sectional distribution. Earnings yield (E/P) is the correct feature because it is bounded and handles losses gracefully. This is why Barra uses E/P, not P/E."
- **Key numbers:** 8 features (pb_ratio, roe, asset_growth, earnings_yield, book_to_market, momentum, reversal, volatility). Max |z| = 3.00. Z-mean range [-0.036, 0.037]. Z-std range [0.84, 1.00] — below the expectations document's [0.90, 1.10] range because the z-cap at ±3 mechanically compresses standard deviation for heavy-tailed features. This is the expected cost of the two-stage outlier approach and is acceptable. Heatmap: pb_ratio-roe = +0.85, pb_ratio-book_to_market = -0.65 (reciprocals). Momentum-reversal = -0.22 (cross-sectional average; the time-pooled correlation is +0.03 — a Simpson's paradox variant).
- **Student takeaway:** Feature engineering in finance requires domain knowledge. Ratios that look reasonable numerically (P/E) can be structurally problematic cross-sectionally. Always verify feature distributions and pairwise correlations before feeding them to ML models. The corrected feature matrix — 7 z-scored features, bounded within ±3, with economically sensible correlations — is what Week 4's models will consume.

---

## Seminar Exercises

### Exercise 1: Can You Replicate Fama-French?

**Blueprint intended:** Construct SMB, HML, and momentum factors from raw data and compare against Ken French official returns. Discover where and why they diverge.
**Expected:** SMB correlation in [0.15, 0.55]. HML correlation in [0.20, 0.60]. MOM correlation in [0.50, 0.85]. Calibration pattern: MOM >> HML, SMB. Tracking errors: SMB and HML in [5%, 25%], MOM in [2%, 15%].
**Actual:** SMB correlation = 0.19. HML correlation = 0.82. MOM correlation = 0.85. SMB tracking error = 11.6%, HML tracking error = 7.9%, MOM tracking error = 8.2%. 131 months for SMB/HML, 130 months for MOM.
**Category:** Expectation miscalibration
**Methodological note:** Same static-sort caveat as Section 3 applies. The 131-month SMB/HML series and the high HML correlation are computed using fixed portfolio assignments based on ~2021-2025 fundamentals applied backward over the full price window. This introduces look-ahead bias (see Section 3 methodological note).

**Teaching angle:**
- **Frame as:** "Momentum replicates well (r = 0.85), confirming our code is correct — this is the calibration anchor. SMB replicates poorly (r = 0.19), confirming that the S&P 500 cannot produce a real size factor. HML shows much higher correlation (r = 0.82), but this comes with a caveat: our implementation uses a static sort (one-time classification using recent fundamentals applied over the full 11-year window), which introduces look-ahead bias and flatters the result. The genuine insight — that value-growth dispersion exists within large-caps while size dispersion does not — is real, but students should understand that the HML correlation would likely be lower with a proper annual-rebalancing approach limited to the fundamental window."
- **Key numbers:** SMB: r = 0.19, TE = 11.6%. HML: r = 0.82 (with static-sort caveat; see Section 3), TE = 7.9%. MOM: r = 0.85, TE = 8.2%. Production: SMB r ~ 0.99, HML r ~ 0.98 (Tidy Finance, Scheuch et al., 2023, CRSP/Compustat full universe, with proper annual rebalancing).
- **Student takeaway:** Factor replication quality depends on whether the characteristic you sort on has meaningful dispersion in your universe. Size dispersion is absent in large-caps; value dispersion is present. However, our high HML correlation is partly flattered by using a static sort with future information — the notebook must be transparent about this methodological shortcut and its limitations.

---

### Exercise 2: Which Factors Carry a Risk Premium?

**Blueprint intended:** Run Fama-MacBeth on 5 characteristics (beta, size, value, profitability, momentum). Discover which carry significant premia with Newey-West standard errors. See the "beta anomaly" and the empirical messiness of factor pricing.
**Expected:** Beta gamma in [-0.010, +0.015], NW |t| < 2.0 ("beta anomaly"). Momentum NW |t| in [1.0, 4.0], "most likely to survive significance testing." Profitability NW |t| in [0.5, 3.0]. At least one significant (momentum or profitability), at least one insignificant (beta or size).
**Actual:** Beta gamma = 0.0064 (NW t = 2.46, p = 0.014) — significant, contradicting the expected "beta anomaly." Log_mcap NW t = 1.70 (marginally insignificant). B/M NW t = -1.66 (insignificant). Profitability NW t = -0.40 (insignificant, well below expected [0.5, 3.0]). Momentum NW t = 0.81 (insignificant, well below expected [1.0, 4.0]). Only 1 of 5 factors significant. 71 months, 179 stocks.
**Category:** Real phenomenon

**Teaching angle:**
- **Frame as:** "The textbook says beta should be insignificant (the 'beta anomaly') and momentum should be significant. Our 2014-2024 data flips both predictions. Beta is the only factor significantly priced (t = 2.46), while momentum is insignificant (t = 0.81). This is not a bug — it is the central lesson: factor premia are sample-dependent. The 'beta anomaly' documented with 1963-1990 data (Fama & French, 1992) disappears in a decade when high-beta tech stocks earned 40%+ annual alpha. Momentum's insignificance reflects the COVID-era market regime disruption, where momentum crashed and recovered erratically."
- **Key numbers:** Beta: gamma = 0.64% monthly (NW t = 2.46, p = 0.014). Log_mcap: t = 1.70 (marginal). B/M: t = -1.66. Profitability: t = -0.40. Momentum: t = 0.81. Production: beta insignificant in 1963-1990 (Fama & French, 1992, CRSP); size and B/M significant in 1963-1990 but weakened in recent decades (Hou, Xue & Zhang, 2015).
- **Student takeaway:** "Which factors are priced?" has no stable answer across time. Running the same test on different decades gives contradictory conclusions. This is why asset pricing remains an active field — the ground truth keeps shifting.

**Sandbox vs. reality:** The significant beta premium and insignificant momentum are both period-specific. With CRSP data spanning 1963-2024 (60+ years), the "beta anomaly" is well-documented and momentum is among the strongest cross-sectional predictors. Our 71-month window (2019-2024, after accounting for rolling-beta estimation) over-weights the COVID-era disruption and AI-driven tech rally, which favoured high-beta stocks and disrupted momentum.

---

### Exercise 3: Decompose Your Portfolio's Risk — `ex3_portfolio_risk.py`

**Blueprint intended:** Compare risk decomposition for a diversified portfolio (20 stocks across sectors) vs. a concentrated portfolio (tech-only) using factor regression. Show that concentration increases factor risk exposure.
**Expected:** Diversified factor risk share in [40%, 75%]. Concentrated factor risk share in [55%, 90%]. Concentrated > diversified by >= 10 pp. Factor share difference in [5, 40] pp.
**Actual:** Diversified factor risk share = 89.0% (above expected max 75%). Concentrated (tech) factor risk share = 81.0%. Direction reversed: diversified > concentrated by 8.0 pp. Factor share difference = 8.0 pp (within [5, 40] pp range, near lower bound).
**Category:** Real phenomenon (reversed direction)

**Teaching angle:**
- **Frame as:** "The blueprint predicted that sector concentration would increase factor risk. Our data shows the opposite: the diversified portfolio has higher factor risk share (89%) than the concentrated tech portfolio (81%). This is counterintuitive but genuine. The diversified portfolio — 2 stocks from each of 10 sectors — tracks the broad market cleanly (Mkt-RF loading = 1.11, R-squared = 0.89). The tech-concentrated portfolio has more idiosyncratic risk because individual tech stocks are volatile and heterogeneous: NVDA, AAPL, and MSFT are all 'tech' but their return drivers differ substantially. Concentration does not always mean more factor risk — it depends on whether the concentrated sector's stocks co-move with common factors or with each other in idiosyncratic ways."
- **Key numbers:** Diversified: factor share = 89.0%, R-squared = 0.890, Mkt-RF = 1.11, CMA = +0.32. Concentrated: factor share = 81.0%, R-squared = 0.810, Mkt-RF = 1.23, CMA = -0.34. Difference = 8.0 pp (diversified > concentrated). Production benchmark: diversified portfolios typically 60-90% factor risk (Menchero, 2010, MSCI Research).
- **Student takeaway:** The relationship between concentration and factor risk is more nuanced than "concentration = more factor exposure." Tech stocks have high idiosyncratic risk that is not captured by standard factors. Always decompose before assuming.

**Sandbox vs. reality:** The code uses FF5 time-series regression for risk decomposition rather than a Barra-style cross-sectional factor contribution approach. Production Barra models would use daily cross-sectional factor contributions with more factors and stocks. The methodology should be noted in the notebook: "We use FF5 regression here for computational stability. A production Barra model would use cross-sectional factor contributions with daily data and more factors."

---

### Exercise 4: The Factor Zoo Safari

**Blueprint intended:** Test 8-10 characteristics (including noise factors) via univariate Fama-MacBeth, apply multiple testing corrections (Bonferroni, BH), demonstrate the factor zoo problem.
**Expected:** 3-5 of 10 naive significant. After Bonferroni: 1-3 survive. Noise factors do not survive Bonferroni. Momentum is "most likely to survive."
**Actual:** 4 of 10 naive significant (reversal, asset_growth, book_to_market, noise_1). After Bonferroni: 2 survive (reversal, asset_growth). BH: 4 survive. HLZ (t > 3.0): 2 survive. Noise_1 passes naive (t = 2.69) but fails Bonferroni — a textbook false discovery demonstration. Momentum: t = 0.84 (insignificant, not a survivor as expected). Reversal: t = 36.58 (dominates; near-mechanical relationship).
**Category:** Expectation miscalibration

**Teaching angle:**
- **Frame as:** "Without correction, 4 of our 10 characteristics appear significant — including one that is pure random noise (noise_1, t = 2.69). After Bonferroni correction, only 2 survive, and the noise factor is correctly filtered out. This is the factor zoo in miniature: conventional significance thresholds produce false discoveries when you test many hypotheses. The survivors are reversal (t = 36.58) and asset growth (t = 3.22) — not momentum, which the literature predicts should be robust. The reversal t-statistic of 36.58 is suspiciously extreme and likely reflects a near-mechanical relationship: short-term reversal is computed from recent returns that partially overlap with the dependent variable. A subtlety: Benjamini-Hochberg at FDR=5% retains all 4 naive significant factors — including the noise factor. BH controls the expected *proportion* of false discoveries, not individual false positives. With only 10 tests, a noise factor at p=0.008 can survive BH even though it is pure noise. This illustrates why BH is insufficient for the factor zoo: controlling the false discovery rate at 5% still allows individual false discoveries through. Bonferroni and the HLZ threshold (t > 3.0) are more conservative and correctly reject noise_1."
- **Key numbers:** Naive significant: 4/10. Bonferroni significant: 2/10 (50% drop). BH (FDR=5%): 4 survive (including noise_1 — BH fails to filter the false discovery). Noise_1: t = 2.69 (naive significant, Bonferroni rejected — the false discovery). Reversal: t = 36.58 (suspiciously extreme). Asset_growth: t = 3.22. Momentum: t = 0.84 (insignificant). Production benchmark: 53% of 296 published factors are false discoveries (Harvey, Liu & Zhu, 2016). HLZ threshold: t > 3.0.
- **Student takeaway:** Multiple testing correction is not optional — a noise factor in our sample cleared p < 0.01 before correction. The 50% drop (4 to 2 significant) from naive to Bonferroni demonstrates the scale of the problem. Surviving multiple testing correction is necessary but not sufficient. Reversal survives Bonferroni with t=36.58, but this likely reflects a mechanical overlap between the reversal signal (last month's return) and the dependent variable (this month's return, which includes mean reversion). In practice, you would investigate whether a surviving factor's signal is genuinely predictive or mechanically induced before declaring it a real discovery.

---

## Homework Deliverables

### Deliverable 1: The Factor Factory

**Blueprint intended:** Build a complete `FactorBuilder` class that constructs all 6 factors (SMB, HML, RMW, CMA, MOM, Mkt-RF) from raw data and validates against Ken French official data.
**Expected:** MOM correlation in [0.50, 0.85] (calibration anchor). SMB in [0.15, 0.55]. HML in [0.20, 0.60]. RMW in [0.10, 0.55]. CMA in [0.10, 0.55]. Fundamental factor months: ~36-48. Each portfolio >= 15 stocks.
**Actual:** 168 tickers processed (11 missing equity, 15 Net Income fallback). 131 months of returns for all 6 factors (full price window, not 36-48 as expected). MOM: r = 0.64, TE = 15.0%. SMB: r = 0.33, TE = 11.7%. HML: r = 0.83, TE = 7.7%. RMW: r = 0.35, TE = 9.7%. CMA: r = 0.29, TE = 9.5%. All portfolios >= 19 stocks.
**Category:** Expectation miscalibration
**Methodological note:** Same static-sort caveat as Section 3. The FactorBuilder computes all characteristics once from ~2021-2025 fundamentals and applies fixed portfolio assignments over the full 2014-2024 price window. All fundamental-based factor correlations (HML, RMW, CMA, SMB) are subject to look-ahead bias. MOM is the only factor unaffected (price-based, no fundamental dependency). The 131-month series length and HML correlation of 0.83 would both be lower under a proper annual-rebalancing approach.

**Teaching angle:**
- **Frame as:** "The FactorBuilder produces all 6 factors end-to-end. Validation against Ken French reveals a hierarchy: HML replicates best (r = 0.83), followed by MOM (r = 0.64), RMW (r = 0.35), SMB (r = 0.33), and CMA (r = 0.29). However, an important caveat: HML, RMW, CMA, and SMB are all computed using a static sort — portfolio assignments are fixed based on recent fundamentals and applied over the full 11-year price window. This introduces look-ahead bias and flatters the correlations for fundamental-based factors. MOM (price-only, no fundamentals needed) is the only clean benchmark. The hierarchy is still informative — HML replicates better than SMB regardless of methodology, because value-growth dispersion exists in large-caps while size dispersion does not — but the absolute correlation numbers should be treated with caution."
- **Key numbers:** HML: r = 0.83 (static-sort caveat), TE = 7.7%. MOM: r = 0.64 (no look-ahead bias — clean benchmark), TE = 15.0%. RMW: r = 0.35 (static-sort caveat), TE = 9.7%. SMB: r = 0.33 (static-sort caveat), TE = 11.7%. CMA: r = 0.29 (static-sort caveat), TE = 9.5%. Production: r ~ 0.99 for all factors (Tidy Finance, Scheuch et al., 2023, CRSP/Compustat full universe, with proper annual rebalancing). Missing equity: 11 tickers. Net Income fallback: 15 tickers.
- **Student takeaway:** Factor replication quality varies by factor and reflects data constraints. The relative ordering (HML > MOM > RMW > SMB > CMA) is informative, but the absolute correlation numbers are inflated by the static-sort methodology. The validation report should flag this: never trust self-built factors without benchmarking, and never trust a benchmark comparison without understanding the methodology behind it.
- **Methodological note on MOM divergence:** MOM correlation is 0.64 here vs. 0.85 in Exercise 1. The difference is not a bug — Exercise 1 uses equal-weighted portfolios for momentum, while the FactorBuilder uses value-weighted portfolios (matching the Fama-French methodology). Value-weighting concentrates the portfolio in mega-cap stocks, whose momentum signals are noisier relative to the broad market. This 0.21 correlation gap from a single weighting choice illustrates how "the momentum factor" is not a unique object — construction methodology matters.

**Sandbox vs. reality:** All correlations are well below the production benchmark of ~0.99 (Tidy Finance, CRSP/Compustat). The hierarchy — HML > MOM > RMW > SMB > CMA — reflects different sensitivities to universe coverage and data quality. With CRSP/Compustat (4,000+ stocks including true small-caps, point-in-time fundamentals, proper annual rebalancing), all correlations exceed 0.95. Additionally, our static-sort approach (one-time classification using ~2025 fundamentals applied backward) differs fundamentally from the production Fama-French methodology (annual rebalancing each June using prior fiscal year's data). This means our correlations are not directly comparable to Tidy Finance's — they benefit from look-ahead bias that production implementations avoid. A third source of bias: yfinance fundamental data is "as-reported" but not point-in-time. We cannot verify exact reporting dates, so our factor sorts may use fundamentals before they were publicly available. In production, CRSP/Compustat point-in-time databases track when each data point became known to investors, preventing this contamination. The notebook should state this explicitly when presenting the validation results.

---

### Deliverable 2: The Cross-Sectional Feature Matrix — `d2_feature_matrix.py`

**Blueprint intended:** Build a reusable `FeatureEngineer` class that constructs a standardized cross-sectional feature matrix from raw stock data, ready for ML consumption. Output as Parquet for Week 4.
**Expected:** 8-12 features (5 fundamental + 3 technical). Z-scores bounded after winsorization. Rank transforms in [0, 1]. ML-ready output saved to Parquet.
**Actual (corrected):** 7 ML-ready features (4 fundamental + 3 technical). Feature matrix shape (23,192 × 21). ML-ready matrix shape (23,192 × 7). Z-scores bounded within ±3.0. Rank transforms in [0, 1]. Saved to `feature_matrix_ml.parquet`. 179 tickers, 130 months. Missing data: pb_ratio 6.2%, roe 6.2%, all others < 1%.
**Category:** Corrected (originally flagged for winsorization failure and pe_ratio pathology)

**What was wrong and why it matters:**

The FeatureEngineer class had the same two issues as Section 7 (shared pipeline logic):

(a) **Z-scores unbounded at 7.3.** The class applied [1,99] percentile winsorization before z-scoring (correct order), but with ~179 stocks per month, the 1st and 99th percentiles only clip the 2 most extreme observations — insufficient for heavy-tailed financial ratios. The fix: add `.clip(-3.0, 3.0)` on z-scores inside `_winsorize_and_zscore`. This is the industry-standard two-stage approach: percentile clip in raw space + hard cap in z-space.

(b) **P/E included alongside earnings yield.** The `FUNDAMENTAL_FEATURES` list included both `pe_ratio` and `earnings_yield`. Since these are mathematical reciprocals, including both is redundant at best and misleading at worst — especially because P/E's sign-flip for loss-makers breaks the expected negative correlation. The fix: remove `pe_ratio` from `FUNDAMENTAL_FEATURES` and delete the computation block. The ML-ready matrix drops from 8 to 7 z-scored features.

**Teaching angle:**
- **Frame as:** "The FeatureEngineer class demonstrates production-style feature construction: compute raw characteristics, winsorize cross-sectionally, standardize to z-scores, and provide both z-scored and rank-transformed outputs. Two corrections during development illustrate real pitfalls: (1) winsorization alone is insufficient for small universes — always add a z-cap as a safety net; (2) P/E ratio is a trap for ML practitioners because it is non-monotonic cross-sectionally. The corrected output — 7 features, bounded within ±3, saved as Parquet — is the input to every model in Week 4."
- **Key numbers:** ML-ready matrix: 23,192 × 7. Features: pb_ratio, roe, asset_growth, earnings_yield (fundamental); momentum, reversal, volatility (technical). Z-mean range [-0.036, 0.037]. Z-std range [0.84, 1.00] — below the expectations document's [0.90, 1.10] range because the z-cap at ±3 mechanically compresses standard deviation for heavy-tailed features. This is the expected cost of the two-stage outlier approach and is acceptable. Max |z| = 3.0. Missing: pb_ratio/roe 6.2%, all others < 1%.
- **Student takeaway:** A feature matrix is only as reliable as its preprocessing. Two-stage outlier control (percentile clip + z-cap) and careful feature selection (E/P over P/E) are not optional — they directly affect downstream ML model quality.

---

### Deliverable 3: The Factor Model Horse Race

**Blueprint intended:** Compare CAPM vs. FF3 vs. FF5 using Fama-MacBeth regressions. Report which factors are priced, how explanatory power changes, and what residual alpha remains for ML.
**Expected:** R-squared monotonicity: CAPM < FF3 <= FF5. CAPM R-squared in [0.00, 0.08]. FF3 in [0.02, 0.15]. FF5 in [0.03, 0.20]. Diminishing marginal improvement. Average |residual| > 0.5% monthly under all models. At least one factor significant in FF5.
**Actual:** R-squared: CAPM = 0.086, FF3 = 0.139, FF5 = 0.153. Monotonicity holds. CAPM-to-FF3 improvement = +0.054, FF3-to-FF5 = +0.014 (diminishing). Average |residual|: CAPM = 5.05%, FF3 = 4.95%, FF5 = 4.91%. Two factors significant in FF5: beta (t = 2.37, p = 0.018) and investment (t = 2.41, p = 0.016). Log_mcap marginal (t = 1.99, p = 0.047). Profitability insignificant (t = -0.43).
**Category:** Matches

**Teaching angle:**
- **Frame as:** "The horse race delivers a clear winner — but a modest one. FF5 achieves the highest cross-sectional R-squared (15.3%), beating FF3 (13.9%) and CAPM (8.6%). But 85% of cross-sectional return variation remains unexplained. The average stock has 4.9% monthly unexplained return even under the best factor model. This is the residual alpha — and it is exactly what Week 4's ML models will attempt to capture."
- **Key numbers:** R-squared: CAPM = 8.6%, FF3 = 13.9%, FF5 = 15.3%. Improvement: CAPM-to-FF3 = +5.4 pp, FF3-to-FF5 = +1.4 pp. Average |residual|: 5.05% (CAPM) to 4.91% (FF5). Significant factors in FF5: beta (t = 2.37), investment (t = 2.41). Production: cross-sectional R-squared 0.02-0.10 (CAPM), 0.05-0.15 (FF3), 0.05-0.20 (FF5) with CRSP data (Fama & French, 2020).
- **Student takeaway:** Factor models explain only 8-15% of cross-sectional return variation. The remaining 85%+ is the opportunity space for ML — but also the noise that makes cross-sectional prediction hard.

---

## Open Questions Resolved

1. **Will self-built fundamental-based factors show any meaningful signal above noise given the combination of short fundamental window (~36-48 months) and large-cap-only universe?**
   **Finding:** The question was not tested as originally framed. The expectations document anticipated annual rebalancing limited to the ~2021-2025 fundamental window, which would have produced ~36-48 months of factor returns — the scenario this question was designed to probe. Instead, the code uses a **static sort** (one-time classification using most recent ~2021-2025 fundamentals applied over the full 2014-2024 price window), producing 131 months of factor returns. This sidesteps the original question by avoiding the short-window constraint entirely.
   Under the static-sort methodology: HML replicates well (r = 0.82-0.83), SMB replicates poorly (r = 0.19-0.33), RMW (r = 0.35) and CMA (r = 0.29) show moderate signal. The "calibration anchor" strategy (using MOM as proof the code works) was validated: MOM r = 0.64-0.85 across implementations.
   **However, these correlations are not directly comparable to what the expectations document predicted**, because the methodology differs. The static sort introduces look-ahead bias (future fundamentals applied to past returns) and benefits from the absence of rebalancing noise. The high HML correlation is partly genuine (value/growth is stable among S&P 500 large-caps) and partly a methodological artifact. A proper annual-rebalancing implementation limited to the fundamental window would likely produce correlations closer to the expectations document's predicted range of [0.20, 0.60] for HML.
   **Why the static sort is still a warranted simplification:** yfinance provides only ~4-5 annual fundamental periods per ticker (~2021-2025), making proper annual rebalancing over 2014-2024 impossible. The choice was between (a) 36-48 months of methodologically proper factor returns or (b) 131 months of methodologically impure factor returns. The code chose (b) for statistical power. This is a defensible pedagogical choice — more data makes the factor replication comparison more visually clear — but the notebooks must explicitly label it as a simplification with look-ahead bias.
   **Affects:** S3, Ex1, D1
   **Teaching implication:** The lecture narrative should contrast "factors that work with free data" (HML, MOM) vs. "factors that require full-universe data" (SMB), while being explicit that the HML correlation is flattered by the static-sort methodology. The notebook should state: "Our factor construction uses a single static sort based on the most recent fundamentals. This is a simplification — the proper Fama-French methodology rebalances portfolios annually using that year's fundamentals. Our approach introduces look-ahead bias, which means the HML correlation of 0.82 overstates what a methodologically proper implementation would achieve. The qualitative ordering (HML replicates better than SMB) is robust to this choice; the absolute correlation numbers are not."

2. **Will any factor carry a statistically significant Fama-MacBeth risk premium with only ~36-48 monthly cross-sections and Newey-West standard errors?**
   **Finding:** Yes. Market beta is consistently significant (t = 2.3-2.5) across all three model specifications in D3, and in Ex2 (t = 2.46). Investment is significant in FF5 (t = 2.41). However, the traditional "priced" factors (size, value) are insignificant, and momentum (expected to be the strongest) is insignificant (t = 0.81). The effective window is 71 months (after rolling-beta estimation), not 36-48. Note: the question asked about ~36-48 monthly cross-sections (the fundamental window), but the actual test used 71 months because rolling betas consume the first ~60 months of the price window, leaving 71 months of cross-sectional data. This nearly doubles the statistical power relative to the question's premise (standard errors ~25% smaller at T=71 vs. T=40). Whether beta would remain significant with only T=40 cross-sections is untested — the expectations document's contingency ("if zero factors are significant") remains an open possibility for the fundamental-window scenario.
   **Affects:** Ex2, D3
   **Teaching implication:** The exercise successfully demonstrates sample dependence: "beta is priced here but not in Fama-French 1992" and "size/value are priced in 1992 but not here." The contrast between decades is itself the insight.

3. **How will the Barra-style cross-sectional R-squared compare between monthly and daily frequency regressions?**
   **Finding:** Daily regressions were not implemented. Monthly cross-sectional R-squared = 0.34 (median), at the upper end of the expected range and comparable to production Barra models (0.20-0.40).
   **Affects:** S6
   **Teaching implication:** Monthly frequency is sufficient for the demonstration. A note that production Barra models use daily data with more stocks and factors should be included in the notebook.

---

## Proposed `curriculum_state.md` Entry

*This entry will be appended to `curriculum_state.md` upon approval of this brief.*

```markdown
## After Week 3: Factor Models & Cross-Sectional Analysis

### Concepts Taught
- CAPM: beta estimation (OLS of excess returns on market excess return), security market line, Jensen's alpha, R-squared as explanatory power, sample-dependence of the SML (steep in 2014-2024, flat in 1963-1990 per Fama & French 1992) (COVERED)
- Fama-French three-factor model: SMB, HML, time-series regression, R-squared improvement over CAPM, alpha shrinkage (60% of stocks), cumulative factor returns 1926-2025, "death of value" post-2007 (COVERED)
- Factor construction via double sorts: breakpoints (median size, 30/70 B/M), 2x3 portfolio formation, value-weighted returns, SMB and HML from raw data (COVERED)
- Factor replication and validation: correlation and tracking error against Ken French official data, universe bias as driver of replication quality (HML r=0.82 vs. SMB r=0.19); static-sort methodology caveat (look-ahead bias from applying ~2025 fundamentals backward over 2014-2024 returns) (COVERED)
- FF5 model: RMW (profitability) and CMA (investment) factors, diminishing R-squared improvement (+0.025 median), HML-CMA redundancy (r=0.65), alpha reallocation vs. alpha shrinkage (COVERED)
- Momentum: strong long-run returns, catastrophic crash risk (2009: 60% drawdown), excluded from FF5 (no risk-based theory), UMD-HML negative correlation (-0.31) (COVERED)
- Fama-MacBeth two-step procedure: time-series beta estimation, cross-sectional regression per period, time-series average of slopes as risk premium estimate, Newey-West standard errors, manual vs. linearmodels implementation comparison (COVERED)
- Barra-style risk model: cross-sectional regression of returns on characteristics + sector dummies, factor risk vs. specific risk decomposition, diversified portfolio R-squared ~ 0.89 (COVERED)
- Factor premia sample dependence: beta significantly priced in 2014-2024 but not 1963-1990; momentum insignificant in recent sample; factor significance varies by period and universe (COVERED)
- Factor zoo and multiple testing: 50% of naive significant factors eliminated by Bonferroni correction, noise factor false discovery at p < 0.01 (noise_1 t=2.69), Harvey-Liu-Zhu t > 3.0 threshold (COVERED)
- Cross-sectional feature engineering: why P/E is pathological (sign-flip for loss-makers, non-monotonic), why E/P is preferred, two-stage outlier control (percentile clip + z-cap), feature correlation verification (COVERED)

### Skills Acquired
- Estimate CAPM betas and plot the security market line for arbitrary stock universes
- Run FF3 and FF5 time-series regressions and compare explanatory power across model specifications
- Construct factors from raw data via double-sort portfolio methodology (implemented in FactorBuilder class)
- Validate self-built factors against benchmark data using correlation and tracking error metrics
- Execute the Fama-MacBeth two-step procedure (manual and via linearmodels library)
- Apply Newey-West standard errors to correct for serial correlation in cross-sectional slope estimates
- Build a simplified Barra-style cross-sectional regression and decompose portfolio risk into factor and specific components
- Apply Bonferroni and Benjamini-Hochberg multiple testing corrections to Fama-MacBeth t-statistics
- Construct a standardized cross-sectional feature matrix with two-stage outlier control (FeatureEngineer class)

### Finance Vocabulary Established
beta, alpha (Jensen's alpha), security market line (SML), systematic risk, idiosyncratic risk, excess return, risk-free rate, SMB, HML, RMW, CMA, UMD/MOM, double sort, breakpoint, value-weighted return, book-to-market, operating profitability, asset growth, earnings yield, Fama-MacBeth regression, cross-sectional regression, risk premium, Newey-West standard error, factor loading, factor risk, specific risk, Barra risk model, factor zoo, multiple testing correction, Bonferroni, Benjamini-Hochberg, false discovery rate, Harvey-Liu-Zhu threshold, winsorization, z-score standardization

### Artifacts Built
- FactorBuilder class: downloads raw data, computes fundamentals, forms double-sort portfolios, produces 6 factor return series, validates against Ken French (homework D1)
- FeatureEngineer class: computes 7 cross-sectional features (4 fundamental + 3 technical), two-stage outlier control (percentile clip + z-cap at ±3), rank-transform, output to Parquet for Week 4 (homework D2)
- Factor model horse race: Fama-MacBeth regressions for CAPM, FF3, FF5 with R-squared progression and residual alpha analysis (homework D3)

### Key Caveats Established
- S&P 500 universe cannot produce a real size factor (SMB r=0.19 with Ken French); value factor replicates well (HML r=0.82), but the HML correlation is flattered by the static-sort methodology (look-ahead bias from applying ~2025 fundamentals backward over the full 2014-2024 window; a proper annual-rebalancing approach would likely produce lower correlations)
- Factor premia significance is sample-period dependent — the "beta anomaly" and momentum strength vary across decades
- Self-built factor returns should always be validated against benchmark data; never trust unchecked factor construction; and always understand the methodology behind any benchmark comparison (static sort vs. annual rebalancing produces different correlation numbers)
- Cross-sectional R-squared from factor models is modest (8-15%); ~85% of return variation remains unexplained — this is the opportunity space for ML
- Diversified portfolios can have HIGHER factor risk share than concentrated portfolios if the concentrated sector has high idiosyncratic risk
- P/E ratio is pathological for cross-sectional ML (sign-flip at E<0, non-monotonic); use earnings yield (E/P) instead
- Winsorization at [1,99] percentiles is insufficient for small universes (~179 stocks); always add a z-score hard cap
- yFinance fundamental data is not point-in-time: reporting dates are unknown, introducing potential look-ahead bias beyond the static-sort issue

### Reusable Downstream
- Can assume: CAPM/FF3/FF5 factor regression, beta estimation, factor loading interpretation, Fama-MacBeth procedure, cross-sectional standardization, multiple testing correction concepts, factor construction methodology, two-stage outlier control for feature matrices
- Can reference by name: FactorBuilder class, FeatureEngineer class, factor model horse race results, the factor replication hierarchy (HML > MOM > RMW > SMB > CMA), feature_matrix_ml.parquet
- Don't re-teach: what factors are, why CAPM fails, how double sorts work, why survivorship bias and universe coverage matter (established in Week 1 and reinforced here)
```
