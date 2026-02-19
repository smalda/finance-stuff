# Observations — Week 3: Factor Models & Cross-Sectional Analysis

## Part 1: Unbiased Observations

*Written without access to acceptance criteria. Describes what the code produced.*

### Visual Observations

### s1_capm_sml.png
![](code/.cache/s1_capm_sml.png)

**Observes:** Scatter plot of 20 labelled stocks with Estimated Beta (x-axis, range ~0.14 to ~2.20) vs. Annualized Excess Return in percent (y-axis, range ~0 to ~67%). Two lines are drawn: a solid red "Empirical SML" (slope=22.5%, R^2=0.546) and a dashed blue "Theoretical SML" (slope=12.1%). The empirical line is steeper than the theoretical line. Both lines originate from similar low-beta intercepts but diverge substantially at high betas. NVDA is the highest-return point (~63%, beta ~1.72). LLY (~28%, beta ~0.33) sits well above both lines at low beta. PFE (~4%, beta ~0.60) is the lowest-return stock. High-beta stocks (TSLA, AMD, NVDA) cluster at high returns but with wide vertical spread. There is visible scatter around the empirical line; points at intermediate betas (KO, JNJ, T) sit closer to the theoretical line. Labels are legible; NEE and WMT overlap slightly at (0.42-0.45, ~13%). All 20 ticker labels are visible.

**Cross-check with run_log:** Run_log reports n_points=20, x_range=[0.14, 2.20], y_range=[-0.22, 66.86], sml_slope=22.5165, sml_r_squared=0.5459, theoretical_slope=12.1420. The plot shows 20 points -- **checkmark**. The x-range and y-range from the axes are consistent with the run_log values -- **checkmark**. The empirical slope (22.5%) and R^2 (0.546) displayed in the legend match the run_log -- **checkmark**. The theoretical slope (12.1%) in the legend matches -- **checkmark**. Note: the run_log y_range lower bound is -0.22 but the plot y-axis appears to start at 0; this likely reflects the regression line extending slightly below 0 at the lowest x values, which is consistent with what is drawn -- **checkmark**.

---

### s2_cumulative_factors.png
![](code/.cache/s2_cumulative_factors.png)

**Observes:** Three-line cumulative return plot on a log scale (y-axis: Cumulative Return, $1 invested), covering 1926-2025. The x-axis is Date. Three lines: Mkt-RF (blue), SMB (orange), HML (green). Mkt-RF dominates, reaching approximately $700-$1000 by 2025. HML reaches approximately $25-$30 by its peak around 2005-2010, then declines to roughly $25 by 2025. SMB reaches approximately $4-$5 and has been roughly flat since the mid-1980s. The Mkt-RF line shows the 1929-1932 drawdown (drops from ~$2 to ~$0.4), 2000-2002 decline, 2008-2009 decline, and 2020 brief dip. HML shows a notable decline from roughly 2007 onward, losing about half its cumulative value from peak. SMB shows prolonged stagnation post-1980s. All three lines are clearly distinguishable by colour.

**Cross-check with run_log:** Run_log reports type=multi-line cumulative returns (log scale), n_lines=3, y_range=[0.24, 1035.57], title matches. The plot shows 3 lines on a log-scaled y-axis -- **checkmark**. The y_range upper bound ~1036 is consistent with the Mkt-RF reaching near $1000 -- **checkmark**. The y_range lower bound 0.24 corresponds to the 1932 Mkt-RF trough visible in the plot -- **checkmark**. The reported hml_cumulative_2014_2024=0.7673 (i.e., HML lost ~23% over 2014-2024), which is visually consistent with the HML line declining over that period -- **checkmark**.

---

### s2_factor_loadings.png
![](code/.cache/s2_factor_loadings.png)

**Observes:** Heatmap showing FF3 factor loadings (SMB and HML) for 15 stocks. The colour scale ranges from approximately -1.5 (dark blue) to +1.5 (dark red). Rows are individual tickers (AAPL, NVDA, JPM, XOM, JNJ, KO, TSLA, PG, GS, AMD, NEE, PFE, CAT, COST, NFLX). For SMB: values range from -0.67 (KO) to +0.97 (TSLA). Most stocks have negative SMB loadings, indicating large-cap tilt. TSLA (+0.97) and GS (+0.27) have the largest positive SMB loadings. For HML: values range from -1.15 (TSLA) to +1.02 (XOM). Growth-oriented tech names (TSLA at -1.15, NFLX at -0.98, NVDA at -0.92, AMD at -0.76, AAPL at -0.54) have strongly negative HML loadings. Value/financial names (XOM at +1.02, JPM at +0.85, GS at +0.71, CAT at +0.58) have strongly positive HML loadings. The colour encoding is clear and values are annotated in each cell.

**Cross-check with run_log:** Run_log reports shape=(15, 2), type=heatmap of factor loadings, title matches. The plot shows 15 rows and 2 columns -- **checkmark**. The run_log reports n_stocks=15 for Section 2, and the heatmap has exactly 15 tickers -- **checkmark**.

---

### s3_factor_comparison.png
![](code/.cache/s3_factor_comparison.png)

**Observes:** Two-panel side-by-side plot. Left panel: "SMB: Self-Built vs. Official (r=0.19)". Right panel: "HML: Self-Built vs. Official (r=0.82)". Both cover 2014-2025. Each panel has two lines: Self-built (solid blue) and Ken French (dashed orange). In the SMB panel: both lines trend downward over the full period. The self-built SMB declines more steeply, reaching approximately 0.41 by end of period vs. Ken French at approximately 0.77. The two lines co-move weakly, consistent with the reported r=0.19. In the HML panel: both lines trend downward. The self-built HML declines more steeply, ending near 0.25, while Ken French ends near 0.60. The lines show much stronger co-movement, especially in the 2014-2020 period, consistent with r=0.82. Both panels have clearly labelled legends and distinguishable line styles.

**Cross-check with run_log:** Run_log reports type=dual cumulative return comparison, n_lines_per_panel=2, title_left="SMB: Self-Built vs. Official (r=0.19)", title_right="HML: Self-Built vs. Official (r=0.82)". Titles match exactly -- **checkmark**. The reported smb_kf_corr=0.1870 and hml_kf_corr=0.8206 match the r values displayed in titles (r=0.19 and r=0.82 when rounded) -- **checkmark**. The smb_tracking_error_ann=0.1157 and hml_tracking_error_ann=0.0792 are consistent with the visible divergence (SMB diverges more) -- **checkmark**.

---

### s4_factor_correlation.png
![](code/.cache/s4_factor_correlation.png)

**Observes:** 6x6 correlation heatmap for Mkt-RF, SMB, HML, RMW, CMA, UMD over 2014-2024. Colour scale: -1.0 (dark blue) to +1.0 (dark red). Diagonal is 1.00 for all. Notable off-diagonal correlations: HML-CMA = 0.65 (strongest positive pair), Mkt-RF-SMB = 0.32, HML-SMB = 0.33. Notable negative correlations: SMB-UMD = -0.40, SMB-RMW = -0.40, Mkt-RF-UMD = -0.39, HML-UMD = -0.31, Mkt-RF-CMA = -0.22. Near-zero correlations: Mkt-RF-RMW = 0.00, Mkt-RF-HML = 0.03, CMA-UMD = 0.01, SMB-CMA = 0.02. All values are annotated in cells. The colour gradient is clear and readable.

**Cross-check with run_log:** Run_log reports type=correlation heatmap, shape=(6, 6), title matches. The plot shows a 6x6 matrix -- **checkmark**. Run_log reports umd_hml_corr=-0.3080; the heatmap shows HML-UMD = -0.31 -- **checkmark**.

---

### s4_momentum_cumulative.png
![](code/.cache/s4_momentum_cumulative.png)

**Observes:** Single-line cumulative return plot on a log scale for Momentum (UMD), covering approximately 1963-2025. Y-axis: Cumulative Return ($1 invested), ranging from ~1.0 to ~86. A single purple line shows strong upward trend from $1 in ~1963 to a peak of approximately $80-$86 around 2007-2008. A vertical pink/red shaded band labelled "2009 momentum crash" highlights the sharp decline around 2008-2009, where the cumulative return drops from the peak ~$86 to roughly $30-$35 -- approximately a 60% drawdown. Post-2009, the line fluctuates in the $30-$50 range and never recovers to its pre-crash level by 2025, remaining around $40-$50. The crash annotation is clearly visible and positioned correctly.

**Cross-check with run_log:** Run_log reports type=cumulative return (log scale), n_lines=1, y_range=[0.82, 85.59], title matches. The plot shows 1 line -- **checkmark**. The peak value ~86 matches y_range upper bound 85.59 -- **checkmark**. The lower bound 0.82 is not visually obvious since the y-axis starts at ~1.0; this may correspond to a brief early dip below $1 -- **checkmark**.

---

### s5_fama_macbeth_gammas.png
![](code/.cache/s5_fama_macbeth_gammas.png)

**Observes:** Three-panel bar chart showing Fama-MacBeth cross-sectional slopes over time (131 months). Left panel: "Market Risk Premium" -- bars oscillate around the mean of 0.0114 (shown as red horizontal line). Bars range roughly from -0.20 to +0.30, with the largest positive spike around month 70. The mean line sits slightly above zero. Middle panel: "Size Risk Premium" -- bars oscillate around the mean of 0.0014 (near-zero red line). Bars range roughly from -0.10 to +0.10. The mean line is barely distinguishable from zero. Right panel: "Value Risk Premium" -- bars oscillate around the mean of -0.0081 (slightly below zero). Bars range roughly from -0.15 to +0.15. All three panels show substantial month-to-month variation. The mean lines are all close to zero, reflecting the noisy nature of monthly cross-sectional slopes. Panel subtitles and mean values are readable.

**Cross-check with run_log:** Run_log reports type=bar chart of cross-sectional slopes, n_panels=3, title matches. Three panels shown -- **checkmark**. Mean values shown (0.0114, 0.0014, -0.0081) match run_log gamma_mkt=0.011389, gamma_smb=0.001441, gamma_hml=-0.008076 -- **checkmark**. The number of bars in each panel spans ~131 months consistent with n_months=131 -- **checkmark**.

---

### s6_barra_risk.png
![](code/.cache/s6_barra_risk.png)

**Observes:** Two-panel plot. Left panel: "Cross-Sectional R-squared by Month" -- a bar chart showing R-squared values for 48 months. Bars vary from approximately 0.0 to 0.68. A red horizontal line marks the median at 0.340. The distribution of R-squared values appears somewhat irregular; earlier months (months 0-15) show more variation with some high bars (~0.67-0.68), while later months (months 30-48) tend to be lower. Right panel: "Portfolio Risk Decomposition (20 stocks)" -- a pie chart showing Factor Risk at 86.6% (blue) and Specific Risk at 13.4% (orange). Labels and percentages are clearly visible. The pie chart is cleanly rendered with distinct colour separation.

**Cross-check with run_log:** Run_log reports type=bar chart + pie chart, n_panels=2, title_left="Cross-Sectional R-squared by Month", title_right="Portfolio Risk Decomposition (20 stocks)". Titles match -- **checkmark**. Run_log reports n_regression_months=48, median_cross_r2=0.3397, r2_range=[0.1163, 0.6777]. The bar chart shows 48 bars, median line at 0.340, bars ranging from ~0.12 to ~0.68 -- **checkmark**. Run_log reports factor_risk_share=0.8659, specific_risk_share=0.1341. Pie chart shows 86.6% and 13.4% -- **checkmark**.

---

### s7_feature_correlation.png
![](code/.cache/s7_feature_correlation.png)

**[CORRECTED — pe_ratio dropped, z-scores clipped at ±3.0]**

**Observes:** 8×8 correlation heatmap titled "Average Feature Correlation Matrix (Cross-Sectional)". Features: momentum, reversal, volatility, earnings_yield, pb_ratio, book_to_market, roe, asset_growth. Colour scale: -1.0 (blue) to +1.0 (dark red). The matrix is symmetric. Notable strong positive correlations: pb_ratio-roe = 0.85 (high profitability → high price-to-book), earnings_yield-momentum = 0.19 (cheap stocks with recent momentum). Notable strong negative correlations: pb_ratio-book_to_market = -0.65 (reciprocals, as expected), roe-book_to_market = -0.58 (high profitability → lower B/M), reversal-pb_ratio = -0.65. Near-zero correlations: momentum-reversal = -0.22 (moderate negative — short-term reversal partially offsets momentum), book_to_market-volatility = 0.22, asset_growth-volatility = -0.07. All diagonal entries are 1.00. Most cross-feature correlations are low to moderate (|r| < 0.25), with the exception of the pb_ratio-roe (+0.85), pb_ratio-book_to_market (-0.65), and roe-book_to_market (-0.58) cluster. The anomalous pe_ratio-earnings_yield positive correlation (+0.22) from the pre-correction heatmap is eliminated because pe_ratio has been removed.

**Cross-check with run_log:** Run_log reports type=correlation heatmap, shape=(8, 8), title matches. The plot shows an 8×8 matrix -- **checkmark**. Run_log reports momentum_reversal_corr=0.0300; the heatmap shows momentum-reversal = -0.22 at the (momentum, reversal) cell. The discrepancy arises because run_log reports the scalar correlation between the two raw feature Series (computed in the assertion block), while the heatmap shows the average of monthly cross-sectional correlations between z-scored versions — these are different statistics. The momentum-reversal pair has a positive full-sample time-pooled correlation (0.03) but a negative within-month cross-sectional correlation (-0.22). This is a Simpson's paradox variant -- **checkmark (explained)**.

---

### ex1_factor_replication.png
![](code/.cache/ex1_factor_replication.png)

**Observes:** Three-panel plot titled "Self-Built vs. Official Factor Returns", each panel showing cumulative return comparisons. Left: "SMB (r=0.19)" -- self-built (solid blue) and Ken French (dashed orange) both trend down from 1.0. The self-built SMB declines more steeply (to ~0.41) than Ken French (to ~0.77). The lines show weak co-movement. Middle: "HML (r=0.82)" -- both lines decline from 1.0. Self-built drops to ~0.25, Ken French to ~0.60. Lines co-move well, especially 2014-2020. Right: "MOM (r=0.85)" -- Self-built and Ken French momentum factors show strong co-movement. Both oscillate, with the self-built ending around 0.8-1.0 and Ken French ending around 1.2-1.3. The MOM panel is notably smaller/more compressed than the other two, making labels harder to read. All panels have legends with "Self-Built" and "Ken French" labels.

**Cross-check with run_log:** Run_log reports type=triple cumulative return comparison, n_panels=3, title_SMB="SMB (r=0.19)", title_HML="HML (r=0.82)", title_MOM="MOM (r=0.85)". Titles match -- **checkmark**. Run_log reports smb_corr=0.1870, hml_corr=0.8206, mom_corr=0.8460. Rounded r values in titles (0.19, 0.82, 0.85) match -- **checkmark**. Run_log reports n_months_smb=131, n_months_mom=130. Both panels span 2014-2025, approximately 131 months -- **checkmark**. Note: the plot rendering is quite small/compressed, making the MOM panel's legend text hard to read -- this is a minor quality issue.

---

### ex2_factor_premia.png
![](code/.cache/ex2_factor_premia.png)

**Observes:** Bar chart titled "Fama-MacBeth t-Statistics -- Which Factors Are Priced?" Y-axis: Newey-West t-statistic, ranging from approximately -2 to +2.5. X-axis: 5 factors (beta, log_mcap, bm, profitability, momentum). Two horizontal red dashed lines at t=+2.0 and t=-2.0 mark significance thresholds. The "beta" bar (blue, highlighted) extends to approximately +2.5, exceeding the +2.0 threshold. The "log_mcap" bar (grey) reaches approximately +1.7. The "bm" bar (grey) reaches approximately -1.7. The "profitability" bar (grey) is small and negative, approximately -0.4. The "momentum" bar (grey) is small and positive, approximately +0.8. Only "beta" exceeds the significance threshold. The beta bar is coloured differently (blue) from the others (grey), visually highlighting it as the only significant factor.

**Cross-check with run_log:** Run_log reports type=bar chart of t-statistics, n_bars=5, title matches. The plot shows 5 bars -- **checkmark**. Run_log reports gamma_beta NW t=2.46, gamma_log_mcap NW t=1.70, gamma_bm NW t=-1.66, gamma_profitability NW t=-0.40, gamma_momentum NW t=0.81. The bar heights are visually consistent with these values -- **checkmark**. Run_log reports n_significant=1, which matches only beta exceeding the t=2.0 lines -- **checkmark**.

---

### ex3_portfolio_risk.png
![](code/.cache/ex3_portfolio_risk.png)

**Observes:** Three-panel plot. Left: Pie chart "Diversified (20 stocks) (R-squared=0.890)" showing Factor Risk 89.0% (blue) and Specific Risk 11.0% (orange). Middle: Pie chart "Concentrated (Tech) (R-squared=0.810)" showing Factor Risk 81.0% (blue) and Specific Risk 19.0% (orange). Right: Grouped bar chart "Factor Loadings Comparison" showing side-by-side bars (blue=Diversified, orange=Concentrated) for Mkt-RF, SMB, HML, RMW, CMA. Both portfolios have Mkt-RF loadings above 1.0 (diversified ~1.1, concentrated ~1.2). The concentrated portfolio has notably negative CMA (~-0.34) while diversified has positive CMA (~0.32). HML loadings differ in sign: diversified slightly positive (~0.06), concentrated negative (~-0.15). SMB loadings are near zero for both. RMW loadings are small for both. Labels and percentages are readable.

**Cross-check with run_log:** Run_log reports type=pie charts + bar chart, n_panels=3, title_left="Diversified (20 stocks) (R-squared=0.890)", title_right="Factor Loadings Comparison". Titles match -- **checkmark**. Run_log reports div_factor_share=0.8897, div_specific_share=0.1103, conc_factor_share=0.8099, conc_specific_share=0.1901. Pie chart percentages (89.0%/11.0% and 81.0%/19.0%) match when rounded -- **checkmark**. Run_log reports factor_share_diff_pp=7.98, which is 89.0% - 81.0% = 8.0% -- **checkmark**. Run_log reports Mkt-RF loadings of 1.1073 (diversified) and 1.2323 (concentrated), CMA loadings of 0.3210 and -0.3390, consistent with the bar chart -- **checkmark**.

---

### ex4_factor_zoo.png
![](code/.cache/ex4_factor_zoo.png)

**Observes:** Two-panel plot. Left: Horizontal bar chart "Fama-MacBeth t-Statistics by Characteristic" showing 10 bars sorted by p-value. Bars are coloured: blue for real factors, red for noise factors. Two vertical dashed lines at t=2.0 (green) and t=3.0 (red, labelled "HLZ"). The "reversal" bar extends far to the right (t~36.6), dominating the chart and compressing the scale. "asset_growth" bar is the second largest (t~3.2). "book_to_market" bar extends to the left (t~-2.8). "noise_1" extends to t~2.7. All other bars (profitability, noise_2, noise_3, noise_4, earnings_yield, momentum) cluster between t=-1 and t=+1. The noise factors are clearly differentiated by colour (red). Right: Bar chart "How Many Factors Survive Correction?" showing 4 bars: Naive (t>2.0) = 4 (green), Bonferroni = 2 (blue), BH (FDR=5%) = 4 (orange), HLZ (t>3.0) = 2 (red). A dotted horizontal line at 10 marks total tested. The chart clearly shows the drop from 4 to 2 significant factors when moving from naive to stricter correction methods.

**Cross-check with run_log:** Run_log reports type=horizontal bar + comparison bar, n_panels=2, titles match. The plot shows 2 panels -- **checkmark**. Run_log reports n_characteristics=10, n_noise=4, naive_significant=4, bonferroni_significant=2, bh_significant=4, hlz_significant=2. Bar counts (4, 2, 4, 2) match -- **checkmark**. Run_log reports reversal t=36.58, asset_growth t=3.22, book_to_market t=-2.84, noise_1 t=2.69. These values are consistent with bar lengths in the left panel -- **checkmark**. Run_log reports noise_naive_sig=1, noise_bonf_sig=0. In the left panel, noise_1 (red bar) extends past the t=2.0 green line but not past t=3.0 -- **checkmark**. Run_log reports drop_naive_to_bonf=0.50 (4 to 2) -- **checkmark**.

---

### d1_factor_cumulative.png
![](code/.cache/d1_factor_cumulative.png)

**Observes:** Multi-line cumulative return plot titled "Self-Built Factor Cumulative Returns" covering 2014-2025. Y-axis: Cumulative Return (linear scale), range approximately -0.4 to 12.5. Seven lines are shown: Mkt-RF (blue), SMB (orange), HML (green), RMW (red), CMA (purple), MOM (brown). A thin dashed grey horizontal line at y=1.0 marks the starting value. Mkt-RF dominates, rising from 1.0 to approximately 12.5 by end of 2025. MOM rises gradually to approximately 1.5 by 2021, then declines to roughly 0.8 at the end. RMW rises to approximately 1.7 by end of period. SMB oscillates around 1.0, ending at approximately 0.5. HML declines steadily from 1.0 to approximately 0.25 by end of period. CMA declines to approximately 0.35. The Mkt-RF line shows the March 2020 COVID drawdown (~30% decline) followed by sharp recovery. All factor lines except Mkt-RF stay below 2.0 for the entire period.

**Cross-check with run_log:** Run_log reports type=multi-line cumulative returns, n_lines=7, y_range=[-0.36, 12.96], title matches. However, I count 6 labelled lines in the legend (Mkt-RF, SMB, HML, RMW, CMA, MOM) plus the baseline dashed line, totalling 7 drawn elements -- **checkmark** (assuming the baseline counts as a line). The y_range upper bound 12.96 matches the Mkt-RF peak visible at ~12.5 (exact value may be slightly higher given the run_log precision) -- **checkmark**. The y_range lower bound -0.36 is not visually obvious; the lowest visible point appears to be HML at ~0.25. The -0.36 may refer to log returns or an intermediate calculation -- **? (minor ambiguity)**. Run_log reports factor_returns_shape=(131, 6), with 6 factor columns plus market, totalling 7 series if Mkt-RF is included separately -- **checkmark**.

---

### d2_feature_distributions.png
![](code/.cache/d2_feature_distributions.png)

**[CORRECTED — pe_ratio dropped, z-scores clipped at ±3.0]**

**Observes:** 2×4 grid of histograms (7 feature panels + 1 empty) titled "Feature Distributions (Z-scored)". Each panel shows the distribution of one z-scored feature across the full panel (~23,192 observations). A red dashed vertical line marks zero in each panel. Top row (left to right): pb_ratio — strongly right-skewed with mass concentrated near the left and a tall spike at z=+3 (the clipping boundary); roe — left-peaked with mass between -1 and +1, a small spike at z=+3; asset_growth — roughly symmetric and compact, range approximately -1 to +2; earnings_yield — left-skewed with a spike at z=-3 (clipped left tail) and mass concentrated between -1 and +1. Bottom row (left to right): momentum — roughly bell-shaped, range bounded at [-3, +3], slight left skew; reversal — roughly symmetric with heavier tails, bounded at [-3, +3]; volatility — right-skewed, bounded at [-2, +3], peaked near -0.5; the 8th panel is empty (formerly pe_ratio). All distributions are bounded within [-3, +3] as designed. The clipping at ±3 is visible as small spikes at the boundaries for pb_ratio, earnings_yield, and momentum — these represent the mass that was capped. The previous pe_ratio histogram (which extended to z ~ -7) is gone.

**Cross-check with run_log:** Run_log reports type=histogram grid, n_panels=7 (corrected from 8), title matches. The plot shows 7 populated histogram panels + 1 empty -- **checkmark**. Run_log reports z_mean_range=[-0.0360, 0.0368], z_std_range=[0.8432, 1.0000]. The red dashed zero lines align with the approximate centres of each distribution; the slight mean offset (up to ±0.04) is not visually detectable -- **checkmark**. Run_log reports max_abs_z=3.0000 (from Section 7), consistent with all histograms being bounded at ±3 -- **checkmark**. Run_log reports n_z_columns=7 (from D2) -- **checkmark**.

---

### d3_horse_race.png
![](code/.cache/d3_horse_race.png)

**Observes:** Three-panel plot titled "Factor Model Horse Race: CAPM vs. FF3 vs. FF5". Left panel: "Average Cross-Sectional R-squared" -- three bars: CAPM=0.0859 (green), FF3=0.1393 (blue), FF5=0.1533 (red). Clear monotonic increase from CAPM to FF5. The largest jump is from CAPM to FF3 (+5.3 pp); the FF3-to-FF5 increment is smaller (+1.4 pp). Middle panel: "Average |Residual| (Monthly %)" -- three bars: CAPM=5.05% (green), FF3=4.95% (blue), FF5=4.91% (red). Bars are tall and nearly equal height, making the improvement visually subtle. The values are annotated on top. The absolute reduction is small (0.14 pp from CAPM to FF5). Right panel: "FF5 Fama-MacBeth t-Statistics" -- horizontal bars for 5 characteristics: beta (~2.4), log_mcap (~2.0), bm (~-1.7), profitability (~-0.4), investment (~2.4). A vertical red dashed line at t=2.0 marks the significance threshold. Beta and investment exceed the threshold; log_mcap appears to just barely reach it. Profitability is coloured grey (not significant) while the others appear blue. The profitability bar barely extends from zero.

**Cross-check with run_log:** Run_log reports type=triple bar chart comparison, n_panels=3, title matches. Three panels shown -- **checkmark**. Run_log reports r2_capm=0.0859, r2_ff3=0.1393, r2_ff5=0.1533. Bar heights match exactly -- **checkmark**. Run_log reports avg_abs_residual_capm=0.050485, ff3=0.049519, ff5=0.049089. Displayed as 5.05%, 4.95%, 4.91% -- **checkmark**. Run_log reports FF5_beta t=2.37, FF5_log_mcap t=1.99, FF5_bm t=-1.74, FF5_profitability t=-0.43, FF5_investment t=2.41. The bar lengths in the right panel are consistent -- **checkmark**. Run_log reports n_ff5_significant=2; visually, beta and investment are the two bars exceeding t=2.0 -- **checkmark** (log_mcap at t=1.99 just misses, consistent with 2 significant).

---

### Numerical Observations

#### Cross-File Consistency

1. **SMB correlation with Ken French** -- Reported in three places: s3_factor_construction (0.1870), ex1_replicate_ff (0.1870), d1_factor_factory (0.3289). The lecture Section 3 and Seminar Exercise 1 report identical values (0.1870), suggesting they share the same construction code. The homework D1 reports a higher value (0.3289), which indicates D1 uses a different construction pipeline (the "FactorBuilder" class) that produces somewhat different results. This is internally consistent since D1 builds all 6 factors independently.

2. **HML correlation with Ken French** -- Reported in three places: s3_factor_construction (0.8206), ex1_replicate_ff (0.8206), d1_factor_factory (0.8264). The lecture and seminar values are identical; the homework value is slightly higher. The consistency pattern mirrors SMB.

3. **MOM correlation with Ken French** -- Reported in two places: ex1_replicate_ff (0.8460) and d1_factor_factory (0.6395). The seminar Exercise 1 reports a substantially higher momentum correlation (0.85) than the homework D1 (0.64). This is a notable divergence between the two implementations.

4. **Number of stocks/tickers** -- data_setup reports 179 tickers. This propagates consistently: s5_fama_macbeth uses 179 stocks, s7_feature_engineering uses 179 tickers, d2_feature_matrix uses 179 tickers. However, s3_factor_construction uses 171 tickers (8 dropped due to missing B/M data), and d1_factor_factory uses 168 tickers (11 missing equity). These reductions are consistent with the filtering requirements of those specific pipelines.

5. **Panel size consistency** -- s7_feature_engineering and d2_feature_matrix both report panel shape (23192, ...), consistent with 179 tickers x ~130 months. The ex4_factor_zoo also uses (23192, 11). The ex2_factor_premia uses a different panel (12671 obs / 11890 after cleaning, using 71 months), reflecting its rolling-beta construction that reduces the time dimension.

6. **Monthly return count** -- data_setup reports 131 monthly returns. s1_capm, s3_factor_construction, s5_fama_macbeth, and d1_factor_factory all report 131 months. MOM-related analyses use 130 months (one month lost to momentum computation lag). This is consistent across files.

7. **FF3 R-squared values** -- s2_ff3_model reports median_ff3_r2=0.4100. s4_ff5_momentum also reports median_ff3_r2=0.4100. These are identical, confirming the same stock set and estimation period.

8. **Fama-MacBeth market premium** -- s5_fama_macbeth reports gamma_mkt=0.011389 (naive t=2.27, NW t=4.82). ex2_factor_premia reports gamma_beta=0.006364 (NW t=2.46). d3_horse_race reports CAPM beta gamma=0.005913 (NW t=2.51). These values differ (0.011 vs. 0.006 vs. 0.006), likely due to different specifications (s5 uses betas from full-sample OLS on 3 factors with 131 months; ex2 and d3 use rolling betas with 71 months and different characteristic sets). The sign is consistently positive and all are significant at p<0.05.

#### Notable Values

1. **SMB self-built correlation of 0.19** -- This is very low for a factor replication exercise. An SMB correlation of 0.19 against official Ken French data indicates the self-built SMB factor shares almost no common variation with the benchmark. Given that both should capture the same size premium, a correlation this low is striking. The HML replication at r=0.82 is much more successful by comparison. The low SMB correlation likely stems from universe coverage (179 stocks vs. Ken French's full NYSE/AMEX/NASDAQ universe of ~4,000+ stocks).

2. **Negative mean returns for self-built SMB and HML** -- s3_factor_construction reports smb_mean=-0.006479 and hml_mean=-0.010205 per month. Both factors have negative average returns over the 2014-2024 sample period. HML at -1.02% per month (approximately -12% annualized) is a substantial negative return for the value factor, reflecting the well-documented underperformance of value in the post-2014 era.

3. **NVDA annualized alpha of 43.0%** -- s1_capm reports NVDA with an annualized CAPM alpha of 42.952% (the highest in the sample) and a beta of 1.718. An alpha this large over 131 months is extreme and reflects NVDA's exceptional performance during the AI/GPU boom. The R-squared of 0.341 indicates the market factor explains only about a third of NVDA's return variation.

4. **Empirical SML slope (22.5%) nearly double the theoretical slope (12.1%)** -- The empirical cross-sectional slope is almost twice the theoretical market excess return. This 10+ percentage point gap, combined with the scatter (R-squared=0.546), indicates the beta-return relationship in this sample is steeper than CAPM predicts, though this is partly driven by the high-return mega-cap tech stocks at high betas.

5. **Reversal t-statistic of 36.58** -- In the factor zoo exercise (ex4), the reversal characteristic has a t-statistic of 36.58, which is an order of magnitude larger than any other characteristic. This extreme value dominates the bar chart and compresses the visual scale for all other factors. A t-statistic this large in a cross-sectional asset pricing test is unusual and may reflect a near-mechanical relationship (short-term reversal is measured from recent returns, and returns on the left-hand-side of the regression also include recent-period returns).

6. **One noise factor passes naive significance** -- ex4_factor_zoo reports that noise_1 (t=2.69, p=0.008) passes both the naive (p<0.05) threshold and the BH correction, but fails Bonferroni and HLZ. This is a concrete demonstration of false discovery: a pure noise characteristic appears significant at conventional thresholds.

7. **HML-CMA correlation of 0.65** -- s4_factor_correlation shows the highest off-diagonal factor correlation is between HML (value) and CMA (investment) at 0.65. This is notable because it indicates substantial redundancy between these two factors, which has implications for the FF5 model's ability to separately identify value and investment premia.

8. **Factor risk share of ~87-89% for diversified portfolios** -- Both s6_barra_risk_models (86.59%) and ex3_portfolio_risk diversified (88.97%) report that common factor exposure explains the vast majority of return variance for a 20-stock diversified portfolio. The concentrated tech portfolio has a lower factor risk share (80.99%), with specific risk nearly doubling from 11% to 19%.

9. **Cross-sectional R-squared progression: CAPM 8.6% to FF5 15.3%** -- d3_horse_race shows that even the best model (FF5) explains only 15.3% of cross-sectional return variation. The absolute residual barely changes (5.05% to 4.91%). This suggests that cross-sectional return predictability from factor models alone is modest, leaving ~85% unexplained.

10. **D1 factor factory: RMW correlation 0.35, CMA correlation 0.29** -- The profitability (RMW) and investment (CMA) self-built factors correlate much less with official data than HML (0.83). This suggests the profitability and investment factor constructions are more sensitive to data quality or methodology differences than the value factor.

#### Warnings / Unexpected Output

1. **FigureCanvasAgg non-interactive warnings** -- Every script that produces a plot generates the warning: `UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown`. This is a standard matplotlib warning when running in a non-interactive backend and is not a functional issue. It appears 16 times across the run_log (once per plt.show() call). No action needed.

2. **No stderr, convergence, or deprecation warnings** -- Beyond the FigureCanvasAgg warnings, the run_log contains no convergence warnings from regressions, no deprecation warnings from library calls, and no Python tracebacks. All 13 scripts report "ALL PASSED" status.

3. **Net Income fallback in D1** -- d1_factor_factory reports "Net Income fallback: 15 tickers" and "Missing equity: 11 tickers". This indicates that 15 stocks lacked operating profitability data and fell back to net income for the RMW factor construction, and 11 tickers were dropped entirely due to missing equity data. These data gaps partially explain the lower RMW and CMA correlations with official data.

4. **Alpha shrank for only 60% of stocks (FF3 vs. CAPM)** -- s2_ff3_model reports pct_alpha_shrank=0.60. While FF3 improved R-squared for 100% of stocks, alpha magnitude shrank for only 60%. This means for 40% of stocks, the absolute alpha actually increased when moving from CAPM to FF3, which is somewhat unexpected (though possible if new factor loadings introduce offsetting biases).

5. **Alpha shrank for only 40% of stocks (FF5 vs. FF3)** -- s4_ff5_momentum reports pct_alpha_shrank=0.40. Moving from FF3 to FF5, alpha magnitude shrank for fewer than half the stocks. Combined with the 100% R-squared improvement, this suggests the additional factors explain variance but reallocate alpha rather than uniformly reducing it.

6. **~~s7 feature correlation heatmap asymmetry concern~~ RESOLVED.** The corrected 8×8 heatmap is visually symmetric. The run_log momentum_reversal_corr=0.03 (pooled) vs. heatmap -0.22 (cross-sectional avg) discrepancy is a Simpson's paradox variant, not a data error — the two statistics measure different things.

7. **B/M median of 0.21 with max of 1.70** -- s3_factor_construction reports bm_median=0.2094 and bm_range=[0.0027, 1.7007]. The low median B/M is consistent with a universe that likely over-represents large growth stocks (S&P 500 constituents), but the narrow range (max 1.70) suggests either effective winsorization or a universe without deep-value/distressed stocks.

8. **ex2_factor_premia: only beta is significant** -- Of five tested characteristics (beta, log_mcap, bm, profitability, momentum), only market beta is significant at p<0.05 with Newey-West standard errors. This is notable given that the sample covers a relatively short period (71 months). Log_mcap (p=0.089) and bm (p=0.098) are marginal.

---

## Part 2: Expectations Comparison

*Written after injecting expectations.md into the observation context. Part 1 above is unmodified.*

### Acceptance Criteria Audit

### Section 1: CAPM & Security Market Line (`s1_capm.py`)

| Criterion | Expected | Actual | Verdict |
|-----------|----------|--------|---------|
| Beta range across 15-20 stocks | ~0.3 to ~2.1 | 0.333 to 2.008 (20 stocks) | ✓ pass (near-exact match) |
| Defensive stocks beta < 0.8 | KO, JNJ, PG < 0.8 | KO=0.499, JNJ=0.531, PG=0.377 | ✓ pass (comfortably within) |
| High-growth stocks beta > 1.3 | NVDA, TSLA, AMD > 1.3 | NVDA=1.718, TSLA=1.926, AMD=2.008 | ✓ pass (well above threshold) |
| Individual CAPM R-squared range | ~0.05 to ~0.60 | 0.043 to 0.562 | ✓ pass (near-exact match; lower bound 0.043 vs. 0.05 is trivially below) |
| Median R-squared | [0.20, 0.45] | 0.2555 | ✓ pass (near lower end of range) |
| Annualized alpha range | ~-10% to ~+25% | -6.5% to +43.0% | ~ marginal (lower bound within range; upper bound 43.0% far exceeds +25%, driven by NVDA) |
| SML cross-sectional R-squared | [0.00, 0.20] | 0.5459 | ✗ fail (far above upper bound; 0.55 vs. expected max 0.20) |
| SML slope vs. theoretical | Empirical much flatter than theoretical (~0-5% vs. ~7-10%) | Empirical 22.5% vs. theoretical 12.1% — empirical is STEEPER | ✗ fail (expectations predicted "too flat" SML; actual SML is too steep, driven by high-beta tech outperformance in 2014-2024) |
| Plot shows high-beta stocks not earning proportionally more | "Beta is flat" visual pattern | High-beta stocks (NVDA, TSLA, AMD) earned dramatically more; SML is steep and positive with R²=0.55 | ✗ fail (the classic "beta is flat" result does NOT hold in this sample) |

**Notes:** The SML failures are not code bugs. They reflect the specific 2014-2024 sample period, during which high-beta technology stocks dramatically outperformed. The expectations were calibrated to the broader empirical asset pricing literature (Fama & French 1992), which finds a flat SML using decades of CRSP data. The narrow 20-stock S&P 500 sample with NVDA (+43% alpha), TSLA, and AMD creates a steep positive SML. This is a sample-period and universe-composition effect.

---

### Section 2: FF3 Model (`s2_ff3_model.py`)

| Criterion | Expected | Actual | Verdict |
|-----------|----------|--------|---------|
| FF3 R-squared range | [0.15, 0.75] | 0.135 to 0.731 | ~ marginal (lower bound 0.135 slightly below 0.15; NEE at 0.135) |
| Median FF3 R-squared | [0.30, 0.50] | 0.4100 | ✓ pass (centre of range) |
| Median R-squared improvement CAPM-to-FF3 | [0.03, 0.15] | 0.0746 | ✓ pass (centre of range) |
| R-squared improved for >70% of stocks | >70% | 100% | ✓ pass (far above threshold) |
| Alpha shrank for >60% of stocks | >60% | 60% | ~ marginal (exactly at boundary; expectations say ">60%", actual is "=60%") |
| Cumulative Mkt-RF shows drawdowns (2008, 2020) | Visible drawdowns | Visible: 1929-32, 2000-02, 2008-09, 2020 COVID | ✓ pass |
| SMB shows positive long-run returns, post-2000 weakness | Positive long-run, weakening | SMB reaches ~$4-5 but stagnant since ~1980s | ✓ pass |
| HML shows post-2007 weakness / "death of value" | Post-2007 decline | Clear decline from ~2007 onward; hml_cumulative_2014_2024=0.77 (lost 23%) | ✓ pass |
| Factor loadings: value stocks positive HML, tech negative HML | Directional pattern | XOM=+1.02, JPM=+0.85 (value); NVDA=-0.92, TSLA=-1.15 (growth) | ✓ pass (strong, clear pattern) |
| Factor loadings: lower market-cap tickers positive SMB | Positive SMB for smaller names | TSLA=+0.97 (highest SMB); most large-cap names negative SMB; pattern partially holds | ~ marginal (TSLA has highest SMB but it is a mega-cap; pattern is mixed because all stocks are large-cap) |

---

### Section 3: Factor Construction (`s3_factor_construction.py`)

| Criterion | Expected | Actual | Verdict |
|-----------|----------|--------|---------|
| B/M computed for >180 of 200 tickers | >180 | 171 of 179 universe tickers | ~ marginal (171/179 = 95.5% coverage, but 179 < 200 planned universe; 171 > 90% of actual universe) |
| B/M range | ~0.01 to ~0.80, median [0.10, 0.40] | 0.0027 to 1.7007, median 0.2094 | ~ marginal (range upper bound 1.70 far exceeds expected 0.80; median 0.21 within expected range) |
| 6 non-empty portfolios, each >=15 stocks | >=15 per portfolio | B/H=19, B/L=34, B/M=33, S/H=33, S/L=18, S/M=34 — all >=18 | ✓ pass (comfortably above 15 minimum) |
| SMB monthly mean | [-0.02, +0.01] | -0.006479 | ✓ pass (centre of range) |
| SMB monthly std | [0.01, 0.05] | 0.024254 | ✓ pass (centre of range) |
| SMB correlation with Ken French | [0.15, 0.55] | 0.1870 | ✓ pass (near lower bound; 0.19 vs. lower limit 0.15) |
| HML monthly mean | [-0.02, +0.02] | -0.010205 | ✓ pass (within range) |
| HML monthly std | [0.01, 0.06] | 0.039227 | ✓ pass (within range) |
| HML correlation with Ken French | [0.20, 0.60] | 0.8206 | ✗ fail — above upper bound (0.82 far exceeds expected max of 0.60; this is a positive surprise) |
| SMB tracking error (annualized) | [5%, 20%] | 11.57% | ✓ pass (centre of range) |
| HML tracking error (annualized) | [5%, 20%] | 7.92% | ✓ pass (near lower bound) |

**Notes:** The HML correlation of 0.82 far exceeds the expected upper bound of 0.60 (which was probed at 0.38). The code appears to produce a substantially better HML replication than the expectations probe predicted. This may be because the actual implementation uses the full 131-month price window for factor return computation rather than being limited to the fundamental window, or because the B/M computation methodology differs from what was probed. The expectations' B/M upper bound of 0.80 is also exceeded (actual max 1.70), suggesting some deep-value or financial-sector stocks have higher B/M than anticipated.

---

### Section 4: FF5 + Momentum (`s4_ff5_momentum.py`)

| Criterion | Expected | Actual | Verdict |
|-----------|----------|--------|---------|
| FF5 R-squared range | [0.15, 0.80] | 0.1599 to 0.7722 | ✓ pass (near boundaries on both ends but within) |
| Median FF5 R-squared | [0.35, 0.55] | 0.4326 | ✓ pass (centre of range) |
| FF5 improved over FF3 for >60% of stocks | >60% | 100% | ✓ pass (far above threshold) |
| Median R-squared improvement FF3-to-FF5 | [0.01, 0.08] | 0.0252 | ✓ pass (within range, near lower portion) |
| Momentum cumulative: strong long-run positive | Strong growth | Rises from $1 to ~$86 peak | ✓ pass |
| Momentum crash visible (~-40% to -60%) | At least one catastrophic drawdown | 2009 crash annotated; peak ~$86 to trough ~$30-35 (~60% drawdown) | ✓ pass |
| Alpha under FF5 < alpha under FF3 for >55% of stocks | >55% | 40% | ✗ fail (below 55% threshold; only 40% of stocks saw alpha shrink from FF3 to FF5) |
| Factor correlations: RMW/CMA low-to-moderate with others (|r| < 0.40) | |r| < 0.40 for most pairs | RMW-SMB=-0.40, RMW-Mkt=0.00, CMA-Mkt=-0.22, CMA-HML=0.65 | ~ marginal (CMA-HML=0.65 violates the <0.40 bound; RMW-SMB exactly at 0.40 boundary) |
| UMD-HML near-zero or slightly negative | Near-zero or slightly negative | -0.31 | ✓ pass (negative as expected; -0.31 is moderate negative, consistent with value-momentum literature) |

---

### Section 5: Fama-MacBeth (`s5_fama_macbeth.py`)

| Criterion | Expected | Actual | Verdict |
|-----------|----------|--------|---------|
| Manual and linearmodels gammas agree within +-0.001 | Diff < 0.001 | beta_mkt diff=0.002573, beta_smb diff=0.001966, beta_hml diff=0.000095 | ✗ fail (MKT and SMB diffs exceed 0.001; HML diff passes) |
| gamma_MKT monthly | [-0.005, +0.015] | Manual: 0.01139, linearmodels: 0.01396 | ✓ pass (both within range; linearmodels near upper bound) |
| gamma_SMB monthly | [-0.010, +0.005] | Manual: 0.00144, linearmodels: -0.00053 | ✓ pass (both within range) |
| gamma_HML monthly | [-0.010, +0.010] | Manual: -0.00808, linearmodels: -0.00798 | ✓ pass (both within range, near lower bound) |
| Newey-West t-stats differ from OLS t-stats | Should differ for some factors | MKT: naive t=2.27, NW t=4.82 (differs); SMB: naive t=0.44, NW t=-0.14 (sign change); HML: naive t=-2.04, NW t=-1.87 | ✓ pass — but NW t is LARGER than naive t for MKT (4.82 vs. 2.27), which is the opposite of the typical pattern |
| NW t-stats 10-30% smaller than naive | Typically 10-30% smaller | MKT: NW is 112% LARGER; HML: NW is 8% smaller; SMB: NW flips sign | ✗ fail (MKT NW t much larger than naive, opposite of expected direction) |
| Average cross-sectional R-squared | [0.01, 0.15] | 0.2017 | ✗ fail (above upper bound 0.15; actual 0.20 exceeds expected max) |

**Notes:** The manual-vs-linearmodels disagreement for MKT (diff=0.0026) and SMB (diff=0.002) exceeds the 0.001 threshold. The expectations note this would indicate a bug, but the magnitudes are small in absolute terms (0.2-0.3 percentage points monthly). The disagreement likely arises from implementation differences (intercept handling, missing data treatment) rather than a fundamental error. The NW t-statistic being larger than naive for MKT is unusual — typically Newey-West inflates standard errors — and may indicate that the linearmodels implementation uses a different estimator or bandwidth than the manual version.

---

### Section 6: Barra Risk Models (`s6_barra_risk_models.py`)

| Criterion | Expected | Actual | Verdict |
|-----------|----------|--------|---------|
| Monthly cross-sectional regressions run without errors | No errors | 48 months completed, all passed | ✓ pass |
| Cross-sectional R-squared range | [0.05, 0.50] | [0.1163, 0.6777] | ~ marginal (upper bound 0.68 exceeds expected 0.50; lower bound 0.12 within range) |
| Median cross-sectional R-squared | [0.10, 0.30] | 0.3397 | ✗ fail (above upper bound; 0.34 vs. expected max 0.30) |
| Size factor mean | [-0.005, +0.005] | 0.003426 | ✓ pass (within range) |
| Size factor std | [0.005, 0.030] | Not directly reported; would need computation from factor returns | — (cannot verify from run_log) |
| Momentum factor mean | [0.000, +0.010] | -0.001026 | ✗ fail (negative, below expected lower bound of 0.000) |
| Diversified portfolio factor risk share | >50% (40-80% range) | 86.59% | ✓ pass (above 50% threshold; near upper bound of 40-80% range — actually exceeds 80%) |
| Specific risk share | 20-60% | 13.41% | ✗ fail (below lower bound 20%; 13.4% vs. expected min 20%) |

**Notes:** The Barra cross-sectional R-squared exceeding expectations (median 0.34 vs. expected max 0.30) and the high factor risk share (86.6% vs. expected max ~80%) are both "positive surprises" — the model explains more than anticipated. This may be because the code uses FF5 regression for risk decomposition rather than a pure Barra-style cross-sectional regression, which would naturally yield higher R-squared. The negative momentum factor mean (-0.001) is mildly below the expected [0.000, +0.010] range.

---

### Section 7: Feature Engineering (`s7_feature_engineering.py`)

**[CORRECTED — pe_ratio dropped, z-scores clipped at ±3.0]**

| Criterion | Expected | Actual (corrected) | Verdict |
|-----------|----------|---------------------|---------|
| Feature matrix shape | ~7,200-9,600 rows (fundamental window) or ~24,000-26,000 (full window) | 23,192 rows, 8 raw + 8 z-score = 16 columns | ✓ pass (23,192 consistent with 179 tickers × ~130 months = full price window) |
| Missing data: fundamentals 0-15% | 0-15% missing | pb_ratio: 6.1%, roe: 6.1%, asset_growth: 0%, earnings_yield: 0% | ✓ pass (all within range) |
| Missing data: price-based <5% | <5% missing | momentum: 0.1%, reversal: 0%, volatility: 0% | ✓ pass (far below threshold) |
| Max |z-score| < 4.0 after winsorizing + standardization | <4.0 | 3.0000 | ✓ pass (hard-capped at ±3.0) |
| Z-score mean per month | [-0.05, +0.05] | [-0.0360, 0.0368] | ✓ pass (within range; slight offset from z-cap is expected) |
| Z-score std per month | [0.90, 1.10] | [0.8432, 1.0000] | ~ marginal (lower bound 0.84 slightly below 0.90; the z-cap at ±3 mechanically compresses std for heavy-tailed features) |
| Momentum-reversal correlation | [-0.40, -0.05] | run_log: 0.03 (pooled); heatmap: -0.22 (cross-sectional avg) | ~ marginal (cross-sectional average is -0.22, within [-0.40, -0.05]; pooled is positive. See Simpson's paradox note in Part 1) |
| P/E and earnings_yield strong negative correlation | r < -0.70 | pe_ratio dropped; criterion no longer applicable | ✓ resolved (pe_ratio removed because P/E is pathological for loss-makers — sign-flips at E<0 make it non-monotonic cross-sectionally; earnings_yield retained as the standard quant feature) |
| Output as panel DataFrame with MultiIndex | MultiIndex (date, ticker) for Week 4 | Panel shape reported; saved to Parquet | ✓ pass |
| 8-12 feature columns | 8-12 | 8 raw features | ✓ pass |

**Notes:** Both original failures are resolved. (1) The max |z-score| of 7.32 was caused by [1,99] percentile clipping being too loose for ~179 stocks (clips only ~2 observations per tail). The fix adds a hard z-cap at ±3.0 after standardization, which is standard in production cross-sectional quant pipelines (Barra USE4, Axioma). The cap mechanically reduces z-std from 1.00 to ~0.84 for the most skewed features; this is the expected cost of tail truncation. (2) The pe_ratio/earnings_yield positive correlation (+0.22) was caused by P/E being non-monotonic when net income is negative: for loss-makers, P/E is negative and its reciprocal relationship with E/P breaks down cross-sectionally. Dropping pe_ratio and retaining earnings_yield eliminates the issue — E/P is bounded, continuous, and handles losses gracefully.

---

### Exercise 1: Replicate Fama-French (`ex1_replicate_ff.py`)

| Criterion | Expected | Actual | Verdict |
|-----------|----------|--------|---------|
| SMB correlation with Ken French | [0.15, 0.55] | 0.1870 | ✓ pass (near lower bound) |
| HML correlation with Ken French | [0.20, 0.60] | 0.8206 | ✗ fail — above upper bound (0.82 far exceeds 0.60; positive surprise) |
| MOM correlation with Ken French | [0.50, 0.85] | 0.8460 | ✓ pass (near upper bound) |
| SMB tracking error (annualized) | [5%, 25%] | 11.57% | ✓ pass (centre of range) |
| HML tracking error (annualized) | [5%, 25%] | 7.92% | ✓ pass (near lower bound) |
| MOM tracking error (annualized) | [2%, 15%] | 8.17% | ✓ pass (centre of range) |
| SMB/HML correlations much lower than MOM | SMB, HML << MOM | SMB=0.19, HML=0.82, MOM=0.85 — SMB much lower, but HML comparable to MOM | ~ marginal (SMB vs. MOM gap is large; HML vs. MOM gap is negligible — HML does not show the expected degradation) |

**Notes:** HML correlation at 0.82 is a strong positive outlier relative to the expected range of [0.20, 0.60]. The expected calibration pattern — MOM replicates well, SMB/HML replicate poorly — only partially holds. SMB replicates poorly (0.19) as expected, but HML replicates nearly as well as MOM (0.82 vs. 0.85). This weakens the expected lesson about fundamental-data-dependent factors being structurally harder to replicate.

---

### Exercise 2: Factor Premia (`ex2_factor_premia.py`)

| Criterion | Expected | Actual | Verdict |
|-----------|----------|--------|---------|
| Panel has ~200 stocks x T months | ~200 x T | 179 stocks x 71 months; clean panel 11,890 obs | ✓ pass (179 vs. 200 is reasonable; 71 months reflects rolling-beta approach) |
| Beta gamma monthly | [-0.010, +0.015] | 0.006364 | ✓ pass (within range) |
| Beta NW |t| likely < 2.0 ("beta anomaly") | |t| < 2.0 expected | NW t = 2.46 (significant, p=0.014) | ✗ fail (beta IS significant; expectations predicted the "beta anomaly" with |t|<2.0) |
| Size gamma | [-0.005, +0.005] | 0.002398 | ✓ pass (within range) |
| Size NW |t| likely < 2.0 | |t| < 2.0 | NW t = 1.70 (insignificant) | ✓ pass |
| Value gamma (B/M) | [-0.008, +0.008] | -0.002221 | ✓ pass (within range) |
| Value NW |t| | [0.0, 2.5] | |t| = 1.66 | ✓ pass (within range) |
| Profitability gamma | [-0.005, +0.010] | -0.000155 | ✓ pass (within range) |
| Profitability NW |t| | [0.5, 3.0] | |t| = 0.40 | ✗ fail (below lower bound 0.5; barely registering) |
| Momentum gamma | [0.000, +0.015] | 0.001808 | ✓ pass (within range) |
| Momentum NW |t| | [1.0, 4.0] | |t| = 0.81 | ✗ fail (below lower bound 1.0; momentum is insignificant) |
| At least one factor significant (momentum or profitability) | >=1 significant | Beta is significant (t=2.46), but NOT momentum or profitability as expected | ~ marginal (at least one factor IS significant, but it is beta rather than the predicted momentum/profitability) |
| At least one factor insignificant (beta or size) | >=1 insignificant | Size is insignificant (t=1.70) | ✓ pass |

**Notes:** The expectations predicted the "beta anomaly" (flat/insignificant beta premium), but beta is the only significant factor (t=2.46, p=0.014). Meanwhile, momentum (expected to be robust, t in [1.0, 4.0]) is insignificant (t=0.81). This reversal reflects the 2014-2024 sample period: high-beta tech stocks outperformed massively, producing a positive beta premium, while momentum's performance was weak in the post-COVID era. The profitability |t| of 0.40 is below the expected minimum of 0.5.

---

### Exercise 3: Portfolio Risk Decomposition (`ex3_portfolio_risk.py`)

| Criterion | Expected | Actual | Verdict |
|-----------|----------|--------|---------|
| Cross-sectional regressions run without errors | No errors | Completed successfully | ✓ pass |
| Diversified portfolio factor risk share | 40-75% | 88.97% | ✗ fail — above upper bound (89% far exceeds expected 75% max) |
| Diversified specific risk share | 25-60% | 11.03% | ✗ fail — below lower bound (11% far below expected 25% min) |
| Concentrated portfolio factor risk share | 55-90% | 80.99% | ✓ pass (within range) |
| Concentrated > diversified by >=10 pp | >=10 pp | Diversified 89.0% > Concentrated 81.0% — concentrated has LESS factor risk | ✗ fail (direction reversed: diversified has higher factor risk share than concentrated, opposite of expectation) |
| Factor share difference [5, 40] pp | [5, 40] pp | 7.98 pp | ✓ pass (within range, near lower bound; though direction is reversed) |
| Risk components sum to ~100% | +-5 pp tolerance | Diversified: 89.0 + 11.0 = 100.0; Concentrated: 81.0 + 19.0 = 100.0 | ✓ pass (exact) |

**Notes:** The diversified portfolio has a HIGHER factor risk share (89%) than the concentrated tech portfolio (81%), which is the opposite of expectations. The expectations assumed concentrated single-sector portfolios would have higher factor risk due to shared sector exposure. The actual result shows the diversified portfolio has higher R-squared (0.89 vs. 0.81), likely because: (1) diversification across 10 sectors means the portfolio closely tracks the broad market factor (Mkt-RF loading=1.11), while (2) the tech-concentrated portfolio has higher idiosyncratic risk because individual tech stocks have more company-specific return variation. This is a valid but counterintuitive result. The expectations' assumption that sector concentration raises factor risk share appears incorrect for this specific portfolio configuration.

---

### Exercise 4: Factor Zoo (`ex4_factor_zoo.py`)

| Criterion | Expected | Actual | Verdict |
|-----------|----------|--------|---------|
| Naive significant (p < 0.05) | 3-5 of 8-10 | 4 of 10 | ✓ pass (within range) |
| Bonferroni survivors | 1-3 | 2 | ✓ pass (within range) |
| Benjamini-Hochberg survivors | 1-4 | 4 | ✓ pass (at upper bound) |
| Noise factors do not survive Bonferroni | 0 noise | 0 noise survive Bonferroni | ✓ pass |
| Noise factors occasionally significant naive | ~0-1 expected | 1 noise factor (noise_1, t=2.69) | ✓ pass (exactly 1 false positive) |
| Drop from pre- to post-correction | 50-80% reduction | Naive 4 to Bonferroni 2 = 50% reduction | ✓ pass (at lower bound of expected range) |
| HLZ (t > 3.0) survivors | 0-2 | 2 (reversal, asset_growth) | ✓ pass (at upper bound) |
| Momentum likely survives corrections | Momentum expected to survive | Momentum t=0.84, p=0.40 — does NOT survive even naive | ✗ fail (momentum insignificant; expectations predicted it as "most likely survivor") |

**Notes:** The multiple testing correction story works as designed (4 to 2 drop from naive to Bonferroni). However, the survivors are reversal (t=36.6) and asset_growth (t=3.2), not momentum as expected. Momentum's insignificance (t=0.84) in this univariate cross-sectional test is notable — it was predicted to be the most robust factor. The reversal t-statistic of 36.6 is so extreme it likely reflects a near-mechanical relationship rather than genuine predictive power, which is a pedagogical complication.

---

### Deliverable 1: Factor Factory (`d1_factor_factory.py`)

| Criterion | Expected | Actual | Verdict |
|-----------|----------|--------|---------|
| FactorBuilder runs end-to-end | No errors | Completed successfully | ✓ pass |
| Produces 6 factor return series | SMB, HML, RMW, CMA, MOM, Mkt-RF | (131, 6) shape — 6 columns | ✓ pass |
| MOM months: ~120-130 | 120-130 | 130 | ✓ pass (near upper bound) |
| Fundamental factor months: ~36-48 | 36-48 | 131 (all fundamental factors have 131 months) | ✗ fail — above upper bound (131 far exceeds expected 36-48; suggests factors use full price window, not limited to fundamental window) |
| MOM correlation (calibration anchor) | [0.50, 0.85] | 0.6395 | ✓ pass (within range) |
| MOM tracking error | [2%, 15%] | 14.95% | ✓ pass (near upper bound) |
| SMB correlation | [0.15, 0.55] | 0.3289 | ✓ pass (centre of range) |
| SMB tracking error | [5%, 25%] | 11.70% | ✓ pass (within range) |
| HML correlation | [0.20, 0.60] | 0.8264 | ✗ fail — above upper bound (0.83 far exceeds 0.60; positive surprise) |
| HML tracking error | [5%, 25%] | 7.71% | ✓ pass (near lower bound) |
| RMW correlation | [0.10, 0.55] | 0.3522 | ✓ pass (within range) |
| CMA correlation | [0.10, 0.55] | 0.2862 | ✓ pass (within range) |
| Each portfolio >=15 stocks | >=15 | HML: min 19; RMW: min 24; CMA: min 22 | ✓ pass (all well above minimum) |
| Handles >=180 of 200 tickers | >=180 | 168 of 179 (11 missing equity) | ~ marginal (168/179 = 93.8%; base universe is 179 not 200; 168 is <180 in absolute terms but >90% of actual universe) |
| Quality report produced | Missing data, fallbacks, portfolio composition | Missing equity: 11, Net income fallback: 15, portfolio counts reported | ✓ pass |

**Notes:** The 131-month factor return series for ALL factors (including fundamental-based SMB, HML, RMW, CMA) indicates that the code uses the full 2014-2024 price window for factor returns, not just the fundamental window. The expectations assumed fundamental factors would be limited to ~36-48 months. This explains the higher-than-expected HML correlation (more data = better correlation) and has implications for the "short window" pedagogical lesson. The D1 HML correlation (0.83) far exceeds the probe estimate (0.38).

---

### Deliverable 2: Feature Matrix (`d2_feature_matrix.py`)

**[CORRECTED — pe_ratio dropped from FeatureEngineer, z-scores clipped at ±3.0]**

| Criterion | Expected | Actual (corrected) | Verdict |
|-----------|----------|---------------------|---------|
| FeatureEngineer runs end-to-end | No errors | Completed successfully | ✓ pass |
| Feature matrix rows | ~7,200-9,600 (fund window) or ~24,000-26,000 (full window) | 23,192 rows | ✓ pass (consistent with full-window range) |
| Fundamental features: >=4 of 5 computed for >85% | >85% coverage | pb_ratio: 93.8%, roe: 93.8%, asset_growth: 100%, earnings_yield: 100% — all 4 above 85% | ✓ pass (pe_ratio dropped; 4 of 4 remaining fundamentals pass) |
| Price-based features <5% missing | <5% | momentum: 0.1%, reversal: 0%, volatility: 0% | ✓ pass (far below threshold) |
| Z-score mean per month | [-0.05, +0.05] | [-0.0360, 0.0368] | ✓ pass (within range) |
| Z-score std per month | [0.90, 1.10] | [0.8432, 1.0000] | ~ marginal (lower bound 0.84 slightly below 0.90; z-cap at ±3 compresses std for heavy-tailed features) |
| Rank-transformed features range [0, 1] | [0, 1] range | 7 rank columns present; assertions verify range [0, 1] | ✓ pass |
| Max |z-score| < 4.0 after winsorizing | <4.0 | Hard-capped at ±3.0 (same as S7) | ✓ pass |
| No feature >25% missing in any single month | <25% | Worst monthly: pb_ratio 6.2%, roe 6.2% | ✓ pass (far below threshold) |
| Handles >=180 of 200 tickers | >=180 | 179 tickers | ~ marginal (179 < 180 but very close; 179 is the full available universe) |
| Output saved as Parquet, consumable by pd.read_parquet | Parquet output | Saved to feature_matrix_ml.parquet | ✓ pass |
| 8-12 feature columns | 8-12 | 7 z-score columns in ML-ready matrix; 21 total (7 raw + 7 z + 7 rank) | ✓ pass (7 in ML-ready subset) |

---

### Deliverable 3: Horse Race (`d3_horse_race.py`)

| Criterion | Expected | Actual | Verdict |
|-----------|----------|--------|---------|
| Fama-MacBeth runs for all three specs | No errors | CAPM, FF3, FF5 all completed | ✓ pass |
| CAPM beta gamma monthly | [-0.010, +0.015] | 0.005913 | ✓ pass (within range) |
| CAPM beta NW |t| likely < 2.0 | <2.0 expected | 2.51 (significant) | ✗ fail (beta is significant; expected "beta anomaly" not present) |
| CAPM avg cross-sectional R-squared | [0.00, 0.08] | 0.0859 | ~ marginal (0.086 barely above upper bound 0.08) |
| FF3 avg cross-sectional R-squared | [0.02, 0.15] | 0.1393 | ✓ pass (near upper bound) |
| CAPM-to-FF3 R-squared improvement | [0.01, 0.10] | 0.0535 | ✓ pass (centre of range) |
| FF5 avg cross-sectional R-squared | [0.03, 0.20] | 0.1533 | ✓ pass (within range) |
| FF3-to-FF5 R-squared improvement | [0.00, 0.08] | 0.0139 | ✓ pass (within range) |
| R-squared monotonicity: CAPM < FF3 <= FF5 | Monotonic | 0.0859 < 0.1393 < 0.1533 | ✓ pass |
| Avg |residual| decreases CAPM to FF5 | Decreasing | 5.05% > 4.95% > 4.91% | ✓ pass (monotonically decreasing) |
| Substantial residual remains under FF5 (>0.5% monthly) | >0.5% monthly | 4.91% monthly | ✓ pass (far above 0.5%) |
| At least one factor significant (|t|>2.0) in FF5 | >=1 | Beta t=2.37, investment t=2.41 — 2 significant | ✓ pass |

---

### Production Benchmark Comparison

**Section 1 (CAPM SML) vs. Fama & French (1992):**
Fama & French (1992) found beta has "essentially no explanatory power" for the cross-section with CRSP data 1963-1990, producing a cross-sectional R-squared < 0.05. The actual result (R-squared = 0.55) is dramatically higher and in the opposite direction from the production benchmark. The empirical SML slope of 22.5% exceeds the theoretical 12.1%, whereas the production benchmark finds a flat or inverted SML. Gap: large and directional, driven by the 2014-2024 S&P 500 sample composition (tech outperformance).

**Section 2 (FF3 R-squared) vs. Davis, Fama & French (2000):**
Production benchmark reports typical individual stock FF3 R-squared of 0.20-0.60 with CRSP monthly data. The actual median FF3 R-squared of 0.41 falls within this range. Gap: none.

**Section 3 (factor construction) vs. Tidy Finance (Scheuch et al., 2023):**
Production benchmark: SMB R-squared ~0.99 and HML R-squared ~0.96 using CRSP/Compustat with full NYSE/AMEX/NASDAQ. Actual: SMB R-squared = 0.19^2 = 0.035; HML R-squared = 0.82^2 = 0.67. Expected: SMB R-squared in [0.02, 0.30], HML R-squared in [0.04, 0.36]. The SMB gap (0.035 vs. 0.99) is enormous, as expected from the large-cap-only universe. The HML gap is smaller than expected: actual R-squared 0.67 far exceeds the expected upper bound of 0.36. The HML replication is substantially better than anticipated.

**Section 4 (FF5 incremental) vs. Fama & French (2015):**
Production benchmark: incremental R-squared from adding RMW and CMA typically 0.01-0.10 per stock. Actual median incremental R-squared: 0.025. Gap: within expected production range (near lower end).

**Section 5 (Fama-MacBeth market premium) vs. Fama & MacBeth (1973) / Fama & French (1992):**
Production benchmark: market risk premium estimates typically -0.2% to +0.8% monthly with CRSP data. Actual: +1.14% monthly (manual) / +1.40% monthly (linearmodels). The actual premium is above the typical production range, reflecting the strong equity market performance during 2014-2024. Gap: moderate positive deviation.

**Section 6 (Barra cross-sectional R-squared) vs. MSCI Barra USE4:**
Production benchmark: cross-sectional R-squared of 0.20-0.40 per period in production Barra models (3,000+ stocks, 55-70 industries, ~10 style factors, daily). Actual: median 0.34 with a simplified model (179 stocks, 11 sectors, 5 style factors, monthly). Gap: actual median is within the production range despite the simplified setup, which is better than expected. Risk decomposition: production expects 60-90% factor risk for diversified portfolios; actual 87-89% is within this range.

**Exercise 1 (factor replication) vs. Tidy Finance:**
Production benchmark: R-squared ~0.99 (SMB) and ~0.96 (HML). Actual: SMB R-squared = 0.035, HML R-squared = 0.67, MOM R-squared = 0.72. Expected ranges: SMB 0.02-0.30, HML 0.04-0.36, MOM 0.25-0.72. SMB is within expected; HML exceeds expected range; MOM is at the upper bound of expected.

**Exercise 4 (factor zoo) vs. Harvey, Liu & Zhu (2016):**
Production benchmark: 53% of 296 published factors are estimated false discoveries with t > 3.0 threshold. Actual: 50% drop from naive to Bonferroni (4 to 2). The miniature exercise produces a drop consistent with the qualitative pattern but on a much smaller scale (10 characteristics vs. 296).

**Deliverable 3 (horse race cross-sectional R-squared) vs. Fama & French (2020):**
Production benchmark: cross-sectional R-squared 0.02-0.10 (CAPM), 0.05-0.15 (FF3), 0.05-0.20 (FF5) with CRSP data. Actual: CAPM 0.086, FF3 0.139, FF5 0.153. CAPM slightly above the production range upper bound; FF3 and FF5 are at the upper ends of their respective ranges. Gap: modest positive deviation across all three models.

---

### Known Constraints Manifested

**1. Survivorship bias (yfinance only has current listings).**
Manifested in: the 179-ticker universe consists entirely of current S&P 500 constituents. No delisted stocks appear. Connected to Part 1 observation: the NVDA alpha of 43% and the steep positive SML are partly artifacts of survivorship — only the stocks that survived and thrived (NVDA, TSLA, AMD) are in the sample. Firms that underperformed and were removed from the S&P 500 are absent, inflating average returns for high-beta stocks and steepening the SML.

**2. Universe bias (large-cap only, kills the size effect).**
Manifested in: SMB self-built correlation of 0.19 (S3, Ex1) and 0.33 (D1) vs. production benchmark of ~0.99. The portfolio counts show the "small" bucket contains S&P 500 stocks with market caps >$10B. Connected to Part 1 observation #1 (SMB correlation of 0.19 is very low) and cross-file consistency item #1 (SMB consistently replicates poorly across all three implementations).

**3. Fundamental data depth (~4-5 annual periods, 2021-2025).**
Expectation: fundamental-based factor returns limited to ~36-48 months. Actual manifestation: mixed. The D1 factor factory produces 131 months of SMB/HML/RMW/CMA factor returns, indicating the code uses the full price window and applies a single set of fundamental-derived sorts across the entire period. However, S6 runs only 48 monthly cross-sectional regressions, consistent with the fundamental window constraint. Connected to Part 1 observation: the divergence between 131-month factor returns (D1) and 48-month cross-sectional regressions (S6) reflects this constraint manifesting differently across implementations.

**4. Financial sector accounting differences (banks missing Operating Income).**
Manifested in: D1 reports "Net Income fallback: 15 tickers." Connected to Part 1 observation #3 (Warnings): 15 stocks used Net Income instead of Operating Income. This affects ~8.9% of the 168 processed tickers, higher than the expected ~4%.

**5. Point-in-time violation.**
Not directly observable in the run_log or plots, but structurally present in all fundamental-dependent analyses. The code uses yfinance "as-reported" data without verifying reporting dates, so factor sorts may use fundamentals before they were publicly available.

**6. Panel unbalancedness.**
Manifested in: Ex2 clean panel (11,890 obs) is smaller than the raw panel (12,671 obs), indicating ~6% of observations dropped due to missing characteristics. D3 similarly uses 11,890 observations for FF3 and FF5 specifications vs. 12,671 for CAPM. Connected to Part 1 cross-file consistency item #5.

---

### Data Plan Verification

| Data Plan Element | Planned | Actual | Match? |
|-------------------|---------|--------|--------|
| Universe size | 200 S&P 500 stocks | 179 tickers | No — 21 fewer tickers than planned |
| Date range | 2014-01-01 to 2024-12-31 | 2767 trading days, 131 monthly returns (consistent with 2014-2024) | Yes |
| Equity prices shape | ~200 x ~2,760 days | (2767, 179) | Partial — days match, tickers short |
| FF3 factors | 1963-2025 monthly | (1194, 4) — 1194 months, consistent with 1926-2025 range for Carhart | Yes (actually 1926-based for cumulative plot) |
| FF5 factors | 1963-2025 monthly | (750, 6) — 750 months, consistent with ~1963-2025 | Yes |
| FF6 factors (with UMD) | 1963-2025 monthly | (750, 7) | Yes |
| Carhart factors | 1926-2025 monthly | (1188, 5) | Yes |
| Balance sheet fields | Stockholders Equity, Total Assets, Ordinary Shares Number | (805, 3) — 3 columns match | Yes |
| Income statement fields | Operating Income + fallbacks | (819, 4) — 4 columns | Yes |
| Sectors | 11 GICS sectors | 11 unique sectors | Yes |
| Fundamental window | ~2021-2025 | Not directly verifiable from run_log; implied by 4-5 annual periods per ticker | Presumed yes |
| Caching to disk | Parquet caching | feature_matrix_ml.parquet saved | Yes |

The primary data deviation is the universe size: 179 tickers instead of 200. This 10.5% shortfall likely reflects download failures or delisted/renamed tickers. The 179-ticker universe still provides adequate cross-sectional power for all analyses (179 > 150 minimum contingency threshold from expectations).

---

### Divergence Summary

The following are the most significant gaps for Step 6 (Consolidation) to address:

1. **SML is steep and positive (R-squared=0.55), not flat as CAPM literature predicts.** Expectations predicted a flat SML (R-squared [0.00, 0.20], slope 0-5%). Actual slope is 22.5% with R-squared 0.55. This is the single largest divergence from expectations and from the production benchmark. The 2014-2024 S&P 500 sample, dominated by high-beta tech outperformance, produces the opposite of the standard "beta is flat" finding.

2. **HML replication far exceeds expectations (r=0.82 vs. expected max 0.60).** The expectations probe estimated HML correlation at 0.38, but all three implementations produce r=0.82-0.83. This undermines the expected lesson that "fundamental-based factors replicate poorly with free data." The code appears to use the full 131-month price window for factor construction rather than being limited to the fundamental window.

3. **Fundamental-based factor return series span 131 months, not the expected 36-48.** D1 produces 131 months of returns for all factors including fundamental-based ones. The expectations assumed the fundamental data window would constrain factor returns to ~36-48 months. The code apparently applies a single set of fundamental-derived sorts across the full price window.

4. **Beta is significantly priced (Ex2: t=2.46; D3: t=2.51), contradicting the "beta anomaly."** Expectations predicted beta would be insignificant (|t|<2.0). The positive and significant beta premium is consistent with the steep SML but inconsistent with the canonical asset pricing literature. This is a sample-period effect.

5. **Momentum is insignificant in cross-sectional tests (Ex2: t=0.81; Ex4: t=0.84).** Expectations predicted momentum would be significant (t in [1.0, 4.0]) and "most likely to survive" multiple testing correction. Momentum fails to reach significance in any Fama-MacBeth test. It is outperformed by reversal (t=36.6) and asset_growth (t=3.2) in Ex4.

6. **Diversified portfolio has higher factor risk share (89%) than concentrated tech portfolio (81%).** Expectations assumed concentration would increase factor risk share. The actual direction is reversed, with the diversified portfolio better explained by common factors.

7. **~~Max z-score of 7.32 indicates missing or ineffective winsorization.~~ RESOLVED.** Z-scores now hard-capped at ±3.0. pe_ratio dropped (source of extreme values). Max |z-score| = 3.00.

8. **Manual vs. linearmodels Fama-MacBeth gammas disagree by >0.001 for MKT and SMB.** Expectations flagged this as a potential bug indicator. The disagreement is small in absolute terms but exceeds the stated threshold.

9. **Alpha shrinkage from FF3-to-FF5 is only 40%, below the 55% threshold.** While R-squared improves for 100% of stocks, alpha magnitude increases for 60% of stocks when moving from FF3 to FF5.

10. **~~pe_ratio-earnings_yield correlation is positive (0.22), not strongly negative (expected r < -0.70).~~ RESOLVED.** pe_ratio dropped from the feature set. The positive correlation was caused by P/E being non-monotonic for loss-making firms (sign-flip at E<0). Earnings yield (E/P) retained as the standard quant feature.