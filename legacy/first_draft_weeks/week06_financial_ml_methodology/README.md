# Week 6 — Financial ML Methodology: Labeling, CV, and Backtesting

> **This is the week that separates people who "do ML in finance" from people who do financial ML correctly. Every shortcut you took in Weeks 4-5 comes back to haunt you.**

## Prerequisites
Weeks 4-5 are essential: the cross-sectional prediction framework, expanding-window CV, XGBoost/LightGBM models, and SHAP analysis. Students must have a working pipeline for feature construction, model training, and evaluation. They should also understand Sharpe ratio, maximum drawdown, and transaction costs from Week 3. Week 2's stationarity and volatility concepts are referenced when discussing volatility-scaled barriers.

## The Big Idea

For the past two weeks, you've been building cross-sectional prediction models with a dirty secret: your methodology was wrong. Not slightly wrong — fundamentally wrong in ways that would invalidate your results at any serious quant firm. Let's count the sins.

Sin #1: You labeled your target as "next month's return" — a fixed-threshold, fixed-horizon label. But returns aren't symmetric around zero, and the right holding period isn't always one month. A stock that hits +5% on day 3 and then crashes back to -2% by day 21 gets labeled as a -2% return. Your model learns to avoid this stock, even though a real trader would have taken the 5% profit and moved on. The label is punishing the model for not having a crystal ball about the exact day to exit.

Sin #2: You used expanding-window cross-validation, which is better than shuffled k-fold but still leaks information. If your label at time t is a 21-day return (from t to t+21), and your training set includes data from time t-5, then the training label at t-5 overlaps with the test label at t by 16 days. They share information. Your model has seen data that's contaminated by the test period. The CV score is inflated.

Sin #3: You treated every sample equally, but they're not equally informative. When a stock is in a quiet period, the next day's label is almost entirely determined by market noise. When it's in an earnings week, the label carries genuine information. Weighting all observations equally means the model spends most of its capacity learning noise.

This week fixes all three problems with Lopez de Prado's financial ML methodology: triple-barrier labeling (adaptive labels that respect the reality of trading), purged cross-validation (CV that accounts for label overlap and temporal contamination), and meta-labeling (a second model that filters the first model's predictions by predicting which trades will be profitable). These ideas come from Chapters 3, 4, 5, and 7 of *Advances in Financial Machine Learning*, and they represent Lopez de Prado's most influential contribution to the field.

The payoff is a methodology that's honest. Your evaluation metrics will drop — they always do when you remove information leakage — but what remains is real. A strategy that survives purged CV is a strategy you can actually trade. One that only works with standard CV is a fairy tale.

## Lecture Arc

### Opening Hook

"In 2019, a team at a mid-tier quant fund built what they thought was a breakthrough strategy: a neural network predicting 5-day stock returns with a Sharpe ratio of 3.2 in backtest. Beautiful SHAP analysis. Passed every traditional check. They allocated $200 million. In the first quarter of live trading, the Sharpe was 0.4 — not 3.2, not 2.0, but 0.4. They lost $12 million before pulling the plug. The post-mortem found two problems: their labels overlapped (5-day returns with daily rebalancing = 80% overlap between consecutive labels), and their cross-validation didn't account for this overlap. The CV score was inflated by a factor of 8. Every number they'd calculated was wrong — not because the model was bad, but because the evaluation methodology was broken. This week, you'll learn the methodology that would have caught this before a single dollar was deployed."

### Section 1: Why Standard ML Methodology Fails in Finance
**Narrative arc:** We catalog the specific ways that standard ML practice goes wrong when applied to financial data. The setup: everything you learned in an ML course about train/test splits, cross-validation, and labeling needs to be modified for finance. The tension: the modifications aren't cosmetic — they change your results by 2-8x. The resolution: Lopez de Prado's methodology fixes each problem specifically.

**Key concepts:**
- Label overlap: when labels share time periods, they share information
- Serial correlation in features: adjacent time periods have correlated inputs
- Non-stationarity: the relationship between features and labels changes over time
- The look-ahead problem: labels computed from future data can't be known at decision time
- The triple sin: wrong labels + leaky CV + uniform weighting = inflated results

**The hook:** "Let me show you a number that should scare you. Take your Week 5 XGBoost model. Evaluate it with standard 5-fold CV. Note the IC. Now evaluate it with expanding-window CV. Note the IC — it's lower, because you've removed temporal leakage. Now evaluate it with purged expanding-window CV (which we'll build today). Note the IC — it's even lower, because you've removed label overlap leakage. The difference between the first number and the last is the amount of self-deception in your methodology. For most financial models, the first number is 2-5x the last. That gap is pure fiction."

**Key formulas:**
"The overlap problem, formalized. If your label is a T-bar return:

$$y_t = \frac{P_{t+T}}{P_t} - 1$$

Then labels $y_t$ and $y_{t+1}$ share T-1 days of price data. The information shared is:

$$\text{Overlap}(y_t, y_{t+1}) = \frac{T - 1}{T}$$

For 21-day returns (monthly): 95% overlap between consecutive daily labels.
For 5-day returns: 80% overlap.

This means consecutive labels are highly correlated — not because the model is good, but because they're constructed from nearly identical data. If both labels are in your training set, you have massive redundancy. If one is in your training set and the other is in your test set, you have information leakage."

**Code moment:** Demonstrate the leakage quantitatively. Compute the autocorrelation of 21-day forward return labels — it will be extremely high (>0.9 at lag 1). Then show: the same model evaluated with standard 5-fold CV vs. expanding-window CV vs. purged expanding-window CV. Three numbers, declining, with the gap quantifying the total leakage.

**"So what?":** "Every backtest result you've produced in Weeks 4-5 is inflated — maybe by 30%, maybe by 300%. The expanding-window CV you used was better than shuffled CV, but it still didn't account for label overlap. Starting today, we fix this permanently."

### Section 2: Triple-Barrier Labeling
**Narrative arc:** The first of Lopez de Prado's three innovations. The setup: fixed-horizon labels (e.g., "21-day return") ignore how real traders operate. The tension: a trader would take profit at +5% or stop loss at -3%, not wait exactly 21 days regardless of what happens. The resolution: triple-barrier labels adapt to each stock's price path and volatility.

**Key concepts:**
- The three barriers: profit-take (upper), stop-loss (lower), time (maximum holding period)
- Dynamic barrier sizing: barriers scaled by recent volatility
- Label values: +1 (profit-take hit first), -1 (stop-loss hit first), 0 (time barrier expires)
- Why this produces better labels: captures the actual outcome of a trade
- Connection to options: triple-barrier is economically similar to a long straddle with an expiry

**The hook:** "Imagine you buy a stock. It goes up 5% in a week, then crashes 10% over the next three weeks. A fixed 21-day label says this trade returned -5% — a loss. But any competent trader would have taken the 5% profit and moved on. The fixed-horizon label is punishing you for not having a crystal ball about day 22. Triple-barrier labeling says: 'If the price hits +5% (profit-take) or -3% (stop-loss) before 21 days (time expiry), record which barrier was hit first.' Now the label reflects the outcome of a realistic trade, not a forced holding period."

**Key formulas:**
"The triple-barrier label for a position entered at time t₀ at price P₀:

Define the three barriers:
- Upper barrier: $P_0 \times (1 + \tau_{\text{PT}})$, where $\tau_{\text{PT}}$ is the profit-take threshold
- Lower barrier: $P_0 \times (1 - \tau_{\text{SL}})$, where $\tau_{\text{SL}}$ is the stop-loss threshold
- Time barrier: $t_0 + T_{\text{max}}$

The label:

$$y_{t_0} = \begin{cases} +1 & \text{if upper barrier hit first} \\ -1 & \text{if lower barrier hit first} \\ \text{sign}(r_{t_0, t_0+T_{\text{max}}}) & \text{if time barrier hit first} \end{cases}$$

The key innovation: make the barriers dynamic by scaling with volatility:

$$\tau_{\text{PT}} = m_{\text{PT}} \times \sigma_{t_0, \text{daily}} \quad \text{and} \quad \tau_{\text{SL}} = m_{\text{SL}} \times \sigma_{t_0, \text{daily}}$$

where $\sigma_{t_0, \text{daily}}$ is the trailing 20-day standard deviation of daily returns, and $m_{\text{PT}}, m_{\text{SL}}$ are multipliers (typically 1-3). This means:
- In calm markets (low $\sigma$): tight barriers (small moves count)
- In volatile markets (high $\sigma$): wide barriers (only large moves count)

This is adaptive labeling — the definition of 'meaningful move' changes with market conditions."

**Code moment:** Build the triple-barrier labeler from scratch:

```python
def triple_barrier_label(prices, t0, pt_mult=2.0, sl_mult=2.0, max_days=21):
    """Label a position entered at time t0."""
    daily_vol = prices.pct_change().rolling(20).std().loc[t0]
    upper = prices.loc[t0] * (1 + pt_mult * daily_vol)
    lower = prices.loc[t0] * (1 - sl_mult * daily_vol)

    # Forward path
    path = prices.loc[t0:][:max_days+1]

    # Check which barrier is hit first
    pt_hit = path[path >= upper].index.min()
    sl_hit = path[path <= lower].index.min()

    if pd.isna(pt_hit) and pd.isna(sl_hit):
        # Time barrier: sign of final return
        return np.sign(path.iloc[-1] / path.iloc[0] - 1)
    elif pd.isna(sl_hit) or (not pd.isna(pt_hit) and pt_hit <= sl_hit):
        return 1  # Profit-take
    else:
        return -1  # Stop-loss
```

Apply to SPY and compare the label distribution to fixed-threshold labels. Show that triple-barrier labels have a different class balance (it depends on the multipliers) and produce different training signals.

**"So what?":** "Triple-barrier labels produce a classification problem instead of a regression problem. The model predicts whether a trade will hit the profit-take or stop-loss — which is closer to the actual decision a trader makes. In the homework, you'll see that models trained on triple-barrier labels produce more stable predictions than models trained on fixed-horizon returns."

### Section 3: Sample Uniqueness and Weighting
**Narrative arc:** The solution to the overlap problem identified in Section 1. The setup: we've created triple-barrier labels, but their durations overlap in time. The tension: overlapping labels create redundant samples that mislead the model and inflate CV scores. The resolution: compute each sample's uniqueness and weight accordingly.

**Key concepts:**
- Label span: the time period each label covers (from entry to barrier hit)
- Concurrent labels: how many other labels overlap with this one
- Average uniqueness: 1 / (average number of concurrent labels)
- Sample weights: proportional to uniqueness
- Why this matters: without weighting, redundant samples dominate the training set

**The hook:** "If you label a daily time series with 21-day returns, you get one label per day. But each label covers 21 days, and consecutive labels share 20 of those 21 days. You have 252 labels per year, but the information content is closer to 252/21 = 12 independent observations. Your model sees 252 'samples' but only has 12 independent data points. The other 240 are copies with slight modifications. Sample weighting fixes this by telling the model how unique each observation really is."

**Key formulas:**
"For label $y_t$ spanning the time interval $[t, t + h_t]$, define the concurrency at time $s$:

$$c_s = \sum_{t: s \in [t, t+h_t]} 1$$

This counts how many labels are active at time $s$. The uniqueness of label $y_t$ is:

$$u_t = \frac{1}{h_t} \sum_{s=t}^{t+h_t} \frac{1}{c_s}$$

A uniqueness of 1.0 means no overlap. A uniqueness of 0.05 means this label shares 95% of its time span with other labels. Sample weight is then:

$$w_t = u_t \times |\text{return}_t|$$

The $|\text{return}_t|$ term upweights labels with larger absolute returns — these carry more information than labels near zero."

**Code moment:** Compute average uniqueness for SPY with 21-day forward returns. Show that it's around 0.05 — meaning each label is only 5% unique. Then recompute with triple-barrier labels (which have variable duration). The uniqueness will be higher for triple-barrier labels, because labels in volatile periods hit barriers quickly (short duration, less overlap) while labels in quiet periods take longer (more overlap).

**"So what?":** "Sample weighting is the invisible fix that makes everything downstream work better. Without it, your model is fitting to 252 'samples' that contain 12 independent observations. That's an effective overfitting ratio of 21:1. No amount of regularization compensates for training on 21 copies of each data point."

### Section 4: Purged K-Fold Cross-Validation
**Narrative arc:** The methodological centerpiece of this week — the CV method that accounts for label overlap. The setup: standard k-fold CV splits data into random folds, ignoring temporal structure. Expanding-window CV respects time but doesn't account for label overlap. The tension: if a training sample's label extends into the test period, the model has indirectly seen the test data. The resolution: purge the training set by removing all samples whose labels overlap with the test period, and add an embargo period for extra safety.

**Key concepts:**
- The information leakage mechanism: training label at time t spans to t+21; test starts at t+5; 16 days of overlap
- Purging: remove all training samples whose labels overlap with the test set
- Embargo: additionally remove a buffer of samples after each purge boundary
- Combinatorial Purged Cross-Validation (CPCV): generate multiple backtest paths from a single CV structure
- The bias-variance tradeoff in purging: more aggressive purging = less leakage but fewer training samples

**The hook:** "Here's a thought experiment. Your test set starts on January 1, 2023. Your training set ends on December 31, 2022. Looks clean, right? But your training labels are 21-day forward returns. The label for December 15, 2022 covers Dec 15 to Jan 5, 2023. Five days of that label are in the test period. Your model has been trained on data that directly tells it what happens in the first week of the test period. That's information leakage, and it's invisible unless you specifically check for it. Purged CV removes these contaminated samples from the training set."

**Key formulas:**
"For a test set covering the period $[t_{\text{test,start}}, t_{\text{test,end}}]$, the purging rule removes from the training set all samples $i$ where:

$$[t_i, t_i + h_i] \cap [t_{\text{test,start}}, t_{\text{test,end}}] \neq \emptyset$$

That is: if any part of sample $i$'s label period overlaps with the test period, remove it.

The embargo extends this by removing an additional $\epsilon$ time units after the purge boundary:

$$[t_i, t_i + h_i] \cap [t_{\text{test,start}} - \epsilon, t_{\text{test,end}} + \epsilon] \neq \emptyset$$

where $\epsilon$ is typically set to the maximum label duration (e.g., 21 days for monthly labels).

In CPCV, we create all $\binom{N}{k}$ ways to hold out $k$ of $N$ groups as test sets. This generates many backtest paths, and their distribution tells you about the strategy's robustness."

**Code moment:** Build purged k-fold CV from scratch:

```python
class PurgedKFold:
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y, label_end_times):
        """
        X: features (DatetimeIndex)
        label_end_times: Series mapping each sample's index to its label end time
        """
        indices = np.arange(len(X))
        fold_size = len(X) // self.n_splits
        embargo = int(len(X) * self.embargo_pct)

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = min((fold + 1) * fold_size, len(X))
            test_idx = indices[test_start:test_end]

            # Purge: remove training samples whose labels overlap with test
            test_start_time = X.index[test_start]
            test_end_time = X.index[test_end - 1]

            train_idx = []
            for i in indices:
                if i in test_idx:
                    continue
                if label_end_times.iloc[i] < test_start_time:
                    # Label ends before test starts: safe
                    train_idx.append(i)
                elif X.index[i] > test_end_time:
                    # Sample starts after test ends: check embargo
                    if i >= test_end + embargo:
                        train_idx.append(i)

            yield np.array(train_idx), test_idx
```

Then compare model performance with standard k-fold, expanding-window, and purged k-fold. The IC will drop with each successive methodology. Plot all three to show the magnitude of the leakage.

**"So what?":** "The IC difference between standard CV and purged CV is the amount of information leakage in your evaluation. For models with overlapping labels (which is almost all financial models), this difference can be 30-300%. If you're using standard CV, you're dramatically overstating your model's skill. Purged CV tells you the truth."

### Section 5: Meta-Labeling — The Model Behind the Model
**Narrative arc:** The most creative and practically powerful idea from Lopez de Prado. The setup: you have a primary model (e.g., a moving average crossover) that generates trade signals. Some of its signals are good and some are bad. The tension: can you build a second model that predicts which signals will be profitable? The resolution: meta-labeling — a classifier that predicts the outcome of the primary model's trades, enabling adaptive position sizing.

**Key concepts:**
- Primary model: generates directional signals (buy/sell)
- Meta-labeling model: predicts whether the primary model's next trade will be profitable
- The separation of direction and size: the primary model picks direction, the meta model picks size
- Why this is powerful: it converts a binary direction problem into a probability estimation problem
- Position sizing: high meta-label probability → full position; low probability → small or no position
- Connection to ensemble methods: meta-labeling is a learned ensemble

**The hook:** "Here's a result that surprises everyone the first time they see it: a simple moving average crossover (50/200 day) — the most basic trend-following signal imaginable — has a hit rate of about 52-54% on the S&P 500. That's barely better than a coin flip. But combined with a meta-labeling model that filters out the bad trades, the profit factor jumps from roughly 1.1 to 1.5 or higher. The meta-model doesn't make you right more often. It makes you wrong less expensively — it learns to skip the trades where the primary model is confident but wrong."

**Key formulas:**
"The meta-labeling setup:

1. **Primary model** generates signal: $s_t \in \{-1, +1\}$ (short or long)

2. **Triple-barrier label** applied to positions taken at signal times: $y_t \in \{0, 1\}$ (trade unprofitable or profitable, from the primary model's perspective)

3. **Meta-model** predicts: $\hat{p}_t = P(y_t = 1 | \mathbf{x}_t, s_t)$ — the probability that the primary model's trade will be profitable

4. **Position sizing**:

$$\text{position}_t = s_t \times \hat{p}_t$$

If the meta-model is confident the trade will be profitable ($\hat{p}_t$ high), take a full position. If uncertain ($\hat{p}_t$ near 0.5), take a small position or skip.

The beauty: we separate the direction decision (primary model, simple, interpretable) from the sizing decision (meta-model, complex, ML-driven). The primary model can be a human analyst's intuition. The meta-model adds quantitative rigor to that intuition."

**Code moment:** Build the full meta-labeling pipeline:

```python
# Step 1: Primary model — 50/200 day moving average crossover
primary_signal = np.where(sma_50 > sma_200, 1, -1)

# Step 2: Triple-barrier labels for trades at primary signal changes
labels = []
for t in signal_change_dates:
    label = triple_barrier_label(prices, t, pt_mult=2, sl_mult=2, max_days=21)
    labels.append({'date': t, 'label': 1 if label == primary_signal[t] else 0})

# Step 3: Features for meta-model
# Use: recent volatility, momentum strength, volume surge, RSI, etc.
meta_features = build_features(prices, signal_change_dates)

# Step 4: Meta-model (Random Forest or XGBoost)
meta_model = RandomForestClassifier(n_estimators=500, max_depth=5)
meta_model.fit(meta_features_train, labels_train, sample_weight=uniqueness_train)

# Step 5: Position sizing
meta_prob = meta_model.predict_proba(meta_features_test)[:, 1]
position_size = primary_signal_test * meta_prob
```

Show the results: primary model alone vs. primary + meta-model. The meta-model should improve the profit factor and reduce the maximum drawdown, even if it doesn't improve the hit rate much.

**"So what?":** "Meta-labeling is the most practical idea in this entire course. Take any trading model — fundamental, technical, ML, or human intuition — and wrap a meta-labeling model around it. The meta-model learns when the primary model is likely to be wrong, and adjusts position size accordingly. This is used at production quant funds, and it's the primary technique for managing model risk."

### Section 6: The Multiple Testing Problem (Preview)
**Narrative arc:** A brief but essential section that introduces the danger of running many backtests. The setup: you've tested OLS, Ridge, Lasso, Elastic Net, Random Forest, XGBoost, LightGBM, neural nets, with various feature sets and hyperparameters. The tension: the more strategies you test, the more likely you are to find one that works by chance. The resolution: the deflated Sharpe ratio (covered fully in Week 18) accounts for this selection bias.

**Key concepts:**
- The multiple testing problem: N hypotheses → some will be "significant" by chance
- How many strategies have you really tested? Count every model, hyperparameter, and feature set
- The Bonferroni correction (too conservative) and False Discovery Rate (better)
- Preview of the deflated Sharpe ratio (Week 18)
- Why honest accounting of N is the hardest part

**The hook:** "Let me ask you an uncomfortable question: how many strategies have you tested so far in this course? Count honestly. Every model (OLS, Ridge, Lasso, Elastic Net, RF, XGBoost, LightGBM = 7). Every hyperparameter configuration in your Optuna search (50+ trials). Every feature set variant you tried before settling on the final one (at least 3-5). You've easily tested 100+ strategies. At a 5% significance level, you'd expect 5 of them to look 'significant' by pure chance. Is your best result genuine signal, or is it the best of 100 random draws?"

**Key formulas:**
"Under the null hypothesis (no strategy has genuine skill), the maximum Sharpe ratio across N independent strategy tests is approximately:

$$E[\max SR] \approx \sqrt{2 \ln N}$$

For N = 100 strategies: $E[\max SR] \approx \sqrt{2 \ln 100} \approx 3.03$. That means even if no strategy has any skill, you'd expect the best one to have a Sharpe ratio of about 3.0 just from selection bias.

The deflated Sharpe ratio (Bailey & Lopez de Prado, 2014) adjusts for this:

$$\text{DSR} = P\left[ \hat{SR}_0 > SR^* \sqrt{V[\hat{SR}_0]} + SR_0 \right]$$

We'll implement this fully in Week 18. For now, the takeaway: if you tested N strategies and your best Sharpe is below $\sqrt{2 \ln N}$, you may have found nothing."

**Code moment:** Run a simulation: generate 100 strategies with zero signal (random predictions) and record the maximum Sharpe ratio. Repeat 1000 times. Plot the distribution of maximum Sharpe ratios. Students will see that even with zero signal, the best-of-100 Sharpe is typically 1.5-3.0. This makes the "how many backtests did you run" question very concrete.

**"So what?":** "This is the most inconvenient truth in quantitative finance. The more models you test, the more likely you are to find something that works by accident. There's no cure — only honesty. When you report your results, report N (the number of things you tried). In Week 18, we'll formalize this with the deflated Sharpe ratio."

### Closing Bridge

"Take stock of what happened this week. You replaced fixed-horizon labels with triple-barrier labels that adapt to volatility and respect the reality of trading. You replaced naive CV with purged CV that accounts for the label overlap that's been silently inflating your results. You learned to weight samples by uniqueness, because 252 daily labels per year contain maybe 12 independent observations. And you discovered meta-labeling — a technique that wraps any primary trading model in an ML filter that learns when the model is right and when it's wrong.

Your IC numbers probably dropped this week compared to Week 5. Good. The old numbers were inflated by information leakage. The new numbers are real.

Next week, we move to Module 3: Deep Learning. We'll build the Gu-Kelly-Xiu neural network in PyTorch and see whether it can beat the trees you built this week — when evaluated with the rigorous methodology you built this week. Trees remain the production default, but neural nets have their moment. The question isn't 'can deep learning work for finance?' — that's been settled. The question is 'when does the added complexity justify itself?' We'll find out."

## Seminar Exercises

### Exercise 1: Triple-Barrier Label Sensitivity — How Do Barrier Parameters Change Everything?
**The question we're answering:** How sensitive are triple-barrier labels to the choice of barrier multiplier and max holding period, and do the "right" parameters depend on the asset?

**Setup narrative:** "The lecture built triple-barrier labels for SPY with one set of parameters. But barrier design is a researcher degree of freedom — and like all such degrees, it can be a source of overfitting if you're not careful. You're going to systematically vary the parameters and see how the label distribution, class balance, and downstream model performance change."

**What they build:** Apply triple-barrier labeling to 5 diverse assets (SPY, TSLA, TLT, GLD, IWM) with a parameter sweep: profit-take multiplier in {1.0, 1.5, 2.0, 3.0}, stop-loss multiplier in {1.0, 1.5, 2.0, 3.0}, max holding period in {5, 10, 21, 42} trading days. For each combination: record label distribution (% profit-take, % stop-loss, % time expiry), average label duration, average uniqueness. Build a heatmap for each asset: (PT multiplier × SL multiplier) → class balance ratio. Then pick the "fairest" parameter set per asset (closest to 50/50 class balance for the binary outcome) and note how it differs across assets.

**What they'll see:** Tight barriers (multiplier=1.0) produce many barrier hits but high label noise (random touches of tight bands). Wide barriers (multiplier=3.0) produce few barrier hits and many time-expiry labels. The "sweet spot" differs by asset: volatile assets (TSLA) need wider barriers; low-vol assets (TLT) need tighter ones. Asymmetric barriers (e.g., PT=2.0, SL=1.5) produce skewed class balance — useful for modeling directional bias, but dangerous if unintended.

**The insight:** "Barrier parameters are not free — they embed assumptions about the trading strategy. Choosing PT=SL assumes symmetric risk preferences. Choosing max_period=21 assumes monthly rebalancing. These are design choices, not technical defaults, and they should be justified by the trading strategy the model serves. A model tuned on 'optimal' barrier parameters that happen to maximize backtest Sharpe is overfitting the label definition."

### Exercise 2: Combinatorial Purged CV — How Many Backtest Paths Is Your Strategy?
**The question we're answering:** How robust is your model across different train/test splits, and does combinatorial purged CV reveal fragility that a single expanding window hides?

**Setup narrative:** "The lecture demonstrated purged CV with a single fold structure. But one backtest path can be lucky. Combinatorial Purged CV (CPCV) generates many backtest paths from the same data — and their distribution tells you whether your strategy is robust or fragile. You're about to discover how much your results depend on which specific months landed in the test set."

**What they build:** Implement CPCV with N=6 temporal groups and k=2 held-out groups, producing $\binom{6}{2} = 15$ backtest paths. For each path: train the Week 5 XGBoost on the 4 non-held-out groups (with purging and embargo), predict on the 2 held-out groups, compute the Sharpe ratio of the resulting backtest. Plot the distribution of 15 Sharpe ratios. Then repeat with k=3 (producing 20 paths) and compare the distribution width. Also compute: what fraction of paths have Sharpe > 0? What's the median vs. mean Sharpe?

**What they'll see:** The 15 Sharpe ratios will not be identical — they'll range from perhaps -0.3 to +1.5. Some paths include favorable periods (trending markets where momentum works) and some include unfavorable ones (momentum crashes). The distribution width tells you about strategy robustness: a tight distribution means the strategy works consistently; a wide distribution means it depends on which months you happened to test on. The fraction with Sharpe > 0 is a more honest assessment than the single-backtest Sharpe.

**The insight:** "A single backtest Sharpe is a point estimate from one draw. CPCV gives you a distribution. If 12 of 15 paths have positive Sharpe, your confidence is much higher than if 8 of 15 do. The distribution width is your model's fragility metric — it tells you how much of your 'edge' is specific to the test period vs. genuinely robust. This is the precursor to the deflated Sharpe ratio in Week 18."

### Exercise 3: Meta-Labeling Stress Test — Multiple Primary Models
**The question we're answering:** Does meta-labeling improve any primary model, or does it only help certain types of strategies?

**Setup narrative:** "The lecture demonstrated meta-labeling with a single primary model (SMA crossover). But if meta-labeling only helps simple trend-following, that's a limited result. You're going to test it on three fundamentally different primary models and see if the improvement generalizes."

**What they build:** Implement three primary models for SPY: (1) SMA 50/200 crossover (trend-following), (2) RSI(14) mean-reversion (buy when RSI < 30, sell when RSI > 70), (3) the Week 5 XGBoost cross-sectional model's top-quintile signal adapted to SPY. For each primary model: apply triple-barrier labeling at entry points, train a meta-model (XGBoost) with the same feature set (volatility, momentum strength, volume surge, RSI, distance to 52-week high), and compute position sizing = signal × meta_probability. Compare: each primary model alone vs. with meta-labeling. Report hit rate, profit factor, Sharpe, max drawdown for all six variants (3 × {with, without}).

**What they'll see:** Meta-labeling will improve profit factor and reduce drawdown for all three primary models — but the magnitude of improvement will vary. Trend-following (SMA) benefits the most, because it has long holding periods with many "bad" trades that the meta-model learns to skip. Mean-reversion (RSI) benefits less, because its trades are shorter and more binary. The ML primary model benefits modestly — it's already "smart," so the meta-model has less room to filter. The meta-model's feature importances will differ across primary models: for SMA, volatility is key (trend-following fails in choppy markets); for RSI, volume is key (mean-reversion fails on genuine breakouts).

**The insight:** "Meta-labeling is model-agnostic — it improves any primary model, but by different amounts. The improvement is largest when the primary model is simple and generates many low-quality trades. For sophisticated primary models (like your XGBoost), the meta-model is more of a refinement. This tells you where to invest your effort: if your primary model is simple, meta-labeling is high-ROI. If it's already complex, the marginal gain is smaller."

### Exercise 4: Demonstrating the Information Leakage
**The question we're answering:** Can we build a model that looks great in standard CV but is actually useless?

**Setup narrative:** "We're going to deliberately construct a leaking feature to show exactly how information leakage works — and how purged CV catches it."

**What they build:** Create a synthetic leaking feature: the 5-day forward return lagged by 3 days. This feature directly contains future information (it 'knows' what happened 3 days from now, but the label covers 21 days, so there's 18 days of genuine forward information and 3 days of overlap). Train a model with this feature. Evaluate with standard CV (the model looks incredible) and purged CV (the model looks mediocre, because the leaking feature's advantage is reduced when overlapping samples are purged).

**What they'll see:** With standard CV, the leaking feature will be the top feature and the model will have IC > 0.10. With purged CV, the leaking feature's importance will drop dramatically and the IC will fall to a realistic 0.02-0.04. This is a vivid demonstration of how leakage inflates results and how purging catches it.

**The insight:** "Purged CV isn't just a theoretical improvement — it's a leakage detector. If your model's performance drops significantly when you switch from standard to purged CV, that's a red flag. The performance drop tells you how much of your model's 'skill' was actually information leakage."

## Homework: "The Meta-Labeling Pipeline"

### Mission Framing

This is the homework where everything comes together. You're going to build a complete, methodologically rigorous financial ML pipeline — from signal generation to meta-labeling to backtesting — using every technique from this week. The pipeline will be the template for everything you build for the rest of the course and, frankly, for the rest of your career if you work in this field.

The mission is specific: take a simple primary model (50/200 day moving average crossover), wrap it with meta-labeling, evaluate with purged k-fold CV, and compare against the primary model alone. The question isn't whether meta-labeling helps — the literature says it does. The question is: by how much? And does the improvement survive honest evaluation with purged CV, sample weighting, and transaction costs?

You'll build two reusable classes — `TripleBarrierLabeler` and `PurgedKFold` — that will be used whenever pipeline rigor matters in subsequent weeks, and especially in the Week 18 capstone. Think of them as infrastructure investments: they cost time now but save time (and mistakes) later.

### Deliverables

1. **Implement a primary model: 50/200 day simple moving average crossover.** Apply to SPY daily data from 2005-2024. Buy when SMA(50) crosses above SMA(200); sell when it crosses below. Document the signal: number of trades, average holding period, hit rate, and profit factor.

2. **Implement triple-barrier labeling with dynamic barriers.** Barriers = [1.5, 2.0, 2.5] × trailing 20-day volatility. Max holding period = 21 trading days. Apply to the primary model's trade entry points. Report: label distribution (% profit-take, % stop-loss, % time expiry), average label duration, and average uniqueness.

3. **Train a meta-labeling model (Random Forest or XGBoost).** Features: trailing 20-day volatility, trailing 20-day momentum (return), 5-day volume relative to 20-day average, RSI(14), distance to 52-week high, and the primary model's signal duration (how many days the signal has been active). Target: binary label (trade profitable or not, based on triple-barrier). Use sample weighting by average uniqueness.

4. **Implement purged k-fold CV** with at least 5 folds and embargo period = maximum label duration. Use this for all model evaluation. Report: accuracy, precision, recall, and F1 for the meta-model. Compare to standard (non-purged) k-fold — how much does purging reduce the metrics?

5. **Show the full pipeline: primary signal → meta-label filter → position sizing → backtest.** Position size = primary_signal × meta_probability. Compare three variants:
   - Strategy A: primary model alone (equal position size on all trades)
   - Strategy B: primary model + meta-label binary filter (only trade when meta_prob > 0.5)
   - Strategy C: primary model + meta-label position sizing (position = meta_prob)

6. **Evaluate all three strategies.** Report: Sharpe ratio, hit rate, profit factor, maximum drawdown, Calmar ratio. Include 10 bps round-trip transaction costs. Which strategy is best after costs?

7. **Build reusable `TripleBarrierLabeler` and `PurgedKFold` classes.** These should be general-purpose: `TripleBarrierLabeler` takes a price series, entry points, and barrier parameters; `PurgedKFold` takes a dataset with label end times and produces purged train/test splits. Both should be importable as standalone Python classes.

### What They'll Discover

- The SMA crossover (primary model alone) has a Sharpe ratio of about 0.3-0.5 on SPY — barely above zero. It's a mediocre strategy. The hit rate is around 52-54%. Most of the profit comes from a few large trends; the rest of the trades are noise.

- The meta-labeling model (Strategy B or C) will improve the profit factor from roughly 1.1 to 1.3-1.6. The Sharpe ratio will improve by 0.2-0.5 points. But the improvement comes almost entirely from skipping bad trades, not from entering better trades. The number of trades will decrease, and the average profit per trade will increase.

- Purged CV will produce lower metrics than standard CV — typically 20-50% lower. This is the honest assessment. If you used standard CV, you'd overestimate your strategy's edge and deploy with too much capital.

- The meta-model's feature importance will show that volatility and volume are the most important predictors of trade success. Volatility tells the model whether the market is in a regime where the SMA crossover works (trending) or doesn't (mean-reverting). Volume tells the model whether there's conviction behind the move. These are economically sensible features, which is reassuring.

- The choice of barrier multiplier matters: tighter barriers (1.5×) produce more labels but weaker signals. Wider barriers (2.5×) produce fewer labels but stronger signals. There's a sweet spot around 2.0× for most assets.

### Deliverable
A complete Jupyter notebook containing: the full meta-labeling pipeline, purged CV implementation, three-strategy comparison, `TripleBarrierLabeler` and `PurgedKFold` classes, and a discussion of results. The classes should be clean, documented, and importable.

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Standard ML methodology failures in finance | Demo: show label overlap autocorrelation, quantify leakage across 3 CV methods | Not covered (done in lecture) | Reference: use as motivation for rigorous methodology |
| Triple-barrier labeling | Demo: build labeler from scratch for SPY, compare to fixed labels | Exercise 1: parameter sensitivity sweep across 5 assets, heatmap of class balance | Build: `TripleBarrierLabeler` class with dynamic barriers for homework pipeline |
| Sample uniqueness & weighting | Demo: compute average uniqueness for SPY, compare fixed vs. triple-barrier | Not covered (done in lecture) | Integrate: use uniqueness weights in meta-model training |
| Purged k-fold CV | Demo: build `PurgedKFold` from scratch, compare to standard CV | Exercise 2: implement CPCV with N=6/k=2, analyze distribution of 15 Sharpe ratios | Build: reusable `PurgedKFold` class for all subsequent evaluations |
| Meta-labeling | Demo: full pipeline with SMA crossover primary model | Exercise 3: test meta-labeling on 3 different primary model types, compare improvement | At scale: full meta-labeling pipeline for homework backtest |
| Information leakage detection | Demo: show IC inflation from standard to purged CV | Exercise 4: deliberately inject leaking feature, show purged CV catches it | Integrate: quantify leakage gap in homework strategy |
| Multiple testing problem (preview) | Demo: simulate max Sharpe from 100 zero-signal strategies | Not covered (preview for W18) | Reference: count total strategies tested, compute E[max SR] |
| Position sizing from meta-probabilities | Demo: position = signal × meta_probability | Exercise 3: compare binary filter vs. continuous sizing | At scale: compare 3 variants (no filter, binary, continuous) in homework |

## Key Stories & Facts to Weave In

1. **Lopez de Prado's career and methodology.** Marcos Lopez de Prado managed over $13 billion in assets at AQR Capital Management and Guggenheim Partners. His book *Advances in Financial Machine Learning* (2018) was controversial in academia but embraced by practitioners. His central thesis: most financial ML research uses wrong methodology, and the results are therefore unreliable. Triple-barrier labeling, purged CV, and meta-labeling are his proposed solutions. Whether you agree with every detail, the core insight is unchallengeable: methodology matters more than models.

2. **The backtest overfitting crisis.** Bailey, Borwein, Lopez de Prado, and Zhu (2014) showed that with modern computing power, it's trivially easy to find a strategy with a Sharpe ratio above 2.0 from historical data — even if no genuine strategy exists. They estimated that more than 50% of published backtest results in finance are likely false positives. The paper, "Pseudo-Mathematics and Financial Charlatanism," is one of the most provocative titles in academic finance. It argues that backtesting without controlling for multiple testing is "pseudo-mathematics" — it has the appearance of rigor without the substance.

3. **The $200 million lesson.** Multiple quant funds (names usually withheld for obvious reasons) have deployed strategies that worked brilliantly in backtest and failed in live trading. The most common cause: label overlap leakage in cross-validation. A 5-day return label with daily rebalancing creates 80% overlap between consecutive labels. Standard k-fold CV doesn't account for this, so it reports a model that's 3-5x better than reality. The gap is only discovered when live P&L doesn't match the backtest.

4. **The Medallion Fund's approach to methodology.** Renaissance Technologies is famously secretive, but former employees have described their approach as "fanatically rigorous about evaluation methodology." They reportedly use hundreds of millions of simulated strategies to calibrate their expectations about what "significant" means. Their effective N (number of strategies tested) is enormous — which is why they require extremely high bars for statistical significance before deploying capital.

5. **Meta-labeling in production.** Multiple hedge funds (including Man AHL and systematic divisions of larger firms) use some form of meta-labeling in production. The primary model might be a fundamental analyst's stock picks, a systematic momentum strategy, or an ML model. The meta-model wraps around it and adjusts position size. This is one of the cleanest applications of ML in practice because it doesn't require replacing existing processes — it augments them.

6. **The volatility scaling insight.** Dynamic barriers (scaling by volatility) solve a problem that most ML practitioners don't even know exists: in a constant-threshold world, all your labels during low-volatility periods are "no move" and all your labels during high-volatility periods are "large move." The model learns to distinguish calm from volatile markets instead of learning which direction the market moves. Volatility-scaled labels normalize for this, producing labels that are comparable across regimes.

7. **Hudson & Thames' MLFinLab.** The commercial implementation of Lopez de Prado's methods is MLFinLab by Hudson & Thames. It costs $7,500/year for an academic license. We build everything from scratch because: (a) it's the only way to understand how the methods work, (b) our implementations will be tailored to our use cases, and (c) $7,500 is a lot of money for a student. The classes you build this week are functionally equivalent to the core of MLFinLab.

8. **CPCV and the number of backtest paths.** Combinatorial Purged Cross-Validation (CPCV) with N=6 groups and k=2 held-out produces $\binom{6}{2} = 15$ backtest paths. With N=10 and k=2: 45 paths. With N=10 and k=3: 120 paths. Each path is a valid purged backtest. The distribution of Sharpe ratios across these paths tells you about the strategy's robustness — a wide distribution suggests the strategy is fragile and depends on the specific training/test split. A tight distribution suggests genuine, stable signal.

## Cross-References
- **Builds on:** The cross-sectional prediction pipeline from Weeks 4-5 (features, models, evaluation). The expanding-window CV from Week 4 (purged CV is the corrected version). The GARCH volatility model from Week 2 (used for dynamic barrier sizing). Risk metrics from Week 3 (Sharpe, drawdown, Calmar used in the comparison).
- **Sets up:** The `TripleBarrierLabeler` and `PurgedKFold` classes are used in the homework for every subsequent week that involves model evaluation. The meta-labeling framework is revisited in Week 11 (uncertainty-filtered trading, which is conceptually similar to meta-labeling) and Week 18 (the full pipeline). The multiple testing discussion is resolved in Week 18 with the deflated Sharpe ratio.
- **Recurring thread:** The "honest evaluation" thread reaches its climax here. Week 4 introduced expanding-window CV (better than shuffled CV). This week introduces purged CV (better than expanding-window CV). Week 18 will add the deflated Sharpe ratio (accounting for multiple testing). Each step removes a layer of self-deception from the evaluation.

## Suggested Reading
- **Lopez de Prado, *Advances in Financial Machine Learning*, Chapters 3-5, 7** — The source material for this entire week. Chapter 3: triple-barrier labeling. Chapter 4: sample uniqueness and weighting. Chapter 5: fractional differentiation (revisited from Week 2). Chapter 7: purged cross-validation. Dense but essential.
- **Lopez de Prado, *Causal Factor Investing* (2023)** — Lopez de Prado's philosophical evolution. He now argues that even rigorous backtesting isn't enough — you need causal reasoning to distinguish genuine factors from spurious correlations. Preview of Week 18's causal inference discussion.
- **Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"** — The paper that quantifies how many strategies you can test before selection bias dominates. Short, readable, and deeply unsettling. You'll implement the DSR in Week 18, but read the paper now to understand why honest accounting of N matters.
