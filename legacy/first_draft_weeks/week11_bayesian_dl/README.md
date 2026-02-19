# Week 11 — Bayesian Deep Learning & Uncertainty Quantification

> **Your model says "buy." How much does it believe itself? Three lines of code give you the answer.**

## Prerequisites
- **Week 7 (Feedforward Nets):** The Gu-Kelly-Xiu neural net with dropout. You built this model. You used dropout as regularization during training. This week, you leave dropout ON at inference — and that one change turns your point-prediction model into an uncertainty-aware model. You need the trained model, the architecture code, and the expanding-window evaluation framework.
- **Week 5 (XGBoost):** Your XGBoost pipeline is the performance benchmark. Deep ensembles (this week) formalize what you already did informally in Week 7: train multiple models and use their disagreement as an uncertainty measure.
- **Week 3 (Portfolio/Risk):** Position sizing, Sharpe ratio, risk metrics. Uncertainty estimates change how you SIZE positions, not just which ones you take. This week we introduce the Kelly criterion — the mathematically optimal bet sizing formula — and it requires knowing your uncertainty.
- **Week 4 (Linear Models):** Ridge regression. We'll show that Ridge regression IS Bayesian linear regression with a Gaussian prior — the regularization parameter is the prior precision. This connection bridges classical ML and Bayesian inference.

## The Big Idea

Every model you've built in this course has a fatal flaw: it's overconfident. Your neural net from Week 7 predicts "+1.5% return for AAPL" and gives no indication of whether that prediction is backed by clear, consistent signals or whether the model is extrapolating wildly into unfamiliar territory. Your XGBoost from Week 5 predicts "short TSLA" and doesn't tell you whether the features look like training data (high confidence) or like nothing it's ever seen (low confidence). You treat both predictions the same — equal position size, equal conviction. This is exactly wrong.

In traditional ML, we optimize for accuracy. In financial ML, we optimize for risk-adjusted returns. And risk adjustment requires knowing not just WHAT to predict but HOW CERTAIN you are. A prediction of "+2% return with 95% confidence" calls for a large position. A prediction of "+2% return with 55% confidence" calls for a small position or no position at all. The Kelly criterion — the mathematically optimal bet sizing formula — requires both the expected return AND the uncertainty. Without uncertainty, you're flying blind.

The beautiful irony is that you already have the tools. Dropout, which you added to your Week 7 model to prevent overfitting, is mathematically equivalent to approximate Bayesian inference. Leave dropout on at inference time, run the same input through the model 100 times, and you get 100 different predictions. The mean is your point estimate. The standard deviation is your uncertainty. Three lines of code. Yarin Gal and Zoubin Ghahramani proved this in 2016, and it remains one of the most elegant results in modern machine learning.

The alternative — deep ensembles — is equally simple and empirically better calibrated. You trained 5 models with different seeds in Week 7. The mean of their predictions is your ensemble prediction. The standard deviation is your uncertainty. You already did this. You just didn't know it was Bayesian.

This week adds the missing dimension to every model you've built. By the end, you'll have uncertainty-aware predictions that enable you to: size positions optimally with the Kelly criterion, detect regime changes (when model uncertainty spikes across the entire universe), and filter out low-confidence predictions before they cost you money.

## Lecture Arc

### Opening Hook

"On March 9, 2020 — six days before the COVID crash bottomed out — a standard neural net trained on 10 years of US equity data would have predicted returns as usual. The predictions might have been wrong, but they would have LOOKED normal — the model would have shown the same level of confidence as it did in January 2020, when everything was calm. An MC Dropout version of the exact same model would have been screaming. The epistemic uncertainty — the model's measure of its own confusion — would have spiked to 3-4x normal levels, because the input features (VIX at 54, daily moves of 7-10%, correlations spiking to 1) looked nothing like training data. The model was saying, in the clearest possible terms: 'I have no idea what's happening.' A strategy that listened — that reduced position sizes when uncertainty was high — would have dodged the worst of a 34% drawdown. Today, we build that strategy."

### Section 1: Two Types of Uncertainty — Why the Distinction Matters

**Narrative arc:** We establish the fundamental distinction between aleatoric (data) and epistemic (model) uncertainty, and show that they have different practical implications for trading.

**Key concepts:** Aleatoric uncertainty, epistemic uncertainty, irreducible vs. reducible uncertainty, heteroskedasticity.

**The hook:** "There are two reasons your model might be wrong about AAPL's next-month return. Reason 1: AAPL's return is inherently noisy — even with perfect information, you can't predict it exactly. That's aleatoric uncertainty. It's the irreducible noise in financial returns, and no amount of data or model complexity can eliminate it. Reason 2: your model has never seen features like AAPL's current features before — maybe the stock just had a 20% one-day move and your training data has no similar examples. That's epistemic uncertainty. It's the model's ignorance, and it CAN be reduced with more data or a better model. The practical difference: aleatoric uncertainty should inform your risk management (set wider stop-losses). Epistemic uncertainty should inform your position sizing (trade smaller or not at all)."

**Key formulas:**

Total predictive uncertainty decomposes as:

$$\text{Var}[\hat{r}] = \underbrace{\mathbb{E}[\sigma^2_{\text{aleatoric}}]}_{\text{data noise}} + \underbrace{\text{Var}[\mu_{\text{model}}]}_{\text{model uncertainty}}$$

"Read this equation carefully. The first term is the average noise in the data — it doesn't change no matter how many times you run the model. The second term is the variance of the model's mean prediction across different model instantiations — it shrinks as you see more data or get a better model."

For MC Dropout, these terms are estimated as:

$$\text{Aleatoric} \approx \frac{1}{T} \sum_{t=1}^{T} \hat{\sigma}^2_t \quad \text{(average predicted variance)}$$

$$\text{Epistemic} \approx \frac{1}{T} \sum_{t=1}^{T} (\hat{\mu}_t - \bar{\mu})^2 \quad \text{(variance of predicted means)}$$

where $T$ is the number of forward passes and $\hat{\mu}_t$, $\hat{\sigma}^2_t$ are the mean and variance predicted in pass $t$.

"The aleatoric term captures: 'Returns are noisy.' The epistemic term captures: 'My model doesn't know what to do with these inputs.' For trading, the epistemic term is more actionable — it tells you when to step aside."

**Code moment:**

```python
# Conceptual illustration
# Aleatoric: flip a fair coin — P(heads) = 0.5 ± 0. You KNOW it's 50/50.
#            More data doesn't help. The uncertainty is in the outcome, not the model.

# Epistemic: you find a weighted coin. Is P(heads) = 0.6? 0.7? 0.8?
#            You're uncertain about the MODEL. More flips reduce this uncertainty.

# In finance:
# Aleatoric: AAPL's daily returns have ~2% daily std. No model eliminates this.
# Epistemic: The model has never seen VIX at 80. It doesn't know what to predict.
```

**"So what?":** "If your model has high aleatoric uncertainty for a stock, that stock is inherently risky — set tighter risk limits. If your model has high epistemic uncertainty, the model is confused — reduce position size or don't trade. These are different actions for different types of uncertainty. Conflating them costs money."

### Section 2: MC Dropout — Bayesian Inference for Free

**Narrative arc:** The central technique of this week. We show that dropout at inference is mathematically equivalent to approximate variational inference, then implement it in three lines.

**Key concepts:** Dropout as approximate posterior, variational inference, Monte Carlo sampling, Gal & Ghahramani (2016).

**The hook:** "Yarin Gal was a PhD student at Cambridge when he proved something that surprised the deep learning community: if you have a neural network with dropout, you already have a Bayesian model. You just don't know it yet. Training with dropout is equivalent to performing variational inference on the network weights. And running inference with dropout left ON is equivalent to sampling from the approximate posterior. The result is uncertainty estimates that are calibrated, computationally free (if you already use dropout), and require exactly zero changes to your training procedure."

**The mathematical argument (simplified):**

Training a neural net with dropout minimizes:

$$\mathcal{L} = \frac{1}{N} \sum_{i} \ell(y_i, f_{\theta}(x_i)) + \lambda \|\theta\|^2$$

where the dropout randomly zeros out neurons. Gal showed this is equivalent to minimizing the KL divergence between an approximate posterior $q(\theta)$ (a Bernoulli distribution over weight masks) and the true posterior $p(\theta | D)$:

$$\mathcal{L}_{\text{VI}} = \text{KL}[q(\theta) \| p(\theta)] - \mathbb{E}_{q(\theta)}[\log p(D | \theta)]$$

"The L2 regularization term ($\lambda \|\theta\|^2$) corresponds to a Gaussian prior on the weights. The dropout rate $p$ determines the variational family. The training loss you've been minimizing all along IS a variational objective. Gal just recognized it."

**Inference with MC Dropout:**

At inference, instead of the standard procedure (dropout OFF, single forward pass), you keep dropout ON and run $T$ forward passes:

$$\hat{r}_i^{(t)} = f_{\theta, z_t}(x_i), \quad z_t \sim \text{Bernoulli}(1-p), \quad t = 1, ..., T$$

Point estimate: $\bar{r}_i = \frac{1}{T} \sum_t \hat{r}_i^{(t)}$

Uncertainty: $\sigma_i = \sqrt{\frac{1}{T} \sum_t (\hat{r}_i^{(t)} - \bar{r}_i)^2}$

"Each forward pass uses a different random dropout mask — effectively sampling a different sub-network from the ensemble of all possible sub-networks. The variance of predictions across masks IS the epistemic uncertainty."

**Code moment — the three-line version:**

```python
# Standard inference (no uncertainty)
model.eval()
prediction = model(X_test)

# MC Dropout inference (uncertainty for free)
model.train()  # <-- THIS is the key line. Keeps dropout active.
preds = torch.stack([model(X_test) for _ in range(100)])  # 100 forward passes
mean_pred = preds.mean(dim=0)
uncertainty = preds.std(dim=0)
```

"That's it. `model.train()` keeps dropout active. `torch.stack` collects 100 stochastic predictions. `.mean()` and `.std()` give you the point estimate and uncertainty. Your Week 7 model with Dropout(0.5) becomes a Bayesian model with these three lines."

**How many forward passes (T)?**

"T=100 is the standard. T=50 is often sufficient. T=30 gives noisier uncertainty estimates but is 3x faster. For the small models we use (1,337 parameters), T=100 takes about 200 milliseconds on M4. Not a bottleneck."

**"So what?":** "MC Dropout is the practical default at quant funds. Zero additional training cost. Zero architectural changes. Minimal inference overhead. The only requirement is that your model already uses dropout — and if it doesn't, it should, because dropout is good regularization regardless. You get uncertainty for free, and 'free' is the most compelling price in all of finance."

### Section 3: Deep Ensembles — Better Calibration, More Compute

**Narrative arc:** Deep ensembles are the alternative to MC Dropout — empirically better calibrated but M-times more expensive to train. We compare the two approaches honestly.

**Key concepts:** Ensemble uncertainty, calibration, proper scoring rules, Lakshminarayanan et al. (2017).

**The hook:** "In 2017, Lakshminarayanan, Pritzel, and Blundell at DeepMind published a paper with a deliberately boring title: 'Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.' The approach is exactly as boring as the title suggests: train M models with different random seeds, average their predictions, use their disagreement as uncertainty. No Bayesian math, no variational inference, no fancy priors. Just train more models. And yet this simple approach is consistently better calibrated than MC Dropout, Bayes by Backprop, and every other Bayesian approximation tested."

**Key formulas:**

Ensemble prediction and uncertainty:

$$\hat{r}_i^{\text{ens}} = \frac{1}{M} \sum_{m=1}^{M} f_{\theta_m}(x_i)$$

$$\sigma_i^{\text{ens}} = \sqrt{\frac{1}{M} \sum_{m=1}^{M} (f_{\theta_m}(x_i) - \hat{r}_i^{\text{ens}})^2}$$

"When all M models agree on a prediction, $\sigma^{\text{ens}}$ is small — the ensemble is confident. When models disagree, $\sigma^{\text{ens}}$ is large — the ensemble is uncertain. Unlike MC Dropout (which samples sub-networks from a single model), deep ensembles sample completely independent models. This means the diversity of predictions is typically higher, and the uncertainty estimates are better calibrated."

**MC Dropout vs. Deep Ensembles — the honest comparison:**

| Property | MC Dropout | Deep Ensembles |
|----------|-----------|---------------|
| Training cost | 1x (single model) | Mx (train M models) |
| Inference cost | T forward passes (fast) | M forward passes (same) |
| Calibration | Good | Better |
| Diversity of predictions | Limited (sub-networks) | Higher (independent inits) |
| Captures multi-modality? | No | Yes |
| Ease of implementation | 3 lines of code | Train M models, average |

"For a 3-layer financial model that trains in 3 minutes, an ensemble of M=5 trains in 15 minutes. That's the total additional cost. In return, you get better uncertainty estimates AND a better point prediction (ensemble averaging). In industry, most teams use both: ensemble of M=5 models, each with MC Dropout at inference."

**Code moment:**

```python
# Deep ensemble
models = []
for seed in range(5):
    torch.manual_seed(seed)
    model = GKXNet(n_features).to(device)
    train_model(model, train_loader, val_loader)
    models.append(model)

# Ensemble prediction + uncertainty
with torch.no_grad():
    all_preds = torch.stack([m(X_test) for m in models])
    ensemble_mean = all_preds.mean(dim=0)     # point prediction
    ensemble_std = all_preds.std(dim=0)       # epistemic uncertainty

# MC Dropout ON TOP of ensemble (belt + suspenders)
for m in models:
    m.train()  # enable dropout
mc_preds = torch.stack([
    torch.stack([m(X_test) for _ in range(20)])  # 20 MC samples per model
    for m in models
])  # shape: (5, 20, n_stocks)
full_uncertainty = mc_preds.reshape(-1, mc_preds.shape[-1]).std(dim=0)
```

**"So what?":** "Deep ensembles are the gold standard for predictive uncertainty in neural networks. MC Dropout is the practical standard. Use ensembles if you can afford M× training time (usually yes for small financial models). Use MC Dropout if you can't. Use both if you want the most reliable uncertainty estimates."

### Section 4: Bayesian Linear Regression — The Ridge Connection

**Narrative arc:** We show that Ridge regression, which students already know, is secretly Bayesian. This provides the intuitive bridge between classical ML and Bayesian inference.

**Key concepts:** Conjugate priors, Gaussian prior on weights, posterior distribution, Ridge = MAP with Gaussian prior.

**The hook:** "Here's something that will change how you think about Ridge regression. When you set the regularization parameter $\alpha$ in Ridge, you're secretly choosing the variance of a Gaussian prior on the weights. A large $\alpha$ means 'I believe the weights should be close to zero' — a strong prior. A small $\alpha$ means 'I'll let the data decide' — a weak prior. Ridge regression gives you the maximum a posteriori (MAP) estimate under this prior. But Bayesian linear regression gives you the entire posterior distribution — not just the most likely weights, but the full range of plausible weights. The posterior is available in closed form. No MCMC. No approximation. Exact Bayesian inference, for free."

**Key formulas:**

Setup: $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$, where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 I)$.

Prior: $\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$

Posterior (closed-form because the Gaussian prior is conjugate to the Gaussian likelihood):

$$\boldsymbol{\beta} | \mathbf{y}, \mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}_{\text{post}}, \boldsymbol{\Sigma}_{\text{post}})$$

where:

$$\boldsymbol{\Sigma}_{\text{post}} = \left(\frac{1}{\sigma^2} \mathbf{X}^T \mathbf{X} + \frac{1}{\tau^2} \mathbf{I}\right)^{-1}$$

$$\boldsymbol{\mu}_{\text{post}} = \boldsymbol{\Sigma}_{\text{post}} \cdot \frac{1}{\sigma^2} \mathbf{X}^T \mathbf{y}$$

"Compare the posterior mean to the Ridge solution:"

$$\boldsymbol{\beta}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

"They're the same thing when $\alpha = \sigma^2 / \tau^2$. Ridge gives you the MAP estimate. Bayesian linear regression gives you the MAP estimate PLUS the posterior covariance $\boldsymbol{\Sigma}_{\text{post}}$ — which tells you how uncertain each coefficient is."

**Why this matters for factor models:**

"In a factor model, the coefficients represent factor loadings — how much a stock's return depends on the market factor, the size factor, the momentum factor. The posterior distribution tells you not just the best estimate of each loading but how confident you are in it. A coefficient of 0.5 with posterior std of 0.1 is a reliable factor. A coefficient of 0.5 with posterior std of 0.8 is noise."

**Code moment:**

```python
import numpy as np

def bayesian_linear_regression(X, y, sigma2=1.0, tau2=1.0):
    """Exact Bayesian linear regression with conjugate Gaussian prior."""
    n, p = X.shape
    # Posterior covariance
    Sigma_post = np.linalg.inv(X.T @ X / sigma2 + np.eye(p) / tau2)
    # Posterior mean (= Ridge solution with alpha = sigma2/tau2)
    mu_post = Sigma_post @ (X.T @ y / sigma2)
    return mu_post, Sigma_post

# Compare to Ridge
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)  # alpha = sigma2/tau2
ridge.fit(X, y)
# ridge.coef_ == mu_post (up to numerical precision)
```

**"So what?":** "Bayesian linear regression is the cleanest illustration of what Bayesian methods buy you: not just a point estimate, but a full posterior distribution that quantifies uncertainty in every parameter. For neural networks, exact posteriors are intractable — that's why we use approximations like MC Dropout and ensembles. But for factor models (which are linear), you get exact Bayesian uncertainty for free."

### Section 5: Applications — What Uncertainty Enables

**Narrative arc:** We connect uncertainty estimates to three concrete, money-making applications: Kelly criterion position sizing, regime detection, and stock filtering.

**Key concepts:** Kelly criterion with uncertainty, epistemic uncertainty as regime detector, confidence-filtered trading.

**Application 1: Kelly Criterion with Uncertainty**

**The hook:** "The Kelly criterion says: bet a fraction $f^* = \mu / \sigma^2$ of your bankroll, where $\mu$ is expected return and $\sigma^2$ is variance. But which variance? If you only have aleatoric uncertainty, $f^*$ might be large — the model is 'confident' in a prediction with high expected return. But if you include epistemic uncertainty, $f^*$ shrinks — the model is saying 'I think the return is high, but I'm not sure.' The difference between using aleatoric-only vs. total uncertainty for Kelly sizing is the difference between aggressive betting on every prediction and cautious betting on confident predictions."

**Key formulas:**

Kelly with total uncertainty:

$$f^* = \frac{\hat{\mu}_i}{\hat{\sigma}^2_{\text{aleatoric},i} + \hat{\sigma}^2_{\text{epistemic},i}}$$

"When epistemic uncertainty is low (model is confident), $f^*$ is large — full conviction. When epistemic uncertainty is high (model is confused), $f^*$ shrinks toward zero — the model's ignorance reduces the bet. This is exactly the behavior you want: aggressive when confident, cautious when uncertain."

**Code moment:**

```python
# Kelly criterion with uncertainty
def kelly_weights(mean_pred, aleatoric_var, epistemic_var):
    """Position size = expected return / total variance."""
    total_var = aleatoric_var + epistemic_var
    kelly_f = mean_pred / (total_var + 1e-8)
    # Half-Kelly is standard practice (less aggressive)
    return kelly_f * 0.5
```

**Application 2: Regime Detection via Epistemic Uncertainty**

**The hook:** "Here's a feature nobody expected: the average epistemic uncertainty across your stock universe is a regime indicator. In calm markets, most stocks look like training data — uncertainty is low and stable. Before a crisis, features start moving outside the training distribution — uncertainty creeps up. During a crisis, everything is out-of-distribution — uncertainty spikes. You didn't build a regime detection model. You built a prediction model that admits when it's confused. Same thing."

**Code moment:**

```python
# Universe-level uncertainty over time
avg_uncertainty = uncertainty_per_stock.groupby('date').mean()
avg_uncertainty.plot(title='Average Epistemic Uncertainty Over Time')
# Label: COVID crash, 2022 rate hikes, SVB collapse
```

"Plot this. You'll see spikes in late February 2020 (pre-COVID), early 2022 (rate hike cycle), and March 2023 (SVB collapse). The uncertainty signal leads the crisis by days to weeks — it's a genuine early warning system."

**Application 3: Uncertainty-Filtered Trading**

**The hook:** "The simplest application: don't trade stocks where the model is uncertain. Split your universe into 'confident' (bottom 50% of epistemic uncertainty) and 'uncertain' (top 50%). Trade only the confident half. The Sharpe ratio almost always improves — you're removing the predictions that are most likely to be wrong."

**Code moment:**

```python
# Split by uncertainty
confident_mask = epistemic_uncertainty < epistemic_uncertainty.median()
uncertain_mask = ~confident_mask

sharpe_all = compute_sharpe(portfolio_all)
sharpe_confident = compute_sharpe(portfolio_confident)
sharpe_uncertain = compute_sharpe(portfolio_uncertain)

# Typical result: sharpe_confident > sharpe_all > sharpe_uncertain
```

**"So what?":** "Uncertainty quantification changes the economics of your strategy. Without it, you're trading blindly — equal conviction on every prediction. With it, you concentrate capital on high-conviction predictions and step aside when the model is confused. In a world where transaction costs eat into marginal trades, being able to say 'I'll pass on this one' is worth more than having a slightly better prediction for every stock."

### Section 6: Calibration — Does Your Uncertainty Mean What You Think It Means?

**Narrative arc:** Uncertainty estimates are only useful if they're calibrated — if a "90% prediction interval" actually contains the true value 90% of the time. We test this.

**Key concepts:** Calibration, reliability diagrams, prediction intervals, coverage probability.

**The hook:** "Here's the embarrassing truth about most uncertainty estimates: they're not calibrated. A model that says '90% prediction interval: [-2%, +4%]' might contain the true return only 70% of the time — the intervals are too narrow. Or 98% of the time — the intervals are too wide. Under-coverage means your uncertainty is overconfident — dangerous. Over-coverage means it's too conservative — you'll trade too little. Calibration analysis tells you which failure mode you have."

**Key formulas:**

For a nominal coverage level $\alpha$ (e.g., 90%), the prediction interval is:

$$[\bar{r}_i - z_{\alpha/2} \cdot \sigma_i, \quad \bar{r}_i + z_{\alpha/2} \cdot \sigma_i]$$

The actual coverage is:

$$\hat{C}(\alpha) = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[r_i \in [\bar{r}_i - z_{\alpha/2} \cdot \sigma_i, \bar{r}_i + z_{\alpha/2} \cdot \sigma_i]]$$

"Perfect calibration: $\hat{C}(\alpha) = \alpha$ for all $\alpha$. The reliability diagram plots $\hat{C}$ vs. $\alpha$ — a perfectly calibrated model follows the diagonal."

**Code moment:**

```python
def reliability_diagram(predictions, uncertainties, actuals, n_bins=10):
    """Plot calibration: does 90% PI contain 90% of actuals?"""
    coverages = []
    nominals = np.linspace(0.1, 0.9, n_bins)
    for alpha in nominals:
        z = norm.ppf(0.5 + alpha/2)
        lower = predictions - z * uncertainties
        upper = predictions + z * uncertainties
        actual_coverage = ((actuals >= lower) & (actuals <= upper)).mean()
        coverages.append(actual_coverage)

    plt.plot(nominals, coverages, 'o-', label='Model')
    plt.plot([0, 1], [0, 1], '--', label='Perfect calibration')
    plt.xlabel('Nominal coverage')
    plt.ylabel('Actual coverage')
```

"MC Dropout typically produces under-confident intervals (too wide — actual coverage exceeds nominal). Deep ensembles are usually better calibrated but slightly under-confident. Neither is perfect. The reliability diagram tells you exactly how to adjust."

**"So what?":** "Uncalibrated uncertainty is worse than no uncertainty — it gives you false confidence in your confidence. Always check calibration before using uncertainty for position sizing or filtering. If 90% intervals actually cover 75% of the time, your Kelly positions are too large."

### Section 7: What Doesn't Work — Honest Assessment

**Narrative arc:** We close with an honest discussion of what Bayesian deep learning can't do, and where the practical limits are.

**Key concepts:** Full Bayesian neural networks, Bayes by Backprop, variational inference, HMC, computational limits.

**The hook:** "I should be honest about what we're NOT doing. Full Bayesian inference over neural network weights — placing a prior on every weight and computing the exact posterior — is computationally intractable for any network larger than a few hundred parameters. Bayes by Backprop (Blundell et al., 2015) doubles the parameter count and is notoriously unstable. Hamiltonian Monte Carlo gives excellent posteriors but takes hours for a single layer. The 'Bayesian deep learning' we're doing is an approximation — MC Dropout approximates the posterior with a Bernoulli distribution, and deep ensembles don't explicitly compute a posterior at all. These approximations are good enough for practical applications. But they're not Bayesian inference in the way a statistician would use the term."

| Method | Accuracy | Computational Cost | Practical for Finance? |
|--------|----------|-------------------|----------------------|
| MC Dropout | Good | ~Free (just keep dropout on) | Yes - default |
| Deep Ensembles | Better | M× training | Yes - standard |
| Bayes by Backprop | Moderate | 2× parameters, unstable | Rarely |
| HMC / NUTS | Excellent | Hours for small nets | Research only |
| Laplace Approximation | Good | Post-hoc, cheap | Emerging |

"The practical sweet spot: MC Dropout for development and rapid iteration. Deep ensembles for production. Everything else is for papers, not portfolios."

**"So what?":** "Bayesian deep learning in finance is not about computing exact posteriors. It's about getting useful uncertainty estimates with minimal overhead. MC Dropout and deep ensembles are 'good enough Bayesian inference' — they won't satisfy a statistician, but they'll save your portfolio."

### Closing Bridge

"You now have models that see each stock independently (feedforward nets), process temporal sequences (LSTMs), leverage pre-trained representations (foundation models), read text (NLP embeddings), and quantify their own uncertainty (this week). But there's one thing they can't do: see relationships between stocks. When Apple reports strong iPhone demand in China, TSMC's stock moves — because TSMC manufactures Apple's chips. When oil prices spike, every airline stock drops. When one bank fails, the market panics about all banks. These relationships are real, persistent, and profitable. Next week, we give your models the ability to see the network — Graph Neural Networks encode stock relationships explicitly, turning your universe from a bag of independent predictions into a connected graph of interdependent entities."

## Seminar Exercises

### Exercise 1: Uncertainty Decomposition — Which Stocks Are Inherently Noisy vs. Unfamiliar?
**The question we're answering:** The lecture showed the 3-line MC Dropout trick and the aleatoric/epistemic decomposition formula. Can you decompose uncertainty for your actual stock universe and discover which stocks are inherently noisy (high aleatoric) vs. unfamiliar to the model (high epistemic)?
**Setup narrative:** "The lecture gave you the formula. Now you apply it to real data and make it actionable. Some stocks are just noisy — meme stocks, biotech, crypto-correlated names. Others are calm but occasionally look alien to the model (e.g., when a blue chip enters unfamiliar territory). You need to tell the two apart because they demand different portfolio actions."
**What they build:** Modify the Week 7 model to output both $\hat{\mu}$ and $\hat{\sigma}$ (heteroscedastic output head, per Kendall & Gal 2017). Run MC Dropout (T=100). Compute aleatoric and epistemic uncertainty separately for each stock-month. Create a 2D scatter: aleatoric (x) vs. epistemic (y). Label quadrants: "calm + confident," "noisy + confident," "calm + confused," "noisy + confused." Analyze which stocks populate each quadrant.
**What they'll see:** Large-cap blue chips cluster in "calm + confident." Meme stocks (GME, AMC) in "noisy + confident" (the model knows they're noisy). Stocks undergoing regime change (e.g., banks during SVB crisis) in "calm + confused" (historically stable but now out-of-distribution). The quadrant matters for portfolio construction.
**The insight:** Aleatoric and epistemic uncertainty are different and demand different responses. High aleatoric = set wider stop-losses. High epistemic = reduce position size. The decomposition turns a single model into a nuanced risk management tool.

### Exercise 2: Calibration Across Market Regimes — When Does Your Uncertainty Lie?
**The question we're answering:** The lecture showed reliability diagrams as a global check. But financial models face radically different conditions across regimes. Are your uncertainty estimates still calibrated during crises, or do they break exactly when you need them most?
**Setup narrative:** "A globally calibrated model might be well-calibrated in calm markets and badly miscalibrated in crises. Since crises are when uncertainty matters most, you need to check calibration regime-by-regime."
**What they build:** Compute MC Dropout and deep ensemble uncertainty over the 2020-2024 test period. Split into three regimes: calm (VIX<20), elevated (VIX 20-30), crisis (VIX>30). Plot separate reliability diagrams for each regime and each method. Compare 90% coverage per regime.
**What they'll see:** Both methods are reasonably calibrated in calm markets (~88-92% actual coverage for 90% nominal). During crises, MC Dropout becomes under-covered (~75-80% for 90% nominal — intervals too narrow). Deep ensembles degrade less (~82-87%). Neither is perfectly calibrated in crises.
**The insight:** Uncertainty calibration breaks during crises — exactly when you need it most. This is because crises produce tail events that exceed the model's learned variance. Practical fix: inflate prediction intervals by a regime-dependent factor, or use a heavier-tailed distribution (Student-t instead of Gaussian) for the prediction intervals.

### Exercise 3: Bayesian Linear Regression for Factor Models
**The question we're answering:** Can we get exact uncertainty on factor model coefficients without any approximation?
**Setup narrative:** "Unlike neural networks, linear models have exact Bayesian solutions. We're going to fit a factor model with a Gaussian prior on the coefficients and get the full posterior distribution — not an approximation, but the actual answer. Then we'll show that the posterior mean equals Ridge regression. The 'mystery' of regularization dissolves: it's just a prior."
**What they build:** Implement Bayesian linear regression from scratch (5 lines of numpy). Fit a 5-factor model for 10 stocks. Plot posterior distributions for each factor loading. Show that changing prior precision changes the posterior (and equals changing Ridge alpha).
**What they'll see:** Factor loadings with tight posteriors (well-determined) vs. wide posteriors (uncertain). Market beta is always well-determined. Smaller factors (profitability, investment) often have wide posteriors — the data can't distinguish their effect from zero.
**The insight:** Bayesian linear regression tells you which factors are real and which are noise — for a specific stock, in a specific time period. Ridge regression hides this information behind a point estimate.

### Exercise 4: Dynamic Uncertainty Thresholds — Adaptive Filtering Across Regimes
**The question we're answering:** The lecture showed a static 50th-percentile filter for confident vs. uncertain stocks. But in a crisis, ALL stocks have high uncertainty. Should you trade nothing? Or should the threshold adapt to the current regime?
**Setup narrative:** "A static filter — 'trade only below-median uncertainty' — works in calm markets. But during COVID, the median uncertainty tripled. A static threshold would have pulled you out of the market entirely, missing the V-shaped recovery. You need an adaptive threshold."
**What they build:** Implement three filtering strategies: (A) static filter at the unconditional median, (B) rolling adaptive filter at the 30-day rolling median (always trades the calmer half of the CURRENT regime), (C) percentile-based filter that normalizes uncertainty by its trailing 60-day distribution (z-score filter). For each, construct long-short quintile portfolios. Compare Sharpe, max drawdown, and turnover across strategies.
**What they'll see:** Strategy A exits entirely during crises (misses V-recovery). Strategy B maintains exposure but reduces it (trades the relatively-calmer stocks within the crisis). Strategy C is the most stable — by normalizing to the trailing distribution, it always trades a consistent fraction. Strategy C typically has the best Sharpe because it avoids both over-trading in calm periods and complete exit in crises.
**The insight:** Static uncertainty thresholds are fragile. Adaptive thresholds — normalized to the current regime — give you the benefits of uncertainty filtering without the regime-dependent failure modes. This is how production systems implement the idea.

## Homework: "Uncertainty-Aware Stock Prediction"

### Mission Framing

Your Week 7 model has been making predictions for months. Some were right. Some were wrong. The problem is that you never knew which was which before the fact. You treated every prediction with equal conviction, sizing every position the same way, trading every stock regardless of how confident the model was. That changes today.

You're going to add uncertainty quantification to your existing model — with almost zero additional code — and then build three applications on top of it. First, a calibration analysis to verify that your uncertainty estimates mean what they say. Second, a regime detection system that uses universe-level uncertainty to flag stress periods. Third, an uncertainty-filtered strategy that proves, on real data, that trading only confident predictions improves risk-adjusted returns. The punchline: three lines of code (keeping dropout on at inference) improve your strategy more than any architecture change you've made all course.

### Deliverables

1. **MC Dropout implementation.** Take your Week 7 Gu-Kelly-Xiu model. Ensure it has Dropout(p=0.5) after each hidden layer (matching your Week 7 architecture). At inference, use `model.train()` to keep dropout active. Run T=100 forward passes per stock per month. Compute: mean prediction (point estimate), prediction standard deviation (epistemic uncertainty). Store both for the entire out-of-sample period.

2. **Deep ensemble comparison.** Train M=5 versions of the model with different random seeds. For each stock-month, compute ensemble mean and ensemble std. Compare ensemble uncertainty to MC Dropout uncertainty: are they correlated? Which is larger?

3. **Calibration analysis.** Construct 50%, 80%, and 90% prediction intervals using the MC Dropout uncertainty. Compute actual coverage: what fraction of realized returns fall within each interval? Plot the reliability diagram. Is the model over-confident, under-confident, or well-calibrated? Repeat for the deep ensemble.

4. **Uncertainty over time.** Compute the average epistemic uncertainty across all stocks at each month in the out-of-sample period. Plot this time series. Annotate known stress events: COVID (March 2020), 2022 rate hike cycle (March-September), SVB collapse (March 2023), Middle East tensions (October 2023). Does uncertainty spike before or during these events?

5. **Uncertainty-filtered strategy.** Construct three portfolios using your model's predictions:
   - **Strategy A:** Long top quintile, short bottom quintile, ALL stocks, equal-weight.
   - **Strategy B:** Same, but only trade stocks in the BOTTOM 50% of epistemic uncertainty (confident predictions).
   - **Strategy C:** Same, but only trade stocks in the TOP 50% of epistemic uncertainty (uncertain predictions).
   Compare: Sharpe ratio, maximum drawdown, IC, hit rate. Report in a table.

6. **Kelly-criterion position sizing.** Replace equal-weight positions in Strategy A with Kelly-weighted positions: $w_i = \hat{\mu}_i / (\hat{\sigma}^2_{\text{aleatoric}} + \hat{\sigma}^2_{\text{epistemic}})$, normalized to sum to 1 per side. Use half-Kelly (multiply weights by 0.5) for conservatism. Compare Kelly-weighted vs. equal-weight Sharpe.

7. **Deliverable:** Notebook + reliability diagram + uncertainty time series plot + strategy comparison table + Kelly analysis.

### What They'll Discover

- MC Dropout adds literally 3 lines of code to an existing model and produces informative uncertainty estimates. The correlation between uncertainty and absolute error is typically 0.20-0.35.
- Deep ensembles produce better-calibrated uncertainty (closer to the diagonal on the reliability diagram) but are 5x more expensive to train. For their small model, this means 15 minutes instead of 3 — not a real constraint.
- The uncertainty time series spikes during known stress periods. The COVID spike begins in late February 2020 — about 2 weeks before the worst of the crash (March 16-23). This isn't a prediction of the crash; it's the model saying "my inputs look nothing like training data."
- Strategy B (confident-only) has 0.1-0.3 higher Sharpe than Strategy A (all stocks) and significantly smaller max drawdown. Strategy C (uncertain-only) has the worst performance — confirming that uncertainty is anti-correlated with prediction quality.
- Kelly-criterion sizing with uncertainty typically improves Sharpe by 0.05-0.15 over equal-weight. The improvement comes from reducing position sizes during uncertain periods and concentrating capital during confident periods.

### Deliverable

Final notebook: `hw11_uncertainty_aware.ipynb` containing MC Dropout implementation, calibration analysis, uncertainty time series, strategy comparison, and Kelly analysis.

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Aleatoric vs. epistemic uncertainty | Demo: explain with coin-flip analogy, show decomposition formula | Exercise 1: decompose for real stocks, build 2D scatter with labeled quadrants | Integrate: track both types separately over the OOS period |
| MC Dropout | Demo: 3-line implementation, explain Gal & Ghahramani equivalence to variational inference | Not re-implemented (done in lecture) | At scale: run T=100 passes per stock-month over entire OOS period |
| Deep ensembles | Demo: train 5 seeds, compare to MC Dropout in table | Exercise 2: regime-conditional calibration — test whether ensembles stay calibrated during crises | At scale: train M=5 models, compare uncertainty to MC Dropout |
| Bayesian linear regression (Ridge connection) | Demo: derive posterior, show Ridge = MAP, implement in 5 lines of numpy | Exercise 3: fit 5-factor model for 10 stocks, plot posterior distributions per factor loading | Not covered in HW (neural net focus) |
| Calibration and reliability diagrams | Demo: explain coverage, show reliability diagram code | Exercise 2: plot regime-conditional reliability diagrams (calm/elevated/crisis) | At scale: construct 50%/80%/90% prediction intervals, compute actual coverage |
| Kelly criterion with uncertainty | Demo: derive $f^* = \mu / \sigma^2$, show code for half-Kelly | Not covered (done in lecture) | At scale: Kelly-weighted vs. equal-weight portfolio comparison |
| Regime detection via universe-level uncertainty | Demo: explain concept, show plotting code | Not covered (foreshadowed for homework) | Build: plot average epistemic uncertainty over time, annotate crisis events |
| Uncertainty-filtered trading | Demo: show static 50th-percentile split with code | Exercise 4: implement adaptive threshold (rolling median, z-score), compare three filter strategies | At scale: construct confident/uncertain/all portfolios, report Sharpe and drawdown |

## Key Stories & Facts to Weave In

1. **Gal & Ghahramani (2016) — the dropout-as-Bayesian-inference paper.** Yarin Gal was a PhD student at Cambridge. His supervisor, Zoubin Ghahramani, is one of the world's leading Bayesian ML researchers. Their 2016 ICML paper showed that training with dropout is mathematically equivalent to approximate variational inference. The result surprised the deep learning community — dropout had been seen as a regularization hack, not a principled Bayesian method. The paper has been cited over 7,000 times and spawned an entire subfield of "efficient Bayesian deep learning."

2. **The COVID uncertainty spike (February-March 2020).** MC Dropout uncertainty on US equities began rising in late February 2020, about 2 weeks before the S&P 500's bottom on March 23. The signal was not "a crash is coming" — it was "my inputs look nothing like anything I've seen before." VIX was at levels not seen since 2008. Daily moves of 5-10% were occurring multiple times per week. Cross-stock correlations spiked to nearly 1. All of these features were outside the training distribution, and the model's uncertainty reflected this honestly.

3. **Lakshminarayanan et al. (2017) — deep ensembles.** Published at NeurIPS by a team at DeepMind. The paper's central claim was deliberately provocative: ensembles of neural networks, trained with different random seeds, provide better uncertainty estimates than any sophisticated Bayesian method tested. No variational inference. No MCMC. Just train more models. The result has held up remarkably well — as of 2025, deep ensembles remain the standard for uncertainty quantification in production systems.

4. **Kelly criterion at Renaissance Technologies.** The Kelly criterion was developed by John Kelly at Bell Labs in 1956 for optimal gambling strategies. It tells you the optimal fraction of your bankroll to bet: $f^* = \mu / \sigma^2$. Renaissance Technologies — the most successful quant fund in history, with 66% annual returns over 30 years — reportedly uses a variant of Kelly criterion for position sizing. The critical input is the denominator: $\sigma^2$. If you underestimate variance (no epistemic uncertainty), you bet too much. Kelly with calibrated uncertainty is Kelly with the right denominator.

5. **The connection between Ridge and Bayesian inference.** When Hoerl and Kennard introduced Ridge regression in 1970, they motivated it as a fix for multicollinearity — adding $\lambda I$ to $X^TX$ makes it invertible. The Bayesian interpretation (Ridge = MAP with Gaussian prior) was recognized later. The regularization parameter $\lambda$ is the ratio of noise variance to prior variance: $\lambda = \sigma^2 / \tau^2$. Cross-validation for $\lambda$ is implicitly choosing how strong your prior is. Understanding this connection makes regularization less mystical and more principled.

6. **Uncertainty at quant funds — the practitioner's view.** At Two Sigma and Citadel, model uncertainty is used for three purposes: (1) position sizing — scale down positions when the model is uncertain; (2) risk monitoring — track average model uncertainty as a regime indicator; (3) model selection — prefer models with better-calibrated uncertainty, even if point prediction accuracy is similar. MC Dropout is the most common implementation because it requires zero additional training infrastructure.

7. **Kendall & Gal (2017) — "What Uncertainties Do We Need?"** This NeurIPS paper formalized the decomposition into aleatoric and epistemic uncertainty for deep learning. The key contribution: a neural network can output both a point prediction AND an aleatoric uncertainty estimate by predicting both $\mu$ and $\sigma$ (heteroscedastic output). Combined with MC Dropout for epistemic uncertainty, you get a full decomposition. For finance, this means the model can say: "AAPL is inherently noisy (high aleatoric) but I'm confident in my prediction (low epistemic)" vs. "AAPL is calm (low aleatoric) but I've never seen inputs like these (high epistemic)."

## Cross-References
- **Builds on:** Week 7's neural net with dropout (the exact model you add MC Dropout to) and its seed-averaged ensemble (deep ensembles formalize this). Week 5's XGBoost baseline (performance benchmark for uncertainty-aware models). Week 4's Ridge regression (Bayesian linear regression reveals its hidden Bayesian nature). Week 3's position sizing and risk metrics (uncertainty enables proper position sizing via the Kelly criterion, introduced this week).
- **Sets up:** Week 12 (GNN predictions can be uncertainty-quantified with the same MC Dropout approach — dropout in the graph attention layers). Week 13 (RL agents can use model uncertainty to adjust exploration vs. exploitation). Week 18 (capstone — uncertainty-filtered strategies are a legitimate component of a full trading system).
- **Recurring thread:** The "honest assessment" theme. MC Dropout and deep ensembles are approximations. They're not full Bayesian inference. We're honest about what they can and can't do. This mirrors the course-wide insistence on honest benchmarking — claim only what you can prove, and report limitations alongside results.

## Suggested Reading
- **Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"** — the paper that started it all. Read the first 5 pages for the main result. The proof that dropout equals variational inference is in Section 3.
- **Lakshminarayanan et al. (2017), "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"** — the paper that made ensembles the practical standard. Read Sections 1-3. The comparison to MC Dropout and Bayes by Backprop in Section 4 is the most important empirical result.
- **Kendall & Gal (2017), "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"** — the paper that introduced the aleatoric/epistemic decomposition for deep learning. Written for vision, but the framework applies directly to finance. Read Sections 1-3.
