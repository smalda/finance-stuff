# Week 7 — Feedforward Neural Networks for Asset Pricing

> **A three-layer net with 56 neurons beat every linear model ever published on 30,000 stocks over 60 years. This week, you build it.**

## Prerequisites
- **Week 4 (Linear Models):** The cross-sectional prediction framework — features as firm characteristics, expanding-window CV, out-of-sample IC as the metric. You need this because the neural net is solving the exact same problem, just with a richer function class.
- **Week 5 (Trees/XGBoost):** Your XGBoost pipeline is the benchmark the neural net must beat. You'll need the feature matrix, the evaluation framework, and the SHAP-based understanding of which features matter.
- **Week 3 (Portfolio/Risk):** Sharpe ratio, long-short portfolio construction, transaction costs. We'll use these to evaluate whether the neural net's statistical improvement translates to economic improvement.
- **Basic PyTorch familiarity:** Tensors, autograd, `nn.Module`, `nn.Linear`, `nn.ReLU`. If you've trained an image classifier, you have everything you need. If not, spend 30 minutes on the PyTorch "60 Minute Blitz" tutorial before class.

## The Big Idea

Every model you've built so far has a ceiling. Ridge regression can only learn linear relationships between features and returns — if momentum matters more when volatility is high, Ridge can't capture that unless you hand-engineer the interaction term. XGBoost captures nonlinear interactions beautifully, but it does so through axis-aligned splits — it partitions feature space into rectangles. Neural networks draw arbitrary curves.

In 2020, Shihao Gu, Bryan Kelly, and Dacheng Xiu published what became the single most cited paper in empirical asset pricing of the decade. They took 30,000 US stocks, 60 years of monthly data, 94 firm characteristics, and ran a horse race between 9 model classes: OLS, Ridge, Lasso, Elastic Net, PCR, PLS, Random Forest, Gradient Boosted Trees, and neural networks of increasing depth. The neural net won. Not by a landslide — the monthly out-of-sample R² went from 0.40% (Ridge) to 0.70% (3-layer net). That sounds tiny until you realize that in finance, doubling the R² can double the Sharpe ratio of a long-short portfolio. We're talking about the difference between a mediocre fund and a very good one.

This week is about two transitions. First, you move from scikit-learn to PyTorch — a shift that changes not just the library but the mindset. In sklearn, you call `.fit()` and get a model. In PyTorch, you build the model brick by brick, define your own training loop, control every gradient update. It's more work, but it gives you something sklearn never will: the ability to define custom loss functions, architectures, and training procedures tuned specifically for financial data. Second, you move from "ML applied to finance" to "deep learning for asset pricing" — a field with its own conventions, pitfalls, and folklore.

The punch line you'll prove by the end of this week: a 3-layer feedforward net with batch normalization and dropout, trained with proper temporal cross-validation, captures nonlinear factor interactions that both linear models and trees partially miss. It's not magic — it's function approximation with a better inductive bias for smooth, high-dimensional interactions. But "not magic" is still worth billions of dollars a year to the funds that deploy it.

## Lecture Arc

### Opening Hook

"In 2020, Gu, Kelly, and Xiu published a paper with a deliberately understated title: 'Empirical Asset Pricing via Machine Learning.' The results were anything but understated. They tested every major ML model class on 30,000 US stocks over 60 years — the largest horse race in the history of empirical finance. The winner was a neural network with three hidden layers and a grand total of 56 neurons. Not a transformer. Not a 100-billion-parameter behemoth. Fifty-six neurons. It earned a monthly out-of-sample R² of 0.70%, which doesn't sound like much until you do the portfolio math: the long-short decile spread was 2.1% per month — 25.2% per year before costs. Today, we replicate that architecture, train it on your data, and find out whether the result holds up when you're the one writing the code."

### Section 1: From sklearn to PyTorch — Why Bother?

**Narrative arc:** We start with the comfort of sklearn, show where it breaks, and make the case that PyTorch's extra complexity buys you things you genuinely need for financial ML.

**Key concepts:** Tensors, autograd, `nn.Module`, custom `Dataset`/`DataLoader`, MPS acceleration on Apple Silicon.

**The hook:** "Your sklearn pipeline from Week 5 works. XGBoost is fast, handles missing values, gives you feature importance for free. So why would you voluntarily move to a framework where you have to write your own training loop, manage your own batching, and debug gradient explosions at 2 AM? Three reasons. First: custom loss functions. In sklearn, you minimize MSE or log-loss. In PyTorch, you can minimize the negative Sharpe ratio of the portfolio implied by your predictions — the objective function IS the trading objective. Second: architecture flexibility. You can build networks that share layers across time periods, that attend to specific stocks, that output both a prediction and an uncertainty estimate. Third: the entire frontier of financial ML — foundation models, attention mechanisms, graph neural networks — lives in PyTorch. If you don't speak PyTorch, you can't read the menus at the restaurants where the interesting food is being served."

**Key formulas:**

The feedforward network computes a nested composition of affine transformations and nonlinearities. Start simple — a single hidden layer:

$$\hat{r}_{i,t+1} = \mathbf{w}_2^T \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{x}_{i,t} + \mathbf{b}_1) + b_2$$

where $\mathbf{x}_{i,t}$ is the feature vector for stock $i$ at time $t$ (your 20+ characteristics from Week 4), and $\hat{r}_{i,t+1}$ is the predicted next-period return. The ReLU introduces nonlinearity — without it, this is just fancy linear regression. Stack three layers and you get the Gu-Kelly-Xiu architecture:

$$\hat{r}_{i,t+1} = f_3(f_2(f_1(\mathbf{x}_{i,t})))$$

where each $f_k(\mathbf{z}) = \text{ReLU}(\mathbf{W}_k \mathbf{z} + \mathbf{b}_k)$ except the final layer, which is linear (no activation — we want unbounded predictions).

**Code moment:** Build a minimal PyTorch model and compare to sklearn Ridge:

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
```

Show that this is 4 lines of actual architecture. The ceremony is in the training loop, not the model.

**"So what?":** "If this looks like a standard MLP from any deep learning course — it is. The finance-specific parts aren't in the architecture. They're in the training procedure: how you split data, what you optimize, and how you evaluate. That's where we spend the rest of the lecture."

### Section 2: The Gu-Kelly-Xiu Architecture — Anatomy of a Winner

**Narrative arc:** We dissect the specific architecture that won the horse race — not because it's the only option, but because it's the benchmark everyone cites and because its design choices encode deep lessons about financial data.

**Key concepts:** Layer widths (32-16-8 bottleneck), batch normalization, dropout, Huber loss, ensemble over seeds.

**The hook:** "The Gu-Kelly-Xiu paper tested 1- through 5-layer networks, with hidden sizes ranging from 32 to 256. The winner? Three layers, 32-16-8. A funnel. The width shrinks as you go deeper. This is not accidental — it forces the network to compress information at each stage, discarding noise and keeping signal. Financial data is roughly 99% noise and 1% signal. A wide network memorizes the noise. A narrow funnel is forced to find the signal."

**Key formulas:**

Batch normalization normalizes each mini-batch's activations before the nonlinearity:

$$\hat{z}_j = \frac{z_j - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$\tilde{z}_j = \gamma \hat{z}_j + \beta$$

Intuition: financial features have wildly different scales. Market cap might be $10^{11}$, momentum might be 0.03. Without batch norm, the network's gradients are dominated by whichever feature has the largest magnitude. Batch norm makes the network scale-invariant — it doesn't matter whether you feed in market cap in dollars or billions.

Dropout randomly zeros out neurons during training with probability $p$:

$$\tilde{h}_j = \begin{cases} 0 & \text{with probability } p \\ h_j / (1-p) & \text{with probability } 1-p \end{cases}$$

The $(1-p)$ scaling ensures the expected activation magnitude stays the same. Why does this matter for finance? Because financial datasets have a terrible signal-to-noise ratio. Without dropout, a neural net will memorize the noise in training data — it'll learn that "when this specific combination of 20 features takes these specific values, the return is +3%" even though that combination was pure coincidence. Dropout forces the network to be robust: no single neuron can carry the whole prediction.

Gu-Kelly-Xiu use Huber loss instead of MSE:

$$L_{\text{Huber}}(\hat{r}, r) = \begin{cases} \frac{1}{2}(\hat{r} - r)^2 & \text{if } |\hat{r} - r| \leq \delta \\ \delta |\hat{r} - r| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

"Why Huber? Because financial returns have fat tails. MSE squares the error, so a single outlier return — a stock that jumps 50% on an earnings surprise — can dominate the entire gradient update. Huber loss acts like MSE for small errors and MAE for large errors. It says: 'learn from the normal returns, don't let the outliers hijack your gradients.'"

**Code moment:** The full Gu-Kelly-Xiu architecture in PyTorch:

```python
class GKXNet(nn.Module):
    def __init__(self, n_features, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1)
        )
```

"Count the parameters. With 20 input features: 20×32 + 32 + 32×16 + 16 + 16×8 + 8 + 8×1 + 1 = 1,337 parameters. That's it. This model that beat every linear model in the history of asset pricing has fewer parameters than most image classifiers have in a single layer."

**"So what?":** "The architecture is boring on purpose. The lesson from Gu-Kelly-Xiu is not 'use a clever architecture.' It's 'use a standard architecture with proper regularization and proper evaluation.' The magic is in the methodology, not the model."

### Section 3: Training Pitfalls — Everything That Can Go Wrong (and Will)

**Narrative arc:** We enumerate the ways financial training differs from standard ML training, then show the correct approach for each.

**Key concepts:** Temporal train/val/test split, `SequentialSampler`, early stopping on temporal validation, learning rate scheduling.

**The hook:** "Here's a reliable way to fool yourself: shuffle your financial data before splitting into train/test. Your model will see January 2020 returns in training and predict February 2020 returns in testing. Sounds fine? It's not. The model learns that 'when the VIX is at 82 and everything is crashing, returns are negative' from the March 2020 training data, then 'predicts' the crash in March 2020 test data. You get an out-of-sample R² of 5% and think you're a genius. In reality, your model has seen the future. Every quant who's been embarrassed by a live deployment failure has this story."

**Key formulas:**

The expanding-window protocol:

$$\text{Train}_t = \{(x_{i,s}, r_{i,s+1}) : s \leq t - \text{embargo}\}$$
$$\text{Val}_t = \{(x_{i,s}, r_{i,s+1}) : t - \text{embargo} < s \leq t\}$$
$$\text{Test}_t = \{(x_{i,t}, r_{i,t+1})\}$$

The embargo gap between train and validation prevents label leakage when labels overlap (e.g., if you're predicting 5-day returns, you need at least a 5-day embargo).

**Code moment:** Show the WRONG way (shuffled `DataLoader`) and the RIGHT way (sequential, temporally ordered):

```python
# WRONG — this is how you fool yourself
loader = DataLoader(dataset, batch_size=256, shuffle=True)  # NEVER do this

# RIGHT — respect temporal ordering
from torch.utils.data import SequentialSampler
loader = DataLoader(dataset, batch_size=256, sampler=SequentialSampler(dataset))
```

Then show the expanding-window training loop skeleton. Early stopping monitors validation IC (not loss!) on the most recent temporal holdout.

**"So what?":** "The training loop is where most financial ML projects quietly fail. Not because the code crashes — it doesn't. Because the evaluation lies to you. Every line in this training loop exists to prevent a specific type of self-deception."

### Section 4: Financial Loss Functions — Beyond MSE

**Narrative arc:** MSE is fine as a starting point, but we can do better by aligning the loss function with what we actually care about: ranking stocks correctly, not predicting returns accurately.

**Key concepts:** MSE, Huber loss, weighted MSE (market-cap weighting), IC-based loss, Sharpe-ratio loss.

**The hook:** "Your model predicts that Stock A will return +1.2% and Stock B will return +0.8%. The actual returns are +5.0% and +0.3%. Under MSE, your model gets brutally punished for being wrong about Stock A's magnitude (3.8% error). But as a portfolio manager, you don't care about the magnitude — you care that the model correctly ranked A above B. You'd buy A and sell B, and you'd make money. MSE doesn't know that. IC-based losses do."

**Key formulas:**

The information coefficient (Spearman rank correlation between predicted and actual returns):

$$\text{IC}_t = \text{corr}_{\text{rank}}(\hat{r}_{1:N,t}, r_{1:N,t})$$

A differentiable approximation for use as a loss function uses soft sorting:

$$\mathcal{L}_{\text{IC}} = -\text{corr}(\hat{r}, r) \approx -\frac{\sum_i (\hat{r}_i - \bar{\hat{r}})(r_i - \bar{r})}{\sqrt{\sum_i (\hat{r}_i - \bar{\hat{r}})^2 \cdot \sum_i (r_i - \bar{r})^2}}$$

We negate it because we want to maximize correlation. This is just Pearson correlation as a loss — not exactly Spearman (which requires ranking, which isn't differentiable), but a good approximation for training.

**Code moment:** Implement IC loss:

```python
def ic_loss(predictions, targets):
    """Negative Pearson correlation as a loss function."""
    p = predictions - predictions.mean()
    t = targets - targets.mean()
    return -(p * t).sum() / (p.norm() * t.norm() + 1e-8)
```

Compare training with MSE vs. IC loss. Show that IC loss produces slightly lower R² but higher IC — it trades prediction accuracy for ranking accuracy, which is what the portfolio cares about.

**"So what?":** "The loss function is the contract between you and the optimizer. MSE says 'predict the number.' IC loss says 'rank the stocks.' Sharpe loss says 'make money.' In finance, you should optimize what you actually want. Most people optimize MSE because it's the default. Defaults are how you lose money quietly."

### Section 5: Ensembles — The Free Lunch

**Narrative arc:** A single neural net is sensitive to its random seed. An average of 5 nets with different seeds is more stable, more accurate, and (as we'll see in Week 11) gives you uncertainty estimates for free.

**Key concepts:** Seed ensembles, hyperparameter ensembles, temporal ensembles, prediction averaging.

**The hook:** "Train the same architecture 5 times with different random seeds. You'll get 5 different sets of predictions with ICs ranging from, say, 0.030 to 0.045. That's a 50% spread from the same model on the same data. The only thing that changed was the random initialization. Now average the 5 predictions. The ensemble IC is typically 0.042 — near the top of the range, and more stable month-to-month. In the 2024 follow-up paper 'Large (and Deep) Factor Models,' Kelly and co-authors show that ensembling over 10 random seeds was worth 0.5-1.0% annual Sharpe improvement. Literally free money."

**Key formulas:**

Ensemble prediction is just the mean:

$$\hat{r}_i^{\text{ensemble}} = \frac{1}{M} \sum_{m=1}^{M} \hat{r}_i^{(m)}$$

Why does averaging help? The bias-variance decomposition:

$$\text{MSE}(\hat{r}^{\text{ensemble}}) = \text{Bias}^2 + \frac{1}{M}\bar{\sigma}^2 + \left(1 - \frac{1}{M}\right)\bar{\rho}\bar{\sigma}^2$$

The variance term shrinks as $1/M$ — but only the uncorrelated part. If all models make the same errors ($\bar{\rho} = 1$), ensembling doesn't help. In practice, financial models trained with different seeds have $\bar{\rho} \approx 0.6$-$0.8$, so ensembling reduces variance substantially.

**Code moment:** Train 5 models, collect predictions, average:

```python
models = []
for seed in range(5):
    torch.manual_seed(seed)
    model = GKXNet(n_features).to(device)
    train_model(model, train_loader, val_loader)
    models.append(model)

# Ensemble prediction
with torch.no_grad():
    preds = torch.stack([m(X_test) for m in models])
    ensemble_pred = preds.mean(dim=0)
    ensemble_std = preds.std(dim=0)  # <-- free uncertainty estimate!
```

"That `ensemble_std` is a preview of Week 11. When 5 models disagree about a stock, the standard deviation is high — the prediction is uncertain. When they agree, it's low. We'll exploit this systematically in Week 11."

**"So what?":** "Ensembling is the closest thing to a free lunch in ML. It costs M× training time, but financial models are small — 5 training runs on a 3-layer net take minutes, not hours. The payoff is more stable predictions, higher IC, and a free uncertainty signal. There's no reason not to do it."

### Section 6: Neural Net vs. XGBoost — The Honest Comparison

**Narrative arc:** We compare the neural net head-to-head against XGBoost from Week 5. The result is nuanced: the neural net wins on some metrics, ties on others, and loses on training convenience. The real winner is the ensemble of both.

**Key concepts:** Model comparison methodology, performance across regimes, complementarity of model classes.

**The hook:** "Here's the dirty secret of the Gu-Kelly-Xiu paper: Figure 4 shows that gradient boosted trees (GBT) achieved an out-of-sample R² of 0.54% versus the neural net's 0.70%. Both crushed linear models (0.40%). But the gap between tree and net is much smaller than the gap between linear and tree. In the 2024 follow-up, the authors showed that the gap narrows further with more features and that ensembling a tree model with a neural net outperforms either alone. The models make different mistakes — trees overfit to local interactions, nets overfit to smooth trends — and their errors are partially uncorrelated."

**Code moment:** Side-by-side comparison table from the actual experiment:

```python
results = pd.DataFrame({
    'Model': ['Ridge', 'XGBoost', 'Neural Net (1 seed)', 'NN Ensemble (5 seeds)', 'XGBoost + NN Ensemble'],
    'OOS IC': [0.028, 0.038, 0.035, 0.041, 0.044],
    'OOS R²': ['0.35%', '0.52%', '0.48%', '0.62%', '0.68%'],
    'Long-Short Sharpe': [0.8, 1.2, 1.1, 1.4, 1.5],
})
```

"These numbers are illustrative — your exact results depend on the universe, features, and time period. But the ranking is remarkably consistent across studies."

**"So what?":** "Don't throw away your XGBoost pipeline. Extend it. The best production systems at quant funds don't pick one model — they blend predictions from multiple model classes. Your Week 5 XGBoost and your Week 7 neural net are not competitors. They're teammates."

### Closing Bridge

"You've now built a feedforward neural network that sees each stock as an independent observation — a bag of 20 features with no memory, no sequence, no relationships. But stocks have history. The return at time $t$ depends on what happened at $t-1$, $t-2$, and $t-20$. Volatility clusters. Momentum persists. These are sequential patterns, and feedforward nets are blind to them. Next week, we give the network a memory. LSTMs and GRUs process sequences of observations, and we'll aim them at one of the most practically important targets in all of finance: volatility forecasting. The GARCH model you built in Week 2 is about to meet its first real competitor."

## Seminar Exercises

### Exercise 1: Stress-Test the GKX Architecture Across Universes
**The question we're answering:** The lecture demonstrated the GKX 32-16-8 architecture on our standard universe. Does the architecture's advantage over Ridge survive when you change the stock universe, or is it specific to the data we happened to show?
**Setup narrative:** "You saw the GKX net beat Ridge in the lecture. But was that a lucky draw of stocks? Science demands replication. You're going to train the same architecture on THREE different sub-universes — large-cap tech, small-cap value, and a random sample — and see whether the neural net's edge is universal or context-dependent."
**What they build:** Using the lecture's `GKXNet` class (already built), train it on three sub-universes of 50 stocks each: (a) mega-cap tech, (b) small-cap value, (c) random sample from S&P 500. Compare IC vs. Ridge for each.
**What they'll see:** The neural net beats Ridge in all three, but the margin varies: largest for the random sample (most heterogeneous features), smallest for small-cap value (fewer nonlinear interactions among value stocks). IC range: 0.028-0.045 (NN) vs. 0.022-0.030 (Ridge).
**The insight:** The GKX architecture's advantage is real but not uniform. It helps most when the cross-section is heterogeneous — many different types of stocks with different factor exposures — because that's where nonlinear interactions are richest.

### Exercise 2: Temporal Train/Val/Test Split in PyTorch
**The question we're answering:** How do you implement expanding-window cross-validation in PyTorch without accidentally leaking future information?
**Setup narrative:** "This is the exercise where you learn that 80% of financial ML papers have subtly wrong evaluation. The difference between 'shuffle then split' and 'split then never shuffle across time' is the difference between a publication and a retraction."
**What they build:** A custom `TemporalSplitter` that yields train/val/test indices respecting temporal ordering, plus an embargo gap.
**What they'll see:** Compare IC with shuffled splits vs. temporal splits. Shuffled: IC ~ 0.06 (too good to be true). Temporal: IC ~ 0.035 (honest).
**The insight:** If your IC looks surprisingly high, your evaluation is probably wrong.

### Exercise 3: Loss Function Comparison
**The question we're answering:** Does optimizing for ranking (IC) instead of prediction accuracy (MSE) improve portfolio performance?
**Setup narrative:** "MSE trains the model to be a good forecaster. IC loss trains it to be a good ranker. You're about to find out which skill makes more money."
**What they build:** Train the same architecture with MSE, Huber, and IC loss. Evaluate each on IC, R², and long-short portfolio Sharpe.
**What they'll see:** IC loss achieves the highest IC (0.04+) but the lowest R². MSE achieves the highest R² but lower IC. Huber is in between.
**The insight:** R² and IC measure different things. R² cares about levels; IC cares about ranks. For portfolio construction (where you're long the top and short the bottom), ranks are what matter. Optimize what you trade on.

### Exercise 4: Seed Ensemble and Its Surprising Stability
**The question we're answering:** How much does the random seed affect your results, and does averaging fix it?
**Setup narrative:** "You trained a model with seed=42 and got IC=0.038. Your colleague used seed=7 and got IC=0.031. Who has the better model? Neither — they have the same model with different luck."
**What they build:** Train 5 models (seeds 0-4), record individual ICs, compute ensemble IC. Plot monthly IC over time for individual models vs. ensemble.
**What they'll see:** Individual model ICs bounce around. The ensemble IC is smoother and typically matches or exceeds the best individual model.
**The insight:** A single neural net's IC is partly skill, partly luck. The ensemble distills the skill. This is why production systems at quant funds never deploy a single model.

## Homework: "Deep Cross-Sectional Model"

### Mission Framing

You've spent the last two weeks building cross-sectional alpha models with linear models and trees. This week, you add the third pillar: neural networks. But this isn't just "try another model" — it's a genuine experiment with a genuine question: does the neural net capture something that trees and linear models miss, and if so, what?

The Gu-Kelly-Xiu paper says yes — the neural net wins. But their experiment used 94 features on 30,000 stocks over 60 years with institutional-grade CRSP/Compustat data. You're working with 20 features on 200 stocks from yfinance. The question isn't whether you can replicate their exact numbers — you can't, and that's fine. The question is whether the fundamental insight holds: does a simple feedforward net extract signal from nonlinear feature interactions that simpler models miss?

Your mission is to build the net, race it against your XGBoost from Week 5, and then do something neither model can do alone: ensemble them. If the ensemble beats both individuals, you've just demonstrated that the models are capturing complementary patterns. That's the real result.

### Deliverables

1. **Implement the Gu-Kelly-Xiu architecture in PyTorch.** Three hidden layers (32-16-8), ReLU activations, batch normalization after each layer, dropout(0.5) after each activation. Output layer is a single linear neuron (no activation — unbounded predictions). Use Huber loss. Training should use Adam with learning rate 1e-3, decayed by 0.5 every 10 epochs. Early stopping on validation IC with patience of 10 epochs. Training should run on MPS if available, CPU otherwise. (2-3 minutes per training run on M4.)

2. **Train with expanding-window cross-validation.** Use the same temporal splits from Week 4-5. Train on all data up to month $t$, validate on the most recent 12 months before $t$, predict month $t+1$. Retrain every 12 months (not every month — that's too expensive for a neural net). The embargo gap should be at least 1 month.

3. **Compare against your Week 5 best model (XGBoost).** Report out-of-sample IC, rank IC, R², and long-short decile portfolio Sharpe. Include 10 bps round-trip transaction costs. The comparison should be on the exact same test periods — no cherry-picking. Present results as a table with one row per model.

4. **Implement an ensemble of 3 models: your neural net (average of 5 seeds), XGBoost, and LightGBM.** Simple average of z-scored predictions. Report the same metrics. The key question: does the ensemble beat every individual model?

5. **Regime analysis.** Split your out-of-sample period into "high volatility" months (VIX > 20) and "low volatility" months (VIX <= 20). Report IC separately for each regime and each model. The question: does the neural net win more in high-vol or low-vol environments?

6. **Deliverable:** A Jupyter notebook containing all code, a comparison table, a monthly IC time series plot (all models + ensemble), and a one-paragraph conclusion about when and why the neural net adds value.

### What They'll Discover

- The neural net's individual-seed IC varies by 20-50% depending on the random seed. This should make you deeply suspicious of any paper that reports results from a single run.
- The ensemble of neural net + XGBoost + LightGBM typically beats every individual model by 0.005-0.010 IC — a meaningful improvement that costs nothing but training time.
- The neural net tends to outperform trees most during regime transitions (e.g., the onset of COVID, the 2022 rate hike cycle). It captures smooth nonlinear interactions that trees, with their axis-aligned splits, approximate crudely.
- The neural net is worse than XGBoost at handling missing features and feature noise. If you have noisy data, trees are more robust.

### Deliverable

Final notebook: `hw07_deep_cross_sectional.ipynb` containing the full pipeline, all model code, the comparison table, monthly IC plot, regime analysis, and written conclusions.

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| PyTorch basics (Tensors, `nn.Module`, autograd) | Demo: build `SimpleNet`, compare to sklearn Ridge | Not covered (done in lecture) | Prerequisite: used throughout |
| GKX 32-16-8 architecture (batch norm, dropout, Huber loss) | Demo: build full `GKXNet`, count parameters, explain design choices | Exercise 1: stress-test across 3 stock sub-universes | At scale: full implementation with expanding-window CV and early stopping |
| Temporal train/val/test splitting | Demo: show WRONG (shuffled) vs. RIGHT (sequential) split, expanding-window skeleton | Exercise 2: build `TemporalSplitter` from scratch, compare shuffled vs. temporal IC | Integrate: used for all model training and evaluation |
| Financial loss functions (MSE, Huber, IC loss) | Demo: implement IC loss, compare MSE vs. IC conceptually | Exercise 3: train same architecture with MSE, Huber, and IC loss; compare portfolio Sharpe | Integrate: use Huber loss in final model, report IC metric |
| Seed ensembles and prediction stability | Demo: train 5 seeds, show IC spread, introduce `ensemble_std` | Exercise 4: full ensemble analysis — plot monthly IC for individuals vs. ensemble | At scale: build 3-model ensemble (NN + XGBoost + LightGBM) with z-scored averaging |
| Neural net vs. XGBoost comparison | Demo: show side-by-side comparison table (illustrative numbers) | Not covered (lecture provides context) | At scale: head-to-head on same test periods with IC, R², Sharpe, and transaction costs |
| Regime analysis (high-vol vs. low-vol) | Not covered (foreshadowed in closing bridge) | Not covered | Build: split OOS into VIX regimes, report IC per regime per model |

## Key Stories & Facts to Weave In

1. **Gu-Kelly-Xiu (2020) — the paper that settled the debate.** Monthly out-of-sample R² of 0.70% for their 3-layer net vs. 0.40% for Ridge. Tested on 30,000 US stocks from 1957 to 2016. The long-short decile portfolio earned 2.1% per month. Published in the Review of Financial Studies, the most prestigious empirical finance journal. Every quant fund read it.

2. **The 2024 sequel: "Large (and Deep) Factor Models."** Kelly, Kuznetsov, Malamud, and Xu pushed the depth to 100+ layers (with residual connections) and showed that performance keeps improving with depth — a result that shocked the finance community, which had assumed that shallow nets were optimal for tabular financial data.

3. **Two Sigma's shift.** In 2024, Two Sigma — one of the world's largest quant funds ($60B+ AUM) — publicly stated they were shifting from traditional ML to foundation models for signal extraction. Their internal research teams now use neural architectures as the backbone, with tree models as "second opinions."

4. **The signal-to-noise problem in numbers.** A typical daily stock return has a standard deviation of about 2% and a predictable component of about 0.02%. That's a signal-to-noise ratio of 1:100. Monthly returns are slightly better: 5% std, 0.5% predictable — 1:10. This is why financial ML is so hard. Image classification operates at signal-to-noise ratios of 100:1 or better. You're working at the opposite extreme.

5. **Why 32-16-8 and not 256-128-64?** In the Gu-Kelly-Xiu experiments, wider networks had higher in-sample R² but lower out-of-sample R². The 32-16-8 bottleneck forces information compression at each layer — a form of regularization that works hand-in-hand with dropout and batch norm. In a regime where signal-to-noise is 1:100, capacity control is more important than capacity.

6. **The Huber loss origin story.** Peter Huber proposed it in 1964 as a "robust estimator" — a loss function that isn't dominated by outliers. In financial data, where a single stock can jump 50% on a takeover announcement, this isn't a nice-to-have. Without Huber loss, a single outlier return can hijack an entire gradient update, pushing the model toward explaining that one anomalous observation instead of learning the general pattern.

7. **Production neural nets at quant funds.** At Renaissance Technologies, neural networks are one component of a larger prediction system that includes signal processing, information theory, and statistical mechanics. At D.E. Shaw, neural nets are used for "anomaly detection" — identifying market conditions where standard models break down. Neither firm publishes papers, so we rely on patent filings and employee interviews.

## Cross-References
- **Builds on:** Week 4's cross-sectional prediction framework (features, IC, expanding-window CV). Week 5's XGBoost pipeline (the benchmark to beat). Week 3's portfolio construction methodology (Sharpe ratio, long-short portfolios).
- **Sets up:** Week 8 (LSTMs add sequence modeling to the same PyTorch framework). Week 9 (foundation models extend the neural architecture to pre-trained representations). Week 11 (the ensemble + dropout from this week become the basis for uncertainty quantification — MC Dropout is literally "dropout left on at inference," using the same Dropout layers you build today). Week 12 (GNNs replace the feedforward layers with graph-aware message passing, using the same PyTorch skills).
- **Recurring thread:** The "honest evaluation" theme from Week 6 (purged CV, proper methodology) continues here — temporal splitting, early stopping on validation IC, and the seed-sensitivity analysis all reinforce the lesson that methodology matters more than architecture.

## Suggested Reading
- **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning"** — the paper this week replicates. Read the first 15 pages for the setup and Table 6 for the punchline. Skip the appendix unless you want to see all 94 features.
- **Kelly, Kuznetsov, Malamud & Xu (2024), "Large (and Deep) Factor Models"** — the sequel that pushes depth to 100+ layers. Read Section 4 on the depth-performance curve — it overturns the conventional wisdom that shallow nets are optimal for tabular data.
- **Zhang & Zohren (2024), "Deep Learning in Quantitative Trading," Chapter 4** — the most accessible recent textbook treatment of feedforward nets for asset pricing. Better pedagogy than the original paper.
