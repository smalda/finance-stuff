# Week 14 — Derivatives Pricing with Neural Networks

> **Black-Scholes is the most successful wrong model in history. Neural networks can learn why it's wrong — and what to do about it.**

## Prerequisites
- **Week 7 (Feedforward Nets):** PyTorch fluency — building architectures, training loops, custom loss functions. This week is PyTorch-heavy. You'll implement Black-Scholes as a differentiable computation graph and then train nets to replicate and surpass it.
- **Week 2 (Time Series & Stationarity):** Volatility estimation and the concept that volatility changes over time. The entire weakness of Black-Scholes is that it assumes volatility is constant. You know it isn't — Week 2 proved that.
- **Week 8 (LSTM/GRU):** Understanding that sequential models can capture time-varying dynamics. The Heston model (stochastic volatility) is, in some sense, the "LSTM of options pricing" — it lets volatility have memory.
- **General calculus:** Partial derivatives. The Greeks are just partial derivatives of the option price with respect to different inputs. If you know what a gradient is (and you do — you've been backpropagating through neural nets all course), you know what the Greeks are.

## The Big Idea

Options are the most mathematical instruments in finance. A stock is simple: you own a piece of a company, and its value goes up or down. An option is a *contract* — the right, but not the obligation, to buy (or sell) a stock at a fixed price by a certain date. That single sentence contains enough complexity to keep an army of quantitative analysts employed, and in 1973, Fischer Black and Myron Scholes figured out how to price these contracts using a single, elegant partial differential equation. They won the Nobel Prize for it (well, Scholes did — Black died before the award).

The Black-Scholes formula is a triumph of 20th-century mathematics. It's also wrong. It assumes volatility is constant (it isn't — we proved that in Week 2). It assumes returns are Gaussian (they aren't — we proved that in Week 1). It assumes you can trade continuously and for free (you can't — we proved that in Week 15... well, we will). Despite being wrong, it's extraordinarily useful. Every options trader in the world uses it, not because they believe it's true, but because it provides a common language: the *implied volatility* surface. When a trader says "vol is 25," they mean "the Black-Scholes formula, plugged in backwards, says volatility is 25%." It's wrong, but it's wrong in a structured way that everyone understands.

Here's where neural networks enter. A trained neural net can learn the Black-Scholes pricing function from data — and then go beyond it. Feed it (stock price, strike, time to expiry, volatility, interest rate) and it outputs the option price. Train it on synthetic Black-Scholes data, and it replicates the formula almost perfectly. Train it on *real* market data, and it learns the deviations — the smile, the skew, the term structure — that Black-Scholes ignores. Better yet, `torch.autograd` gives you all the Greeks (delta, gamma, vega, theta) for free, as a byproduct of the forward pass. No finite differences, no analytical derivations — just one call to `.backward()`.

This week is one of the cleanest applications of deep learning in the entire course, because we have an analytical ground truth to compare against. When your neural net agrees with Black-Scholes, you know it's learning correctly. When it disagrees, you know it's learning something *more*. And the practical applications are real: JP Morgan, Goldman Sachs, and Citadel all use neural networks to speed up Greeks computation for complex derivatives where analytical solutions don't exist. A Monte Carlo simulation that takes 10 seconds can be replaced by a neural net forward pass that takes 10 microseconds. In a trading environment where prices change every millisecond, that's the difference between live hedging and flying blind.

## Lecture Arc

### Opening Hook

On May 6, 2010 — the Flash Crash — the Dow dropped 1,000 points in five minutes and then recovered almost completely within the next fifteen. Market makers pulled their quotes. Liquidity evaporated. And every options desk on Wall Street had the same problem: their Greeks were stale. The Black-Scholes delta they computed at 2:30 PM assumed a world that no longer existed at 2:35 PM. Recomputing Greeks for a portfolio of 100,000 options using Monte Carlo takes minutes. The desks that had invested in neural-network-based Greek approximations — fast, differentiable, computed in milliseconds — were the ones that could rehedge in real time. The rest just watched their P&L scream.

### Section 1: Options Crash Course — What You're Pricing
**Narrative arc:** Most ML engineers have never seen a derivative. We build the concept from first principles using an analogy to insurance, then show the payoff diagrams that make options pricing a well-posed mathematical problem.

**Key concepts:** Call option, put option, strike price, expiry, payoff function, moneyness, time value, intrinsic value.

**The hook:** Here's a concept that might feel alien at first: imagine you could pay $5 today for the *right* — not the obligation — to buy Apple stock at $190 anytime in the next three months, regardless of what happens to the price. If Apple goes to $220, you exercise your right, buy at $190, and pocket $30 minus the $5 you paid. If Apple drops to $160, you shrug, tear up the contract, and you've lost only the $5. That contract is a *call option*, and the $5 you paid is the *premium*. The question that launched a billion-dollar industry: what should that $5 be?

**Key formulas:**

Call option payoff at expiry:

$$\text{Payoff} = \max(S_T - K, 0)$$

where S_T is the stock price at expiry and K is the strike price. You make money when the stock ends above the strike. Otherwise, the option expires worthless.

Put option payoff:

$$\text{Payoff} = \max(K - S_T, 0)$$

The reverse — you make money when the stock drops below the strike.

The option *price* (premium) before expiry is always higher than the intrinsic value max(S-K, 0), because there's still time for the stock to move. This "time value" is what Black-Scholes quantifies.

**Code moment:** Plot the payoff diagram for a call option: x-axis is stock price at expiry, y-axis is profit/loss. The hockey-stick shape. Then overlay the Black-Scholes price curve *before* expiry — it's the smooth version of the hockey stick. The gap between the two is time value. Output: students see that option pricing is literally "predict the smooth curve that converges to the hockey stick as time passes."

**"So what?":** For an ML engineer, an option payoff is just a piecewise linear function, and option pricing is learning a smooth approximation that accounts for uncertainty. You already know how to do this. The finance just gives it a fancy name.

### Section 2: Black-Scholes — The Beautiful Wrong Answer
**Narrative arc:** We derive the Black-Scholes formula by *building intuition*, not solving PDEs. The key insight: if you can perfectly hedge an option by dynamically trading the underlying stock, the option price is determined by no-arbitrage. The tension: the formula assumes constant volatility, Gaussian returns, and continuous trading — none of which are true.

**Key concepts:** Geometric Brownian motion, risk-neutral pricing, no-arbitrage, the Black-Scholes PDE, the closed-form solution.

**The hook:** In 1973, the same year Black and Scholes published their formula, the Chicago Board Options Exchange (CBOE) opened — the first organized options exchange. Texas Instruments marketed a calculator that could compute Black-Scholes prices on the trading floor. The formula didn't just describe reality — it *created* reality. Traders used it to price options, and the market prices converged toward the formula's predictions, in a self-fulfilling prophecy that would last until the 1987 crash exposed its fatal flaw.

**Key formulas:**

The Black-Scholes formula for a European call option:

$$C(S, K, T, \sigma, r) = S \cdot N(d_1) - K e^{-rT} \cdot N(d_2)$$

where:

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

Read it piece by piece:
- $S \cdot N(d_1)$: the expected value of receiving the stock, weighted by the probability you'll exercise (in the risk-neutral world).
- $K e^{-rT} \cdot N(d_2)$: the present value of paying the strike price, weighted by the probability of exercise.
- The call price is the difference: what you expect to get minus what you expect to pay.

The inputs: S (stock price), K (strike), T (time to expiry in years), sigma (volatility), r (risk-free rate). Five numbers in, one number out. This is a *function* — and neural nets are universal function approximators.

**Code moment:** Implement Black-Scholes in PyTorch, *not* NumPy. This is the key move:

```python
import torch
from torch.distributions import Normal

def black_scholes_call(S, K, T, sigma, r):
    """Black-Scholes in PyTorch — fully differentiable."""
    d1 = (torch.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)
    N = Normal(0, 1)
    return S * N.cdf(d1) - K * torch.exp(-r * T) * N.cdf(d2)
```

By using PyTorch tensors instead of NumPy arrays, the entire computation graph is tracked. That means we can compute *any* derivative with respect to *any* input using `.backward()`. One function, all Greeks, for free.

**"So what?":** Black-Scholes in PyTorch isn't just a reimplementation — it's a paradigm shift. Instead of deriving each Greek analytically (which is tedious and error-prone for complex models), you get them all via autograd. This is the same trick that powers all of deep learning, applied to derivatives pricing. And it generalizes: any model you can express as a differentiable computation graph gives you Greeks for free.

### Section 3: The Greeks via Autograd — Your First "Aha" Moment
**Narrative arc:** We show that the Greeks — delta, gamma, theta, vega — are just partial derivatives that PyTorch already knows how to compute. Students will compute them in one line of code and compare against the analytical values. The match is exact. This is the "okay, this is actually cool" moment.

**Key concepts:** Delta (dC/dS), Gamma (d²C/dS²), Theta (dC/dT, with sign convention), Vega (dC/dsigma), Rho (dC/dr).

**The hook:** Every options desk computes Greeks hundreds of thousands of times per day. For exotic options — barrier options, Asian options, baskets — there are no closed-form Greeks. The industry standard: compute the price, bump the input by a tiny amount, recompute, take the ratio. Bump-and-recompute. For five Greeks, that's five extra pricing calls per option. For a portfolio of 50,000 options, that's 250,000 pricing calls — every time the market moves. With autograd, you get all five Greeks from a single forward pass plus one backward pass. That's not an incremental improvement. That's a 5x speedup with higher accuracy, because you're not approximating a derivative with a finite difference — you're computing the *exact* derivative.

**Key formulas:**

The analytical Greeks for Black-Scholes:

$$\Delta = \frac{\partial C}{\partial S} = N(d_1)$$

$$\Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{N'(d_1)}{S \sigma \sqrt{T}}$$

$$\Theta = \frac{\partial C}{\partial T} = -\frac{S N'(d_1) \sigma}{2\sqrt{T}} - rK e^{-rT} N(d_2)$$

$$\mathcal{V} = \frac{\partial C}{\partial \sigma} = S \sqrt{T} N'(d_1)$$

where $N'(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$ is the standard normal density.

Each Greek tells you something specific: Delta says "if the stock moves $1, how much does your option move?" Gamma says "how fast does Delta change?" Vega says "if implied volatility rises 1 percentage point, how much does your option gain or lose?" These are the sensitivities that drive hedging decisions.

**Code moment:**

```python
S = torch.tensor(100.0, requires_grad=True)
K = torch.tensor(105.0)
T = torch.tensor(0.25)
sigma = torch.tensor(0.20, requires_grad=True)
r = torch.tensor(0.05)

price = black_scholes_call(S, K, T, sigma, r)
price.backward()

print(f"Price: {price.item():.4f}")
print(f"Delta (autograd): {S.grad.item():.4f}")
print(f"Vega  (autograd): {sigma.grad.item():.4f}")
```

For Gamma (second derivative), use `torch.autograd.grad` with `create_graph=True`:

```python
delta = torch.autograd.grad(price, S, create_graph=True)[0]
gamma = torch.autograd.grad(delta, S)[0]
```

Output: students compare autograd Greeks against the analytical formulas. They match to 6+ decimal places. The visual should be a table with "Analytical" and "Autograd" columns — identical numbers. The point lands: autograd isn't an approximation. It's exact.

**"So what?":** If this seems like a parlor trick for Black-Scholes (which already has analytical Greeks), wait until Section 5. When we price options under the Heston stochastic volatility model — which has no closed-form Greeks — autograd gives us the Greeks anyway. For free. That's the real payoff.

### Section 4: Training a Neural Net to Learn Black-Scholes
**Narrative arc:** Now we do the "obvious but illuminating" thing: train a neural network to learn the Black-Scholes pricing function from data. It works almost perfectly — which is both unsurprising (neural nets are universal approximators) and useful (the trained net is 1000x faster than calling the analytical formula on complex inputs).

**Key concepts:** Function approximation, training data generation, input normalization for financial data, extrapolation behavior, architecture choices.

**The hook:** Goldman Sachs reported in 2021 that they replaced part of their Monte Carlo pricing engine with neural network surrogates. The accuracy loss: less than 0.01%. The speedup: 1,000-10,000x. For a bank that computes derivatives prices millions of times per day, that translates to infrastructure savings in the millions of dollars per year — and, more importantly, the ability to compute risk in real-time instead of waiting for overnight batch jobs.

**Key formulas:**

Training data generation — we create 500,000 random option scenarios:

$$S \sim \text{Uniform}(50, 200), \quad K \sim \text{Uniform}(50, 200)$$
$$T \sim \text{Uniform}(0.01, 2.0), \quad \sigma \sim \text{Uniform}(0.05, 0.80)$$
$$r \sim \text{Uniform}(0.00, 0.10)$$

Labels: $y = \text{BS}(S, K, T, \sigma, r)$.

Input normalization matters: we use moneyness $m = S/K$ instead of raw S and K. This is the financial ML insight — the option price depends on S and K only through their ratio (plus a small adjustment for the interest rate). Using moneyness reduces the input dimensionality from 5 to 4 effective degrees of freedom and dramatically improves generalization.

The architecture: 4 hidden layers, 128 units each, ReLU activations, trained with MSE loss. Surprisingly, this simple architecture achieves R² > 0.9999 on the test set. The Black-Scholes function is smooth and well-behaved — a neural net's dream.

**Code moment:**

```python
class OptionsPricer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),  # (moneyness, T, sigma, r)
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Softplus()  # prices are always positive
        )

    def forward(self, x):
        return self.net(x)
```

The Softplus output activation enforces non-negativity (option prices can't be negative — that would be free money). Train for 100 epochs, plot predicted vs. actual on 50K test samples. Output: a scatter plot so tight it looks like a single line at y=x. R² = 0.9999. The net has learned Black-Scholes.

But then — the extrapolation test. Show the net options with moneyness > 2.5 (far out of the money) and T < 0.01 (near expiry). The neural net starts to diverge from Black-Scholes. Not by much — but visibly. This is the "neural nets don't extrapolate" lesson from Week 7, applied to a new domain.

**"So what?":** Learning Black-Scholes is the warm-up. The neural net replicates a known function faster than the analytical formula (for batch pricing) and with free Greeks via autograd. But the real value comes when we move to models *without* analytical solutions — where the neural net isn't replicating a formula but *replacing* one that doesn't exist.

### Section 5: Beyond Black-Scholes — Neural Nets for Stochastic Volatility
**Narrative arc:** The Heston model allows volatility to be stochastic — it has its own random process. This is more realistic than Black-Scholes but much harder to price (no simple closed form, Monte Carlo is slow). A neural net trained on Heston-generated data gives us fast pricing *and* Greeks via autograd. This is where the practical payoff lands.

**Key concepts:** Stochastic volatility, the Heston model, Monte Carlo pricing, variance reduction, neural net as surrogate model.

**The hook:** After the 1987 crash — when the S&P 500 fell 22% in a single day — traders noticed something that Black-Scholes couldn't explain. Options with low strike prices (puts that protect against crashes) became permanently more expensive relative to at-the-money options. This "volatility skew" persists to this day. Black-Scholes says implied volatility should be the same at all strikes. The market says otherwise. The Heston model, published in 1993 by Steven Heston, was the first widely-adopted model that could reproduce the skew — by letting volatility itself be random.

**Key formulas:**

The Heston model:

$$dS = \mu S \, dt + \sqrt{v} \, S \, dW_1$$
$$dv = \kappa(\theta - v) \, dt + \xi \sqrt{v} \, dW_2$$
$$\text{Corr}(dW_1, dW_2) = \rho$$

In words: the stock price S follows geometric Brownian motion (like Black-Scholes), but the variance v is itself a random process. It mean-reverts to theta with speed kappa, and has its own volatility xi (the "vol of vol"). The correlation rho between stock returns and volatility changes is typically negative — stocks fall when volatility rises (the "leverage effect").

There's no simple closed-form price. You either use the characteristic function (Fourier inversion — fast but tricky) or Monte Carlo simulation (slow but straightforward):

```python
# Monte Carlo Heston pricing (simplified)
for path in range(n_paths):
    S, v = S0, v0
    for t in range(n_steps):
        dW1, dW2 = correlated_normals(rho)
        v = max(v + kappa * (theta - v) * dt + xi * sqrt(v) * sqrt(dt) * dW2, 0)
        S = S * exp((mu - v/2) * dt + sqrt(v) * sqrt(dt) * dW1)
    payoffs[path] = max(S - K, 0)
price = exp(-r * T) * payoffs.mean()
```

This takes ~10 seconds for 100K paths. A neural net forward pass takes ~10 microseconds. The speedup is 1,000,000x.

**Code moment:** Generate 200K Heston option prices via Monte Carlo (this takes a few minutes — show a progress bar). Train the same neural net architecture on Heston data. Then compute the implied volatility surface: for a grid of (moneyness, time-to-expiry) pairs, compute the Heston price via the neural net, then invert Black-Scholes to get the implied volatility. Plot the surface as a 3D mesh. Output: the volatility smile/skew appears naturally. The surface is NOT flat (which is what Black-Scholes would give). It curves up for out-of-the-money puts (the skew) and for short-dated options (the smile). This is what real options markets look like.

**"So what?":** Neural net surrogates for complex pricing models aren't an academic curiosity — they're how modern options desks actually work. Train the net offline on Monte Carlo data (slow, expensive, do it once). Deploy the net for real-time pricing and hedging (fast, cheap, do it millions of times). The quality of your surrogate determines the quality of your hedge, and autograd gives you Greeks that are as good as the pricing model allows.

### Section 6: Deep Hedging — Learning to Hedge Without a Model
**Narrative arc:** The culmination. Instead of learning a pricing function and then deriving a hedge, what if we learn the hedge directly? Buehler et al. (2019) showed this is not only possible but superior — because the learned hedge can account for transaction costs, discrete rebalancing, and non-Gaussian returns. This is the cutting edge.

**Key concepts:** Deep hedging, end-to-end learning, convex risk measures, hedging under transaction costs.

**The hook:** The Buehler et al. (2019) paper from JP Morgan Quant Research starts with a deceptively simple question: if you sell an option and need to hedge it, what's the optimal sequence of trades? Black-Scholes says: hold Delta shares of the stock and rebalance continuously. But "continuously" is a fantasy — you rebalance once a day, or once an hour, and each rebalance costs money in transaction costs. The deep hedging framework says: forget the formula. Train an LSTM (or any sequence model) to minimize the hedging error directly. The input is the market history; the output is the hedge position at each time step; the loss is the P&L variance of the hedged portfolio. It learns to hedge. And it hedges better than Black-Scholes delta, because it knows about transaction costs and discrete rebalancing.

**Key formulas:**

The deep hedging objective:

$$\min_\theta \; \rho\left( -H_T + \sum_{t=0}^{T-1} \delta_t^\theta (S_{t+1} - S_t) - \sum_{t=0}^{T-1} c \cdot |\delta_t^\theta - \delta_{t-1}^\theta| \cdot S_t \right)$$

In words: minimize a risk measure rho of (the option payoff you owe, minus the gains from your hedging trades, minus the transaction costs). The hedge ratio delta_t at each step is the output of a neural network that takes the market history as input.

The network learns: when to hedge aggressively (high gamma positions near expiry), when to under-hedge (transaction costs exceed the risk reduction), and how to manage the tradeoff between tracking error and trading costs. No formula tells it this — it discovers it from data.

**Code moment:** A simplified deep hedging setup: train an MLP (not LSTM, for simplicity) to hedge a European call option under Black-Scholes dynamics with proportional transaction costs. Compare: (a) Black-Scholes delta hedge, (b) deep hedge with no transaction costs (should match BS), (c) deep hedge with 10 bps transaction costs (should deviate from BS). Output: plot the hedge ratio delta_t as a function of moneyness. Without costs, the deep hedge matches the BS delta curve exactly. With costs, it creates a "band" — the deep hedge doesn't rebalance when the moneyness hasn't changed much (because the cost exceeds the benefit). This is the "transaction cost band" that practitioners use heuristically, but the neural net discovers it from data.

**"So what?":** Deep hedging is the most convincing application of neural networks in derivatives. It's not replacing a formula with a black box — it's solving a problem that formulas *can't* solve (optimal hedging under realistic market conditions). JP Morgan, Goldman Sachs, and Morgan Stanley all have deep hedging programs in production or in advanced pilots. If you want to work at an options desk, this is the skill that gets you hired.

### Closing Bridge

This week showed you the cleanest application of neural networks in finance: a domain where we have analytical ground truth (Black-Scholes), where autograd gives us sensitivities for free, and where the practical speedup (1,000,000x for complex models) translates directly to real-world value. Next week shifts to the opposite end of the spectrum: instead of the mathematical elegance of derivatives, we'll confront the raw, chaotic world of high-frequency trading and market microstructure — where the data is measured in microseconds, the infrastructure costs millions, and the question isn't "can my neural net compute a Greek?" but "can my Python script even receive a market data message before the opportunity is gone?"

## Seminar Exercises

### Exercise 1: Autograd Greeks at Scale — Sensitivity Surfaces
**The question we're answering:** How do the Greeks behave across the full (moneyness, time-to-expiry) surface, and where do they become dangerous?

**Setup narrative:** The lecture demonstrated autograd Greeks for a single option scenario and verified the numbers match. That was the proof of concept. Now you'll use the *same* autograd machinery at scale: compute Greeks across a 2D grid of (moneyness, time-to-expiry) and visualize the sensitivity surfaces that options traders actually watch. The goal isn't "can autograd compute a Greek?" (you saw that it can) but "what do the Greeks *look like* across the full parameter space, and where should a trader be worried?"

**What they build:** Using the lecture's PyTorch BS implementation, compute delta, gamma, vega, and theta on a 50x50 grid of (moneyness from 0.8 to 1.2) x (time-to-expiry from 1 week to 1 year). Produce four 3D surface plots. Identify the "danger zones": where gamma explodes (near ATM, near expiry), where theta decays fastest, where vega is highest (long-dated ATM). Annotate each surface with the risk interpretation.

**What they'll see:** Gamma spikes to extreme values for ATM options near expiry — the "gamma trap" that destroyed LTCM. Theta decay accelerates as expiry approaches (the "theta burn" that option sellers exploit). Vega peaks for long-dated ATM options. The surfaces are beautiful and informative — this is what an options desk monitors in real time.

**The insight:** Autograd Greeks aren't just faster — they're a *visualization and risk management tool*. The surfaces reveal where hedging is most expensive (high gamma), where time decay is most painful (near-expiry ATM), and where volatility exposure is concentrated (long-dated). These surfaces, not individual Greek values, are what practitioners actually use.

### Exercise 2: Architecture Shootout for Neural Pricing
**The question we're answering:** How does architecture choice affect a neural pricer's accuracy, speed, and Greek quality?

**Setup narrative:** The lecture trained one architecture (4-layer MLP, 128 units, Softplus) and showed it works beautifully. But is that the *best* architecture? In practice, options desks that deploy neural pricers care about three things: pricing accuracy, inference speed (they need millions of prices per second), and Greek smoothness (choppy Greeks produce choppy hedges). These three objectives trade off against each other, and the architecture is the lever.

**What they build:** Using the lecture's 500K BS training set, train four architectures: (a) shallow-wide (2 layers, 512 units), (b) deep-narrow (6 layers, 64 units), (c) the lecture's 4x128 baseline, (d) a residual network (4 layers, 128 units, skip connections). For each, report: R² on 50K test samples, inference time for a batch of 100K options, and Greek smoothness (measure the L2 norm of the delta surface gradient — smoother is better). Produce a Pareto frontier plot: accuracy vs. speed, with Greek smoothness as bubble size.

**What they'll see:** The shallow-wide network is fastest but has the least smooth Greeks (large jumps in delta near ATM). The deep-narrow network has the smoothest Greeks but is slowest. The residual network is the best overall — nearly as fast as the baseline, with noticeably smoother Greeks due to the skip connections. The Pareto frontier reveals a clear architecture-quality tradeoff.

**The insight:** For neural surrogates in production, the architecture isn't just about accuracy — it's about the *quality of the derivatives*. A network with R² = 0.9998 but choppy Greeks is worse than one with R² = 0.9995 and smooth Greeks, because the Greeks drive the hedge, and the hedge drives the P&L.

### Exercise 3: The Implied Volatility Surface
**The question we're answering:** Can we see the famous "volatility smile" in real options data, and can a neural network learn the implied volatility surface?

**Setup narrative:** If Black-Scholes were correct, implied volatility would be the same at every strike and every expiry — a flat surface. It's not. The surface has a characteristic shape: higher implied vol for out-of-the-money puts (the "skew"), a minimum near at-the-money, and a complex term structure across maturities. This surface is the central object that every options trader watches, and learning it is a core application of neural networks in derivatives.

**What they build:** Using CBOE sample data (or synthetic Heston-generated data), compute implied volatilities by inverting Black-Scholes with scipy.optimize. Fit a neural net to the surface: input = (moneyness, time_to_expiry), output = implied_vol. Visualize as a 3D surface plot.

**What they'll see:** The smile/skew pattern. The surface is *not* flat — it has structure that varies across strikes and maturities. The neural net captures this structure with ~1% RMSE. The 3D plot is one of the most beautiful visualizations in the course.

**The insight:** The implied volatility surface is the market's "correction factor" for Black-Scholes. Learning this surface with a neural net gives you a fast, interpolatable model that can be queried at any (strike, expiry) pair — not just the traded ones. This is how desks price options that don't have active quotes.

### Exercise 4: Deep Hedging Starter
**The question we're answering:** Can a neural network learn a hedging strategy that accounts for transaction costs, and does it outperform the textbook Black-Scholes delta hedge?

**Setup narrative:** This is the exercise that connects everything. We simulate 10,000 option lifetimes under Black-Scholes dynamics, then train a small MLP to output the hedge ratio at each rebalancing point. The network minimizes hedging P&L variance minus transaction costs — and it discovers the "no-trade band" that practitioners use heuristically.

**What they build:** A simplified deep hedging framework: simulate GBM paths, compute option payoffs, train an MLP to output hedge ratios that minimize CVaR (conditional value at risk) of hedging error. Compare against BS delta hedge under three cost levels: 0 bps, 5 bps, 20 bps.

**What they'll see:** At 0 bps cost, the deep hedge matches BS delta exactly. At 5 bps, the deep hedge reduces trading frequency by ~30% while maintaining similar hedging accuracy. At 20 bps, the deep hedge discovers a clear no-trade band — it only rebalances when delta has drifted significantly. The hedging cost savings are 15-40%.

**The insight:** Deep hedging isn't magic — it's the network discovering that the optimal hedge under transaction costs is *not* the continuous-time delta. The no-trade band is well-known in options theory (Whalley & Wilmott, 1997), but the network discovers it from data without being told.

## Homework: "The Neural Options Laboratory"

### Mission Framing

You're about to build a complete neural derivatives desk — from pricing to hedging, from Black-Scholes to Heston, from analytical to learned. The first half is satisfying: you'll watch a neural network learn one of the most important formulas in finance and produce Greeks that match the analytical values to six decimal places. The second half is where it gets interesting: you'll move to a model (Heston) where analytical Greeks don't exist, and the neural net stops being a fast approximation and starts being the *only practical way* to get the numbers you need.

The climax is the implied volatility surface. You'll generate Heston option prices, invert Black-Scholes to get implied volatilities (a neat trick: using one model's formula to quote another model's prices), and watch the volatility smile emerge from the data. The neural net will learn this surface, and the result is both beautiful and useful — a differentiable model of how the market actually prices options, ready to be queried at any strike and maturity.

### Deliverables

1. **Black-Scholes in PyTorch (30 min):** Implement the BS formula as a differentiable function. Compute all five Greeks via autograd for a grid of 10,000 option scenarios. Produce a comparison table showing autograd vs. analytical Greeks. Maximum error should be < 1e-5.

2. **Neural BS Pricer (45 min):** Generate 500K training samples from Black-Scholes with randomly sampled inputs. Train a 4-layer MLP (128 hidden units, Softplus output). Report: R², MAE, and max error on 50K test samples. Plot predicted vs. actual. Compute the neural net's Delta and Vega via autograd and compare against analytical values.

3. **Extrapolation Analysis (30 min):** Test your neural BS pricer on three out-of-distribution regimes: (a) deep OTM options (moneyness > 2.0), (b) near-expiry options (T < 1 week), (c) high-volatility options (sigma > 60%). Where does the neural net break down? How does this compare to where Black-Scholes itself breaks down in practice?

4. **Heston Monte Carlo Pricer (45 min):** Implement a Heston Monte Carlo pricer. Parameters: S0=100, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, r=0.05. Generate 200K option prices across random (K, T) pairs using 50K paths each. This is the slow step — expect 5-10 minutes of computation.

5. **Neural Heston Pricer (30 min):** Train the same MLP architecture on Heston-generated data. Since Heston has no closed-form Greeks, the neural net's autograd Greeks are the *only* fast way to get them. Report: pricing accuracy and smoothness of the Delta and Vega surfaces.

6. **Implied Volatility Surface (45 min):** For a grid of (moneyness, T) pairs, compute the Heston price via your neural net, then invert Black-Scholes to get implied volatility. Plot the IV surface as a 3D mesh. Compare against the flat surface that Black-Scholes would produce. Identify: the skew (OTM puts are more expensive), the smile (both tails are elevated for short maturities), and the term structure (the smile flattens for longer maturities).

7. **Deliverable:** Complete notebook with BS autograd Greeks, neural BS pricer with extrapolation analysis, Heston MC pricer, neural Heston pricer with autograd Greeks, IV surface visualization, and a 300-word analysis: "Where do neural net pricers add value vs. analytical formulas?"

### What They'll Discover

- The neural BS pricer achieves R² > 0.9999 in-distribution but drops to R² ~ 0.99 for deep OTM options. This mirrors the real challenge: the "edges" of the pricing surface are where models disagree and where money is made or lost.
- Autograd Greeks from the Heston neural net are smooth and well-behaved, despite the Heston model having no closed-form Greeks. This is the practical payoff: Greeks that would take hours to compute via Monte Carlo bump-and-recompute are available in microseconds.
- The implied volatility surface from Heston shows a clear skew: OTM put implied vols are 5-15 percentage points higher than ATM. This matches real market data and cannot be reproduced by Black-Scholes. The neural net captures this structure.
- The rho parameter (correlation between stock and volatility) controls the skew: more negative rho = steeper skew. Students can verify this by varying rho and watching the IV surface tilt.

### Deliverable
A complete Jupyter notebook containing: differentiable BS implementation with autograd Greeks, neural BS pricer with train/test evaluation, extrapolation analysis, Heston MC generator, neural Heston pricer, IV surface visualization, and a written analysis section. All plots should be publication-quality with labeled axes and legends.

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Black-Scholes in PyTorch | Demo: implement BS, compute single-option Greeks via autograd | Not repeated (done in lecture) | At scale: full 10K-scenario Greek comparison table |
| Autograd Greeks | Demo: delta, vega, gamma for one scenario | Exercise 1: Greek *surfaces* across (moneyness, expiry) grid with risk interpretation | At scale: Heston neural Greeks (no analytical alternative) |
| Neural net as function approximator | Demo: train 4x128 MLP, show R² > 0.9999, extrapolation test | Exercise 2: architecture shootout (4 nets), Pareto frontier of accuracy/speed/smoothness | Build: neural BS pricer + neural Heston pricer, extrapolation analysis |
| Stochastic volatility (Heston) | Demo: Monte Carlo Heston pricing, neural surrogate concept | Not covered (reserved for homework depth) | Build: Heston MC pricer, neural Heston surrogate with autograd Greeks |
| Implied volatility surface | Demo: 3D IV surface from Heston neural net | Exercise 3: fit neural net to IV surface from CBOE/Heston data | Build: full IV surface visualization, identify skew/smile/term structure |
| Deep hedging | Demo: simplified MLP hedge, transaction cost band discovery | Exercise 4: deep hedge under three cost levels, compare vs. BS delta | Integrate: concepts from deep hedging inform the written analysis |
| Options fundamentals (payoffs, moneyness) | Demo: payoff diagrams, time value visualization | Not repeated (foundational, done in lecture) | Prerequisite knowledge applied throughout |

## Key Stories & Facts to Weave In

- **The 1987 Crash and the Birth of the Skew:** Before October 19, 1987, the implied volatility surface was approximately flat — Black-Scholes assumptions held (roughly). After the crash, OTM put implied vols permanently rose by 5-10 percentage points relative to ATM. The market learned that crashes happen, and it priced that fear into options forever. Every options trader since has lived in a world where the skew exists. Black-Scholes cannot produce a skew.

- **Goldman Sachs' Neural Net Pricing Engine (2021):** Goldman reported using neural network surrogates for Monte Carlo pricing in their Securities Division. The accuracy: within 0.01% of the full Monte Carlo price. The speedup: 1,000-10,000x. This enabled real-time risk management for exotic derivatives that previously required overnight batch computation. The internal name for the initiative: "fast Greeks."

- **Texas Instruments and the SR-52 Calculator (1977):** Four years after Black-Scholes was published, TI marketed a calculator programmed with the formula. Options traders on the CBOE floor punched in numbers and got prices in seconds. It was the first "fintech" product, and it changed how options were traded: instead of haggling, traders could agree on a mathematical price. The formula didn't just describe the market — it helped create it.

- **Long-Term Capital Management and the Limits of Greeks (1998):** LTCM's options positions were hedged using delta and vega from analytical models. When Russia defaulted and correlations spiked, their Gamma exposure — the rate of change of their Delta — exploded. They thought they were hedged. They had $125 billion in positions and $4 billion in equity. When the second-order risks materialized, they lost $4.6 billion in less than four months.

- **JP Morgan's Deep Hedging (Buehler et al. 2019):** The paper showed that an LSTM trained end-to-end on hedging objectives outperforms Black-Scholes delta hedging by 5-15% in P&L variance, with the improvement increasing as transaction costs increase. The result: at 20 bps round-trip costs, the deep hedge saves enough in transaction costs to cover the infrastructure cost of running the neural network. At 50 bps, it's not even close.

- **The "Vol Surface Arbitrage" Hunt:** Every major bank has a team that monitors the implied volatility surface for arbitrage — combinations of options that violate no-arbitrage constraints. Neural networks that learn the IV surface must enforce calendar spread and butterfly arbitrage constraints, or they produce nonsensical surfaces. This is an active area of research: how to make neural nets "arbitrage-aware."

## Cross-References
- **Builds on:** Week 7 (PyTorch architecture and training — this is the most PyTorch-intensive week since Week 7), Week 2 (volatility concepts — the entire motivation for moving beyond Black-Scholes is that volatility isn't constant), Week 8 (sequence models — deep hedging is sequential decision-making).
- **Sets up:** Week 16 (Market Making — the Avellaneda-Stoikov model is a continuous-time options-like pricing problem; understanding Greeks helps with inventory management). Week 18 (Capstone — a neural options pricer is a legitimate capstone project component).
- **Recurring thread:** The "analytical vs. learned" comparison continues from Weeks 4-5 (linear models vs. trees) through Week 13 (analytical execution vs. RL). Here the comparison is especially clean because we have a perfect analytical benchmark.

## Suggested Reading
- **Hull, "Options, Futures, and Other Derivatives" (11th edition):** The industry standard textbook. Chapters 13-15 cover Black-Scholes, Greeks, and volatility smiles. Dense but definitive. Read this if you want to actually understand why the formula works, not just use it.
- **Buehler et al., "Deep Hedging" (Quantitative Finance, 2019):** The paper that launched deep hedging as a field. Surprisingly readable for a quant finance paper. Focus on Sections 2 (framework) and 4 (experiments). The key result: the neural hedge beats Black-Scholes delta by 5-15% in P&L variance under realistic transaction costs.
- **Ruf & Wang, "Neural Networks for Option Pricing and Hedging: A Literature Review" (Journal of Computational Finance, 2020):** A comprehensive survey of neural nets in derivatives. Covers the history from the 1990s (when people first tried this) through 2020. Good for understanding what's been tried and what actually works.
