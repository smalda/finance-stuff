# Week 16 — Market Making with ML

> **Market makers earn the spread and pray that inventory doesn't kill them. Avellaneda and Stoikov turned that prayer into a formula. We'll turn the formula into an RL agent.**

## Prerequisites
- **Week 13 (RL for Portfolios):** The MDP framework, PPO and SAC algorithms, Stable-Baselines3, Gymnasium environments, reward shaping. This week is the *application* of everything you learned in Week 13 to a specific, well-defined problem.
- **Week 15 (HFT & Microstructure):** The limit order book, bid-ask spread, order flow, adverse selection, market impact. You need to understand the economic role of market makers and why the spread exists before you can build one.
- **Week 7 (Feedforward Nets):** PyTorch basics. The RL agent's policy is a neural net.
- **Week 3 (Portfolio Theory & Risk):** Sharpe ratio, max drawdown. You'll use these to evaluate your market maker's performance.
- **General probability:** Poisson processes (order arrivals), Brownian motion (price dynamics), exponential decay (fill probabilities). If you know what a Poisson distribution is, you're fine — we'll build the intuition for the rest.

## The Big Idea

Last week, you learned that market makers provide liquidity — they continuously quote bid and ask prices, offering to buy from sellers and sell to buyers, and they earn the spread for their trouble. Virtu Financial did this well enough to have one losing day in five years. Citadel Securities does it well enough to earn $7.5 billion in annual revenue. It sounds like free money. It's not.

The market maker's nightmare has a name: adverse selection. When someone hits your bid (buying from you), it might be because they know something you don't — maybe the stock is about to drop, and they're dumping it on you at a price that's about to become stale. You're now long a stock that's falling. If someone lifts your offer (selling to you), maybe the stock is about to rally and they're front-running the move. You're now short a stock that's rising. In both cases, the "informed trader" took the right side of the trade, and you — the market maker — took the wrong side. The spread you earned is smaller than the loss on the position. This is adverse selection, and it's the central problem of market making.

In 2008, Marco Avellaneda and Sasha Stoikov published a paper that changed how the industry thinks about this problem. They asked: given that you're a market maker facing random order flow and a stochastic price process, what are the *optimal* bid and ask quotes? The answer is a pair of formulas — elegant, closed-form, and intuitive — that tell you exactly how wide your spread should be and how to skew your quotes based on your current inventory. The model has three moving parts: the reservation price (your private "fair value," adjusted for inventory risk), the optimal spread (wider when the market is volatile or illiquid), and the quote placement (symmetric around the reservation price). Three formulas, and they capture 80% of what a real market maker does.

This week, we implement Avellaneda-Stoikov from scratch, build a Monte Carlo simulator, wrap it as a Gymnasium environment, and then train an RL agent to improve on the analytical solution. The 2022 Alpha-AS paper (PLOS ONE) showed that RL beats pure Avellaneda-Stoikov on 24 of 30 test days in terms of Sharpe ratio — but with worse tail risk. That tension — better average performance but worse worst case — is the central question of the homework. Everything runs in simulation on your laptop.

## Lecture Arc

### Opening Hook

In January 2021, GameStop stock went from $20 to $483 in two weeks, driven by a Reddit army of retail traders. Market makers had a catastrophic few days. Citadel Securities reportedly handled 7.4 billion shares in a single day — about 40% of total US equity volume. Their systems worked. But smaller market makers got destroyed: they were short gamma (their inventory was moving against them faster than they could rehedge), and the spreads they'd been quoting were too tight for the volatility. Several market makers for GameStop options stopped providing quotes entirely — the book went empty, just like in the Flash Crash. Avellaneda-Stoikov's model would have told them to widen their spread as volatility exploded. The ones who survived were the ones who did exactly that, either by following the model's logic or by having humans override their algorithms in real time.

### Section 1: Market Making Economics — Earning the Spread
**Narrative arc:** We establish the economic fundamentals of market making using a concrete analogy. The tension: the spread looks like free money but adverse selection makes it a risky business. The resolution: the market maker's job is to set the spread wide enough to compensate for adverse selection, but not so wide that nobody trades.

**Key concepts:** Bid-ask spread, market making as intermediation, adverse selection, inventory risk, the market maker's dilemma.

**The hook:** Think of a market maker as a used car dealer. The dealer buys cars from sellers at the "bid" price ($19,000) and sells them to buyers at the "ask" price ($21,000). The $2,000 spread is the dealer's compensation for three things: (1) the cost of holding inventory (the car sits on the lot, depreciating), (2) the risk that the car's value drops (a recall is announced, the model gets bad reviews), and (3) the risk that the seller knows something the dealer doesn't (the transmission is failing and the seller is dumping it). Replace "car" with "100 shares of AAPL" and "dealer" with "Citadel Securities," and you have market making. The spread is payment for holding risk. The question is: how much risk are you holding, and what spread compensates you fairly?

**Key formulas:**

The market maker's P&L per round trip (simplest case):

$$\text{P\&L} = \underbrace{(p^a - p^b)}_{\text{spread earned}} - \underbrace{\Delta S \cdot q}_{\text{inventory P\&L}}$$

If you buy at the bid and sell at the ask before the price moves, you earn the full spread. But if the mid-price moves against you while you're holding inventory q, the second term can overwhelm the first. For Apple, the spread is about 1 cent. A 10-cent move against you on 1,000 shares costs $100 — wiping out the $10 you earned from the spread on 1,000 round trips.

**Code moment:** A simple simulation to show the market maker's dilemma:

```python
# Simulate 1000 trades: earn spread, lose on adverse selection
spread = 0.01  # $0.01 spread
n_trades = 1000
price_moves = np.random.normal(0, 0.05, n_trades)  # random mid-price changes
inventory = np.zeros(n_trades + 1)

pnl_per_trade = []
for i in range(n_trades):
    # Randomly: buy at bid or sell at ask
    side = np.random.choice([-1, 1])
    inventory[i+1] = inventory[i] + side
    spread_pnl = spread / 2  # half spread per trade
    inventory_pnl = inventory[i] * price_moves[i]
    pnl_per_trade.append(spread_pnl + inventory_pnl)
```

Output: the spread P&L is always positive, but the inventory P&L is a random walk. Total P&L is noisy and sometimes negative for long stretches. The visual: cumulative spread P&L (a straight line going up) vs. cumulative inventory P&L (a random walk) vs. total (the sum — usually positive but with scary drawdowns).

**"So what?":** The market maker's profit comes from the spread, but their risk comes from inventory. Managing that risk — knowing when to widen the spread, when to skew quotes, when to cut positions — is the entire game. Avellaneda and Stoikov formalized this game in 2008.

### Section 2: The Avellaneda-Stoikov Model — Three Formulas That Changed Market Making
**Narrative arc:** This is the mathematical core of the week. We build the three A-S formulas step by step, each with full intuition before the math arrives. The tension: the model makes strong assumptions (Brownian motion, Poisson arrivals). The resolution: despite the assumptions, it captures the essential trade-offs and provides a strong baseline.

**Key concepts:** Reservation price, optimal spread, Poisson fill model, risk aversion parameter, inventory skew.

**The hook:** Marco Avellaneda was a professor at the Courant Institute (NYU) and a quantitative researcher at Finance Concepts. Sasha Stoikov was his student. Their 2008 paper, "High-Frequency Trading in a Limit Order Book," is one of the most cited papers in quantitative finance — not because it's complex (the model fits on a single page), but because it's *useful*. Every market-making desk in the world, from Virtu to Jane Street, uses some variant of Avellaneda-Stoikov or is benchmarked against it. The model is to market making what Black-Scholes is to options pricing: the starting point that everyone agrees on, even though everyone also agrees it's wrong.

**Key formulas:**

**Step 1: The Reservation Price**

The market maker's "private fair value," adjusted for inventory:

$$r(t) = S(t) - q \cdot \gamma \cdot \sigma^2 \cdot (T - t)$$

Build this piece by piece:
- Start with the mid-price $S(t)$. If you had no inventory, this would be your fair value.
- Now add inventory. If you're long $q > 0$ shares, you want to sell — so your "true" price is *lower* than mid (you'd accept a slightly worse price to reduce risk). If $q < 0$ (short), your fair value is *higher* than mid.
- The adjustment $q \cdot \gamma \cdot \sigma^2 \cdot (T-t)$ captures three things: how many shares you hold ($q$), how risk-averse you are ($\gamma$), how volatile the market is ($\sigma^2$), and how much time remains ($T-t$). More inventory, more risk aversion, more volatility, more time remaining — all make the adjustment larger.

When $q > 0$: $r(t) < S(t)$. You're eager to sell, so you price below mid.
When $q < 0$: $r(t) > S(t)$. You're eager to buy, so you price above mid.

**Step 2: The Optimal Spread**

How wide should your quotes be?

$$\delta_a + \delta_b = \underbrace{\gamma \sigma^2 (T-t)}_{\text{volatility penalty}} + \underbrace{\frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{\kappa}\right)}_{\text{liquidity premium}}$$

Two terms, two fears:
- **Volatility penalty** $\gamma \sigma^2 (T-t)$: wider when the market is volatile (you need more cushion against adverse moves) or when there's more time left (more time for things to go wrong).
- **Liquidity premium** $\frac{2}{\gamma} \ln(1 + \gamma/\kappa)$: wider when the order book is thin (low $\kappa$). The parameter $\kappa$ controls how fast fill probability decays with distance from mid — low $\kappa$ means orders far from mid still get filled (thin book), so you need to quote wider.

**Step 3: The Optimal Quotes**

Place bid and ask symmetrically around the reservation price:

$$p^b(t) = r(t) - \frac{\delta_b}{2}, \quad p^a(t) = r(t) + \frac{\delta_a}{2}$$

where $\delta_a = \delta_b = (\delta_a + \delta_b) / 2$ in the symmetric case.

The net effect: when you're long (reservation price below mid), both quotes shift down — your bid is lower (less eager to buy more) and your ask is lower (more eager to sell). When you're short, both quotes shift up. The market maker naturally "leans" against inventory.

**Code moment:** Implement all three formulas in ~20 lines:

```python
def avellaneda_stoikov(S, q, gamma, sigma, T_remaining, kappa):
    """Compute optimal bid and ask quotes."""
    # Reservation price
    r = S - q * gamma * sigma**2 * T_remaining

    # Optimal spread
    spread = gamma * sigma**2 * T_remaining + (2/gamma) * np.log(1 + gamma/kappa)

    # Optimal quotes
    bid = r - spread / 2
    ask = r + spread / 2
    return bid, ask, r, spread
```

Output: for S=100, q=0, gamma=0.1, sigma=2, T_remaining=0.5, kappa=1.5, the optimal spread is about $2.60 and the quotes are symmetric around $100. When q=5 (long 5 shares), the reservation price drops to ~99 and both quotes shift down. Students see the inventory skew in action.

**"So what?":** Three formulas, twenty lines of code, and you have a functional market maker. Everything else — the Monte Carlo simulator, the RL enhancement — is built on top of this core. Avellaneda-Stoikov is remarkable because it gives you an *analytical* baseline: before you train any ML model, you know exactly what the "right" answer looks like under the model's assumptions. When your RL agent deviates from A-S, it's either learning something the model missed, or it's overfitting. Telling the difference is the challenge.

### Section 3: Building the Monte Carlo Simulator
**Narrative arc:** We need a realistic-enough simulation to test our market maker. We build a simple LOB simulator: Brownian mid-price, Poisson order arrivals, and exponential fill probabilities. Simple enough to run fast, realistic enough to capture the essential dynamics.

**Key concepts:** Monte Carlo simulation, Brownian motion for mid-price, Poisson process for order arrivals, exponential fill model, episode structure.

**The hook:** Every market maker, from the solo quant to Citadel Securities, develops their market simulator before they develop their strategy. The simulator is the training ground, the testing environment, and the safety net. A bad simulator is worse than no simulator — it teaches your agent to exploit unrealistic dynamics. Our simulator is deliberately simple: Brownian mid-price (no trends, no jumps), Poisson fills (no order book dynamics), and no adverse selection (informed traders don't exist). Every one of these simplifications makes real market making harder than our simulation. That's intentional — we want to see if RL can improve on A-S even in this *easy* environment. If it can't beat A-S here, it won't beat it anywhere.

**Key formulas:**

Mid-price dynamics (Brownian motion):

$$S_{t+1} = S_t + \sigma \sqrt{\Delta t} \cdot Z_t, \quad Z_t \sim N(0, 1)$$

Fill probability (exponential decay with distance from mid):

$$P(\text{fill at bid}) = A \cdot e^{-\kappa \cdot \delta_b}, \quad P(\text{fill at ask}) = A \cdot e^{-\kappa \cdot \delta_a}$$

where $\delta_b = S_t - p^b$ (distance from mid to bid) and $\delta_a = p^a - S_t$ (distance from mid to ask). The parameter $A$ controls the base arrival rate, and $\kappa$ controls how fast fill probability decays with distance. Tight quotes ($\delta$ small) get filled more often but earn less spread. Wide quotes ($\delta$ large) earn more spread but get filled rarely. This is the fundamental trade-off.

Inventory update: if the bid fills, $q_{t+1} = q_t + 1$ (bought a share). If the ask fills, $q_{t+1} = q_t - 1$ (sold a share). Both can fill in the same timestep (but it's rare).

P&L tracking:

$$\text{PnL}_t = \text{PnL}_{t-1} + \underbrace{\mathbf{1}_{\text{ask filled}} \cdot (p^a_t - S_t)}_{\text{sold above mid}} + \underbrace{\mathbf{1}_{\text{bid filled}} \cdot (S_t - p^b_t)}_{\text{bought below mid}} + \underbrace{q_t \cdot (S_{t+1} - S_t)}_{\text{inventory mark-to-market}}$$

**Code moment:** Build the full simulator in ~80 lines. Run one episode with the A-S strategy. Plot: mid-price, bid/ask quotes, inventory, and cumulative P&L over the 200-step session. Output: students see the quotes dancing around the mid-price, inventory fluctuating between -10 and +10, and P&L generally trending upward with occasional dips when inventory gets large and the price moves against it.

**"So what?":** This simulator is our "gym" — the training environment for both the analytical A-S strategy and the RL agent. Running 10,000 episodes takes about 30 seconds on an M4. That's fast enough for the Monte Carlo analysis and for RL training. In the homework, you'll run parameter sweeps, train RL agents, and compare strategies — all on this foundation.

### Section 4: The Gymnasium Environment — From Simulation to RL
**Narrative arc:** We wrap the simulator as a Gymnasium environment, connecting the market-making problem to the RL framework from Week 13. The state captures what the agent knows; the action determines where it quotes; the reward captures what we want it to learn.

**Key concepts:** gymnasium.Env interface, state design, action space design, reward design, episode termination.

**The hook:** The bridge from "I have a simulator" to "I have an RL environment" is surprisingly short — about 50 lines of code. But those 50 lines contain critical design decisions: what goes in the state (and what doesn't), how to parameterize the action space, and what reward to optimize. Get any of these wrong and the RL agent will learn something useless. We saw this in Week 13 with reward hacking — the same dangers apply here, but the consequences are more visceral because we can compare against the analytical A-S solution.

**Key formulas:**

State vector:

$$s_t = \begin{bmatrix} q_t & \text{(inventory — the most important feature)} \\ S_t - S_0 & \text{(normalized mid-price change)} \\ (T - t) / T & \text{(fraction of time remaining)} \\ \hat{\sigma}_t & \text{(recent realized volatility, trailing 20 steps)} \\ \delta_t^{AS} & \text{(A-S optimal spread — gives the agent a reference point)} \end{bmatrix}$$

Including the A-S spread as a state feature is a deliberate choice: it gives the RL agent a "baseline" to improve upon rather than forcing it to discover the spread from scratch. This is a form of imitation learning — and it dramatically speeds up training.

Action space (continuous, 2-dimensional):

$$a_t = [\delta_b, \delta_a] \in [0.01, 5.0]^2$$

The agent chooses the distance from mid for its bid and ask quotes. Both are clipped to [0.01, 5.0] to prevent nonsensical quotes.

Reward:

$$r_t = \Delta \text{PnL}_t - \phi \cdot q_t^2$$

The first term encourages making money. The second term penalizes large inventory positions, with strength controlled by $\phi$. Without the penalty, the RL agent will learn to accumulate large positions (it doesn't fear inventory risk the way A-S does). With too much penalty, it quotes so wide that it barely trades.

**Code moment:**

```python
class MarketMakingEnv(gymnasium.Env):
    def __init__(self, S0=100, sigma=2, T=1.0, n_steps=200,
                 kappa=1.5, A=0.5, gamma=0.1, phi=0.01):
        self.action_space = gymnasium.spaces.Box(
            low=0.01, high=5.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        # ... store parameters ...

    def step(self, action):
        delta_b, delta_a = action
        bid = self.mid_price - delta_b
        ask = self.mid_price + delta_a
        # Fill logic (Poisson)
        # Inventory update
        # PnL update
        # Mid-price update (Brownian)
        reward = delta_pnl - self.phi * self.inventory**2
        return obs, reward, done, truncated, info
```

Output: the environment runs, random actions produce negative reward (quoting randomly loses money), and students can verify that plugging in A-S quotes produces positive average reward.

**"So what?":** The Gymnasium environment is the bridge between quant finance and RL engineering. Once the simulator is wrapped as an Env, everything from Stable-Baselines3 "just works" — PPO, SAC, A2C, all of them. The hard thinking is in the state/action/reward design, not in the algorithm.

### Section 5: RL vs. Avellaneda-Stoikov — The Alpha-AS Result
**Narrative arc:** We train the RL agent, compare it against A-S, and confront the key finding from the literature: RL can beat A-S on average, but with worse tail risk. This tension is the intellectual climax of the week.

**Key concepts:** RL training dynamics, comparison methodology, Sharpe ratio vs. max drawdown, regime-dependent performance, the bias-variance tradeoff in strategy space.

**The hook:** In 2022, Falces, Aranzabal, and De Lope published "Alpha-AS: RL to Improve the Avellaneda-Stoikov Framework" in PLOS ONE. They trained a Double DQN agent that dynamically adjusted the A-S parameters (gamma and kappa) based on market conditions. The result: on BTC-USD data, the RL-enhanced strategy beat pure A-S on 24 of 30 test days in terms of Sharpe ratio and *dramatically* improved the P&L-to-max-adverse-position ratio — by roughly 20x. But — and this is the critical caveat — the RL agent had worse max drawdown. It made more money on average, but its worst days were worse than A-S's worst days. This is the fundamental tension: RL trades stability for performance.

**Key formulas:**

The Alpha-AS approach: instead of using fixed gamma and kappa, let the RL agent output adjustments:

$$\gamma_t = \gamma_0 + \alpha_\gamma(s_t), \quad \kappa_t = \kappa_0 + \alpha_\kappa(s_t)$$

where $\alpha_\gamma$ and $\alpha_\kappa$ are outputs of the RL agent, bounded to prevent extreme values. The quotes are then computed using the standard A-S formulas with the adapted parameters. This is elegant: the RL agent doesn't replace A-S — it *tunes* A-S in real time.

Alternatively (our approach in the seminar): let the agent directly choose $(\delta_b, \delta_a)$, initialized near the A-S optimal values. This is more flexible but harder to learn.

**Code moment:**

```python
from stable_baselines3 import PPO, SAC

# Train PPO on the market making environment
model = PPO("MlpPolicy", env, verbose=1,
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            gamma=0.99, ent_coef=0.01)
model.learn(total_timesteps=500_000)  # ~10 min on M4
```

After training, evaluate on 1,000 fresh simulated paths. Compare three strategies:
1. Naive symmetric (fixed spread = 1.0, no inventory skew)
2. Avellaneda-Stoikov (analytical optimal)
3. RL agent (PPO)

Output: a comparison table showing mean PnL, Sharpe, max drawdown, mean |terminal inventory|, and % profitable. The RL agent should beat A-S on Sharpe by 10-30% but have 20-50% worse max drawdown. The naive strategy should trail both.

**"So what?":** The RL agent learns something genuinely useful: it adapts spread width to recent volatility more aggressively than A-S (which uses a fixed sigma parameter), and it manages inventory more actively during trending markets. But it occasionally "gambles" on inventory positions that A-S would never take. In production, the choice between A-S and RL is a risk management question, not a performance question: do you want consistent, predictable behavior (A-S) or higher average returns with fatter left tail (RL)?

### Section 6: What the Simulation Hides — Reality vs. Our Model
**Narrative arc:** We close with honesty about what our simulation simplifies. This section prevents students from thinking they've "solved" market making with 200 lines of Python.

**Key concepts:** Model risk, adverse selection, queue position, latency, discrete tick sizes, market impact.

**The hook:** Our simulation assumes five things that are false in real markets: (1) no adverse selection (every order is equally likely to be informed or uninformed), (2) no queue position (your order fills immediately if it's the best price, but in reality there might be 100,000 shares ahead of you), (3) no latency (you update quotes instantly, but in reality there's a delay), (4) no discrete tick sizes (prices move in increments of $0.01, not continuously), and (5) no other market makers (you're the only one quoting, but in reality you're competing against Citadel, Virtu, and fifty other firms). Despite these simplifications, the A-S framework teaches the *decision theory* of market making — when to quote wide, when to skew, when to flatten — and the RL enhancement teaches whether adaptive behavior can improve on static rules.

**Key formulas:** No new formulas. The key comparison is qualitative:

| Aspect | Our Simulation | Reality |
|--------|---------------|---------|
| Mid-price | Brownian motion | Mean-reverting with jumps |
| Fill model | Poisson, symmetric | Adverse selection, queue priority |
| Competitors | None | 50+ market makers |
| Latency | Zero | 1-100 microseconds |
| Tick size | Continuous | $0.01 minimum |
| Data | Simulated | $millions/year for real LOB data |

**Code moment:** Show a simple extension: add adverse selection to the simulator. When an order fills, the mid-price jumps slightly in the adverse direction (filling your bid causes the price to drop, filling your ask causes it to rise). The adverse selection parameter alpha controls the jump size. Rerun the A-S strategy with alpha=0.0 (our original sim) vs. alpha=0.01 vs. alpha=0.05. Output: the Sharpe ratio drops from ~2.0 (no adverse selection) to ~0.5 (moderate) to negative (high adverse selection). This is why market making is hard in real life.

**"So what?":** The simulation teaches you how to think about market making. Production market making requires solving the same optimization problem, but with far more realistic models, real data, and infrastructure that costs millions. The concepts transfer; the code does not.

### Closing Bridge

This week brought together microstructure from Week 15 and RL from Week 13 into the most concrete strategy we've built: a market maker that continuously quotes bid and ask prices, earns the spread, and manages inventory risk. The Avellaneda-Stoikov model gives you the analytical baseline — three formulas that capture 80% of the logic. The RL agent can improve on it — by 10-30% in Sharpe — but at the cost of worse tail risk. That tradeoff is the central lesson, not just for market making but for the entire "analytical vs. learned" debate that's been running through the course. Next week, we leave traditional markets entirely and enter the world of crypto and DeFi — where the exchanges are automated market makers themselves, the order book is a mathematical formula, and the "microstructure" is literally written in smart contract code.

## Seminar Exercises

### Exercise 1: Stress-Testing Avellaneda-Stoikov Under Adversarial Conditions
**The question we're answering:** The lecture showed A-S works in a friendly simulation. Does it survive when conditions get hostile?

**Setup narrative:** The lecture implemented A-S and ran it in the standard Brownian-Poisson environment — and it looked great (Sharpe 1.5-3.0). But the lecture also warned that this simulation is generous: no adverse selection, no competition, symmetric fills. Before we try to improve A-S with RL, we need to know *where* it breaks. This exercise pushes A-S to its limits by modifying the simulation environment in ways that make it more realistic.

**What they build:** Using the lecture's A-S implementation (provided as starter code), run 1,000 episodes under four scenarios: (a) baseline (lecture parameters), (b) adverse selection (mid-price jumps 0.02 against you on each fill), (c) asymmetric order flow (70% of fills are on the wrong side — you buy more than you sell in a falling market), (d) volatility spike (sigma doubles from step 100 onward). Report: mean PnL, Sharpe, max drawdown, and mean |terminal inventory| for each scenario.

**What they'll see:** The baseline confirms the lecture numbers. Adverse selection cuts Sharpe by 50-70%. Asymmetric flow causes dangerous inventory accumulation. The volatility spike catches A-S flat-footed (it uses fixed sigma). The comparison table is the motivation for RL: A-S is good under assumptions, fragile outside them.

**The insight:** A-S is surprisingly effective *in its own model world*. But each relaxation of the model's assumptions degrades performance, because A-S can't adapt. This is the exact gap that RL is designed to fill — and quantifying the gap before training the RL agent makes the comparison meaningful.

### Exercise 2: Parameter Sensitivity — The gamma-kappa Landscape
**The question we're answering:** How sensitive is the A-S model to its two key parameters, and is there an "optimal" region?

**Setup narrative:** Every analytical model has parameters that must be calibrated. A-S has two key ones: gamma (risk aversion) and kappa (order book liquidity). Getting them wrong costs money. This exercise maps the parameter landscape.

**What they build:** A grid search over gamma in {0.01, 0.05, 0.1, 0.5, 1.0} and kappa in {0.5, 1.0, 1.5, 3.0, 5.0}. For each (gamma, kappa) pair, run 500 episodes and record mean Sharpe ratio. Produce a heatmap.

**What they'll see:** Sharpe is highest in a "sweet spot" region — moderate gamma (0.05-0.2) and moderate kappa (1.0-3.0). Very low gamma (not risk-averse enough) leads to large inventory and occasional blowups. Very high gamma (too risk-averse) leads to wide spreads, few fills, and low P&L. The kappa sensitivity is smoother.

**The insight:** The A-S model is robust to parameter choice within a reasonable range — a 2x change in gamma costs maybe 20% Sharpe. But at the extremes, it breaks. This is exactly where RL can help: by adapting gamma and kappa to current market conditions rather than using fixed values.

### Exercise 3: State Representation Ablation for the RL Market Maker
**The question we're answering:** Which state features actually help the RL agent, and which are noise?

**Setup narrative:** The lecture built the Gymnasium environment with a 5-dimensional state: inventory, price change, time remaining, trailing vol, and the A-S optimal spread. The lecture *chose* these features and moved on. But how do we know these are the right features? This exercise runs the ablation study that justifies (or challenges) the state design.

**What they build:** Using the lecture's `MarketMakingEnv` as a base, train five PPO agents (200K steps each, same seed) with different state vectors: (a) full 5-feature state, (b) without A-S spread (can the agent learn its own spread rule?), (c) without trailing vol (can the agent infer volatility from price changes?), (d) inventory only (minimal state — is inventory the only feature that matters?), (e) full state + additional features (rolling mean of recent fills, cumulative PnL). Compare all five on 500 fresh episodes.

**What they'll see:** Removing inventory from the state is catastrophic (the agent can't manage risk). Removing A-S spread costs 10-20% Sharpe (the agent can eventually learn its own spread, but slower). Removing trailing vol costs 5-15% Sharpe. The inventory-only agent is surprisingly decent (Sharpe ~60-70% of the full agent). Adding extra features helps marginally.

**The insight:** Inventory is the single most important state feature — which makes economic sense, because inventory management IS market making. The A-S spread as a state feature is a form of "cheating" (giving the agent the analytical answer), but it speeds up training dramatically. This echoes the general principle from Week 13: state representation design is where the domain knowledge enters RL.

### Exercise 4: RL vs. A-S — Regime-Conditional Analysis
**The question we're answering:** In which *specific* market conditions does RL outperform A-S, and in which does it underperform?

**Setup narrative:** The lecture showed the headline result: RL beats A-S on average Sharpe but has worse tail risk. That's the 30,000-foot view. This exercise zooms in: instead of one aggregate comparison, you'll segment the 500 test episodes by market condition and ask "when does RL's advantage appear, and when does it disappear?" This is the analysis that would inform a production decision.

**What they build:** Train PPO for 500K steps (or use the agent from Exercise 3's best state representation). Run all three strategies (naive, A-S, RL) on 500 shared episodes. Classify each episode by regime: low-vol (sigma in bottom quartile of realized paths), high-vol (top quartile), trending (absolute cumulative return > 1 sigma), mean-reverting (low autocorrelation). For each regime, report: mean PnL, Sharpe, max drawdown. Produce a grouped bar chart: Sharpe by strategy, grouped by regime.

**What they'll see:** RL's advantage over A-S is concentrated in high-vol and trending episodes — exactly where A-S's fixed parameters hurt most. In low-vol, mean-reverting episodes, A-S and RL are nearly identical (the analytical solution is already near-optimal). RL's worst-case drawdowns also cluster in trending episodes (the agent sometimes "bets" on mean reversion that doesn't happen).

**The insight:** The decision "RL vs. A-S" isn't binary — it's regime-dependent. A production market maker might use A-S in calm markets (reliable, predictable) and switch to RL when volatility spikes (adaptive, but with guardrails). The regime analysis transforms the RL-vs-analytics debate from philosophy to data.

## Homework: "Market Making: Analytical vs. Learned"

### Mission Framing

You're about to build a complete market-making operation — from the mathematical model to the Monte Carlo engine to the RL agent — and answer a question that actual market-making desks debate: is the analytical solution good enough, or can machine learning improve on it?

The Avellaneda-Stoikov model is beautiful. Three formulas, clean intuition, proven in practice. But it has a limitation: its parameters are fixed. It doesn't know that volatility just doubled, or that order flow just became one-sided, or that liquidity just evaporated. An RL agent, in principle, can adapt to all of these in real time. The Alpha-AS paper (2022) says the RL agent wins on 24 of 30 days. But it also says the RL agent has worse worst-case performance. Your homework answers: by how much, under what conditions, and what changes when the market regime shifts mid-session?

The most interesting part is the regime test. You'll run 500 simulations where volatility doubles halfway through the session — a crude but effective model of what happens when a central bank announcement hits or a geopolitical event breaks. A-S, with its fixed sigma, will keep quoting as if nothing happened. The RL agent, if it's learned anything useful, will adapt. Whether it adapts *well* is the question you'll answer.

### Deliverables

1. **Avellaneda-Stoikov Implementation (45 min):** Implement the full model: reservation price, optimal spread, optimal quotes, with the Brownian-Poisson simulator. Parameters: S0=100, sigma=2, T=1 (normalized session), N=200 steps, gamma=0.1, kappa=1.5, A=0.5. Verify your implementation produces reasonable behavior on 10 sample paths (positive mean PnL, bounded inventory, quotes tracking mid-price).

2. **Monte Carlo Analysis (30 min):** Run 10,000 A-S simulations. Report: mean PnL, std PnL, Sharpe ratio, max drawdown (worst single-episode draw), mean |terminal inventory|, percentage of episodes that are profitable. These are your baseline numbers. Include histograms of terminal PnL and terminal inventory.

3. **Parameter Sweep (30 min):** Create a heatmap of Sharpe ratio over gamma in {0.01, 0.05, 0.1, 0.5, 1.0} and kappa in {0.5, 1.0, 1.5, 3.0, 5.0}. That's 25 cells, 500 episodes each. Identify the optimal region. Does the optimal gamma change when sigma increases from 2 to 4? (Run a second heatmap with sigma=4.)

4. **Gymnasium Environment (30 min):** Wrap your simulator as `MarketMakingEnv(gymnasium.Env)`. State: [inventory, normalized mid-price change, time remaining, trailing 20-step vol, A-S spread]. Action: [bid_distance, ask_distance], continuous in [0.01, 5.0]. Reward: delta_PnL - 0.01 * inventory^2. Validate with `gymnasium.utils.env_checker.check_env`.

5. **RL Agent Training (45 min):** Train PPO using Stable-Baselines3 with 500K timesteps (~10 minutes on M4). Experiment with two reward functions: (a) delta_PnL - 0.01 * inventory^2, (b) delta_PnL - 0.1 * inventory^2 (10x stronger inventory penalty). Save training curves for both. Which converges faster? Which produces a more conservative agent?

6. **Three-Strategy Comparison (45 min):** Evaluate on 1,000 fresh simulated paths (same paths for all strategies):
   - Strategy 1: Naive symmetric market maker (fixed spread = 1.0, no inventory skew)
   - Strategy 2: Avellaneda-Stoikov with optimal parameters from step 3
   - Strategy 3: RL agent (PPO, best reward function from step 5)
   - Report: mean PnL, Sharpe, max drawdown, mean |terminal inventory|, % profitable episodes
   - Produce: overlaid equity curves for 10 representative paths, comparison table, box plots of terminal PnL distribution

7. **Regime Test (30 min):** Generate 500 paths where sigma doubles (from 2 to 4) at the midpoint of the session (step 100 of 200). Run all three strategies. Which one adapts best? Compare: (a) mean PnL in the first half vs. second half, (b) Sharpe in each half, (c) max drawdown in the second half. The hypothesis: RL should outperform A-S in the second half because it can detect the regime change from the state features.

8. **Deliverable:** Complete notebook with: A-S implementation, Monte Carlo results, parameter heatmaps, Gymnasium environment code, RL training curves, three-strategy comparison table and plots, regime test analysis. Include a 400-word section: "When should a market maker use analytics vs. ML?"

### What They'll Discover

- The A-S model produces Sharpe ratios of 1.5-3.0 in the standard simulation — surprisingly high. This is because the simulation is generous: no adverse selection, no competition, symmetric fill probabilities.
- The RL agent improves Sharpe by 10-30% over A-S but has 20-40% worse max drawdown. The improvement comes from adaptive spread adjustment and more aggressive inventory management.
- The 10x inventory penalty produces a dramatically different RL agent: tighter spreads, faster inventory flattening, lower returns but much lower drawdowns. The two agents demonstrate the full spectrum of risk preferences.
- In the regime test, A-S continues quoting with the same spread after sigma doubles (it uses a fixed sigma). The RL agent — if well-trained — detects the volatility increase via the trailing vol state feature and widens its spread within 5-10 steps. This adaptive behavior is the clearest demonstration of RL's value over analytics.

### Deliverable
A complete Jupyter notebook containing: A-S implementation with verification, 10,000-episode Monte Carlo analysis, parameter heatmap, Gymnasium environment (checked), trained PPO agent with two reward functions, 1,000-episode three-strategy comparison, 500-episode regime test, and a written analysis section. All plots should have labeled axes, legends, and titles.

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Avellaneda-Stoikov model (3 formulas) | Demo: implement A-S in ~20 lines, run sample paths | Exercise 1: stress-test A-S under adversarial conditions (adverse selection, vol spike) | At scale: 10,000-episode Monte Carlo analysis with full statistics |
| Parameter sensitivity (gamma, kappa) | Demo: show quote behavior at different parameter values | Exercise 2: full grid search heatmap over gamma x kappa | At scale: two heatmaps (sigma=2 and sigma=4), identify optimal region |
| Gymnasium environment design | Demo: build `MarketMakingEnv`, show random vs. A-S reward | Exercise 3: state representation ablation (5 variants), measure feature importance | Build: validated environment with `check_env`, ready for RL training |
| RL for market making (PPO) | Demo: train PPO, show 3-strategy comparison table | Exercise 4: regime-conditional analysis — where does RL win/lose vs. A-S? | At scale: 1,000-episode comparison, two reward functions, regime test |
| Reward shaping for inventory control | Demo: PnL - phi * inventory^2 concept | Not covered (applied in Exercise 3 implicitly) | Build: compare phi=0.01 vs. phi=0.1, analyze conservative vs. aggressive agents |
| Simulation limitations | Demo: reality vs. simulation comparison table, adverse selection extension | Exercise 1: quantify performance degradation under realistic conditions | Integrate: regime test (vol doubles mid-session) as bridge toward realism |
| Monte Carlo simulation | Demo: build Brownian-Poisson simulator in ~80 lines | Not repeated (provided as starter code) | At scale: 10,000 episodes, 500-episode regime test |

## Key Stories & Facts to Weave In

- **Avellaneda & Stoikov's Paper (2008):** Published in *Mathematical Finance*, it's one of the most cited papers in quantitative finance. Marco Avellaneda was a professor at NYU's Courant Institute and died in 2022. Stoikov went on to found a trading firm. The paper's genius is its simplicity: a complex problem reduced to three formulas that anyone can implement.

- **GameStop, January 2021:** Market makers for GME options faced unprecedented conditions. Implied volatility exceeded 500%. The bid-ask spread on some GME options went from $0.10 to $5.00 — a 50x widening. Several market makers stopped quoting entirely. Citadel Securities maintained quotes throughout, handling 7.4 billion shares in a single day. The lesson: risk management (knowing when to widen) is more important than optimal quoting.

- **Virtu Financial's IPO and the One Losing Day (2015):** The IPO prospectus disclosed 1 losing day out of 1,238 trading days. Their edge isn't prediction — it's speed, scale, and inventory management. They trade about 25% of US equity ETF volume. Average holding period: measured in seconds to minutes. Average profit per trade: a fraction of a penny. Volume makes it work.

- **The Alpha-AS Paper (PLOS ONE 2022):** Falces et al. tested a Double DQN that dynamically tuned A-S parameters on BTC-USD data. Results: RL beat A-S on 24/30 days for Sharpe ratio. The PnL-to-max-adverse-position ratio improved ~20x. But max drawdown worsened. The paper is refreshingly honest about the trade-offs — rare in academic RL papers.

- **Jane Street's Quantitative Puzzle:** Jane Street, one of the world's most successful market-making firms, is famous for hiring based on quantitative puzzles rather than finance knowledge. Their market-making operation generates an estimated $10+ billion in annual revenue. They don't use A-S directly — their models are proprietary — but every market-making model is, at its core, solving the same inventory-spread optimization problem.

- **Knight Capital's $440M Loss (2012, revisited):** The bug that destroyed Knight wasn't in their market-making model — it was in the deployment. Old test code was accidentally activated, causing the firm to act as a market *taker* (buying at the ask, selling at the bid — the opposite of market making) across 147 stocks simultaneously. They lost $440 million in 45 minutes. A market maker's worst nightmare: becoming a market taker by accident.

## Cross-References
- **Builds on:** Week 13 (RL framework — PPO, SAC, Gymnasium, reward shaping — all applied here), Week 15 (Microstructure — understanding why spreads exist, what adverse selection means, how order flow works), Week 7 (PyTorch — the RL agent is an MLP).
- **Sets up:** Week 17 (Crypto & DeFi — Uniswap's AMM is a *different* kind of market maker, mathematically defined by x*y=k; understanding A-S helps you understand why AMMs have impermanent loss), Week 18 (Capstone — a market-making strategy is a legitimate capstone project).
- **Recurring thread:** The "analytical vs. learned" debate reaches its purest form here. A-S is the analytical solution; RL is the learned one. The homework forces students to confront the tradeoff honestly: RL wins on average, A-S wins on worst case. Which matters more depends on who's writing the check.

## Suggested Reading
- **Avellaneda & Stoikov, "High-Frequency Trading in a Limit Order Book" (Mathematical Finance, 2008):** The foundational paper. Shorter than you'd expect (~15 pages) and more readable than most quant papers. The model derivation is elegant. Read it to understand the assumptions and the logic; our lecture simplifies the math without losing the insight.
- **Falces et al., "Alpha-AS: RL to Improve Avellaneda-Stoikov" (PLOS ONE, 2022):** The paper we're replicating (in simplified form). The comparison methodology is well-done: same data, same metric, honest about when RL wins and when it doesn't. Read Section 4 (results) even if you skip the RL details.
- **Gueant, "The Financial Mathematics of Market Making" (2017):** The textbook treatment of the Avellaneda-Stoikov family of models. More mathematical than our lecture but definitive. Read Chapter 2 if you want the full PDE derivation; skim Chapter 5 for extensions to multi-asset market making.
