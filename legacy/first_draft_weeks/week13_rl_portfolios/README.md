# Week 13 — Reinforcement Learning for Portfolio Management

> **Trading is a sequential decision problem. RL is the framework built for sequential decisions. The marriage should be perfect — and it almost is, except for the divorce proceedings called "non-stationarity."**

## Prerequisites
- **Week 3 (Portfolio Theory & Risk):** Mean-variance optimization, equal-weight and minimum-variance portfolios, Sharpe ratio, max drawdown. You need to know what "good portfolio performance" looks like before you try to learn it.
- **Week 7 (Feedforward Nets):** PyTorch basics — building models, training loops, loss functions. RL agents are neural nets; you need to be comfortable building them.
- **Week 5 (Tree-Based Methods):** XGBoost as a benchmark. Your RL agent will be compared against simpler ML baselines, and you need to know how to build those.
- **Week 4 (Linear Models):** Expanding-window evaluation. RL must be evaluated out-of-sample with the same rigor as any other model.
- **General ML:** Familiarity with policy gradient methods is helpful but not required. We'll build intuition from scratch. If you've trained a neural net to classify images, you have the right mental model — we're just changing the loss function from "cross-entropy" to "cumulative reward."

## The Big Idea

Here's the pitch for reinforcement learning in finance, and it's genuinely compelling: every other ML technique we've used in this course treats trading as a series of independent prediction problems. At time *t*, predict the return. At time *t+1*, predict again. Each prediction is made in isolation, as if the previous one never happened. But that's not how trading works. When you buy 1,000 shares of Apple at 9:31 AM, that decision constrains everything that follows. You now have inventory. You have risk. You have a position that needs managing. The next decision isn't "what does the market do?" — it's "given that I'm already long Apple and the market just dropped 0.5%, what do I do *now*?"

Reinforcement learning is the only ML framework that captures this sequential structure natively. An RL agent maintains a *state* (your current portfolio, recent prices, your P&L so far), takes *actions* (buy, sell, hold, rebalance), receives *rewards* (returns, risk-adjusted returns), and learns a *policy* that maps states to actions. It's the Markov Decision Process formalism applied to trading — and on paper, it's perfect.

In practice, it's a beautiful disaster. The "environment" — meaning the market — is non-stationary (the rules change), partially observable (you can't see other participants' orders), and adversarial (other agents are trying to take your money). The reward signal is delayed and noisy — you won't know if today's trade was good until weeks later. Training is unstable — small hyperparameter changes can swing an RL agent from profitable to catastrophic. And sample efficiency is terrible — you might need millions of simulated episodes to learn something a human trader figures out in a week.

So why teach it? Three reasons. First, RL is the theoretically correct framework for trading, and understanding *why* it's hard teaches you more about financial markets than understanding why it's easy. Second, there are specific niches where RL genuinely works: optimal execution (breaking large orders into smaller pieces), market making (which we'll build in Week 16), and options hedging. Third, frameworks like FinRL and Stable-Baselines3 have made it implementable on a laptop — not production-ready, but enough to build intuition and run meaningful experiments. You'll leave this week understanding both the promise and the pain.

## Lecture Arc

### Opening Hook

In 2019, JPMorgan published a paper called "Deep Hedging" that sent a tremor through the derivatives world. The core claim: a reinforcement learning agent, given nothing but option payoffs and transaction costs, learned a hedging strategy that outperformed the Black-Scholes delta hedge — the strategy that had been the industry standard for 46 years. It didn't just learn to replicate Black-Scholes. It learned to *improve* on it, because it could account for transaction costs, discrete rebalancing, and fat-tailed returns — things the analytical solution assumes away. The quant desks at Goldman Sachs and Citadel took notice. So did every ML researcher who'd been told "RL doesn't work in finance." It works. Just not where most people try it first.

### Section 1: Trading as a Markov Decision Process
**Narrative arc:** We establish the formal framework by showing that every trading decision is actually a state-action-reward triple. The tension: this formalism is elegant, but the "transition function" (how the market moves) is the part we can't model. The resolution: we don't need a perfect model of the market — we just need enough samples to learn a decent policy.

**Key concepts:** Markov Decision Process (MDP), state space, action space, reward function, policy, value function, Bellman equation.

**The hook:** Think about what happens when you place a trade on Robinhood. Before you click "buy," you're in some state: you hold certain positions, the market is at certain prices, your account has a certain value. You take an action: buy 10 shares of AAPL. The market moves — that's the transition. You get a reward: your portfolio went up (or down). Then you're in a new state, and the cycle repeats. Every trading session is an MDP. The question is whether we can learn the optimal policy.

**Key formulas:**

Start with the basics. An MDP is defined by (S, A, P, R, gamma):

- **State** s_t: everything the agent knows at time t. For portfolio management: current weights w_t, recent returns, volatility, maybe features from prior weeks.
- **Action** a_t: the target portfolio weights. This is a *continuous* action space — you're not choosing "buy" or "sell," you're choosing "allocate 23% to AAPL, 15% to MSFT, ..."
- **Reward** r_t: the immediate feedback. Simplest version: portfolio return. Better version: risk-adjusted return.
- **Transition** P(s_{t+1} | s_t, a_t): how the world changes. This is the market. We don't know this function. That's the hard part.

The agent's goal: find a policy pi(a|s) that maximizes expected cumulative discounted reward:

$$J(\pi) = \mathbb{E}_\pi \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

In English: find the action-selection rule that maximizes total returns, where future returns are discounted by gamma (because a dollar today is worth more than a dollar tomorrow, and also because we're less certain about distant rewards).

**Code moment:** We'll define a simple `PortfolioEnv` class that inherits from `gymnasium.Env`. Students see the state/action/reward structure in concrete Python code — `reset()` returns an initial state, `step(action)` returns (next_state, reward, done, info). The first version has just 3 stocks and uses historical returns as the "environment." Output: students see that building the environment is 80% of the RL work.

**"So what?":** Every model we've built so far — XGBoost, LSTM, even the GNN — answers the question "what will the market do?" An RL agent answers a harder question: "given what the market might do and where I am right now, what should *I* do?" That's a fundamentally different — and arguably more useful — question.

### Section 2: The Algorithm Zoo — PPO, SAC, and A2C
**Narrative arc:** We survey the key algorithms without drowning in math. The tension: there are dozens of RL algorithms, and choosing wrong can waste weeks of training time. The resolution: for continuous-action portfolio management, you really only need three — and one of them (PPO) wins most of the time.

**Key concepts:** On-policy vs. off-policy learning, policy gradient methods, actor-critic architecture, entropy regularization.

**The hook:** In 2017, OpenAI released Proximal Policy Optimization (PPO). It was designed for robotics — teaching simulated robots to walk, run, and manipulate objects. By 2020, it was the default algorithm for everything from Dota 2 to portfolio optimization. Not because it's mathematically elegant (it's not — it's a clever hack), but because it's stable. In a field where "my agent diverged and lost imaginary billions" is a normal Tuesday, stability is worth more than optimality.

**Key formulas:**

The policy gradient theorem gives us the direction to improve any policy:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_\pi \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A^\pi(s,a) \right]$$

In words: adjust the policy parameters theta to make good actions (positive advantage A) more likely and bad actions (negative advantage) less likely. The advantage function A(s,a) measures "how much better was this action than what I'd normally do in this state?"

PPO's trick is to clip the policy update so it can't change too much in one step:

$$L^{CLIP}(\theta) = \mathbb{E}\left[ \min\left( r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

where r_t(theta) is the probability ratio between the new and old policy. The clip prevents the catastrophic policy collapses that plague vanilla policy gradient methods.

SAC (Soft Actor-Critic) adds entropy regularization — it rewards the agent for being *uncertain*, which promotes exploration. The objective becomes:

$$J(\pi) = \mathbb{E}\left[ \sum_t r_t + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

The alpha * entropy term says: "all else equal, prefer policies that don't commit too hard." For finance, this is surprisingly useful — it prevents the agent from going all-in on a single position.

A2C (Advantage Actor-Critic) is the simplest: synchronous updates, no replay buffer, fast to train but high variance. Think of it as the "quick and dirty" baseline.

**Code moment:** Show a side-by-side comparison of instantiating all three in Stable-Baselines3:

```python
from stable_baselines3 import PPO, SAC, A2C

ppo_agent = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
sac_agent = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4)
a2c_agent = A2C("MlpPolicy", env, verbose=1, learning_rate=7e-4)
```

Three lines each. The complexity is in the environment, not the algorithm. Students see that SB3 abstracts away the algorithm differences — the real engineering is in reward shaping and state representation.

**"So what?":** PPO is your default choice for portfolio management. It's stable, handles continuous actions, and works with the MlpPolicy out of the box. SAC is worth trying when you want more exploration (it sometimes finds clever strategies PPO misses). A2C is your speed baseline — fast training, useful for hyperparameter sweeps. Don't overthink algorithm selection; overthink your reward function instead.

### Section 3: Why RL Is Hard in Finance (The Honest Section)
**Narrative arc:** We've built the excitement — now we deliver the reality check. This section is crucial because RL papers in finance routinely overfit, and students need to understand *why* before they start trusting their own results.

**Key concepts:** Non-stationarity, partial observability, delayed/sparse rewards, distribution shift, reward hacking, training instability.

**The hook:** In 2020, a team from Oxford published an RL agent for portfolio management that achieved a Sharpe ratio of 4.2 in backtest. For context, Renaissance Technologies' Medallion Fund — widely considered the most successful hedge fund in history — averages about 2.5 after fees. The paper was heavily cited. The strategy, of course, doesn't work out-of-sample. Here's why.

**Key formulas:** No new formulas here — this section is about understanding the failure modes.

**The five enemies of financial RL:**

1. **Non-stationarity.** The MDP assumption requires that the transition function P(s'|s,a) is fixed. Markets violate this constantly. The volatility regime of 2017 (VIX at 10) is nothing like 2020 (VIX at 80). An agent trained on calm markets will freeze in a crisis — or worse, double down.

2. **Partial observability.** The agent sees prices, volumes, maybe some features. It doesn't see the pension fund about to liquidate $2 billion in index futures. It doesn't see the Fed's decision before the announcement. The "state" is always incomplete, and the missing information is often the most important.

3. **Delayed rewards.** In Atari, you see the score immediately. In trading, a position opened today might not pay off (or blow up) for weeks. The credit assignment problem — "which of my 200 trades this month actually made money?" — is brutally hard.

4. **Distribution shift.** Your agent is trained on 2015-2020 data. It's tested on 2021-2024. The distribution has shifted: different volatility, different correlations, different market participants. This is the same problem that plagues all ML in finance, but RL is particularly fragile because the policy is optimized end-to-end for the training distribution.

5. **Reward hacking.** If your reward is raw returns, the agent learns to take maximum leverage. If your reward is Sharpe ratio, the agent might learn to trade infrequently (high Sharpe on 3 trades isn't useful). Designing a reward function that produces sensible behavior is an art, not a science.

**Code moment:** Show a training curve where the agent's performance looks great during training, then collapses on out-of-sample data. The visual is the point — the curve should be viscerally disappointing. Output: students see the gap between training and testing performance and understand it's not a bug, it's the fundamental challenge.

**"So what?":** If you take one thing from this section: never trust an RL backtest that doesn't have a proper out-of-sample test on a *different* market regime. The bar for "my RL agent works" is much higher than "my XGBoost model works," because RL has many more degrees of freedom to overfit.

### Section 4: FinRL — Making It Implementable
**Narrative arc:** After the sobering Section 3, we show that practical frameworks exist. FinRL wraps Stable-Baselines3 with financial environments, making it possible to run meaningful experiments in 50 lines of code. The resolution isn't "RL is easy now" — it's "the engineering barrier is gone, so you can focus on the intellectual challenges."

**Key concepts:** FinRL architecture (data layer, environment layer, agent layer), custom environments, action space design, state representation.

**The hook:** FinRL started as a Columbia University research project in 2020. By 2023, it had 10,000+ GitHub stars and was used by quantitative teams at several hedge funds for prototyping. It's not production code — no hedge fund is running FinRL in production — but it solves the "I spent three weeks building a Gymnasium environment and never got to the actual RL" problem that kills most academic projects.

**Key formulas:** FinRL's state space for portfolio management:

$$s_t = [\text{balance}_t, \; \underbrace{p_{1,t}, \ldots, p_{n,t}}_{\text{prices}}, \; \underbrace{h_{1,t}, \ldots, h_{n,t}}_{\text{holdings}}, \; \underbrace{f_{1,t}, \ldots, f_{m,t}}_{\text{features}}]$$

The action space is target portfolio weights: a_t in R^n, normalized to sum to 1 (or less, with cash). Transaction costs are deducted on rebalancing:

$$\text{cost}_t = \sum_{i=1}^{n} c \cdot |h_{i,t+1} - h_{i,t}| \cdot p_{i,t}$$

where c is the cost per dollar traded (typically 5-20 basis points).

**Code moment:** Show the FinRL pipeline from data download to trained agent in ~30 lines:

```python
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

# Data already processed with features
env = StockTradingEnv(df=processed_data, stock_dim=30, ...)
agent = DRLAgent(env=env)
model_ppo = agent.get_model("ppo", model_kwargs={"n_steps": 2048})
trained_ppo = agent.train_model(model=model_ppo, total_timesteps=100_000)
```

The output: students see that FinRL is a thin wrapper around SB3. The magic isn't in the framework — it's in the environment design and evaluation methodology.

**"So what?":** FinRL gets you from "I want to try RL for portfolio management" to "I have a trained agent" in 30 minutes. The remaining 99% of the work is making sure the agent actually learned something useful rather than memorizing the training period.

### Section 5: Where RL Actually Works in Industry
**Narrative arc:** We close the loop from the opening hook. RL has three proven use cases in finance — not the one most people try first (portfolio management), but optimal execution, market making, and options hedging. This section redirects students from the hardest problem to the problems where RL has genuine demonstrated value.

**Key concepts:** Optimal execution (Almgren-Chriss + RL extensions), market making (preview of Week 16), deep hedging (preview of Week 14).

**The hook:** Citadel Securities — which handles about 25% of all US equity volume — uses RL for optimal execution. Not for deciding *what* to trade, but for deciding *how* to trade it. When a pension fund needs to sell $500 million of Apple stock without crashing the price, the execution algorithm decides: how much to trade in each 5-minute interval, whether to use limit or market orders, when to be aggressive and when to be patient. This is an MDP with real money on the line, and RL outperforms the analytical Almgren-Chriss solution by adapting to real-time liquidity conditions.

**Key formulas:**

The Almgren-Chriss optimal execution framework (a preview of Week 15/16 concepts):

Given X shares to sell over T time periods, the optimal execution schedule x_1, ..., x_T minimizes:

$$\text{Cost} = \underbrace{\sum_t x_t \cdot g(x_t)}_{\text{temporary impact}} + \underbrace{\lambda \sum_t q_t^2 \sigma^2}_{\text{timing risk}}$$

where g(x_t) is the market impact of trading x_t shares and q_t is the remaining inventory. The analytical solution gives a deterministic schedule. The RL enhancement: adapt the schedule in real-time based on observed liquidity, spread, and order flow. When the book is thick, trade more. When it's thin, wait.

**Code moment:** No full implementation here — just a teaser diagram showing where RL fits in the execution pipeline. The key visual is a comparison of "static TWAP" (time-weighted average price, equal chunks every interval) vs. "RL adaptive" execution, where the RL agent clusters trades during high-liquidity periods and lays off during thin markets. Output: students see that RL's value isn't predicting where the price goes — it's *reacting intelligently to real-time conditions*.

**"So what?":** Don't start with "RL for stock picking." Start with "RL for doing something specific better than a fixed rule." Execution, market making, and hedging are all well-defined MDPs with clear reward signals. Portfolio management is the *hardest* RL problem in finance, not the easiest — which is why it's the worst place to start and the best place to learn.

### Section 6: Reward Shaping — The Secret Weapon
**Narrative arc:** This is the most practical section. The choice of reward function changes RL agent behavior more than the choice of algorithm. We'll show three reward functions and the dramatically different policies they produce.

**Key concepts:** Reward shaping, sparse vs. dense rewards, reward hacking, multi-objective rewards.

**The hook:** A team at Man AHL (one of the world's largest quantitative hedge funds, managing $50+ billion) found that their RL agent, trained with a simple return-based reward, learned to take maximum leverage on day one and then sit still for the rest of the episode. Sharpe ratio looked great — one giant bet, one giant return, done. This is reward hacking: the agent found a loophole in your objective, and it exploited it with ruthless efficiency. The fix? A penalty for inventory and a running cost for drawdown.

**Key formulas:**

Three reward functions, from naive to sophisticated:

**Reward A — Simple return:**
$$r_t = R_t^{portfolio} = \sum_i w_{i,t} \cdot r_{i,t}$$

Problem: no risk control. Agent learns to maximize leverage.

**Reward B — Differential Sharpe ratio (Moody & Saffell, 1998):**
$$r_t = \frac{\partial \text{Sharpe}_t}{\partial R_t} \approx \frac{B_{t-1} \Delta A_t - \frac{1}{2} A_{t-1} \Delta B_t}{(B_{t-1} - A_{t-1}^2)^{3/2}}$$

where A_t and B_t are exponential moving averages of returns and squared returns. This provides a dense, continuous signal that encourages Sharpe-maximizing behavior at every step — not just at episode end.

**Reward C — Return minus drawdown penalty:**
$$r_t = R_t^{portfolio} - \lambda \cdot \max(0, \text{MaxDD}_t - \text{MaxDD}_{t-1})$$

This penalizes the agent whenever the drawdown deepens. Lambda controls the trade-off: lambda=0 is pure return-seeking, lambda=1 makes the agent extremely drawdown-averse.

**Code moment:** Train the same PPO agent with all three reward functions. Plot the resulting equity curves side by side. The visual tells the whole story: Reward A produces a volatile equity curve that sometimes makes 200% and sometimes loses 80%. Reward B produces smoother but lower returns. Reward C produces the most "investable" equity curve — steady growth with controlled drawdowns. Output: students see that the reward function is the single most important design decision in financial RL.

**"So what?":** In supervised learning, the loss function is usually obvious (MSE, cross-entropy). In RL for finance, the reward function IS your investment philosophy, encoded in math. Getting it wrong doesn't just produce bad predictions — it produces an agent that actively destroys capital in creative ways you didn't anticipate.

### Closing Bridge

You've now seen RL's promise and its pitfalls for portfolio management. The framework is theoretically perfect — trading *is* sequential decision-making — but the environment is the worst possible one for RL: non-stationary, partially observable, with delayed rewards and adversarial dynamics. The practical takeaway: use RL where it has demonstrated value (execution, hedging, market making), be deeply skeptical of RL for general portfolio management, and always benchmark against simpler approaches. Next week, we shift to a domain where neural networks have a cleaner win: derivatives pricing. When you have an analytical solution (Black-Scholes) to benchmark against, and `torch.autograd` gives you Greeks for free, the gap between "ML experiment" and "useful tool" shrinks dramatically.

## Seminar Exercises

### Exercise 1: Building a Portfolio Gymnasium Environment
**The question we're answering:** How do you formalize "buy and sell stocks to make money" as a Markov Decision Process that a computer can solve?

**Setup narrative:** Every RL project starts with the environment, and most RL projects die there. Building a good Gymnasium environment for finance is harder than it sounds — you need to handle portfolio weights that sum to 1, transaction costs that depend on how much you rebalance, and a state representation that gives the agent enough information without drowning it. We're going to build one from scratch in about 40 lines, then use it for the rest of the seminar.

**What they build:** The lecture showed a 3-stock `PortfolioEnv` with basic state. Now scale up: a `PortfolioEnv(gymnasium.Env)` class with 5 stocks from the Dow 30 and an *enriched* state space — current portfolio weights, trailing 20-day returns, volatilities, *and* pairwise correlation features. Action: target weights (continuous, normalized via softmax). Reward: simple portfolio return minus 10 bps transaction costs. Students also implement a state representation ablation: run the environment with three different state designs (raw prices, returns only, normalized features + correlations) and measure the average reward of a random policy under each.

**What they'll see:** The environment runs, the agent takes random actions, and the baseline reward is slightly negative (random rebalancing pays transaction costs without earning returns — the "monkey with a dartboard but you have to pay commissions" scenario). Crucially, the three state representations produce different reward distributions even under a random policy, because the action normalization interacts with the state differently.

**The insight:** The state representation matters enormously. If you include raw prices in the state, the agent can't generalize across time (prices in 2020 look nothing like 2024). If you include only returns, you lose information about absolute position sizes. The right representation is normalized features — and that's the same lesson from Week 2 on stationarity, applied to RL. The ablation makes this concrete rather than abstract: students *measure* the impact of state design before any agent is trained.

### Exercise 2: Reward Function Design Lab
**The question we're answering:** Can you *design* a reward function that produces a target behavior, and then verify it works?

**Setup narrative:** The lecture *demonstrated* that three reward functions produce different equity curves. That was a show. This exercise is the reverse: you start with a *target behavior* (e.g., "the agent should reduce exposure during drawdowns but stay fully invested during calm markets") and engineer a reward function to produce it. This is the practitioner's problem — not "which of three canned rewards is best?" but "how do I encode my investment philosophy in math?"

**What they build:** Students design *two custom reward functions* beyond the three shown in lecture. Suggestions: (a) a Calmar-ratio-inspired reward that penalizes drawdowns relative to returns, (b) a turnover-penalized reward that charges an explicit cost per rebalance, encouraging the agent to trade only when the signal is strong. Train PPO for 50K timesteps with each. Compare all five reward functions (three from lecture + two custom) using a behavioral fingerprint table: Sharpe, max drawdown, mean turnover, average position concentration (HHI of weights), and time spent at max allocation.

**What they'll see:** Custom rewards produce meaningfully different behaviors. The turnover-penalized agent trades 60-80% less than the raw-return agent. The Calmar-ratio agent may discover a barbell allocation (heavy cash + one concentrated bet). The behavioral fingerprint table reveals that reward design is portfolio management by proxy.

**The insight:** The reward function is not a detail — it's the entire specification of what "good trading" means. In industry, quant PMs spend more time on objective design than on algorithm selection. This exercise moves beyond observing that fact (the lecture demo) to practicing the skill of reward engineering.

### Exercise 3: PPO vs. A2C vs. SAC — The Algorithm Comparison
**The question we're answering:** Given a fixed environment and reward function, which RL algorithm produces the best portfolio manager?

**Setup narrative:** Now we fix the reward (differential Sharpe) and vary the algorithm. This is the comparison everyone wants to do first, but we deliberately did reward shaping first because it matters more. Still, algorithm choice has real effects on training stability and final performance.

**What they build:** Three agents — PPO, A2C, SAC — trained on the same environment with the same reward. Training curves (episode reward over time), final performance on a held-out test period, and a stability analysis (train 5 random seeds each, look at the variance).

**What they'll see:** PPO trains most stably — the training curve is smooth and consistent across seeds. SAC sometimes finds better strategies but has higher variance across seeds. A2C is fastest to train but noisiest. On the test set, PPO and SAC are close, A2C lags.

**The insight:** PPO's dominance in financial RL isn't about theoretical superiority — it's about practical stability. When your training run takes 30 minutes and you can't afford to babysit it, "it reliably works" beats "it sometimes works brilliantly."

### Exercise 4: The Honest Evaluation — Does RL Beat Buy-and-Hold?
**The question we're answering:** After all that training, does our RL agent actually outperform a trivial equal-weight buy-and-hold strategy?

**Setup narrative:** This is the exercise that keeps you honest. We've spent the seminar building environments, shaping rewards, comparing algorithms. But the most important question in quantitative finance is always: "does it beat the dumb benchmark?" Equal-weight buy-and-hold requires zero ML, zero training, zero compute. If RL can't beat it, everything we've done is academic.

**What they build:** A comparison of the best RL agent against: (a) equal-weight buy-and-hold, (b) minimum-variance portfolio (from Week 3), (c) 1/N with monthly rebalancing. Full QuantStats comparison on 2 years of out-of-sample data.

**What they'll see:** On Sharpe ratio, the RL agent is competitive but rarely dominant. On max drawdown, the RL agent often does better — it learns to reduce exposure during volatile periods. On total return, buy-and-hold often wins in bull markets because the RL agent's caution costs it upside. The result is mixed, and that's the honest truth.

**The insight:** RL's edge in portfolio management — to the extent it exists — is in *risk management*, not return generation. The agent learns something useful (when to de-risk), but it doesn't learn to predict returns any better than your XGBoost model from Week 5. This is consistent with where RL works in industry: risk-aware decision-making under sequential constraints, not forecasting.

## Homework: "The RL Portfolio Manager Shootout"

### Mission Framing

Here's your mission, and it's the most fun homework in the course if you enjoy watching artificial agents make terrible financial decisions and then slowly learn better ones.

You're going to build three RL agents — PPO, A2C, and SAC — and let them loose on the Dow 30. Thirty blue-chip stocks, daily data from 2010 to 2024. Each agent gets the same data, the same features, and the same reward function. They'll train on 2010-2021, and then you'll test them on 2022-2024 — a period that includes the 2022 bear market, the AI-fueled rally of 2023, and the rate-cut uncertainty of 2024. A period, in other words, that looks nothing like the training data.

The benchmarks are deliberately humbling: equal-weight buy-and-hold, minimum-variance (from Week 3), and your best ML model from Weeks 5 or 7. If your RL agent can't beat a portfolio that requires zero intelligence, you've learned something important. If it can beat the ML benchmark, you've learned something even more important.

The twist is in the reward shaping experiment. You'll train your best agent with two different reward functions — one that cares only about returns, and one that also penalizes drawdowns — and see if the agent's behavior changes in the way you'd predict. Spoiler: it does, and the way it changes is more interesting than you'd expect.

### Deliverables

1. **FinRL Setup (30 min):** Configure a FinRL environment with Dow 30 stocks, daily data 2010-2024. Features: 20-day momentum, 20-day realized volatility, RSI, volume ratio, MACD. Split: 2010-2021 train, 2022-2024 test. Verify the environment runs with random actions and produces reasonable reward distributions.

2. **Train Three Agents (60 min):** Train PPO, A2C, and SAC using Stable-Baselines3 through FinRL. Use total_timesteps=200_000 for each (about 10-15 minutes per agent on M4). Save training curves. Document hyperparameters (learning rate, n_steps, batch size, gamma).

3. **Benchmark Construction (30 min):** Build three benchmarks for the 2022-2024 test period: (a) equal-weight buy-and-hold of the Dow 30, (b) minimum-variance portfolio re-optimized monthly using the method from Week 3, (c) your best ML-based strategy from Weeks 5 or 7 (XGBoost top-5 strategy, or similar). These are your "intelligence benchmarks."

4. **Head-to-Head Evaluation (45 min):** Evaluate all six strategies (3 RL + 3 benchmarks) on 2022-2024. Report: annualized return, Sharpe ratio, max drawdown, average monthly turnover, and total transaction costs at 10 bps per side. Produce a comparison table and overlaid equity curves.

5. **Reward Shaping Experiment (45 min):** Take your best RL agent. Retrain it with two rewards: (a) simple portfolio return, (b) portfolio return - 0.1 * (increase in max drawdown). Compare the two agents' behaviors: equity curves, drawdown profiles, position concentrations, turnover patterns. The question: does the drawdown penalty actually produce a more investable strategy?

6. **Behavior Analysis (30 min):** For your best RL agent, analyze its behavior during 2022 (bear market) and 2023 (bull market). Plot its portfolio weights over time. Does it reduce equity exposure during drawdowns? Does it concentrate or diversify? Does it trade more during volatile periods? The answer to "did the agent learn something?" lives in these plots, not in the Sharpe ratio.

7. **Deliverable:** Complete notebook with training curves, 6-strategy comparison table, equity curves, reward shaping analysis, and behavior plots. Include a 1-page written section: "What did the RL agent actually learn?"

### What They'll Discover

- The RL agents will likely achieve Sharpe ratios between 0.3 and 0.9 on the test set — competitive with but rarely better than equal-weight buy-and-hold (which gets about 0.5-0.7 on the Dow 30 over 2022-2024).
- The drawdown-penalized agent will have 20-40% lower max drawdown than the return-only agent, but also 10-25% lower total return. This is the explicit risk-return tradeoff encoded in the reward.
- The RL agent's most visible "learned behavior" is reducing exposure during volatile periods — it will reduce position sizes in Q1 2022 (the inflation scare) and Q3 2022 (the rate-hike panic). This is genuinely useful.
- SAC will likely have the best performance on one or two random seeds but the worst on others. PPO will be the most consistent. This mirrors the stability argument from the lecture.

### Deliverable
A complete Jupyter notebook containing: FinRL environment configuration, three trained agents with saved training curves, a 6-strategy comparison table with all metrics, overlaid equity curves, reward shaping comparison plots, agent behavior analysis with portfolio weight evolution, and a written analysis section titled "What Did the RL Agent Actually Learn?" (minimum 300 words).

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| MDP formulation for trading | Demo: build simple `PortfolioEnv` (3 stocks) | Exercise 1: scale to 5 stocks with enriched state + ablation study | At scale: Dow 30 environment via FinRL with full feature set |
| RL algorithms (PPO/SAC/A2C) | Demo: side-by-side SB3 instantiation, conceptual comparison | Exercise 3: train all three, compare training curves + stability across seeds | At scale: 200K-timestep training on Dow 30, multi-seed robustness |
| Reward shaping | Demo: three reward functions, equity curve comparison | Exercise 2: *design* custom reward functions to produce target behaviors | Integrate: reward shaping experiment (return-only vs. drawdown-penalized) on full universe |
| Non-stationarity & RL failure modes | Demo: training vs. test performance gap visualization | Not covered (established in lecture) | At scale: 2022-2024 OOS test spanning bear + bull regimes |
| FinRL framework | Demo: 30-line pipeline from data to trained agent | Not covered (done in lecture) | At scale: full FinRL pipeline with Dow 30, custom features |
| Benchmarking RL vs. simple strategies | Demo: teaser comparison concept | Exercise 4: honest RL vs. buy-and-hold evaluation with QuantStats | At scale: 6-strategy comparison (3 RL + 3 benchmarks) with full metrics |
| Agent behavior analysis | Not covered | Not covered (reserved for homework depth) | Build: portfolio weight evolution, regime-adaptive behavior during 2022 bear market |

## Key Stories & Facts to Weave In

- **JPMorgan's Deep Hedging (2019):** The paper that proved RL could beat analytical solutions for derivatives hedging. The key insight: the RL agent accounts for transaction costs and discrete rebalancing, which Black-Scholes ignores. JP Morgan now uses neural-network-based hedging in production for certain exotic derivatives.

- **Medallion Fund and the RL question:** Renaissance Technologies' Medallion Fund averaged 66% annual return before fees from 1988-2018. Nobody knows exactly what they do, but former employees have hinted at reinforcement-learning-like adaptive systems. The mystery: if RL works, why doesn't everyone use it? Answer: they had 30+ years of proprietary data and infrastructure. You have FinRL and free Yahoo data.

- **Knight Capital, August 2012:** A software deployment activated old test RL-like code that bought high and sold low repeatedly. The firm lost $440 million in 45 minutes and was acquired the following week. The lesson for RL practitioners: an agent that performs actions in production can lose real money at machine speed. Guardrails (position limits, drawdown stops) aren't optional.

- **The OpenAI Five precedent (2018):** OpenAI's Dota 2 agents trained on 180 years of gameplay *per day* using thousands of GPUs. Financial RL has maybe 50 years of daily data for a single stock. The sample efficiency gap explains why RL works for games but struggles in finance — you simply can't get enough diverse experience.

- **Man AHL's reward hacking discovery:** The quantitative hedge fund found that their RL agent, given a Sharpe-maximizing reward, learned to take one massive bet and then stop trading entirely. Technically a high Sharpe ratio. Practically useless. This led to their work on multi-objective reward functions that penalize inactivity, extreme positions, and drawdowns simultaneously.

- **Citadel Securities' execution RL:** The market maker uses RL for optimal execution — deciding how to break large orders into pieces. This is the highest-value RL application in finance: the market impact savings on a $1 billion order can be tens of millions of dollars. The agent doesn't predict direction; it minimizes transaction costs given a direction already decided by humans.

- **FinRL's Columbia origins (2020):** Started by Xiao-Yang Liu at Columbia University's Data Science Institute. The framework made financial RL accessible to researchers without deep engineering resources. By 2024, FinRL had been cited in 200+ papers — making it the de facto standard for academic financial RL, even though no one runs it in production.

## Cross-References
- **Builds on:** Week 3 (portfolio optimization — same asset universe, same metrics), Week 7 (PyTorch skills — RL agents are neural nets), Week 5 (XGBoost benchmark — your "intelligence baseline"), Week 4 (expanding-window evaluation — RL needs proper train/test splits too).
- **Sets up:** Week 16 (Market Making with ML — RL is the natural framework for the Avellaneda-Stoikov problem, and this week gives you the tools), Week 14 (Neural Options — deep hedging is RL for derivatives).
- **Recurring thread:** The "does complexity pay?" question from Weeks 5-7 reaches its climax here. RL is the most complex approach in the course. Whether it justifies that complexity against a simple buy-and-hold benchmark is the honest question every student should answer.

## Suggested Reading
- **Rao, "Foundations of Reinforcement Learning with Applications in Finance" (2022, free PDF):** The most accessible bridge between RL theory and financial applications. Written by a former Goldman Sachs managing director. Read Chapters 1-4 for the MDP framework and Chapter 12 for the trading applications. The free PDF makes this a no-excuses read.
- **Hambly, Xu & Yang, "Recent Advances in Reinforcement Learning in Finance" (2023 survey):** A comprehensive survey of RL in finance that's honest about what works and what doesn't. Read the "challenges" section first — it's the best 10 pages on why financial RL is hard.
- **Fischer, "Reinforcement Learning in Financial Markets — A Survey" (2018):** The earlier survey that established the taxonomy. Useful for historical context and for understanding how the field has (and hasn't) progressed since then.
