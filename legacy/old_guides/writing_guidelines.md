# Notebook Writing Guidelines

> **The voice:** A sharp, opinionated practitioner who respects your intelligence but not your
> existing knowledge of finance — someone who tells you stories, shows you code, and never lets
> you forget that real money is on the line.

---

## The Blend We're Emulating

Our narrative voice is a deliberate mix of four influences:

### Nassim Taleb — the attitude

Opinionated, memorable, anchors everything to real catastrophes. Makes you *feel* why fat tails
matter. Intellectually honest about what we don't know. Delivers one-liners that stick:

> *"LTCM had two Nobel laureates, $125 billion in assets, and models that said a loss this big
> wouldn't happen in the lifetime of the universe. It happened in 4 months."*

We borrow: the conviction, the real-world grounding, the refusal to hand-wave.
We leave behind: the tangents, the combativeness, the 3-page digressions about Levantine merchants.

### Michael Lewis — the storytelling

The best in the business at making finance a page-turner. Characters, tension, "holy shit" moments.
*The Big Short*, *Flash Boys*, *Liar's Poker* — he turns abstract mechanisms into human drama:

> *"When Knight Capital deployed buggy code in 2012, they lost $440 million in 45 minutes.
> That's $10 million per minute. The bug? It reactivated old test code that bought high and
> sold low, over and over."*

We borrow: the storytelling instinct, the human stakes, the concrete examples.
We leave behind: the pure-journalist approach (we need math and code too).

### Jeremy Howard (fast.ai) — the pedagogy

The gold standard for technical education. Top-down, code-first, "let me show you before
I explain." Never more than 2 minutes between idea and code. Builds complexity gradually.
Respects the student's time.

We borrow: show-then-explain structure, gradual complexity, no unnecessary abstraction.
We leave behind: nothing — his pedagogical instincts are nearly flawless.

### Our own addition — the ML engineer's bridge

The audience is ML people learning finance, not the other way around. We constantly bridge
to what they already know:

> *"Think of survivorship bias like training on a dataset where all the failed examples have been
> removed. You'd get great training accuracy and terrible real-world performance. That's exactly
> what happens when you backtest on Yahoo Finance data."*

### Who we are NOT emulating

**Lopez de Prado.** Brilliant methodologist, terrible narrator. Dense, academic, assumes
PhD-level math. Writes to impress peers, not to teach students. His *ideas* are central to
this course; his *prose style* is not.

---

## The 10 Rules

### 1. Lead with a Story, Not a Definition

Every major concept gets introduced through an analogy, a historical event, or a "what would
happen if" scenario *before* the formal version appears.

**Bad:**
> "An exchange maintains an order book — a list of buy (bid) and sell (ask) orders:
> Limit order: sits in the book. Market order: executes immediately."

**Good:**
> "Here's something that might surprise you: when you click 'buy' on Robinhood, your order
> doesn't go to 'the stock market.' It goes to Citadel Securities — a single company that
> handles about 25% of all US equity trades. They look at your order, decide whether to fill
> it themselves, and pocket a fraction of a penny for their trouble.
>
> But let's back up. To understand *why* that system exists, we need to understand the thing
> it replaced: the order book. Think of it as two queues standing face to face..."

### 2. Explain the "Why" Before the "What"

Don't define something and then explain why it matters. Flip it — create the *need* first,
then introduce the concept as the *solution*.

**Bad:**
> "Log returns are additive over time: R = R₁ + R₂ + ... This property is useful for
> multi-period analysis."

**Good:**
> "Quick puzzle. You invest $100. It goes up 10% in January — great, you have $110. Then it
> drops 10% in February. Are you back to $100?
>
> Nope. You're at $99. You *lost* money on two moves that should cancel out. This is the
> compounding trap, and it's bitten more junior quants than any bug in production code.
>
> Log returns fix this. Take the log of 1.10, add the log of 0.90, and you get a small
> negative number — honest about the loss. They're the only return type that adds up cleanly
> across time. Let's prove it with code..."

### 3. Anchor Numbers to Real Life

Every time a number appears, ground it in something tangible. A naked number teaches nothing;
a number with context changes how you think.

- *"The bid-ask spread for Apple is about 1 cent on a $190 share — you'd barely notice. For a
  micro-cap stock, it can be 50 cents on a $5 share — that's 10%. Your model needs to be right
  by more than 10% just to break even on a round trip. Most models aren't right by 1%."*

- *"The S&P 500 has excess kurtosis around 20. A Gaussian distribution has 0. What does that
  mean in practice? It means the October 19, 1987 crash — a 22% single-day drop — had a
  probability of roughly 10⁻¹⁶⁰ under Gaussian assumptions. That's not 'unlikely.' That's
  'the universe isn't old enough for this to happen once.' And yet it did."*

- *"A strategy that turns over daily at 10 bps round-trip burns 25% per year in transaction
  costs alone. For reference, the average hedge fund's gross return is about 10-15%. You'd be
  spending twice your expected revenue on shipping costs. That's not a strategy — that's a
  donation to market makers."*

### 4. Use a Consistent Narrative Voice

The tone is a knowledgeable friend explaining things over coffee — not a textbook.

- **Use "we" and "you":** *"When we feed raw prices into an LSTM, we're asking it to learn
  that 150 and 300 are the same stock at different times. That's a lot to ask."* — not
  "When raw prices are fed into an LSTM, the model must learn price-level invariance."
- **Admit what's hard:** *"If fractional differentiation feels weird right now, good — it is
  weird. The idea that you can take the 0.4th derivative of a time series sounds like
  something a mathematician made up to win an argument. But it works, and we'll see why."*
- **Be opinionated:** *"GARCH(1,1) is the only GARCH variant you need. GARCH(2,1),
  EGARCH, GJR-GARCH — they exist, people publish papers about them, and in practice they
  barely beat (1,1). We'll test this ourselves in the seminar."*
- **Okay to be informal:** Contractions, short sentences, occasional dry humor. Never sloppy.
  *"This function returns NaN for weekends. Because, of course, markets are closed on weekends.
  Unless you're trading crypto, in which case nothing is ever closed, sleep is optional, and
  the concept of 'business days' doesn't apply. We'll deal with that in Week 18."*

### 5. Drop "Did You Know?" Moments

Sprinkle real-world facts that make concepts sticky. These are the bits students remember
at the bar three weeks later. Weave them into the narrative naturally — not as boxed
sidebars, but as the kind of thing you'd say mid-explanation with a raised eyebrow:

- When introducing market makers: *"Citadel Securities — one company — executes about 25% of
  all US equity volume. They're not a hedge fund (that's Citadel LLC, different entity). They're
  the plumbing. Every time your Robinhood order fills, there's a good chance it went through
  them. They made $7.5 billion in revenue in 2022. On pennies per share."*

- When discussing survivorship bias: *"Let's play a game. Name the 10 biggest US companies
  from the year 2000. You'll probably remember Microsoft, GE, Walmart. You probably won't
  remember WorldCom (#20 by market cap, filed the largest bankruptcy in US history two years
  later), or Enron (#7 by revenue, ceased to exist by 2002). If your training data starts in
  2005, these companies simply don't exist. Your model will learn that large-cap stocks always
  survive. They don't."*

- When introducing volatility clustering: *"On February 5, 2018, the VIX — Wall Street's 'fear
  gauge' — doubled in a single day. An ETF called XIV, which bet against volatility, lost 96%
  of its value overnight. It had $1.9 billion in assets that morning. It was liquidated within
  the week. The people who bought it thought volatility was low and would stay low. Volatility
  clustering says: the market remembers its shocks. High vol begets high vol."*

- When discussing Python's speed limitations: *"In the time it takes your Python script to
  import NumPy — about 150 milliseconds — a Xilinx FPGA at the NYSE has already processed
  roughly 150,000 market data messages and placed orders on half of them. That's the gap
  we're working with. It's not a gap you close with better code."*

### 6. Build Code Gradually (No "Download Walls")

Never dump a 40-line function and then explain it. The reader's eyes will glaze over
at line 15 and they'll scroll past the whole thing. Instead, build it in front of them.

**Bad:** A single cell with a complete `make_dollar_bars()` function (40 lines of accumulators,
edge-case handling, and index management) followed by "Let's see the output."

**Good — a three-cell sequence:**

> *"Dollar bars are supposed to sample one bar every time a fixed dollar amount trades.
> Let's see if we can build this in 5 lines:"*
>
> ```python
> # Attempt 1: the napkin version
> dollar_vol = spy['Close'] * spy['Volume']
> cum_dollars = dollar_vol.cumsum()
> bar_ids = (cum_dollars // threshold).astype(int)
> dollar_bars = spy.groupby(bar_ids).agg({'Open': 'first', 'High': 'max', ...})
> ```
>
> *"That actually works — 5 lines. But look at the output. We're losing the proper OHLC
> semantics because `groupby` doesn't know which row is first vs. last. And we get one
> giant bar whenever volume dries up over a holiday. Let's fix both..."*
>
> [Next cell: improved version addressing those two issues]
>
> *"Better. Now let's wrap it in a function so we can reuse it all course..."*

### 7. Every Plot Tells a Story

**Before** every visualization — tell the reader what to look for, and what the "boring"
outcome would be. This way the interesting thing *pops*:

> *"We're about to plot the distribution of daily S&P 500 returns against a Gaussian with
> the same mean and standard deviation. If markets were the well-behaved system that most
> textbooks assume, these two curves would overlap perfectly. They won't. Watch the tails."*

**After** every visualization — don't just describe what we see. Explain what it *costs* you:

> *"See those heavy tails? On March 16, 2020, the S&P dropped 12% in a single day. Under
> Gaussian assumptions, that's a 1-in-10²⁵ event — it shouldn't happen once in trillions of
> universe lifetimes. Under the actual distribution, it's maybe a 1-in-500-year event. If
> your risk model uses Gaussian VaR, it told you that day was impossible. Your portfolio
> disagreed."*

### 8. End Sections with "So What?"

After each major concept, a 1-2 sentence bridge that connects the abstract idea to something
concrete — preferably something that will break your model or lose you money if you ignore it:

> *"So here's what survivorship bias does to your model in practice: it learns that 'buy the
> dip' always works — because every stock in its training set eventually recovered. The ones
> that went to zero? They were quietly removed from the dataset years ago. Your model has
> never seen a company die. It will be very confused when one does."*

> *"Why does non-stationarity matter for you specifically? Because if you train an LSTM on
> raw Apple prices from 2015 ($130) and test on 2024 ($190), the model has never seen numbers
> in the test range. It's extrapolating every single prediction. This is the equivalent of
> training an image classifier on cats and testing on dogs — except it's harder to notice."*

### 9. Progressive Disclosure of Jargon

Don't use 5 financial terms in one paragraph. Your reader is a capable ML engineer — they
know what a loss function is, what cross-validation does, what a transformer is. They do NOT
know what "alpha" means, what a "basis point" is, or why anyone cares about "the bid-ask
spread." Introduce one term, make sure it's anchored, then build.

**Bad:**
> "We compute the IC of our alpha signal using cross-sectional Fama-MacBeth regressions on
> decile-sorted portfolios, controlling for the Fama-French 5-factor exposures."

**Good:**
> "We need a way to measure whether our predictions actually rank stocks correctly. In ML
> you'd use Spearman correlation between predicted and actual values. In finance, they call
> the exact same thing the **information coefficient** (IC). An IC of 0.05 sounds pathetic —
> but in a universe of 500 stocks, rebalanced monthly, it's enough to build a career on.
> We'll see why when we get to the Fundamental Law of Active Management in Week 4."

Never assume the reader knows a financial term unless it was explicitly taught in a
prior week of this course. ML terms can be assumed freely.

### 10. The Rhythm of a Good Lecture Section

Each major section follows this pattern:

1. **Hook** — Why should you care? A provocation, a dollar amount, a disaster. (1-2 sentences)
   > *"Every time your model says 'buy,' someone is quietly taking money from you."*
2. **Intuition** — Analogy, story, or "imagine..." scenario. Make it visceral. (1 paragraph)
   > *"Imagine you're buying a used car. The sticker says $20,000. But you can't pay $20,000 —
   > that's the seller's price. The best offer from buyers is $19,500. That $500 gap is the
   > bid-ask spread. On Wall Street, this gap is pennies — but it's taken from you on every
   > single trade, both in and out."*
3. **Formal concept** — Now the actual definition, formula, or framework. Keep it tight.
   The reader is ready for it because they understand *why* it exists.
4. **Code + visualization** — See it working. Build the code gradually (Rule 6), make the
   plot tell a story (Rule 7).
5. **"So what?"** — Bridge to ML, to real money, to the next concept. (1-2 sentences)
   > *"For a model predicting 0.5% daily returns, a 10 bps spread eats 20% of your edge on
   > every trade. This is why transaction cost modeling isn't optional — it's the difference
   > between a backtest that looks great and a strategy that actually makes money."*

---

## Practical Notes

- **Humor:** Dry, earned, and never punching down. The humor comes from the absurdity of
  finance itself — you don't need to add jokes when the VIX just doubled and erased $2 billion.
  *"The efficient market hypothesis says this can't happen. The market says hold my beer."*
  Never forced, never a pun, never an emoji.

- **Length:** Lecture notebooks should be substantial — students are paying (in time) for
  depth. But every paragraph must earn its seat. If a paragraph doesn't teach something,
  motivate something, or connect something to what comes next, cut it. A 30-cell lecture
  that flows is better than a 15-cell lecture that plods.

- **Formulas and LaTeX:** Don't shy away from math — this is a quant course, and the math
  IS the content. LaTeX should appear generously wherever a formula makes the concept click.
  But math must never arrive cold. The pattern is: **intuition → formula → code → insight.**

  **Bad — formula dropped without context:**
  > $\delta_a + \delta_b = \gamma\sigma^2(T-t) + \frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right)$
  >
  > This is the optimal spread in the Avellaneda-Stoikov model.

  **Good — the formula is the punchline of a story:**
  > *"So the market maker has two enemies: volatility (the price moves against you while you
  > hold inventory) and illiquidity (nobody takes the other side of your trade). The optimal
  > spread has to account for both. Avellaneda and Stoikov showed it's exactly:"*
  >
  > $$\delta_a + \delta_b = \underbrace{\gamma\sigma^2(T-t)}_{\text{volatility penalty}} + \underbrace{\frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right)}_{\text{liquidity premium}}$$
  >
  > *"Read that left to right: the first term says 'widen your spread when vol is high or you
  > have a long time left.' The second says 'widen it more when the book is thin (low κ).'
  > That's the entire intuition — two fears, one formula. Let's implement it:"*
  >
  > ```python
  > optimal_spread = gamma * sigma**2 * (T - t) + (2/gamma) * np.log(1 + gamma/kappa)
  > ```

  The goal: a student who reads only the prose understands the *why*. A student who reads
  only the LaTeX understands the *what*. A student who reads both understands the *how*.
  And the code makes it real.

  **When formulas are dense**, break them into named pieces. Don't write one monster equation —
  build it term by term, labeling each piece with `\underbrace` or introducing sub-expressions
  in sequence:
  > *"Let's build the reservation price step by step. Start with mid-price S(t). Now adjust
  > for inventory: if you're long q shares, you want to sell — so your 'true' price is
  > lower than mid:"*
  >
  > $$r(t) = S(t) - q \cdot \gamma \sigma^2 (T - t)$$
  >
  > *"That's it. When q > 0 (long), the reservation price drops below mid — you're eager to
  > sell. When q < 0 (short), it rises above mid — you're eager to buy. The γσ²(T-t) term
  > controls how aggressively: high risk aversion (γ) or high volatility (σ) means you
  > skew harder."*

- **References:** Never a bare citation. Always a one-line "why you'd read this" that makes
  the student *want* to look it up:
  > *"Gu, Kelly & Xiu (2020) — the paper that ended the debate about whether ML works for
  > stock prediction. Tested 8 models on 30,000 stocks over 60 years. Yes, neural nets win.
  > No, not by as much as you'd think. Still the benchmark everyone cites."*

- **Seminar notebooks:** Leaner prose — these are guided exercises, not lectures. But every
  exercise setup should still motivate *why* we're doing it:
  > *"We're about to compare GARCH against your LSTM from the lecture. The question isn't
  > 'which wins' — it's 'by how much, and is it worth the complexity?'"*

- **Homework outlines:** Structured with numbered steps and concrete deliverables — but the
  intro should make the student *want* to do it, not feel like they're reading a government
  form. Frame the homework as a mission, not a checklist:

  **Bad:**
  > "Homework: Fractional Differentiation Study
  > 1. For a universe of 50 US stocks, find the optimal d* per stock.
  > 2. Build a prediction model using three feature sets..."

  **Good:**
  > "### Your mission: prove that fractional differentiation actually works
  >
  > The lecture made a bold claim — that there's a sweet spot between raw prices (too much
  > memory, non-stationary) and returns (stationary but amnesia). You're going to test that
  > claim on 50 real stocks and see if it holds up or if Lopez de Prado was having us on.
  >
  > 1. For a universe of 50 US stocks, find the minimum d* per stock where..."

  Each step should still be clear and unambiguous — the playfulness lives in the framing,
  not in vague instructions. And toss in a teaser for what they'll discover:
  > *"Step 5 is where it gets interesting — you'll see that the 'optimal' d varies wildly
  > across stocks. Defensive utilities need barely any differencing. Meme stocks are another
  > story entirely."*

- **Homework solutions:** Code-focused, but not silent. The student just spent hours on this —
  reward them with insights they couldn't have gotten on their own. Every solution notebook
  should have 3-5 markdown cells that deliver small "aha" moments:

  > *"Look at the Sharpe ratio for the uncertainty-filtered strategy (B) vs. trading everything
  > (A). The improvement isn't huge — maybe 0.2 Sharpe points. But look at the max drawdown.
  > The filtered strategy dodged the COVID crash almost entirely, because model uncertainty
  > spiked in late February 2020 and the filter pulled you out before March hit. That's not
  > a backtest artifact — that's the model saying 'I have no idea what's happening' and your
  > system listening."*

  > *"Here's a result nobody expects: the simple moving average crossover (our 'dumb' primary
  > model) actually has a decent hit rate on its own — about 52%. The meta-labeling model
  > doesn't improve the hit rate much. What it does is dramatically improve the profit factor,
  > because it learns to skip the trades where the primary model is confident but wrong.
  > It's not making you right more often — it's making you wrong less expensively."*

  Solutions should also flag when results are surprising or counterintuitive, and briefly
  explain *why* — these are the moments that build real understanding.
