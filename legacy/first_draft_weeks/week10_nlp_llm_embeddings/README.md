# Week 10 — NLP for Finance: From FinBERT to LLM Embeddings

> **A 768-dimensional vector extracted from a single earnings call sentence predicts next-month returns better than 94 hand-crafted firm characteristics. Welcome to text-based alpha.**

## Prerequisites
- **Week 7 (Feedforward Nets):** PyTorch fundamentals. You'll load pre-trained transformer models, run inference, and extract embeddings using the same PyTorch framework.
- **Week 5 (XGBoost):** Your tree-based alpha model is the baseline. The key experiment this week is whether adding text features to your XGBoost improves IC. You need the existing feature matrix and evaluation pipeline.
- **Week 4 (Linear Models):** Cross-sectional prediction framework — IC, expanding-window CV, long-short portfolios. Text features plug into this framework.
- **Week 9 (Foundation Models):** The embed-then-predict paradigm. Last week you extracted numerical embeddings from time series foundation models. This week you extract embeddings from text. Same idea, different modality.
- **Basic NLP concepts:** Tokenization, embeddings, transformers. If you've used BERT, GPT, or any Hugging Face model, even once, you have what you need.

## The Big Idea

Every model you've built so far is deaf. It sees prices, volumes, momentum, volatility — but it can't hear what the CEO said on the earnings call. It can't read the 10-K filing that mentions "material weakness in internal controls." It can't process the Reuters headline that says "FDA rejects Pfizer's drug application." Yet these text signals move markets instantly, dramatically, and predictably. When Elon Musk tweeted "Tesla stock price is too high imo" on May 1, 2020, Tesla dropped 12% in a single day — $14 billion in market cap, erased by 8 words.

The evolution of financial NLP is a compressed history of the entire NLP field. Era 1 (2000s-2010s): bag-of-words and dictionary-based sentiment. Researchers counted positive and negative words using hand-curated financial dictionaries (Loughran-McDonald). It worked, barely — IC of maybe 0.01-0.02. Era 2 (2019-2022): FinBERT, a BERT model fine-tuned on financial text. Real contextual understanding — it knows that "the company's risk is not immaterial" is negative, while bag-of-words counts "not" and "immaterial" and gets confused. IC of 0.02-0.03. Era 3 (2023-present): LLM embeddings. Don't use the language model to classify sentiment — use it to embed. Take the text, run it through a large model, extract the 768-dimensional embedding vector, and feed that into your XGBoost or neural net. Chen, Kelly, and Xiu showed in 2023 that this approach significantly outperforms everything that came before.

Why do embeddings beat classification? Information bottleneck theory. When FinBERT classifies a headline as "positive," "negative," or "neutral," it compresses all the information in the text into 2 bits. A 768-dimensional embedding preserves much more: the degree of positivity, the topic, the specificity, the urgency, the similarity to other texts. Your downstream model decides what's relevant, not the NLP model. This is the same insight from Week 9: use the foundation model as a feature extractor, not a predictor.

The practical stakes are staggering. Every major quant fund now has an NLP pipeline. RavenPack, the dominant commercial provider of financial text analytics, processes over 500 million news articles per year and sells sentiment scores to hedge funds for six-figure annual subscriptions. Citadel, Two Sigma, and Renaissance Technologies all have teams dedicated to text-based signals. And in 2024-2025, the frontier moved again: agentic AI systems (AlphaGPT from Man Group, RD-Agent from Microsoft Qlib) that use LLMs not just to read text but to generate trading hypotheses, write code, and run backtests autonomously. You're entering this field at the most exciting time in its history.

## Lecture Arc

### Opening Hook

"On November 9, 2020, Pfizer announced that its COVID-19 vaccine was more than 90% effective. The press release hit at 6:45 AM Eastern, before the US market opened. By 9:30 AM, when the opening bell rang, Pfizer was up 9% in pre-market trading. But here's the part that matters for us: the press release was 400 words long. Within those 400 words, an NLP model could have extracted the key signal — 'efficacy exceeds expectations' — and generated a trading signal in under 100 milliseconds. A human reading the same 400 words takes 2 minutes. The entire pre-market move happened in that gap. Today, we build the system that reads the words."

### Section 1: The Three Eras of Financial NLP

**Narrative arc:** We trace the evolution from naive word-counting to contextual understanding to dense embeddings, showing why each transition was necessary and what it unlocked.

**Key concepts:** Bag-of-words, sentiment dictionaries, contextual embeddings, transfer learning.

**The hook:** "The first serious financial NLP paper used a word list. Tim Loughran and Bill McDonald, in 2011, manually compiled a list of words that are negative in financial contexts: 'liability,' 'litigation,' 'impairment,' 'restructuring.' They counted these words in 10-K filings and found that the count predicted future returns. IC? About 0.01. It took two professors years to compile the list, and the predictive power was barely distinguishable from noise. Fifteen years later, a pre-trained language model achieves 5-10x that IC by reading the same filings — and it wasn't even designed for finance."

**Era 1: Dictionary-Based Sentiment (2000s-2015)**

The Loughran-McDonald dictionary contains ~2,700 negative words and ~350 positive words specific to finance. Crucially, it differs from general-purpose sentiment dictionaries: "liability" is negative in finance but neutral in everyday language. "Outstanding" is positive in everyday language but means "unpaid" in finance ("outstanding debt").

Sentiment score:

$$S_{\text{LM}} = \frac{N_{\text{positive}} - N_{\text{negative}}}{N_{\text{total words}}}$$

"The problem is obvious: 'The company did NOT experience any material losses' has zero negative words if you don't handle negation. And this sentence from an SEC filing — 'Risk factors include but are not limited to: litigation, regulatory changes, competition, and macroeconomic uncertainty' — has 4 negative words but is simply a standard disclosure that every company includes. Dictionary methods can't distinguish informative negative language from boilerplate."

**Era 2: FinBERT (2019-2022)**

FinBERT is BERT (110M parameters) fine-tuned on financial text (Financial PhraseBank, 4,840 sentences labeled by 16 finance experts). It outputs three probabilities: positive, negative, neutral.

"FinBERT understands context. It knows that 'revenue exceeded expectations' is positive and 'the loss exceeded expectations' is negative — same sentence structure, opposite meaning. Dictionary methods would see 'exceeded expectations' in both and give the same score. FinBERT achieves about 87% accuracy on Financial PhraseBank, vs. 72% for dictionary methods."

**Era 3: LLM Embeddings (2023-present)**

"Chen, Kelly, and Xiu's insight: don't classify. Embed. Instead of asking 'is this positive or negative?' extract the full representational vector and let the downstream model figure out what matters."

**Code moment:** All three eras in code:

```python
# Era 1: Dictionary
from collections import Counter
neg_words = load_loughran_mcdonald_negative()
score = sum(1 for w in text.split() if w.lower() in neg_words) / len(text.split())

# Era 2: FinBERT classification
from transformers import pipeline
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
result = finbert("Revenue increased 15% year-over-year")  # {'label': 'positive', 'score': 0.97}

# Era 3: LLM embedding
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("Revenue increased 15% year-over-year")  # shape: (384,)
```

**"So what?":** "Each era increased the information content of the text signal. Dictionary: 1 number (sentiment score). FinBERT: 3 numbers (positive/negative/neutral probabilities). Embeddings: 384-768 numbers (a dense representation of everything in the text). More information means your downstream model has more to work with. The IC improvements follow directly."

### Section 2: FinBERT — Your First Financial NLP Model

**Narrative arc:** We build a FinBERT pipeline end-to-end, from raw headlines to stock-level sentiment scores to IC evaluation.

**Key concepts:** Financial PhraseBank dataset, sentence-level classification, aggregation to stock-day level, evaluation against returns.

**The hook:** "FinBERT is the baseline you need to beat, and it's a higher bar than you might expect. It was published in 2019 by Aristo Doğan, Henry Born, and Ruy Ribeiro at ProsusAI, and it's become the default NLP tool in quantitative finance. Every NLP paper in finance benchmarks against it. It's fast (under 1 second for 100 headlines on CPU), it's accurate (87% on Financial PhraseBank), and it's free (MIT license, Hugging Face). Your LLM embeddings need to beat this to justify their existence."

**Key concepts unpacked:**

**Financial PhraseBank:** 4,840 sentences from English-language financial news, each labeled positive/negative/neutral by 5-8 finance experts. Agreement threshold: >75% of annotators must agree. This is the standard benchmark for financial sentiment classification.

**Aggregation problem:** You get one sentiment score per headline. A stock might have 0-10 headlines per day. How do you aggregate to a stock-day level?

$$\text{Sentiment}_{i,t} = \frac{1}{|\mathcal{H}_{i,t}|} \sum_{h \in \mathcal{H}_{i,t}} (P_{\text{pos}}(h) - P_{\text{neg}}(h))$$

"Average the positive-minus-negative score across all headlines for stock $i$ on day $t$. On days with no headlines, use 0 (neutral). Simple, but it ignores headline importance — a CEO resignation headline should weigh more than a routine analyst update. The LLM embedding approach avoids this problem entirely by preserving the full information in each headline."

**Code moment:**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def score_headlines(headlines):
    """Score a batch of headlines with FinBERT."""
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)  # [positive, negative, neutral]
    # Sentiment = P(positive) - P(negative)
    return (probs[:, 0] - probs[:, 1]).numpy()

scores = score_headlines([
    "Apple reports record quarterly revenue of $123.9 billion",
    "Boeing faces new safety concerns after door plug blowout",
    "Fed holds interest rates steady as expected"
])
# Expected: [0.85, -0.72, 0.02] (approximately)
```

**"So what?":** "FinBERT gives you a single number per headline. That number is useful — IC of 0.02-0.03 on next-day returns — but it's compressing an entire sentence into one dimension. The sentence 'Apple reports record quarterly revenue driven by strong iPhone demand in China despite regulatory headwinds' contains information about revenue growth, product mix, geographic exposure, and risk factors. FinBERT sees all of this and outputs: 'positive.' That's a lot of information destroyed."

### Section 3: The LLM Embedding Revolution — Chen, Kelly, Xiu (2023)

**Narrative arc:** The key paper of this week. We explain what Chen-Kelly-Xiu did, why it works, and what it means for the field.

**Key concepts:** Text embeddings, information bottleneck, dimensionality reduction (PCA), cosine similarity.

**The hook:** "In 2023, Chen, Kelly, and Xiu published 'Expected Returns and Large Language Models.' The title is deliberately provocative — it puts 'returns' and 'LLMs' in the same breath, which six months earlier would have been laughable. Their result: LLM embeddings of financial text significantly outperform all prior NLP methods — including FinBERT — for cross-sectional return prediction. The key: they didn't use the LLM to predict. They used it to embed. The LLM turns text into a 768-dimensional vector. That vector becomes a feature for their prediction model. The prediction model is... a standard linear regression."

**The information bottleneck argument:**

"Why do embeddings beat classification? Think of it this way. Classification forces the NLP model to make a decision: 'This text is positive.' All the nuance — how positive, in what dimension, about which aspect of the business — is discarded. The downstream model never sees it.

Embeddings preserve the full representational power of the language model. A 768-dimensional vector can encode: the topic (earnings vs. litigation vs. product launch), the magnitude (record revenue vs. slight increase), the specificity (concrete numbers vs. vague outlook), the temporal scope (this quarter vs. long-term guidance), and much more. Your downstream model — XGBoost, Ridge, neural net — gets to decide which dimensions are predictive of returns."

Formally, if $z = f_{\text{classify}}(\text{text}) \in \{0, 1, 2\}$ is the classification output and $e = f_{\text{embed}}(\text{text}) \in \mathbb{R}^{768}$ is the embedding, then by the data processing inequality:

$$I(r; e) \geq I(r; z)$$

where $r$ is the future return and $I(\cdot;\cdot)$ is mutual information. The embedding preserves more information about the text, so it preserves at least as much information about returns. In practice, the gap is large.

**Key formulas:**

The embedding-based prediction model:

$$\hat{r}_{i,t+1} = \beta_0 + \boldsymbol{\beta}^T \cdot \text{PCA}_k(e_{i,t}) + \gamma^T x_{i,t}$$

where $e_{i,t}$ is the embedding of stock $i$'s text at time $t$, PCA reduces it from 768 to $k$ dimensions (typically 10-20), and $x_{i,t}$ are traditional features (momentum, size, etc.).

"PCA is essential. You can't feed 768 features into a regression model trained on 200 stocks per month — you'd overfit catastrophically. PCA reduces the embedding to 10-20 orthogonal dimensions that capture the most variance. Chen-Kelly-Xiu found that 20 PCA components captured about 80% of the return-relevant information in the embeddings."

**Code moment:**

```python
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Embed with a sentence transformer (runs on M4 CPU in seconds)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(headlines)  # shape: (n_headlines, 384)

# Aggregate to stock-day level (average embeddings for same stock-day)
stock_day_emb = embeddings_df.groupby(['ticker', 'date']).mean()

# PCA reduction
pca = PCA(n_components=20)
emb_pca = pca.fit_transform(stock_day_emb)

# Combine with traditional features
hybrid_features = np.concatenate([traditional_features, emb_pca], axis=1)
```

**"So what?":** "Chen-Kelly-Xiu showed that adding 20 PCA components of LLM embeddings to a standard cross-sectional model increased out-of-sample R² from about 0.40% to 0.55% — a 38% improvement. That's equivalent to adding a new, independent alpha factor on top of 94 existing ones. No factor discovered in the last decade has produced that magnitude of improvement. And it comes from a pre-trained model that required zero financial training data."

### Section 4: Practical Embedding Approaches — What You Can Run Today

**Narrative arc:** We survey the implementable options, from free and fast (sentence-transformers) to paid and powerful (OpenAI/Anthropic APIs) to bleeding-edge (FinGPT fine-tuning).

**Key concepts:** Sentence-transformers, API-based embeddings, LoRA fine-tuning, compute/cost tradeoffs.

**The hook:** "You have three options for text embeddings, and they span a 1000x cost range. Option A: `sentence-transformers` with `all-MiniLM-L6-v2` — free, runs on your M4 in seconds, 384 dimensions, no API key needed. Option B: OpenAI's `text-embedding-3-small` — $0.02 per million tokens, 1536 dimensions, requires API key. Option C: FinGPT with LoRA fine-tuning — about $300 in compute to fine-tune, produces finance-specific embeddings. Which one should you use? Let's find out."

| Approach | Dimensions | Cost (1M headlines) | Quality (IC) | Runs Locally? |
|----------|-----------|---------------------|-------------|---------------|
| sentence-transformers (MiniLM) | 384 | $0 | 0.035 | Yes (M4) |
| sentence-transformers (mpnet) | 768 | $0 | 0.040 | Yes (M4) |
| OpenAI text-embedding-3-small | 1536 | $0.02 | 0.045 | No (API) |
| OpenAI text-embedding-3-large | 3072 | $0.13 | 0.048 | No (API) |
| FinGPT (LoRA fine-tuned) | 768-4096 | ~$300 one-time | 0.050+ | Yes after training |

"The IC differences are small. For a homework, `sentence-transformers` is the right choice — free, fast, no API dependency. For a production system, the marginal IC from larger models might justify the cost. FinGPT requires significant setup but gives you the most finance-specific embeddings."

**Code moment:** All three approaches side by side:

```python
# Option A: Free, local, fast
from sentence_transformers import SentenceTransformer
model_a = SentenceTransformer('all-MiniLM-L6-v2')
emb_a = model_a.encode(headlines)  # 384-dim, ~5000 headlines/sec on M4

# Option B: Paid API
import openai
response = openai.embeddings.create(
    model="text-embedding-3-small",
    input=headlines[:100]  # batch of 100
)
emb_b = [r.embedding for r in response.data]  # 1536-dim

# Option C: Hugging Face model (larger, better, slower)
model_c = SentenceTransformer('all-mpnet-base-v2')
emb_c = model_c.encode(headlines)  # 768-dim, ~1000 headlines/sec
```

**"So what?":** "The embedding quality matters less than you think. The gap between the cheapest and most expensive option is about 0.015 IC. The gap between using embeddings vs. not using them is 0.035+ IC. In other words, the decision to use embeddings at all is 2x more important than the decision of which model to use. Start with `sentence-transformers`. Upgrade later if the signal is there."

### Section 5: From Text to Trading Signal — The Aggregation Problem

**Narrative arc:** Raw embeddings are per-headline. Trading signals are per-stock-day. Bridging this gap is non-trivial and where much of the alpha is won or lost.

**Key concepts:** Temporal aggregation, attention-weighted aggregation, decay-weighted averages, handling missing text days.

**The hook:** "Apple has 50 news articles per day. A small-cap biotech has 0-2. If you average embeddings per stock-day, Apple's signal is well-estimated and the biotech's is noisy or missing. If you fill missing days with zeros, you're telling the model 'no news means neutral.' Is that true? Sometimes no news IS the signal — companies that suddenly stop generating headlines might be hiding bad news."

**Key approaches:**

**Simple average (baseline):**
$$e_{i,t} = \frac{1}{|\mathcal{H}_{i,t}|} \sum_{h \in \mathcal{H}_{i,t}} f_{\text{embed}}(h)$$

**Exponentially decay-weighted average (better):**
$$e_{i,t} = \frac{\sum_{s \leq t} \sum_{h \in \mathcal{H}_{i,s}} \lambda^{t-s} \cdot f_{\text{embed}}(h)}{\sum_{s \leq t} |\mathcal{H}_{i,s}| \cdot \lambda^{t-s}}$$

where $\lambda \in (0, 1)$ controls the decay rate. "Yesterday's news is more relevant than last week's. A decay of $\lambda = 0.9$ gives yesterday 10x the weight of news from 10 days ago."

**Cosine similarity to prototypes:**
"A clever trick from Chen-Kelly-Xiu: instead of using the raw embedding, compute its cosine similarity to a set of 'prototype' embeddings. The prototypes represent canonical financial narratives: 'earnings beat,' 'earnings miss,' 'regulatory action,' 'management change,' 'product launch.' This reduces the embedding from 768 dimensions to 10-20 interpretable similarity scores."

$$s_{i,t,k} = \frac{e_{i,t} \cdot p_k}{\|e_{i,t}\| \cdot \|p_k\|}$$

where $p_k$ is the embedding of prototype sentence $k$.

**Code moment:**

```python
# Prototype-based features
prototypes = [
    "Company reports record quarterly earnings above analyst expectations",
    "Company announces significant layoffs and cost restructuring",
    "FDA approves company's new drug application",
    "SEC launches investigation into company's accounting practices",
    "Company announces strategic acquisition of competitor",
]
proto_embs = model.encode(prototypes)

# Cosine similarity to each prototype
from sklearn.metrics.pairwise import cosine_similarity
features = cosine_similarity(stock_embeddings, proto_embs)  # shape: (n_stocks, 5)
```

**"So what?":** "The aggregation method determines how much of the text signal actually reaches your prediction model. Simple averaging is a reasonable baseline. Decay-weighting captures the temporal structure of news. Prototype similarity gives you interpretable features. In practice, try all three and keep whatever works best on your validation set."

### Section 6: Signal Decay — How Long Does Text Alpha Last?

**Narrative arc:** Text signals don't last forever. We quantify the decay rate and show that it varies dramatically between FinBERT sentiment and LLM embeddings.

**Key concepts:** IC by horizon, signal decay, half-life of alpha, fast vs. slow information.

**The hook:** "A Reuters headline that says 'Pfizer's drug fails Phase 3 trial' moves the stock in minutes. An analyst's nuanced 40-page report on the company's long-term competitive position takes weeks to be fully reflected in prices. FinBERT captures the headline — fast, binary, decays quickly. LLM embeddings capture the nuance — slower, richer, decays more slowly. The decay rate of your text signal determines how often you need to trade and how much transaction costs eat into your alpha."

**Key formulas:**

IC as a function of forecast horizon:

$$\text{IC}(h) = \text{corr}_{\text{rank}}(\text{signal}_{t}, r_{t \to t+h})$$

Plot IC(h) for h = 1, 2, 5, 10, 20 days. For FinBERT sentiment, IC decays by ~50% from 1-day to 5-day horizon. For LLM embeddings, the decay is slower — maybe 30% over the same window.

"FinBERT captures the immediate reaction. LLM embeddings capture the slow-digesting information. If you're trading daily, FinBERT might be fine. If you're trading monthly, you need the richer representation."

**Code moment:**

```python
# IC by horizon
for horizon in [1, 2, 5, 10, 20]:
    future_return = returns.shift(-horizon)
    ic = signal.corrwith(future_return, method='spearman').mean()
    print(f"Horizon {horizon:2d} days: IC = {ic:.4f}")
```

Expected output pattern:
```
Horizon  1 days: IC = 0.035  (FinBERT: 0.025)
Horizon  2 days: IC = 0.032  (FinBERT: 0.020)
Horizon  5 days: IC = 0.028  (FinBERT: 0.014)
Horizon 10 days: IC = 0.022  (FinBERT: 0.008)
Horizon 20 days: IC = 0.015  (FinBERT: 0.003)
```

**"So what?":** "Signal decay determines trading frequency, which determines transaction costs, which determines whether your alpha is real. A fast-decaying signal (FinBERT) requires high-frequency trading, eating 50+ bps/year in transaction costs. A slow-decaying signal (LLM embeddings) can be traded monthly at 10 bps/year. The slower signal often has higher net alpha after costs, even if the gross IC is similar."

### Section 7: Agentic AI for Quant Research — The 2025 Frontier

**Narrative arc:** We briefly survey the bleeding edge — LLMs that don't just read text but generate hypotheses, write code, and run backtests autonomously.

**Key concepts:** AlphaGPT (Man Group), RD-Agent (Microsoft/Qlib), automated factor discovery, research automation.

**The hook:** "In 2024, Man Group — the world's largest publicly traded hedge fund ($170B+ AUM) — published AlphaGPT, a system where an LLM proposes new trading factors, an evaluator tests them, and the LLM iterates based on the results. In parallel, Microsoft's Qlib team released RD-Agent, which automates the entire research-develop cycle: read a paper, extract the key idea, implement it in code, backtest it, and report whether it works. The quant researcher's job is not being automated away — but the tedious parts of it are. Understanding this frontier is not optional for anyone entering this field."

**The landscape:**

| System | Organization | What It Does | Status |
|--------|-------------|-------------|--------|
| AlphaGPT | Man Group | LLM proposes factors, evaluator tests them, LLM iterates | Published 2024, production use |
| RD-Agent | Microsoft Qlib | Reads papers, implements ideas, backtests, reports | Open-source (GitHub), active development |
| FINMEM | Academic | LLM with layered memory for trading decisions | Research prototype |
| FinGPT | Open-source | Finance-focused LLM fine-tuning framework | Community project, ~$300 fine-tuning |

"We discuss these conceptually, not as homework. Building an agentic system is beyond the scope of a single week. But understanding that these systems exist — and that they work well enough for Man Group to deploy — changes how you think about the value chain in quant research. The future quant doesn't write code from scratch. The future quant supervises AI agents that write code, evaluates their output, and adds the human judgment that the agents can't."

**"So what?":** "Agentic AI is the future of quant research, but it's not yet the present for most teams. The practical takeaway: learn to use LLMs as tools for YOUR research workflow — embedding text, generating feature ideas, debugging code, reviewing papers. The full agentic loop will come later."

### Closing Bridge

"Your models can now see prices, volumes, and text. They process cross-sectional snapshots (feedforward nets), temporal sequences (LSTMs), pre-trained representations (foundation models), and natural language (NLP). But every prediction so far has been a point estimate — a single number with no uncertainty attached. Your model says 'buy AAPL with predicted return +1.5%.' How confident is it? Is this a high-conviction prediction based on clear signals, or a noisy guess in a confusing regime? Without uncertainty quantification, you can't tell the difference — and you'll size both positions the same way, which is exactly wrong. Next week, we add the missing dimension: uncertainty. The tool turns out to be surprisingly simple — you already have it. It's called dropout."

## Seminar Exercises

### Exercise 1: FinBERT Failure Taxonomy — Where Does Sentiment Classification Break?
**The question we're answering:** The lecture demonstrated that FinBERT achieves ~87% accuracy overall. But overall accuracy masks systematic failure patterns. What types of financial language does FinBERT get wrong, and can you predict when to trust it?
**Setup narrative:** "The lecture showed you FinBERT works on average. A portfolio manager doesn't care about averages — they care about the trades that go wrong. You're going to dissect the 13% error rate and build a taxonomy of failure modes."
**What they build:** Score all 4,840 Financial PhraseBank sentences with FinBERT. Separate the errors into categories: (a) negation failures ("not unlikely to face challenges"), (b) hedging/uncertainty ("may potentially see modest improvement"), (c) mixed sentiment ("revenue grew but margins contracted"), (d) financial jargon misread ("outstanding debt" classified as positive). Compute accuracy per category. Build a simple rule-based "confidence filter" that flags sentences likely to be misclassified (e.g., contains negation + hedge words).
**What they'll see:** Accuracy on negation sentences: ~65%. On hedging: ~70%. On mixed sentiment: ~72%. On straightforward positive/negative: ~94%. The confidence filter correctly flags ~40% of misclassifications.
**The insight:** FinBERT's errors are systematic and predictable. Knowing WHEN the classifier fails is almost as valuable as the classifier itself. This motivates embeddings (which preserve nuance) and confidence-filtered pipelines (which avoid low-quality signals).

### Exercise 2: Embedding Headlines with sentence-transformers
**The question we're answering:** Can a generic sentence transformer produce useful financial features?
**Setup narrative:** "This model has never seen a 10-K filing. It was trained on Reddit, Wikipedia, and StackOverflow. We're asking it to understand 'the company reported EBITDA above consensus.' Let's see."
**What they build:** Embed 1,000 Financial PhraseBank sentences with `all-MiniLM-L6-v2`. Visualize with t-SNE. Color by sentiment label. Compute IC: FinBERT sentiment vs. PCA of embeddings.
**What they'll see:** The t-SNE plot shows clear clustering by sentiment — even the generic model separates positive from negative. But within the "positive" cluster, there are sub-clusters (earnings, revenue, growth, acquisitions) that FinBERT would collapse into a single "positive" score.
**The insight:** Even a generic model captures financial text structure. A finance-specific model would be better, but the generic model is surprisingly useful.

### Exercise 3: Text Features + XGBoost — Does Text Help?
**The question we're answering:** Does adding text features to your Week 5 XGBoost improve out-of-sample IC?
**Setup narrative:** "This is the money exercise — literally. You're about to test whether all this NLP work translates to better stock predictions."
**What they build:** Take Week 5 XGBoost pipeline. Add FinBERT sentiment and PCA embeddings as features. Train four variants: (A) price/volume only, (B) +FinBERT sentiment, (C) +PCA embeddings, (D) +both. Compare OOS IC.
**What they'll see:** Model A: IC ~ 0.038. Model B: IC ~ 0.040. Model C: IC ~ 0.044. Model D: IC ~ 0.045. Embeddings add more than sentiment. Combining both adds marginally more.
**The insight:** Text features are additive — they provide information that price/volume features don't capture. Embeddings are strictly better than sentiment as features, because they contain the sentiment information plus additional dimensions. The improvement from B to C is the value of preserving information.

### Exercise 4: Information Bottleneck Visualization
**The question we're answering:** Why do embeddings outperform classification?
**Setup narrative:** "We're going to prove the information bottleneck argument visually. If embeddings preserve more return-relevant information than sentiment labels, we should be able to see it."
**What they build:** For the same headlines, compare: (a) cosine similarity between embeddings vs. (b) agreement in FinBERT labels. Show that headlines with similar embeddings have more similar future returns than headlines with the same FinBERT label.
**What they'll see:** Among "positive" headlines, embedding cosine similarity to a "strong earnings" prototype predicts returns better than the raw "positive" label. The embedding distinguishes between types of positivity.
**The insight:** Classification compresses information. Embeddings preserve it. This is the fundamental reason embeddings win.

## Homework: "Text Alpha: FinBERT vs. LLM Embeddings"

### Mission Framing

Text-based alpha is now the fastest-growing signal category at quantitative firms. RavenPack charges six figures per year for text signals. Bloomberg spends hundreds of millions on NLP infrastructure. You're about to build a version of what they sell — using free data and open-source models — and evaluate it with the same rigor used at production quant funds.

Your mission has two parts. Part 1: build the text pipeline — download headlines, score them with FinBERT, embed them with sentence-transformers, aggregate to stock-day level. Part 2: the shootout — add each text feature set to your existing XGBoost from Week 5 and measure the marginal IC improvement. The question isn't whether text helps (it does — decades of academic evidence). The question is how much it helps, which approach works best, and how fast the signal decays.

If you do this well, you'll have a working text alpha pipeline that could be deployed on real data with minimal modifications. The gap between this homework and a production system is data quality (you're using Financial PhraseBank; they use real-time news feeds) and scale (you're doing 50 stocks; they do 5,000) — not methodology.

### Deliverables

1. **Data preparation.** Download the Financial PhraseBank dataset (available from Hugging Face or direct download). It contains 4,840 sentences from English-language financial news, labeled positive/negative/neutral. Additionally, download SEC EDGAR 8-K filings for 20 stocks using the SEC API or `sec-edgar-downloader`. Extract the first 2-3 sentences from each filing (the "item" description). This gives you a mix of news headlines and regulatory filings.

2. **FinBERT baseline.** Score all headlines/filings with FinBERT. Aggregate to daily stock-level sentiment using simple average. Compute IC against next-day, next-5-day, and next-20-day returns.

3. **LLM embeddings.** Embed all headlines/filings with `sentence-transformers` (use `all-MiniLM-L6-v2` for speed or `all-mpnet-base-v2` for quality). Apply PCA to reduce from 384/768 dimensions to 10 and 20 components. Aggregate to daily stock-level by averaging embeddings, then applying PCA.

4. **Feature comparison.** Using your Week 5 XGBoost pipeline with expanding-window CV, train and evaluate four models:
   - Model A: price/volume features only (your Week 5 baseline)
   - Model B: price/volume + FinBERT sentiment
   - Model C: price/volume + PCA embeddings (20 components)
   - Model D: price/volume + FinBERT + PCA embeddings
   Report OOS IC, R², and long-short portfolio Sharpe for each.

5. **Signal decay analysis.** For both FinBERT sentiment and embedding-based features, compute IC as a function of horizon (1, 2, 5, 10, 20 days). Plot the decay curves. Which signal decays faster? What does this imply for trading frequency?

6. **Attention to data timing.** Ensure you're not using text data before it was actually available. Headlines must be time-stamped, and the prediction target must be the return AFTER the headline publication time. This is the financial NLP equivalent of the temporal splitting from Week 7.

7. **Deliverable:** Notebook + comparison table + signal decay plot + 1-paragraph analysis of when and why each text approach adds value.

### What They'll Discover

- FinBERT sentiment is a real signal: IC of 0.02-0.03 on next-day returns. It's weak but consistently positive across stocks.
- LLM embeddings (PCA to 20 dimensions) achieve IC of 0.03-0.04 — roughly 50% better than FinBERT sentiment. The improvement comes from preserving information that classification destroys.
- Adding text features to XGBoost improves IC by 0.005-0.010 on top of price/volume features. This is a meaningful improvement — equivalent to adding 2-3 new "good" factors.
- FinBERT signal decays by ~50% from 1-day to 5-day horizon. Embedding signal decays by ~30%. Embeddings capture slower-digesting information that takes longer to be reflected in prices.
- The marginal value of adding both FinBERT AND embeddings (Model D vs. C) is small — the embeddings already contain the sentiment information. FinBERT is largely redundant once you have embeddings.

### Deliverable

Final notebook: `hw10_text_alpha.ipynb` containing the full text pipeline, all model comparisons, signal decay analysis, and written conclusions.

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Dictionary-based sentiment (Loughran-McDonald) | Demo: show word-counting code, explain financial-specific vocabulary | Not covered (done in lecture) | Integrate: conceptual understanding for context |
| FinBERT classification | Demo: score 3 example headlines, explain 87% accuracy, show aggregation formula | Exercise 1: full failure taxonomy — categorize errors by type, build confidence filter | At scale: score all headlines/filings, aggregate to stock-day, compute IC across horizons |
| LLM embeddings (sentence-transformers) | Demo: show `encode()` code, explain information bottleneck theory with formulas | Exercise 2: t-SNE visualization, discover sub-clusters within sentiment labels | At scale: embed all text, PCA reduce to 10/20 dims, aggregate to stock-day level |
| Embedding vs. classification (information bottleneck) | Demo: explain data processing inequality, show 3 eras side-by-side | Exercise 4: prove visually — cosine similarity predicts returns better than sentiment labels | Integrate: compare Model B (FinBERT) vs. Model C (embeddings) to demonstrate empirically |
| Aggregation methods (average, decay-weighted, prototypes) | Demo: show three approaches with formulas, explain cosine similarity to prototypes | Not covered (done in lecture) | Integrate: use chosen aggregation in production pipeline |
| Text features + downstream models (XGBoost) | Demo: conceptual pipeline, expected IC improvement table | Exercise 3: train 4 XGBoost variants (price-only, +FinBERT, +embeddings, +both) | At scale: full expanding-window CV with all model variants, report IC/R²/Sharpe |
| Signal decay analysis | Demo: show IC-by-horizon code, expected decay pattern | Not covered (foreshadowed for homework) | Build: compute IC at horizons 1-20 days, plot decay curves for FinBERT vs. embeddings |
| Agentic AI (AlphaGPT, RD-Agent) | Demo: conceptual survey, landscape table | Not covered (conceptual only) | Not covered (awareness only) |

## Key Stories & Facts to Weave In

1. **The Musk tweet that deleted $14 billion (May 1, 2020).** Elon Musk tweeted "Tesla stock price is too high imo" at 11:11 AM ET. Tesla fell 12% by close — $14 billion in market cap erased by 8 words. An NLP model processing Musk's Twitter feed would have detected extreme negative sentiment within milliseconds. The question isn't whether text moves markets. It's whether your model can read fast enough.

2. **Chen, Kelly, Xiu (2023) — the LLM embedding paper.** Published as "Expected Returns and Large Language Models," this paper demonstrated that LLM embeddings of news text significantly outperform all prior NLP methods for cross-sectional return prediction. The key result: adding 20 PCA components of embeddings to a standard factor model increased out-of-sample R² by about 38%. The paper was published in the top finance journal and immediately changed how quant firms think about text data.

3. **RavenPack and the $250K/year text signal.** RavenPack is the dominant commercial provider of financial text analytics. They process over 500 million news articles per year, extracting sentiment, relevance, and novelty scores in real-time. A basic subscription costs approximately $50,000/year. The premium tier — with entity-level sentiment, event detection, and raw NLP features — runs $200,000-$250,000/year. Major hedge funds pay this because the signal is worth it. Your homework replicates a simplified version using free data and open-source models.

4. **The Loughran-McDonald dictionary (2011).** Tim Loughran and Bill McDonald noticed that standard sentiment dictionaries (Harvard General Inquirer, LIWC) gave wrong results on financial text. "Company" is neutral in everyday language but appears in sentences like "the company faces significant litigation risk." They manually curated a list of ~2,700 negative and ~350 positive words specific to finance. This dictionary is still used as a baseline in every financial NLP paper, 15 years later. It's available free at nd.edu/~mcdonald.

5. **Man Group's AlphaGPT (2024).** Man Group, the world's largest publicly traded hedge fund ($170B+ AUM), published a system called AlphaGPT that uses an LLM in a feedback loop: the LLM proposes new trading factors, an evaluator backtests them, and the LLM refines its proposals based on the results. The paper claimed that AlphaGPT-generated factors had comparable IC to human-researched factors. This is arguably the first published evidence that LLMs can contribute to alpha generation, not just alpha extraction.

6. **The Pfizer vaccine announcement — NLP in real time (November 9, 2020).** Pfizer's 400-word press release hit at 6:45 AM ET: "vaccine candidate was found to be more than 90% effective." By 9:30 AM, Pfizer was up 9%, airlines were up 15-20%, and work-from-home stocks (Zoom, Peloton) were down 15-20%. A financial NLP system processing the press release within seconds could have generated signals for dozens of stocks — not just Pfizer, but every company in the "reopening trade." This is the power of text: it creates cross-sectional signals that price/volume data takes hours or days to reflect.

7. **The SEC's EDGAR database — free text at scale.** Every US public company is required to file quarterly (10-Q) and annual (10-K) reports with the SEC. These filings are available free on EDGAR within hours of filing. A typical 10-K is 100-200 pages of dense text — financial statements, risk factors, management discussion. NLP on 10-K filings has been shown to predict returns, earnings surprises, and bankruptcy risk. The data is free. The signal is real. The only barrier is compute.

## Cross-References
- **Builds on:** Week 5's XGBoost pipeline (text features plug directly into your existing model). Week 9's embedding paradigm (same embed-then-predict approach, different data modality). Week 7's PyTorch skills (loading transformer models from Hugging Face). Week 4's cross-sectional framework (IC evaluation, expanding-window CV).
- **Sets up:** Week 11 (uncertainty quantification — how confident is your text-informed prediction?). Week 12 (GNNs can incorporate text embeddings as node features, giving each stock both numerical and textual representations). The capstone (Week 18) may combine text and numerical features in a full strategy.
- **Recurring thread:** The "information preservation" theme — from raw text (infinite dimensions) to sentiment (1 dimension) to embeddings (768 dimensions). Every processing step destroys information. The art is preserving the return-relevant information while discarding noise. This parallels Week 9's discussion of tokenization strategies for time series.

## Suggested Reading
- **Chen, Kelly, Xiu (2023), "Expected Returns and Large Language Models"** — the foundational paper for this week. Read Sections 1-3 for the methodology and Table 4 for the punchline. The information bottleneck argument in Section 2 is the most important theoretical contribution.
- **"A Survey of Large Language Models for Financial Applications" (2024)** — comprehensive taxonomy of how LLMs are used in finance: sentiment analysis, named entity recognition, question answering, report generation, and trading signal extraction. Read for breadth.
- **Loughran & McDonald (2011), "When Is a Liability Not a Liability?"** — the paper that created the standard financial sentiment dictionary. Short, well-written, and a perfect example of why domain-specific NLP matters. Available on SSRN.
