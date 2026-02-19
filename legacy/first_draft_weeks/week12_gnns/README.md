# Week 12 — Graph Neural Networks for Finance

> **Every model you've built treats each stock as an island. In reality, stocks are a network — and the network IS the signal.**

## Prerequisites
- **Week 7 (Feedforward Nets):** PyTorch fluency — `nn.Module`, training loops, MPS acceleration. GNNs are PyTorch modules with the same `.forward()` and `.backward()` mechanics. You're adding graph-aware layers on top of what you already know.
- **Week 4-5 (Linear Models & XGBoost):** The cross-sectional prediction framework — features as firm characteristics, IC evaluation, expanding-window CV. GNNs solve the same cross-sectional problem, but with a relational twist: each stock's prediction is informed by its neighbors' features.
- **Week 3 (Portfolio/Risk):** Factor models, sector decomposition, correlation matrices. You'll use sector classifications and return correlations to construct the graphs that the GNN operates on.
- **Conceptual:** Basic understanding of graphs (nodes, edges, adjacency matrix). If you've used NetworkX or taken a graph theory course, you're set. If not: a graph is a set of nodes (stocks) connected by edges (relationships). That's all you need.

## The Big Idea

Here's a thought experiment. On January 17, 2024, TSMC reported quarterly earnings that beat estimates by 8%. TSMC manufactures chips for Apple, Nvidia, AMD, Qualcomm, and dozens of other tech companies. Within hours, every company in TSMC's supply chain moved — Apple up 2.1%, Nvidia up 3.4%, AMD up 2.8%. Your neural net from Week 7, which treats each stock independently, had no way to connect TSMC's earnings surprise to Apple's return. It would have seen Apple's features — momentum, volatility, volume — and made a prediction based on those alone, missing the most important piece of information: what just happened at Apple's most critical supplier.

Every model you've built so far has this blind spot. Feedforward nets process each stock's feature vector in isolation. LSTMs process each stock's time series independently. Even your ensemble averages predictions without considering that the stocks are connected. But stocks ARE connected. Apple depends on TSMC for chips, Foxconn for assembly, China for 20% of revenue. Banks co-move during crises because they hold each other's debt. When oil prices spike, every airline suffers and every energy company profits. These relationships are not noise — they're the structure of the market, and ignoring them leaves signal on the table.

Graph Neural Networks encode these relationships explicitly. Each stock is a node. Each relationship is an edge. The GNN's message-passing mechanism allows each stock's representation to be updated based on its neighbors' information — TSMC's earnings surprise literally flows through the graph to Apple, Nvidia, and AMD. The question is whether this relational information, on top of the per-stock features you already have, improves predictions enough to justify the added complexity.

The honest answer is nuanced, and that honesty is what separates this lecture from the hype. HIST (2021) achieved IC=0.052 on CSI300, beating LightGBM's 0.040 by 30%. MASTER (AAAI 2024) pushed to IC=0.064, beating XGBoost's 0.051 by 25%. Impressive. But these results use simple feature sets (Alpha360 — basic momentum and volume features). When you use rich, hand-crafted features (Alpha158 — 158 engineered features including complex technical indicators), XGBoost matches or beats GNNs. The GNN's edge comes from encoding relational information that compensates for weak per-node features. If your per-node features are already strong, the relational information adds less.

The practical lesson: the graph construction — choosing WHICH relationships to model — matters more than the GNN architecture. A GAT with the right graph beats a fancier architecture with the wrong graph.

## Lecture Arc

### Opening Hook

"On March 10, 2023, Silicon Valley Bank collapsed. Within 48 hours, First Republic Bank was down 62%, Western Alliance Bancorporation was down 47%, and PacWest Bancorp was down 38%. None of these banks had the same specific problem as SVB — they didn't have the same duration mismatch, the same concentrated depositor base, or the same losses on held-to-maturity securities. They fell because they were CONNECTED to SVB in the market's mind: same sector, similar business model, shared depositor concerns. A contagion graph — banks connected by sector, by size, by deposit concentration — would have propagated SVB's distress signal to every connected node. Your feedforward net from Week 7 saw SVB's collapse and shrugged. A Graph Neural Network would have said: 'if SVB is in trouble, every stock connected to SVB is at risk.' That graph-based reasoning is what we build today."

### Section 1: The Relational View of Markets — Why Stocks Aren't Independent

**Narrative arc:** We make the conceptual case for relational modeling before introducing any technical machinery. Why should you model stock relationships at all?

**Key concepts:** Cross-sectional dependence, information propagation, supply chain effects, sector co-movement, contagion.

**The hook:** "Your cross-sectional model from Week 4 predicts stock $i$'s return using stock $i$'s features. It's a function $f: \mathbb{R}^d \to \mathbb{R}$ — features in, return out. But this ignores a fundamental fact about markets: stock $i$'s return depends on what happens to OTHER stocks. Not just through shared factors (which the Fama-French model captures) but through specific, asymmetric relationships. Apple depends on TSMC (supplier), but TSMC doesn't depend symmetrically on Apple (Apple is one of many customers). Goldman Sachs and JPMorgan co-move because they're both investment banks, but Goldman and Walmart don't. These specific, pairwise relationships form a graph — and a GNN is the architecture designed to learn on graphs."

**Three sources of relational signal:**

1. **Supply chain propagation:** An earnings surprise at a supplier propagates to its customers. Menzly and Ozbas (2010) showed that customer-supplier return predictability is about 2% per month — one of the strongest anomalies in asset pricing.

2. **Sector co-movement:** Banks move together during financial crises. Energy stocks move with oil prices. Tech stocks share macro sensitivity. Sector membership captures this coarse-grained structure.

3. **Statistical co-movement (correlation):** Beyond sector labels, stocks have complex patterns of statistical dependence. Two semiconductor companies co-move more tightly than a semiconductor company and a semiconductor equipment company, even though both are in the same "technology" sector. Rolling correlations capture fine-grained structure that sector labels miss.

**"So what?":** "You've been modeling the forest tree by tree. GNNs let you model the forest as a forest — where each tree's growth depends on its neighbors' shade, water, and soil. The question isn't whether relationships exist (they obviously do). The question is whether ENCODING those relationships in a model improves predictions beyond what per-stock features already capture."

### Section 2: Graph Construction — The Most Important Design Choice

**Narrative arc:** The graph is more important than the GNN architecture. We show three methods for constructing stock graphs, each with different strengths and failure modes.

**Key concepts:** Adjacency matrix, graph sparsity, correlation graphs, sector graphs, supply chain graphs, dynamic graphs.

**The hook:** "If you get the graph wrong, no GNN architecture will save you. Connect every stock to every other stock (a complete graph), and the GNN averages over the entire market — no better than a sector factor. Connect stocks randomly, and you're injecting noise. Connect stocks based on 1-day correlations, and the graph is dominated by spurious relationships. The graph IS the inductive bias. Choose it well."

**Method 1: Correlation graphs**

Compute the rolling 60-day return correlation between every pair of stocks. Threshold at $\rho > 0.5$-$0.7$ to create edges.

$$A_{ij} = \begin{cases} 1 & \text{if } \text{corr}(r_i, r_j) > \tau \\ 0 & \text{otherwise} \end{cases}$$

"A 60-day rolling window balances stability (short windows are noisy) and adaptability (long windows miss regime changes). The threshold $\tau$ controls graph density: at $\tau = 0.5$, a typical S&P 100 graph has ~2,000 edges. At $\tau = 0.7$, it drops to ~500. Sparser is usually better — it forces the GNN to propagate information only through genuinely correlated pairs."

**Code moment:**

```python
import numpy as np
import pandas as pd

def build_correlation_graph(returns, window=60, threshold=0.6):
    """Build adjacency matrix from rolling return correlations."""
    corr_matrix = returns.iloc[-window:].corr()
    adj = (corr_matrix.abs() > threshold).astype(int)
    np.fill_diagonal(adj.values, 0)  # no self-loops
    return adj

adj = build_correlation_graph(returns_df, window=60, threshold=0.6)
print(f"Nodes: {adj.shape[0]}, Edges: {adj.sum().sum() // 2}")
# Typical output: Nodes: 100, Edges: 800-1500
```

**Method 2: Sector/industry graphs**

Connect stocks within the same GICS sector (11 sectors) or sub-industry (157 sub-industries).

$$A_{ij} = \begin{cases} 1 & \text{if } \text{sector}(i) = \text{sector}(j) \\ 0 & \text{otherwise} \end{cases}$$

"Sector graphs are static (sectors don't change often), interpretable (you know why two stocks are connected), and free (GICS codes are available from yfinance or Wikipedia). The downside: they're coarse. Putting Apple and Cisco in the same 'Information Technology' sector doesn't capture the fact that Apple is a consumer electronics company and Cisco sells networking equipment to enterprises."

**Code moment:**

```python
import yfinance as yf

def build_sector_graph(tickers):
    """Connect stocks in the same GICS sector."""
    sectors = {}
    for t in tickers:
        info = yf.Ticker(t).info
        sectors[t] = info.get('sector', 'Unknown')

    n = len(tickers)
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if sectors[tickers[i]] == sectors[tickers[j]]:
                adj[i, j] = adj[j, i] = 1
    return adj
```

**Method 3: Combined (union of edges)**

$$A_{ij}^{\text{combined}} = \max(A_{ij}^{\text{corr}}, A_{ij}^{\text{sector}})$$

"The union captures stocks that are either statistically co-moving OR in the same sector. This is usually the best default — it has higher recall than either individual graph at the cost of more edges."

**Dynamic graphs:**

"A correlation graph computed in January 2020 looks very different from one computed in March 2020. During crises, correlations spike toward 1 — everything moves together. The graph becomes dense, and the GNN's message passing becomes dominated by systemic risk rather than idiosyncratic relationships. Re-computing the graph monthly (or quarterly) captures this evolution. The cost: each graph recomputation triggers a new set of edges, which changes the GNN's message-passing structure."

**"So what?":** "You'll test all three graph types in the homework. The ablation will show that graph choice matters more than architecture choice. A GAT on a good graph beats a fancier GNN on a bad graph, every time."

### Section 3: GNN Architectures for Finance — GCN, GAT, and GraphSAGE

**Narrative arc:** We build from the simplest GNN (GCN) to the recommended default (GAT), explaining what each architecture adds and why GAT is the best default for financial applications.

**Key concepts:** Message passing, graph convolution, attention mechanism, neighborhood aggregation.

**The hook:** "All GNNs share the same fundamental operation: update each node's representation using information from its neighbors. The differences are in HOW they aggregate neighbor information. GCN uses a fixed, symmetric average. GAT learns which neighbors matter more. GraphSAGE samples a random subset of neighbors. For finance, GAT is the best default — because not all stock relationships are equally important, and attention lets the model learn which ones to focus on."

**GCN (Graph Convolutional Network):**

Each layer performs:

$$h_i^{(l+1)} = \text{ReLU}\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{d_i d_j}} W^{(l)} h_j^{(l)}\right)$$

"Read this as: node $i$'s new representation is a weighted average of its neighbors' (and its own) representations, passed through a linear transformation $W$ and a nonlinearity. The $1/\sqrt{d_i d_j}$ normalization ensures that high-degree nodes (stocks with many connections) don't dominate."

"The limitation: every neighbor gets the same weight (up to degree normalization). In finance, this is wrong. If Apple is connected to both TSMC (critical supplier) and a small mutual-fund company (weak correlation-based edge), GCN treats both equally. You need attention."

**GAT (Graph Attention Network) — the recommended default:**

GAT learns attention weights for each edge:

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [W h_i \| W h_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^T [W h_i \| W h_k]))}$$

$$h_i^{(l+1)} = \text{ReLU}\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j^{(l)}\right)$$

"The attention coefficient $\alpha_{ij}$ tells you how much node $i$ 'cares' about node $j$. It's computed from both nodes' features: if TSMC just had a huge earnings surprise (its features are unusual), the attention from Apple to TSMC will be high. If a weakly-connected stock is behaving normally, the attention will be low. This is exactly what you want for financial data: the model dynamically up-weights the most relevant connections."

Multi-head attention extends this by running $K$ independent attention heads:

$$h_i^{(l+1)} = \Big\|_{k=1}^{K} \text{ReLU}\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k W^k h_j^{(l)}\right)$$

"Different heads can capture different types of relationships. Head 1 might attend to sector peers. Head 2 might attend to stocks with similar momentum. Head 3 might attend to high-volatility neighbors. The model discovers the specialization."

**Code moment — GAT in PyTorch Geometric:**

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class StockGAT(nn.Module):
    def __init__(self, n_features, hidden_dim=64, n_heads=4):
        super().__init__()
        self.conv1 = GATConv(n_features, hidden_dim, heads=n_heads, dropout=0.2)
        self.conv2 = GATConv(hidden_dim * n_heads, hidden_dim, heads=1, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        return self.fc(x).squeeze(-1)
```

"That's a complete 2-layer GAT for stock prediction. Layer 1: 4-head attention, expanding features from `n_features` to `hidden_dim × 4`. Layer 2: single-head attention, compressing back to `hidden_dim`. Output: a linear layer predicting one value (next-period return) per node."

**"So what?":** "GAT is the right default for financial GNNs because stock relationships are inherently asymmetric and context-dependent. The attention weights are interpretable — you can extract them and see which connections the model relies on. This interpretability is valuable for debugging and for communicating results to portfolio managers who want to understand why the model is making a particular bet."

### Section 4: PyTorch Geometric — The Practical Toolkit

**Narrative arc:** We cover the essential PyTorch Geometric concepts needed to implement everything: Data objects, edge_index format, batching, and training.

**Key concepts:** `Data` objects, `edge_index` as COO sparse matrix, node features, edge features, mini-batching for graph-level tasks vs. node-level tasks.

**The hook:** "PyTorch Geometric is to GNNs what Hugging Face is to NLP — the library that makes research implementations practical. Without it, you'd be writing sparse matrix operations by hand. With it, a 2-layer GAT for stock prediction is 15 lines of code."

**Key concepts unpacked:**

**The `Data` object:**

```python
from torch_geometric.data import Data

# Build a graph for one time step
data = Data(
    x=node_features,       # shape: (n_stocks, n_features) — feature matrix
    edge_index=edge_index,  # shape: (2, n_edges) — COO format
    y=returns,              # shape: (n_stocks,) — prediction target
)
```

"The `edge_index` is a 2×E matrix where each column `[i, j]` represents a directed edge from node $i$ to node $j$. For undirected graphs (our case), include both `[i, j]` and `[j, i]`. Converting from an adjacency matrix:"

```python
def adj_to_edge_index(adj_matrix):
    """Convert adjacency matrix to PyG edge_index format."""
    src, dst = np.where(adj_matrix > 0)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index
```

**Node-level vs. graph-level tasks:**

"Stock prediction is a NODE-LEVEL task: one prediction per node (stock). Each graph represents the market at one time step, and we predict all stocks' returns simultaneously. This is different from graph classification (one prediction per graph), which is common in chemistry (is this molecule toxic?) but not what we're doing."

**Training loop:**

```python
model = StockGAT(n_features=10, hidden_dim=64, n_heads=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    pred = model(train_data)
    loss = F.mse_loss(pred, train_data.y)
    loss.backward()
    optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_pred = model(test_data)
        ic = spearmanr(test_pred, test_data.y).correlation
```

"The training loop is identical to a standard PyTorch loop. The graph structure is encoded in `edge_index`, not in the training procedure. PyTorch Geometric handles the message passing inside `model(data)` — you don't touch it."

**"So what?":** "PyTorch Geometric reduces the implementation effort for GNNs from 'write a research paper' to 'write 20 lines of code.' The entire pipeline — graph construction, GNN definition, training, evaluation — is under 100 lines. For S&P 100 (100 nodes, ~1,000 edges), training takes seconds per epoch on M4."

### Section 5: State of the Art — HIST, MASTER, and the Honest Benchmark

**Narrative arc:** We survey the best published GNN results for stock prediction, then deflate the hype with an honest assessment of when GNNs actually beat simpler models.

**Key concepts:** HIST, MASTER, Qlib benchmarks, Alpha360 vs. Alpha158 feature sets, the feature richness paradox.

**The hook:** "MASTER, published at AAAI 2024, achieved IC=0.064 on CSI300 stocks — beating XGBoost (IC=0.051) by 25%. That's the headline. Now the fine print: MASTER used the Alpha360 feature set — 360 features, all derived from simple price and volume transformations. When you switch to Alpha158 — 158 more carefully engineered features — XGBoost matches or beats MASTER. The GNN's advantage disappears when the per-node features are strong enough to implicitly capture the relational information."

**HIST (Xu et al., 2021):**

"HIST (Hidden Interconnected Stock Transformer) introduced concept-based graphs: instead of connecting stocks directly, they connect stocks through shared 'concepts' — like industry membership, earnings growth similarity, or volatility regime. Each concept is a hidden node that aggregates information from its associated stocks and feeds it back. On Qlib's CSI300 benchmark, HIST achieved IC=0.052 vs. LightGBM's 0.040 — a 30% improvement."

**MASTER (Li et al., AAAI 2024):**

"MASTER (Market-Guided Stock Transformer) combines a transformer for temporal modeling with a graph network for relational modeling. It's the current SOTA on Qlib benchmarks: IC=0.064 on CSI300. The key innovation: 'market-guided' attention that conditions the graph attention on the overall market state — during crises, the model attends more to sector peers; during calm markets, it attends more to supply chain connections."

**The honest benchmark:**

| Model | Features | IC on CSI300 | Improvement over XGBoost |
|-------|----------|-------------|------------------------|
| XGBoost | Alpha360 | 0.040 | — |
| HIST | Alpha360 | 0.052 | +30% |
| MASTER | Alpha360 | 0.064 | +60% |
| XGBoost | Alpha158 | 0.058 | — |
| HIST | Alpha158 | 0.060 | +3% |
| MASTER | Alpha158 | 0.062 | +7% |

"Read that table carefully. With simple features (Alpha360), GNNs dominate — 30-60% IC improvement. With rich features (Alpha158), the gap shrinks to 3-7%. The explanation: rich features already encode some relational information implicitly (e.g., 'stock's return vs. sector average return'), so the GNN's explicit relational modeling adds less marginal value."

**What this means for your homework:**

"You'll use simple features (5-10 per stock) with S&P 100 stocks. This is the regime where GNNs should shine — your features are simple, and the graph adds relational information the features don't capture. Expect 15-30% IC improvement over MLP/XGBoost. If you see more, check for bugs. If you see less, your graph might not be capturing the right relationships."

**"So what?":** "GNNs are not universally better than XGBoost. They're better when relational information adds value beyond what features already capture. In practice, this means GNNs are most useful for: (1) simple feature sets with rich relational data, (2) ranking tasks where the cross-section matters, and (3) event propagation (earnings surprises, defaults, regulatory actions). If you already have 150+ engineered features per stock, a GNN might not justify its complexity."

### Section 6: Attention Visualization — What the Model Learns

**Narrative arc:** One of the GAT's unique strengths is interpretability — you can extract the attention weights and see which relationships the model focuses on. This matters for trust, debugging, and communication.

**Key concepts:** Attention weight extraction, economic interpretability, network visualization.

**The hook:** "After training, the GAT has learned which stock relationships matter most. You can extract the attention weights and ask: for Apple, which neighbors get the most attention? If the answer is TSMC, Qualcomm, and Google — supply chain partners and competitors — the model has learned economically meaningful relationships. If the answer is a random insurance company, something is wrong with your graph."

**Code moment:**

```python
# Extract attention weights from trained GAT
model.eval()
with torch.no_grad():
    # GATConv stores attention weights with return_attention_weights=True
    out, (edge_index, alpha) = model.conv1(data.x, data.edge_index,
                                            return_attention_weights=True)

# For AAPL (node 0), find top-5 attended neighbors
aapl_edges = (edge_index[1] == 0)  # edges pointing TO AAPL
aapl_neighbors = edge_index[0, aapl_edges]
aapl_attn = alpha[aapl_edges].squeeze()

top5 = aapl_attn.argsort(descending=True)[:5]
for idx in top5:
    neighbor_ticker = tickers[aapl_neighbors[idx]]
    weight = aapl_attn[idx].item()
    print(f"  {neighbor_ticker}: attention = {weight:.4f}")
```

Expected output (illustrative):
```
Top-5 attention neighbors for AAPL:
  MSFT: attention = 0.1842   (tech peer)
  NVDA: attention = 0.1523   (chip customer)
  GOOGL: attention = 0.1204  (competitor in services)
  AMZN: attention = 0.0891   (tech peer)
  AVGO: attention = 0.0734   (chip supplier)
```

"These are economically sensible — the model learned to attend to Apple's competitors and supply chain partners, not random unrelated stocks. This interpretability is a significant advantage of GAT over GCN (which doesn't have attention weights to inspect)."

**Network visualization:**

```python
import networkx as nx

G = nx.from_numpy_array(adj_matrix.numpy())
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_size=50, node_color=sector_colors,
        edge_color='gray', alpha=0.3)
# Highlight AAPL and its top-5 attention neighbors
```

"The resulting plot should show clear sector clustering (financials grouped together, tech grouped together) with some cross-sector edges from the correlation graph."

**"So what?":** "Attention weights turn the GNN from a black box into an interpretable model. A portfolio manager who asks 'why is the model bearish on JPM?' can be shown: 'because the model attended heavily to SVB and First Republic, which are both down.' This interpretability is more valuable in production than marginal IC improvement."

### Section 7: The Complete Pipeline — From Data to Predictions

**Narrative arc:** We tie everything together into a complete pipeline that students can modify and extend.

**Key concepts:** End-to-end pipeline, data flow, temporal aspects, re-computing graphs over time.

**The pipeline:**

```
Daily OHLCV (S&P 100) → Feature Engineering → Graph Construction → GAT → Predictions → IC Evaluation
     ↓                      ↓                       ↓
  yfinance              momentum, vol,          correlation OR
                        volume, RSI             sector OR combined
```

**Temporal considerations:**

"The graph should be re-computed periodically — monthly or quarterly. The features are computed daily. The GAT is retrained at each rebalancing period (monthly). The evaluation uses expanding-window CV: train on data up to month $t$, test on month $t+1$."

"Critically: the correlation graph at time $t$ must use only data up to time $t$ — no future correlations. This is the graph equivalent of the temporal splitting rule from Week 7."

**Code moment — the full pipeline sketch:**

```python
# Full pipeline for one test month
# 1. Compute features
features = compute_features(ohlcv, as_of=test_date)  # (100, 10)

# 2. Build graph (using only past data)
adj = build_correlation_graph(returns[:test_date], window=60, threshold=0.6)
edge_index = adj_to_edge_index(adj)

# 3. Create PyG data object
data = Data(x=features, edge_index=edge_index, y=future_returns)

# 4. Train GAT (on training data)
model = StockGAT(n_features=10, hidden_dim=64, n_heads=4)
train_model(model, train_data_list, val_data)

# 5. Predict
model.eval()
predictions = model(test_data).detach()

# 6. Evaluate
ic = spearmanr(predictions, actual_returns).correlation
```

**"So what?":** "The complete pipeline is about 300 lines of Python — feature engineering, graph construction, GAT definition, training loop, evaluation. For S&P 100, everything runs on M4 in under a minute per epoch. The bottleneck is graph construction (correlation computation), not the GNN itself."

### Closing Bridge

"You've now built models that process individual stocks (Weeks 4-7), temporal sequences (Week 8), pre-trained representations (Week 9), text (Week 10), uncertainty (Week 11), and relational structure (this week). The core toolkit of deep learning for finance is complete. Next week, we move to reinforcement learning for portfolio management. Instead of predicting returns and then constructing portfolios separately, RL learns the optimal trading policy end-to-end: observe the market state, take an action (rebalance the portfolio), receive a reward (portfolio return). It's a fundamentally different paradigm — and it connects directly to the market making problem in Week 16."

## Seminar Exercises

### Exercise 1: Dynamic Graphs Across Regimes — How Does Crisis Rewire the Market?
**The question we're answering:** The lecture demonstrated how to build correlation and sector graphs for one snapshot. But the correlation graph changes dramatically during crises — how does the market's "wiring diagram" shift, and what does that mean for the GNN?
**Setup narrative:** "The lecture built a static graph. Markets are not static. During COVID, correlations spiked to nearly 1 and the graph became a near-complete graph — every stock connected to every other stock. During calm periods, the graph is sparse and sector-structured. You're going to build the same graph at three different moments and see how the market rewires itself."
**What they build:** Using S&P 100 daily returns, build 60-day correlation graphs (threshold=0.6) centered on three dates: (a) January 2020 (calm), (b) April 2020 (post-COVID crash), (c) January 2024 (moderate). For each, compute: number of edges, average degree, graph density, modularity by sector. Visualize all three with NetworkX side-by-side, color-coded by sector.
**What they'll see:** Calm graph: ~800 edges, clear sector clusters, modularity ~0.5. Crisis graph: ~3,000+ edges, sector structure dissolves, modularity drops to ~0.15. Moderate graph: in between. The crisis graph is nearly a complete graph — the GNN's message passing becomes "average everything," which destroys the relational signal.
**The insight:** Graph construction must be regime-aware. During crises, high-threshold graphs (0.7+) or sector-only graphs preserve useful structure. During calm periods, lower thresholds (0.5-0.6) capture fine-grained relationships. The optimal threshold is dynamic, not fixed.

### Exercise 2: The Feature Richness Paradox — When Does the Graph Stop Helping?
**The question we're answering:** The lecture showed that GNNs help most with simple features and least with rich features. At exactly how many features does the GNN's advantage over MLP disappear?
**Setup narrative:** "The lecture gave you the HIST/MASTER benchmark numbers. Now you replicate the finding on your own data. Start with 5 features (where the GNN should dominate) and progressively add more. At some point, the MLP catches up. That crossing point tells you whether your production feature set is 'simple enough' to justify a GNN."
**What they build:** Using the lecture's `StockGAT` class, train both GAT and MLP at five feature-set sizes: (a) 5 basic features (momentum, vol, volume), (b) 10 features (+RSI, reversal, size), (c) 20 features (+sector dummies, cross-sectional ranks), (d) 30 features (+moving average ratios, earnings surprise), (e) 50 features (+all Alpha158-style features you can compute). Plot IC for GAT and MLP as a function of feature count.
**What they'll see:** At 5 features, GAT beats MLP by ~25%. At 10 features, by ~15%. At 20, by ~8%. At 50, the gap shrinks to 2-5% and may not be statistically significant. The crossover (where adding features helps MLP more than GAT) typically happens around 20-30 features.
**The insight:** The GNN's graph-based relational information is partially redundant with rich features. Features like "stock's return vs. sector mean return" encode the same information as a sector graph edge. If your feature engineering is strong, the GNN adds less. This is the practical decision criterion: invest in better features OR a GNN, but probably not both.

### Exercise 3: Ablation — Which Graph Type Helps Most?
**The question we're answering:** Does the graph type matter more than the GNN architecture?
**Setup narrative:** "You have the same GAT, trained on three different graphs: correlation, sector, and combined. If the results differ significantly, it means graph construction is where the real leverage is."
**What they build:** Train GAT on: (a) correlation graph, (b) sector graph, (c) combined graph, (d) no graph (MLP baseline). Same features, same hyperparameters, same evaluation.
**What they'll see:** Typical ranking: combined > correlation > sector > MLP. The combined graph wins because it has the highest recall — it captures relationships that either individual graph misses.
**The insight:** Graph construction is the most important design choice. The difference between the best and worst graph (combined vs. sector-only) is often larger than the difference between GAT and GCN on the same graph.

### Exercise 4: Attention Drift — Do Learned Relationships Change Over Time?
**The question we're answering:** The lecture showed how to extract attention weights at a single point in time. But do the model's learned relationships stay stable, or do they shift as market conditions change?
**Setup narrative:** "A model trained in January 2023 might attend to different relationships than one trained in January 2024. If the attention pattern is stable, the model has learned durable economic relationships. If it drifts, the model is picking up transient correlations. You're going to measure this drift."
**What they build:** Train the GAT at 4 quarterly rebalancing points (Q1-Q4 of a single year). At each point, extract attention weights for AAPL, JPM, and XOM. For each stock, compute: (a) the top-5 neighbors by attention weight at each quarter, (b) the Jaccard similarity of the top-5 sets across quarters (how much overlap?), (c) the rank correlation of attention weights across quarters (how stable are the relative importance values?).
**What they'll see:** For AAPL: top-5 is fairly stable (Jaccard ~0.6-0.8) — MSFT and NVDA appear consistently. For JPM: moderately stable (Jaccard ~0.5-0.7) — banking peers are consistent, but smaller banks rotate in/out. For XOM: less stable (Jaccard ~0.4-0.6) — energy sector attention shifts with oil price dynamics. Rank correlations are typically 0.5-0.7 across quarters.
**The insight:** The GAT learns a mix of durable relationships (sector peers, supply chain) and transient ones (correlation-driven links that shift with market dynamics). The stable core relationships are the ones you can trust for longer-horizon predictions. The transient ones add value for short-term trading but require frequent retraining.

## Homework: "Relational Stock Ranking with Graph Attention Networks"

### Mission Framing

Every model you've built in this course has been solipsistic — each stock lives in its own universe, evaluated on its own features, predicted in isolation. This week, you break that isolation. You're going to build a model that understands that AAPL and TSMC are connected, that JPM and GS move together during banking crises, that an oil price shock affects the entire energy sector simultaneously. The tool is a Graph Attention Network — a neural network that propagates information across a network of stock relationships.

The experiment has three layers. First, you build the graphs — three types of relational structure representing different hypotheses about how stocks are connected. Second, you train the GAT and compare it to an MLP (no graph) and XGBoost (no graph). Third, you analyze the attention weights to check whether the model has learned something economically meaningful or is just fitting noise. The result will be nuanced: the GNN helps, but not by as much as the papers suggest when you have strong features. The real value is in the graph construction, not the GNN architecture.

### Deliverables

1. **Data preparation.** Download 2 years of daily OHLCV data for S&P 100 stocks via yfinance (approximately 2022-2024). Compute 8-10 features per stock per day: 5-day momentum, 10-day momentum, 20-day momentum, 20-day realized volatility, volume ratio (today's volume / 20-day average), RSI (14-day), 5-day reversal (negative of 5-day return), log market cap. Target: next 5-day return rank (cross-sectional rank from 0 to 1).

2. **Graph construction.** Build three graph types:
   - **Correlation graph:** 60-day rolling return correlation, threshold at 0.6. Re-compute monthly.
   - **Sector graph:** same GICS sector = edge. Use yfinance `.info['sector']` or Wikipedia's S&P 100 table.
   - **Combined graph:** union of correlation and sector edges.
   For each graph, report: number of edges, average degree, density.

3. **GAT model.** Implement a 2-layer GAT using PyTorch Geometric: `GATConv(n_features, 64, heads=4)` followed by `GATConv(256, 64, heads=1)` followed by `Linear(64, 1)`. Use dropout(0.2), ReLU activations. Target: next 5-day return rank. Loss: MSE.

4. **Baselines.** (a) MLP with the same features (no graph information). 2 hidden layers (64, 32). (b) XGBoost with the same features.

5. **Training and evaluation.** Use expanding-window CV: train on the first 12 months, validate on months 13-15, test on months 16-24. Retrain every 3 months. Report: IC, RankIC, Precision@10 (did the top-10 predicted stocks actually outperform?), long-short decile portfolio Sharpe.

6. **Ablation.** Train the GAT on each graph type separately and on the combined graph. Report IC for each. Which graph type helps most?

7. **Attention analysis.** For 3 example stocks (e.g., AAPL, JPM, XOM), extract attention weights from the trained GAT. Identify the top-5 attended neighbors. Are they economically sensible? Present as a table or visualization.

8. **Comparison table:**

   | Model | Graph | IC | RankIC | Precision@10 | Sharpe (L/S) |
   |-------|-------|----|----|----|----|
   | MLP | None | | | | |
   | XGBoost | None | | | | |
   | GAT | Correlation | | | | |
   | GAT | Sector | | | | |
   | GAT | Combined | | | | |

9. **Deliverable:** Notebook + comparison table + attention analysis + graph visualizations.

### What They'll Discover

- The GAT on the combined graph typically beats the MLP by 15-30% in IC and XGBoost by 10-20% with these simple features. The graph adds genuinely new information.
- The correlation graph is usually better than the sector graph alone. Statistical co-movement captures fine-grained relationships that coarse sector labels miss.
- The combined graph (union of edges) is usually best — it has the highest recall, capturing both statistical and economic relationships.
- Attention weights are economically interpretable. AAPL attends to tech supply chain partners. JPM attends to banking peers. XOM attends to energy companies. This builds trust in the model.
- With simple features (8-10), the GNN improvement is substantial. If they added 50+ engineered features, the gap would shrink significantly — XGBoost would match the GNN because the rich features implicitly encode relational information.
- Training the GAT on S&P 100 takes seconds per epoch on M4. The graph has 100 nodes and 500-2000 edges. This is computationally trivial compared to the LSTMs from Week 8.

### Deliverable

Final notebook: `hw12_gnn_stock_ranking.ipynb` containing graph construction, GAT implementation, baseline comparisons, ablation study, and attention analysis.

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Graph construction (correlation, sector, combined) | Demo: show code for correlation and sector graphs, explain thresholds and dynamic graphs | Exercise 1: build graphs at 3 dates (calm/crisis/moderate), analyze how crisis rewires the market | At scale: build all 3 graph types for S&P 100, report edges/degree/density |
| GNN architectures (GCN, GAT, GraphSAGE) | Demo: explain message passing, show GCN formula, derive GAT attention, multi-head attention | Not re-implemented (done in lecture) | Integrate: use GAT as the primary architecture |
| GAT implementation in PyTorch Geometric | Demo: full `StockGAT` code, `Data` objects, `edge_index` format, training loop | Exercise 2: test feature richness paradox — sweep 5 to 50 features, find GNN vs. MLP crossover | At scale: implement 2-layer GAT, train with expanding-window CV |
| Graph type ablation | Demo: conceptual discussion of which graph matters most | Exercise 3: train GAT on correlation / sector / combined, compare IC | At scale: full ablation across all graph types, report in comparison table |
| Attention weight interpretation | Demo: show extraction code, expected output for AAPL, network visualization | Exercise 4: measure attention drift across 4 quarters, compute Jaccard stability | At scale: extract and report top-5 neighbors for 3 example stocks |
| The feature richness paradox (Alpha360 vs. Alpha158) | Demo: HIST/MASTER benchmark table, explain when GNNs beat XGBoost | Exercise 2: empirical validation — IC vs. feature count for GAT and MLP | Integrate: contextualize GNN improvement relative to feature richness |
| State of the art (HIST, MASTER) | Demo: survey with honest benchmark comparison table | Not covered (lecture provides context) | Integrate: compare your results to published benchmarks |

## Key Stories & Facts to Weave In

1. **The SVB contagion (March 2023).** When Silicon Valley Bank collapsed on March 10, 2023, the stocks of other regional banks plummeted within hours — First Republic (-62%), Western Alliance (-47%), PacWest (-38%). None had SVB's exact problem (duration mismatch on held-to-maturity securities). They fell because they were in the same "contagion graph" — same sector, similar business model, shared depositor concerns. A GNN with a banking sector graph would have propagated SVB's distress signal to every connected bank node.

2. **TSMC earnings and supply chain propagation (January 2024).** TSMC reported Q4 2023 earnings beating estimates by 8% on January 18, 2024. Within hours: Apple +2.1%, Nvidia +3.4%, AMD +2.8%, Qualcomm +2.5%. These movements were driven entirely by the SUPPLY CHAIN relationship — TSMC manufactures chips for all of them. A GNN with supply chain edges would capture this information propagation explicitly. A standard MLP or XGBoost, treating each stock independently, would miss it entirely.

3. **HIST on Qlib (Xu et al., 2021).** HIST (Hidden Interconnected Stock Transformer) was the first GNN to achieve SOTA on Microsoft's Qlib benchmark for CSI300 stock prediction. IC=0.052 vs. LightGBM's 0.040 — a 30% improvement. The key innovation: "concept-oriented" graphs that connect stocks through shared concepts (industry, earnings growth similarity, volatility regime) rather than direct edges. The paper showed that the graph structure matters more than the GNN architecture.

4. **MASTER at AAAI 2024 (Li et al.).** MASTER (Market-Guided Stock Transformer) is the current SOTA on Qlib benchmarks: IC=0.064 on CSI300. It combines a transformer for temporal modeling with a GNN for relational modeling, conditioned on the overall market state. The "market-guided" attention adjusts automatically: during crises, it attends more to sector peers; during calm markets, it attends more to idiosyncratic supply chain connections. The paper showed that dynamic graph attention (adapting to market conditions) outperforms static attention by 5-10%.

5. **The feature richness paradox.** The most under-reported finding in the GNN-for-finance literature: GNNs help most when features are simple and least when features are rich. On Alpha360 (simple features), GNNs beat XGBoost by 30-60%. On Alpha158 (rich features), XGBoost matches GNNs. The explanation: rich features implicitly encode relational information. A feature like "stock's return vs. sector average return" already captures sector co-movement — the same information the sector graph provides. When features are rich, the graph is redundant. When features are simple, the graph provides information the features lack.

6. **LLM-extracted graphs (FinDKG, ICAIF 2024).** The frontier of graph construction uses LLMs to extract relationships from news text. "Apple partners with TSMC for new 3nm chip production" → edge(AAPL, TSM, type='supply_chain'). These LLM-extracted graphs are dynamic (relationships change as news emerges), typed (supply chain vs. competition vs. partnership), and high-recall (they capture relationships that statistical methods miss). The limitation: LLM-extracted edges can be noisy (hallucinations, misattributions).

7. **PyTorch Geometric efficiency.** For S&P 100 (100 nodes, ~1,000 edges), a 2-layer GAT with 64 hidden units processes one graph snapshot in under 1 millisecond on M4. Training for 100 epochs takes about 5-10 seconds. This is computationally trivial — the bottleneck is data preparation and graph construction, not the GNN. Even for S&P 500 (500 nodes, ~5,000 edges), training takes under a minute. GNNs become computationally expensive only for very large graphs (10,000+ nodes) or very deep architectures (5+ layers).

## Cross-References
- **Builds on:** Week 4-5's cross-sectional framework (same prediction problem, same features, same evaluation). Week 7's PyTorch skills (GNNs use the same `nn.Module`, same training loops). Week 3's sector decomposition and correlation matrices (these become the graph construction inputs). Week 11's MC Dropout (apply dropout at inference in the GAT layers for uncertainty-aware graph predictions).
- **Sets up:** Week 13 (RL for portfolio management — GNN predictions can serve as the state representation for an RL agent, encoding not just individual stock features but relational context). Week 18 (capstone — a GNN-enhanced strategy is a legitimate capstone component, combining graph construction, relational prediction, and portfolio construction).
- **Recurring thread:** The "honest benchmark" theme. GNNs are exciting but they don't universally beat simpler models. The improvement depends on feature richness, graph quality, and the prediction task. We measure the improvement, quantify when it appears, and note when it doesn't. This is the same intellectual honesty we've demanded for every model since Week 4.

## Suggested Reading
- **Feng et al. (2019), "Temporal Relational Ranking for Stock Prediction"** — the first paper to combine GNNs with temporal modeling (LSTM) for stock ranking. Short and influential. Read for the architecture and the motivation.
- **ACM Computing Surveys (2024), "GNN-based Methods for Stock Forecasting: A Systematic Review"** — comprehensive survey of the entire GNN-for-finance literature. Read the taxonomy (Section 3) and the comparison table (Section 5) for breadth.
- **Li et al. (AAAI 2024), "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting"** — the current SOTA. Read Section 3 for the architecture and Section 4.3 for the ablation study showing when graph structure helps and when it doesn't.
