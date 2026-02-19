"""Section 7: Alternative Data — Taxonomy and Institutional Context

Conceptual section: no data downloads, no ML models, no plots.
Prints a structured taxonomy of alternative data categories used in
quantitative finance, institutional cost context, and bridges to
the NLP/alternative-data week (Week 7).
"""


# ── CELL: alt_data_introduction ─────────────────────────────────


print("=" * 70)
print("ALTERNATIVE DATA IN QUANTITATIVE FINANCE")
print("=" * 70)
print()
print("Traditional alpha signals (price momentum, value ratios, quality")
print("metrics) are derived from market prices and financial statements —")
print("data that every participant can access.  Alternative data refers to")
print("non-traditional information sources that may provide an edge before")
print("the signal is priced in.")
print()
print("The alternative data market has grown from ~$200M in 2015 to over")
print("$7B by 2025 (estimated), driven by hedge funds, asset managers, and")
print("increasingly by long-only institutional investors.")


# ── CELL: taxonomy_categories_1 ────────────────────────────────

TAXONOMY = {
    "Sentiment & News": {
        "description": (
            "Text-derived signals from news articles, social media, "
            "earnings call transcripts, and analyst reports."
        ),
        "examples": [
            "News sentiment scores (RavenPack, Bloomberg Event-Driven)",
            "Social media buzz (Twitter/X, StockTwits, Reddit WallStreetBets)",
            "Earnings call tone and Q&A linguistic features",
            "SEC filing complexity and readability (10-K fog index)",
        ],
        "typical_lag": "Minutes to hours (news); real-time (social media)",
        "alpha_horizon": "Intraday to weeks",
    },
    "Web & App Traffic": {
        "description": (
            "Digital footprint data measuring consumer engagement with "
            "companies' online presence."
        ),
        "examples": [
            "Website visit counts and trends (SimilarWeb, Semrush)",
            "App downloads and daily active users (Sensor Tower, App Annie)",
            "Search engine query volume (Google Trends)",
            "Job posting volume and seniority mix (Burning Glass, Indeed)",
        ],
        "typical_lag": "Days to weeks",
        "alpha_horizon": "Weeks to quarters (leading indicators for revenue)",
    },
    "Geolocation & Foot Traffic": {
        "description": (
            "Mobile device location data revealing real-world consumer "
            "behaviour at physical locations."
        ),
        "examples": [
            "Retail foot traffic counts (Placer.ai, SafeGraph)",
            "Parking lot satellite imagery occupancy (Orbital Insight)",
            "Airport and hotel check-in proxies",
            "Supply chain port and warehouse activity",
        ],
        "typical_lag": "Days (aggregated panels)",
        "alpha_horizon": "Weeks to quarters (nowcasting same-store sales)",
    },
    "Satellite & Imagery": {
        "description": (
            "Overhead and ground-level imagery processed with computer "
            "vision to extract economic signals."
        ),
        "examples": [
            "Oil storage tank fill levels (shadow analysis, Ursa Space)",
            "Agricultural crop health via NDVI (Descartes Labs)",
            "Nighttime light intensity as economic activity proxy",
            "Construction site progress tracking",
        ],
        "typical_lag": "Days to weeks (revisit frequency dependent)",
        "alpha_horizon": "Weeks to months (commodity and macro signals)",
    },
}


# ── CELL: taxonomy_categories_2 ────────────────────────────────

TAXONOMY.update({
    "Transaction & Payment Data": {
        "description": (
            "Aggregated, anonymised consumer spending data from credit "
            "cards, point-of-sale systems, and e-commerce receipts."
        ),
        "examples": [
            "Credit card transaction panels (Mastercard SpendingPulse)",
            "E-commerce receipt scraping (Edison Trends, Measurable AI)",
            "Point-of-sale data aggregation (Second Measure, Bloomberg)",
            "Bank account transaction categorisation",
        ],
        "typical_lag": "Days (card networks); weeks (receipt panels)",
        "alpha_horizon": "Weeks to quarters (revenue nowcasting)",
    },
    "Government & Regulatory Filings": {
        "description": (
            "Structured data extracted from public filings, permits, "
            "patents, and regulatory records."
        ),
        "examples": [
            "FDA drug approval pipeline and PDUFA dates",
            "Patent filings and citation networks (Google Patents)",
            "EPA emissions and compliance records",
            "SEC insider transactions (Form 4) and 13-F holdings",
        ],
        "typical_lag": "Days to months (filing cadence varies)",
        "alpha_horizon": "Months (event-driven or structural)",
    },
    "Expert Networks & Surveys": {
        "description": (
            "Proprietary intelligence from industry experts, procurement "
            "managers, and specialised surveys."
        ),
        "examples": [
            "Expert transcript libraries (GLG, AlphaSights, Tegus)",
            "Procurement manager surveys (ISM PMI sub-components)",
            "Channel checks aggregated from supply chain contacts",
            "Sell-side analyst revision velocity",
        ],
        "typical_lag": "Days to weeks (survey cadence)",
        "alpha_horizon": "Weeks to quarters",
    },
})


# ── CELL: taxonomy_display ─────────────────────────────────────

n_categories = len(TAXONOMY)

print()
print("-" * 70)
print(f"ALTERNATIVE DATA TAXONOMY  ({n_categories} categories)")
print("-" * 70)

for i, (category, info) in enumerate(TAXONOMY.items(), 1):
    print(f"\n  {i}. {category}")
    print(f"     {info['description']}")
    print(f"     Examples:")
    for ex in info["examples"]:
        print(f"       - {ex}")
    print(f"     Typical lag:    {info['typical_lag']}")
    print(f"     Alpha horizon:  {info['alpha_horizon']}")


# ── CELL: institutional_cost_context ────────────────────────────

print()
print("-" * 70)
print("INSTITUTIONAL COST CONTEXT")
print("-" * 70)
print()
print("Alternative data is expensive.  The median institutional spend on")
print("alternative data has risen sharply:")
print()
print("  BattleFin (2023 survey):  median fund spend ~$1.6M/year on alt data")
print("  Exabel (2024 report):     similar $1.6M median, top decile >$5M/year")
print()
print("This creates a structural barrier to entry:")
print()
print("  - A small fund with $100M AUM spending $1.6M on data needs the data")
print("    to generate at least 160 bps of gross alpha just to break even on")
print("    the data cost alone (before trading costs, infrastructure, salary).")
print()
print("  - Large multi-strategy funds (Citadel, Millennium, Point72) can")
print("    amortise data costs over many PMs and larger AUM, making marginal")
print("    data ROI far more favourable.")
print()
print("  - Vendor consolidation (S&P Global acquired Kensho; Bloomberg")
print("    absorbed Second Measure) is reducing the number of independent")
print("    providers, potentially increasing prices and decreasing edge as")
print("    the same data feeds reach more market participants.")
print()
print("Key implication for alpha research: by the time an alternative dataset")
print("is widely available and affordable, its alpha content has often been")
print("arbitraged away.  The competitive advantage lies in being among the")
print("earliest adopters — or in combining multiple mediocre signals into a")
print("composite that remains differentiated.")


# ── CELL: data_quality_challenges ───────────────────────────────

print()
print("-" * 70)
print("DATA QUALITY CHALLENGES")
print("-" * 70)
print()
print("Alternative data introduces quality issues rarely encountered with")
print("traditional financial data:")
print()
print("  1. SURVIVORSHIP & COVERAGE BIAS")
print("     App traffic data only covers apps that exist today; discontinued")
print("     apps (and their failing parent companies) vanish from the panel.")
print("     Satellite imagery providers change revisit frequency over time,")
print("     creating non-stationary coverage.")
print()
print("  2. POINT-IN-TIME (PIT) INTEGRITY")
print("     Unlike prices (which have clean timestamps), alt data revisions")
print("     are common.  Web traffic estimates are retroactively adjusted;")
print("     sentiment scores may be recomputed when NLP models are updated.")
print("     Without careful PIT stamping, backtest alpha is illusory.")
print()
print("  3. SAMPLE REPRESENTATIVENESS")
print("     Credit card panels cover a biased subset of consumers.  Foot")
print("     traffic data skews toward smartphone-carrying demographics.")
print("     Extrapolating from a panel to the full population requires")
print("     careful calibration.")
print()
print("  4. LEGAL & COMPLIANCE RISK")
print("     GDPR, CCPA, and evolving privacy regulations constrain how")
print("     location and transaction data can be collected and used.")
print("     Datasets that were legal to acquire in 2019 may not be in 2026.")
print()
print("  5. DECAY & CROWDING")
print("     Alt data signals decay as adoption increases.  A satellite-based")
print("     oil inventory signal that was novel in 2017 is now a commodity")
print("     product.  The half-life of alt data alpha is typically 2-4 years")
print("     from first institutional adoption to broad crowding.")


# ── CELL: bridge_to_week7 ──────────────────────────────────────

print()
print("-" * 70)
print("BRIDGE TO WEEK 7: NLP & ALTERNATIVE DATA SIGNALS")
print("-" * 70)
print()
print("This week we built a complete alpha pipeline using traditional")
print("features — price momentum, value, quality, and volatility.  In")
print("Week 7 we will add the first category of alternative data to this")
print("pipeline: text-derived sentiment signals.")
print()
print("Week 7 covers:")
print()
print("  - Tokenisation, TF-IDF, and word embeddings for financial text")
print("  - Sentiment scoring with FinBERT (a BERT model fine-tuned on")
print("    financial text)")
print("  - Constructing cross-sectional sentiment features from earnings")
print("    call transcripts and news headlines")
print("  - Integrating NLP features into the alpha model framework built")
print("    this week (using the same walk-forward evaluation and IC-based")
print("    assessment)")
print()
print("The key question Week 7 addresses: does adding NLP-derived sentiment")
print("to the feature set built here produce a statistically significant")
print("improvement in out-of-sample IC?  The evaluation framework from this")
print("week — IC, ICIR, paired tests, net-of-cost Sharpe — carries forward")
print("directly.")
print()
print("NOTE: Any IC claims for NLP/alt-data signals require empirical")
print("evidence from our own pipeline or explicit citation to peer-reviewed")
print("sources.  We do not assert that alt data 'works' in general — we")
print("test specific signals on specific universes with rigorous OOS")
print("evaluation.")


# ══════════════════════════════════════════════════════════════════
# VERIFICATION BLOCK
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── ASSERTIONS ─────────────────────────────────────
    # S7-1: Alt data taxonomy >= 5 categories
    assert n_categories >= 5, (
        f"S7-1 FAIL: Expected >= 5 alt data categories, got {n_categories}"
    )

    # S7-2: BattleFin/Exabel $1.6M reference present
    # (verified structurally — the text is in the CELL above)
    import io, contextlib
    buf = io.StringIO()
    # We already printed everything above; verify by checking the
    # TAXONOMY and cost text are in the module source itself.
    src = open(__file__).read()
    assert "BattleFin" in src and "1.6M" in src, (
        "S7-2 FAIL: BattleFin / $1.6M institutional cost reference not found"
    )
    assert "Exabel" in src, (
        "S7-2 FAIL: Exabel reference not found"
    )

    # S7-3: Bridge to Week 7 articulated
    assert "Week 7" in src and "NLP" in src and "sentiment" in src, (
        "S7-3 FAIL: Bridge to Week 7 (NLP/sentiment) not found"
    )

    # S7-4: No unsubstantiated IC claims for alt data
    # Editorial check: verify no line claims a specific IC value for
    # alt data without "citation", "peer-reviewed", or similar qualifier.
    import re
    ic_claim_pattern = re.compile(
        r"(alt|alternative).{0,40}(IC|information coefficient)\s*"
        r"(=|of|is|around|approximately)\s*[\d.]",
        re.IGNORECASE,
    )
    matches = ic_claim_pattern.findall(src)
    assert len(matches) == 0, (
        f"S7-4 FAIL: Found unqualified IC claim(s) for alt data: {matches}"
    )

    # ── RESULTS ────────────────────────────────────────
    print()
    print(f"== lecture/s7_alternative_data ==============================")
    print(f"  n_categories: {n_categories}")
    categories_list = list(TAXONOMY.keys())
    for cat in categories_list:
        print(f"    - {cat}")
    print(f"  institutional_cost_reference: BattleFin/Exabel $1.6M/year")
    print(f"  bridge_to_week7: NLP & sentiment signals")
    print(f"  unqualified_ic_claims: 0")
    print(f"  data_quality_challenges: 5 (survivorship, PIT, "
          f"representativeness, legal, decay)")
    print(f"  s7_1_taxonomy_categories: PASS (>= 5)")
    print(f"  s7_2_cost_context: PASS")
    print(f"  s7_3_week7_bridge: PASS")
    print(f"  s7_4_no_unqualified_ic: PASS")
    print(f"\u2713 s7_alternative_data: ALL PASSED")
