# Week 1 Research Notes: How Markets Work & Financial Data

**Research tier:** LIGHT
**Date:** 2026-02-15
**Queries run:**
1. `financial markets data engineering python 2025 2026 new developments tools polars duckdb`
2. `quantitative finance data course syllabus 2025 2026 university markets data survivorship bias`
3. `market microstructure changes 2025 2026 exchange order types retail trading PFOF regulation`
4. `SEC round lot reform tiered system 2025 2026 tick size access fee Regulation NMS compliance details`
5. `SEC odd lot quote visibility SIP 2026 round lot tiers stock price breakpoints`
6. `UChicago FINM 32900 "Full Stack Quantitative Finance" syllabus` (+ direct fetch of finm-32900.github.io)

## Key Findings

### Latest Papers & Results
- **Ernst & Spatt, "Regulating Market Microstructure" (Annual Review of Financial Economics, Vol. 17, pp. 173-187, Nov 2025; CC BY 4.0):** Survey covering regulation of equity, option, and fixed-income market microstructure. Covers best execution, trade-through rules (Reg NMS), PFOF, tick sizes, access fees, auctions, transparency, and short-selling restrictions. Central thesis: markets are heavily regulated due to externalities, adverse selection, and moral hazard. Notes that equity/option/fixed-income markets have different regulatory regimes. Identifies litigation's growing role in shaping regulatory outcomes. **Directly relevant** as a reference — it covers almost exactly the topics in our Week 1.
- **Boulton, Shohfi & Walz, "How Does Payment for Order Flow Influence Markets? Evidence from Robinhood Crypto Token Introductions" (SEC DERA Working Paper, Jan 2025):** Uses Robinhood Crypto's staggered token introductions as a PFOF shock. Finds crypto PFOF fees are 4.5-45x higher than equity/options PFOF. After the shock: trading volume drops (except BTC/ETH), order imbalances shift to net sales, spreads and volatility rise, increasing daily trading costs by ~$4.8M. **Useful for Week 1** as a concrete example of PFOF's market quality effects — and a bridge to Week 18 (crypto).

### Current State of the Art
- **No paradigm shifts in fundamentals.** Exchanges, order types, bid-ask spreads, market makers, clearing and settlement — all structurally unchanged. The concepts we'd teach are stable.
- **Tiered round-lot sizes (SEC, live Nov 2025):** The SEC replaced the fixed 100-share round lot with price-tiered lots: stocks ≤$250 keep 100-share lots; above $250 the lot shrinks to 40, 10, or 1 share. In practice this only affects ~200 high-priced names (the other ~4,700 NMS stocks are unchanged). The round-lot redefinition went live in November 2025. **Odd-lot quote visibility (May 2026):** SIPs will begin disseminating top-of-book odd-lot quotes priced at or better than the NBBO — including a new "Best Odd-Lot Order" (BOLO) field. This is a big deal for retail investors who trade in odd lots and currently can't see their own liquidity on the consolidated tape.
- **Tick-size and access-fee overhaul (delayed to Nov 2026):** New Rule 612 adds a $0.005 (half-penny) minimum tick for stocks ≥$1 with tight spreads (time-weighted avg quoted spread ≤$0.015). Access fee caps drop from 30 mils to 10 mils per share. A new "fee determinability" rule (610(d), compliance Feb 2026) requires exchange fees to be known at execution time — ending the current practice of month-end tiered rebate schedules. Rule 611 (Order Protection Rule) debate ongoing — could reshape order routing.
- **EU PFOF ban (effective 2025):** Europe banned payment for order flow under MiFID III. US PFOF remains legal. This creates a useful teaching contrast.

### University Course Coverage
- **UChicago FINM 32900 "Full Stack Quantitative Finance" (Winter 2026, Jeremy Bejarano):** Focused on **data engineering and tooling**, not market mechanics. 9-week course covering: Git/reproducible workflows, WRDS/SQL data access, build automation (PyDoit), Jupyter reporting, alternative data sources (Bloomberg, LSEG Datastream, Databento), Python packaging, testing, CI/CD dashboards. Key datasets: CRSP, Compustat, FINRA TRACE, OptionMetrics, EDGAR, NYSE HF data, Treasury auctions. Uses standard Pandas/NumPy/SciPy stack (no Polars/DuckDB). **Takeaway:** This course is complementary to ours — they teach the engineering pipeline, we teach the domain + ML. We should reference their canonical dataset list (CRSP, Compustat, etc.) as "what you'd use at a fund" and mention WRDS as the standard academic data source.
- **Duke MIDS (Spring 2026):** Offering crypto trading and financial crime detection mini-courses — signals growing emphasis on crypto as a distinct asset class and regulatory/compliance applications.
- **University of Bath MSc Financial Mathematics with Data Science:** Combines classical math finance with data science methods — the same blend we're targeting.
- **Survivorship bias remains a standard topic** in CFA curriculum and university courses — no change in how it's taught, but it's still frequently cited as the #1 data pitfall.

### Tools & Libraries
- **Polars + DuckDB are now the standard stack** for financial data engineering alongside (or replacing) pandas. Multiple 2025 tutorials show building research databases with Polars + DuckDB + Parquet in under 10 minutes. This is directly relevant to Week 1's data pipeline exercises.
- **yfinance remains the go-to free data source** for course/educational use, but its reliability continues to be inconsistent (rate limits, schema changes). Worth mentioning alternatives (IBKR API, Databento, Polygon.io) as professional-grade options.
- **Parquet as the default storage format** for financial data — CSV is now clearly legacy for anything beyond toy datasets.

### Paradigm Shifts (if any)
- No fundamental shifts. The biggest change is the SEC round-lot/tick-size reform, which is worth a "did you know?" moment but doesn't change what we teach — it enriches it.

### Practitioner Reality
- **Polars adoption is real but not universal.** Many quant shops still use pandas internally; Polars shows up in new projects and performance-critical pipelines. The practical recommendation is "learn both, prefer Polars for new work."
- **Data vendor landscape is fragmenting:** Databento, Polygon.io, and similar API-first vendors are gaining share vs. legacy terminal-based vendors (Bloomberg, Refinitiv). For a course, yfinance + free sources remain the practical choice, but mentioning the professional landscape matters.

## Implications for This Week's Content
- **Include the SEC round-lot reform** as a "did you know?" moment when teaching order types — it's recent, concrete, and shows that market structure evolves.
- **Mention the EU PFOF ban vs. US status** when discussing market makers and order routing — great teaching contrast.
- **Use Polars alongside pandas** in data pipeline demonstrations. Don't force a full switch (students may know pandas), but show Polars for the performance-critical parts and explain why the industry is moving.
- **Use Parquet as the default storage format** in all data exercises. CSV only for initial download/display.
- **Reference the Annual Reviews microstructure regulation paper** in the reading list.
- **Survivorship bias treatment is fine as planned** — no new developments change the approach, but ground it with a concrete example (e.g., "if you downloaded S&P 500 constituents today and backtested to 2010, you'd miss every company that went bankrupt or was delisted").
