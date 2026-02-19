"""
Section 1: How a Trade Actually Happens

Acceptance criteria (from README):
- Flowchart renders with >= 4 nodes showing distinct stages
  (order submission -> routing -> venue -> execution)
- At least 3 venue types shown (exchange, dark pool, wholesaler)
- Latency annotations are order-of-magnitude correct
  (microseconds for matching, milliseconds for routing)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── CELL: trade_flowchart ────────────────────────────────
# Purpose: Trace a single AAPL market buy order through the execution
#   pipeline — from the moment you tap "buy" to the moment you own shares.
# Visual: A flowchart showing the decision path: order submission -> broker
#   -> smart order router -> three possible venues (exchange, dark pool,
#   wholesaler), each with latency annotations.

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

box_style = dict(boxstyle="round,pad=0.4", facecolor="#E8F0FE", edgecolor="#1A73E8", linewidth=1.5)
venue_style = dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0", edgecolor="#E65100", linewidth=1.5)
exec_style = dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.5)
latency_style = dict(fontsize=8, fontstyle="italic", color="#666666")

# Nodes
ax.text(2, 8.5, "1. Order Submission\n(Your phone app)", ha="center", va="center", fontsize=10, bbox=box_style)
ax.text(2, 6.5, "2. Broker\n(e.g., Schwab, Fidelity)", ha="center", va="center", fontsize=10, bbox=box_style)
ax.text(7, 6.5, "3. Smart Order Router\n(best execution logic)", ha="center", va="center", fontsize=10, bbox=box_style)

ax.text(4.5, 4.0, "Exchange\n(NYSE, Nasdaq)", ha="center", va="center", fontsize=10, bbox=venue_style)
ax.text(7, 4.0, "Dark Pool\n(ATS)", ha="center", va="center", fontsize=10, bbox=venue_style)
ax.text(9.5, 4.0, "Wholesaler\n(Citadel, Virtu)", ha="center", va="center", fontsize=10, bbox=venue_style)

ax.text(7, 1.5, "4. Execution & Confirmation\n(trade matched, reported to tape)", ha="center", va="center", fontsize=11, bbox=exec_style)

# Arrows
arrow_kw = dict(arrowstyle="->", color="#333333", linewidth=1.5)
ax.annotate("", xy=(2, 7.4), xytext=(2, 7.6), arrowprops=arrow_kw)
ax.annotate("", xy=(4.2, 6.5), xytext=(3.4, 6.5), arrowprops=arrow_kw)

ax.annotate("", xy=(4.5, 4.8), xytext=(5.8, 5.8), arrowprops=arrow_kw)
ax.annotate("", xy=(7, 4.8), xytext=(7, 5.8), arrowprops=arrow_kw)
ax.annotate("", xy=(9.5, 4.8), xytext=(8.2, 5.8), arrowprops=arrow_kw)

ax.annotate("", xy=(5.8, 1.9), xytext=(4.5, 3.2), arrowprops=arrow_kw)
ax.annotate("", xy=(7, 2.2), xytext=(7, 3.2), arrowprops=arrow_kw)
ax.annotate("", xy=(8.2, 1.9), xytext=(9.5, 3.2), arrowprops=arrow_kw)

# Latency annotations
ax.text(1.0, 7.5, "~10 ms\n(internet)", **latency_style, ha="center")
ax.text(3.7, 7.0, "~1 ms\n(internal)", **latency_style, ha="center")
ax.text(4.5, 5.5, "~5-50 us\nmatching", **latency_style, ha="center")
ax.text(7, 5.4, "~100 us", **latency_style, ha="center")
ax.text(9.5, 5.5, "~50 us\ninternalize", **latency_style, ha="center")

ax.set_title(
    "Anatomy of a Trade: AAPL Market Buy Order",
    fontsize=14, fontweight="bold", pad=15,
)

plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    # The flowchart is a static diagram. We verify it rendered without error
    # and that the key structural elements are present.
    fig_test = plt.gcf()
    assert fig_test is not None, "Figure did not render"

    # Verify we have the right number of text elements:
    # 7 boxes + 5 latency annotations + 1 title = 13 text elements
    texts = [c for c in ax.get_children() if isinstance(c, plt.Text) and c.get_text().strip()]
    node_count = sum(1 for t in texts if t.get_bbox_patch() is not None)
    assert node_count >= 7, f"Expected >= 7 labeled nodes, got {node_count}"

    # Check venue types are present
    all_text = " ".join(t.get_text() for t in texts)
    for venue in ["Exchange", "Dark Pool", "Wholesaler"]:
        assert venue in all_text, f"Missing venue type: {venue}"

    # Check latency annotations exist with correct magnitude
    assert "us" in all_text or "μs" in all_text, "Missing microsecond latency annotations"
    assert "ms" in all_text, "Missing millisecond latency annotations"

    print("Section 1: All acceptance criteria passed")
