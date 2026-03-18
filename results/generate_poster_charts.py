"""
Poster Chart Generator — Base Llama 3.1 8B Evaluation Results.

Produces six publication-quality figures suitable for inclusion in an A1
academic poster:

  1. Retention Accuracy — within-session vs cross-session (bar chart)
  2. Context-Length Degradation — recall vs conversation length (line chart)
  3. Hallucination Rate — base model vs expected improved model (bar chart)
  4. System Progress — Gantt-style progress overview (horizontal bar)
  5. Key Metrics Summary — radar chart across all five metrics
  6. Key Metrics Summary Table — tabular overview of all metrics

Run:
    python generate_poster_charts.py

Output PNG files are saved to the same directory as this script.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

# ── styling ──────────────────────────────────────────────────────────────────
PALETTE = {
    "ncl_blue":    "#002147",   # Newcastle University navy
    "ncl_gold":    "#D0A650",   # Newcastle University gold
    "accent_red":  "#C0392B",
    "accent_green":"#27AE60",
    "light_grey":  "#F5F5F5",
    "mid_grey":    "#AAAAAA",
    "dark_grey":   "#555555",
    "white":       "#FFFFFF",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "baseline_results.json")
OUTPUT_DIR   = os.path.dirname(__file__)

# ── load data ─────────────────────────────────────────────────────────────────

def load_results() -> dict:
    with open(RESULTS_FILE) as f:
        return json.load(f)

# ── Figure 1 — Retention Accuracy ─────────────────────────────────────────────

def fig_retention(data: dict):
    """Bar chart: within-session vs cross-session retention per scenario."""
    scenarios = []
    within   = []
    across   = []

    for r in data["individual_results"]:
        if r["within_session_retention"] is not None:
            label = r["scenario_id"] + "\n" + r["scenario_name"].replace(" ", "\n", 1)
            scenarios.append(r["scenario_id"])
            within.append(r["within_session_retention"] * 100)
            across.append((r["cross_session_retention"] or 0.0) * 100)

    x = np.arange(len(scenarios))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, within, w, label="Within-Session",
                color=PALETTE["ncl_blue"], alpha=0.9)
    b2 = ax.bar(x + w/2, across, w, label="Cross-Session",
                color=PALETTE["accent_red"], alpha=0.9)

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=PALETTE["accent_red"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"Scenario {s}" for s in scenarios], fontsize=10)
    ax.set_ylabel("Retention Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title("Memory Retention Accuracy — Base Model\n"
                 "(Within-Session vs Cross-Session)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper right", framealpha=0.8)

    ax.axhline(y=0, color="black", linewidth=0.8)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig1_retention_accuracy.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Figure 2 — Context-Length Degradation ─────────────────────────────────────

def fig_context_degradation(data: dict):
    """Line chart: retention accuracy vs number of filler turns."""
    lengths = [5, 10, 15, 20]
    vals = [
        data["context_retention_at_5_turns"]  * 100,
        data["context_retention_at_10_turns"] * 100,
        data["context_retention_at_15_turns"] * 100,
        data["context_retention_at_20_turns"] * 100,
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lengths, vals, marker="o", linewidth=2.5,
            color=PALETTE["ncl_blue"], markersize=9, zorder=3)
    ax.fill_between(lengths, vals, alpha=0.15, color=PALETTE["ncl_blue"])

    for x, y in zip(lengths, vals):
        ax.annotate(f"{y:.0f}%", (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=11, fontweight="bold",
                    color=PALETTE["ncl_blue"])

    ax.set_xlabel("Number of Filler Dialogue Turns After Initial Fact", fontsize=12)
    ax.set_ylabel("Recall Accuracy (%)", fontsize=12)
    ax.set_title("Context-Length Degradation\n"
                 "Base Model — Recall of Fact vs Conversation Length",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(lengths)
    ax.set_xticklabels([f"{l} turns" for l in lengths])
    ax.set_ylim(0, 110)

    # Annotate the problem area
    ax.axhspan(0, 40, alpha=0.05, color=PALETTE["accent_red"])
    ax.text(18.5, 20, "Critical\nloss zone", ha="right", va="center",
            fontsize=9, color=PALETTE["accent_red"], style="italic")

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig2_context_degradation.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Figure 3 — Hallucination Rate ─────────────────────────────────────────────

def fig_hallucination(data: dict):
    """Horizontal bar: hallucination rate comparison."""
    systems = [
        "Base Model\n(Llama 3.1 8B)",
        "Expected with\nHierarchical Memory\n(Target)",
    ]
    rates = [data["avg_hallucination_rate"] * 100,
             35.0]  # 35 % target: estimated ~60 % reduction via fact-grounding in stored memories
    colours = [PALETTE["accent_red"], PALETTE["accent_green"]]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(systems, rates, color=colours, alpha=0.9, height=0.45)

    for bar, rate in zip(bars, rates):
        ax.text(rate + 1.5, bar.get_y() + bar.get_height()/2,
                f"{rate:.0f}%", va="center", fontsize=12, fontweight="bold")

    ax.set_xlim(0, 110)
    ax.set_xlabel("Hallucination Rate (%)", fontsize=12)
    ax.set_title("Hallucination Rate\n"
                 "Base Model vs Expected Improved System",
                 fontsize=13, fontweight="bold", pad=10)

    ax.axvline(x=data["avg_hallucination_rate"]*100, linestyle="--",
               linewidth=1, color=PALETTE["accent_red"], alpha=0.5)

    lower_is_better = mpatches.Patch(color=PALETTE["mid_grey"], label="Lower is better ↓")
    ax.legend(handles=[lower_is_better], loc="lower right", fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig3_hallucination_rate.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Figure 4 — Project Progress ───────────────────────────────────────────────

def fig_progress():
    """Gantt-style chart showing dissertation progress."""
    tasks = [
        # (Task, start_week, duration, colour, status)
        ("Literature Review",        1,  4, PALETTE["accent_green"], "Complete"),
        ("Baseline System Build",    4,  3, PALETTE["accent_green"], "Complete"),
        ("Baseline Evaluation",      6,  2, PALETTE["accent_green"], "Complete"),
        ("Memory Architecture Design",7, 3, PALETTE["ncl_gold"],     "In Progress"),
        ("Episodic Memory Module",   8,  3, PALETTE["mid_grey"],     "Planned"),
        ("Semantic Memory Module",  10,  2, PALETTE["mid_grey"],     "Planned"),
        ("Memory Controller",       11,  2, PALETTE["mid_grey"],     "Planned"),
        ("Integration & Testing",   12,  3, PALETTE["mid_grey"],     "Planned"),
        ("Comparative Evaluation",  14,  2, PALETTE["mid_grey"],     "Planned"),
        ("Dissertation Write-Up",    5, 11, PALETTE["ncl_blue"],     "Ongoing"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_positions = range(len(tasks) - 1, -1, -1)

    status_colours = {
        "Complete":    PALETTE["accent_green"],
        "In Progress": PALETTE["ncl_gold"],
        "Planned":     PALETTE["mid_grey"],
        "Ongoing":     PALETTE["ncl_blue"],
    }

    for i, (task, start, dur, colour, status) in zip(y_positions, tasks):
        ax.barh(i, dur, left=start, height=0.5,
                color=colour, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.text(start + dur/2, i, task,
                ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=1, foreground="black")])

    # Week marker: project started in week 1, poster deadline is week 11
    # (approximate based on module timetable; update if your week count differs)
    current_week = 11
    ax.axvline(x=current_week, linestyle="--", linewidth=2,
               color=PALETTE["accent_red"], zorder=5)
    ax.text(current_week + 0.15, len(tasks) - 0.3, "Now\n(Wk 11)",
            fontsize=9, color=PALETTE["accent_red"], fontweight="bold")

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([t[0] for t in tasks], fontsize=9)
    ax.set_xlabel("Project Week", fontsize=11)
    ax.set_xlim(0, 17)
    ax.set_xticks(range(1, 17))
    ax.set_xticklabels([f"W{w}" for w in range(1, 17)], fontsize=8)
    ax.set_title("Project Progress Overview", fontsize=13,
                 fontweight="bold", pad=10)

    # Legend
    patches = [mpatches.Patch(color=v, label=k)
               for k, v in status_colours.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8, ncol=2)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig4_project_progress.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Figure 5 — Radar / Metrics Summary ────────────────────────────────────────

def fig_radar(data: dict):
    """Radar chart comparing base model metrics to target."""
    labels = [
        "Within-Session\nRetention",
        "Cross-Session\nRetention",
        "Dialogue\nCoherence",
        "Low Hallucination\nRate",
        "Long-Context\nRetention",
    ]

    # Base model scores (0–1, higher = better)
    base_scores = [
        data["avg_within_session_retention"],
        data["avg_cross_session_retention"],
        data["avg_coherence_score"],
        1 - data["avg_hallucination_rate"],          # invert: lower hallucination = better
        data["context_retention_at_20_turns"],        # worst case
    ]

    # Target scores for the improved hierarchical memory system (0–1, higher = better).
    # Axis order: Within-Session Retention, Cross-Session Retention, Dialogue Coherence,
    #             Low Hallucination Rate (1 − hallucination), Long-Context Retention.
    # Targets are conservative estimates based on related RAG / memory-augmented systems
    # from the literature (Lewis et al., 2020; Baddeley, 1992).
    target_scores = [0.90, 0.75, 0.85, 0.60, 0.70]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    base_scores   += base_scores[:1]
    target_scores += target_scores[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    ax.plot(angles, base_scores,   color=PALETTE["ncl_blue"],    linewidth=2, label="Base Model")
    ax.fill(angles, base_scores,   color=PALETTE["ncl_blue"],    alpha=0.25)
    ax.plot(angles, target_scores, color=PALETTE["accent_green"], linewidth=2,
            linestyle="--", label="Target (Hierarchical Memory)")
    ax.fill(angles, target_scores, color=PALETTE["accent_green"], alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7)

    ax.set_title("Performance Comparison\nBase Model vs Target System",
                 fontsize=12, fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig5_metrics_radar.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Figure 6 — Key Metrics Summary Table ─────────────────────────────────────

def fig_summary_table(data: dict):
    """Clean table card of all key metrics for the poster."""
    metrics = [
        ("Within-Session Retention",  f"{data['avg_within_session_retention']*100:.1f}%",  "↑ Higher is better"),
        ("Cross-Session Retention",   f"{data['avg_cross_session_retention']*100:.1f}%",   "⚠ 0% — no persistent memory"),
        ("Dialogue Coherence",        f"{data['avg_coherence_score']*100:.1f}%",           "↑ Higher is better"),
        ("Hallucination Rate",        f"{data['avg_hallucination_rate']*100:.1f}%",        "↓ Lower is better"),
        ("Avg Forgotten / Convo",     f"{data['avg_forgotten_per_conversation']:.1f} facts","↓ Lower is better"),
        ("Recall @ 5 turns",          f"{data['context_retention_at_5_turns']*100:.0f}%",  "Baseline reference"),
        ("Recall @ 10 turns",         f"{data['context_retention_at_10_turns']*100:.0f}%", "Moderate degradation"),
        ("Recall @ 15 turns",         f"{data['context_retention_at_15_turns']*100:.0f}%", "Significant degradation"),
        ("Recall @ 20 turns",         f"{data['context_retention_at_20_turns']*100:.0f}%", "Critical degradation"),
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")

    col_labels = ["Metric", "Base Model Score", "Note"]
    cell_text  = [[m, v, n] for m, v, n in metrics]
    col_widths = [0.38, 0.22, 0.40]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor(PALETTE["ncl_blue"])
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for i in range(1, len(metrics) + 1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(PALETTE["light_grey"] if i % 2 == 0 else PALETTE["white"])

    # Highlight 0% cross-session cell
    table[2, 1].set_facecolor("#FDECEA")
    table[2, 1].set_text_props(color=PALETTE["accent_red"], fontweight="bold")

    ax.set_title("Base Model — Key Evaluation Metrics Summary\n"
                 "(Meta-Llama-3.1-8B, 5 independent runs per scenario)",
                 fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig6_metrics_table.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("Generating Poster Charts — Base Llama 3.1 8B Evaluation")
    print("="*60)

    data = load_results()
    print(f"\nLoaded results: {data['model']} ({data['evaluation_date']})")
    print(f"Generating {6} figures...\n")

    fig_retention(data)
    fig_context_degradation(data)
    fig_hallucination(data)
    fig_progress()
    fig_radar(data)
    fig_summary_table(data)

    print("\n✓ All figures saved to:", OUTPUT_DIR)
    print("\nFigures to include in your poster:")
    print("  fig1_retention_accuracy.png   — Retention accuracy bar chart")
    print("  fig2_context_degradation.png  — Context degradation line chart")
    print("  fig3_hallucination_rate.png   — Hallucination rate comparison")
    print("  fig4_project_progress.png     — Project Gantt / progress overview")
    print("  fig5_metrics_radar.png        — Radar chart: base vs target system")
    print("  fig6_metrics_table.png        — Summary metrics table")
