"""
Generate publication-quality charts and visualisations from the baseline
evaluation results for use in the dissertation poster.

Usage
-----
    python results/generate_poster_charts.py

Output files (saved in results/charts/)
-----------------------------------------
    fig1_retention_comparison.png    – within-session vs cross-session retention per scenario
    fig2_coherence_comparison.png    – within-session vs cross-session coherence per scenario
    fig3_hallucination_rates.png     – hallucination rates by session type and scenario
    fig4_retention_over_turns.png    – retention accuracy decay across conversation turns
    fig5_aggregate_radar.png         – radar chart of all five aggregate metrics
    fig6_performance_table.png       – rendered table suitable for poster inclusion
"""

import json
import os
import sys

# ── dependency check ──────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("matplotlib / numpy not found. Install with:  pip install matplotlib numpy")
    sys.exit(1)

# ── paths ─────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(_THIS_DIR, "baseline_results.json")
CHARTS_DIR = os.path.join(_THIS_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
BLUE = "#2C6FAC"
RED = "#C0392B"
GREEN = "#27AE60"
ORANGE = "#E67E22"
GREY = "#7F8C8D"
LIGHT_BLUE = "#AED6F1"
LIGHT_RED = "#F1948A"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


# ── load data ─────────────────────────────────────────────────────────────────
def load_results(path=RESULTS_JSON):
    with open(path) as f:
        return json.load(f)


# ── chart helpers ─────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 1 – Retention Accuracy Comparison ─────────────────────────────────
def fig1_retention(data):
    scenarios = data["scenario_results"]
    labels = [s["scenario_id"] for s in scenarios]
    within = [s["within_session_retention"] * 100 for s in scenarios]
    cross = [s["cross_session_retention"] * 100 for s in scenarios]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, within, width, label="Within-Session", color=BLUE, alpha=0.85)
    b2 = ax.bar(x + width / 2, cross, width, label="Cross-Session (Baseline)", color=RED, alpha=0.85)

    ax.set_ylabel("Retention Accuracy (%)")
    ax.set_title("Retention Accuracy: Within-Session vs Cross-Session\n(Baseline: No Persistent Memory)")
    ax.set_xticks(x)
    xticklabels = [f"{s['scenario_id']}\n{s['scenario_name'][:22]}…" if len(s['scenario_name']) > 22 else f"{s['scenario_id']}\n{s['scenario_name']}" for s in scenarios]
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend()

    # Annotate bars
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=9, color=BLUE, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        label_text = f"{h:.0f}%" if h > 0 else "0%"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, label_text,
                ha="center", va="bottom", fontsize=9, color=RED, fontweight="bold")

    ax.axhline(y=data["aggregate_metrics"]["mean_within_session_retention"] * 100,
               color=BLUE, linestyle="--", linewidth=1.2, alpha=0.7, label="_nolegend_")
    ax.text(len(labels) - 0.1, data["aggregate_metrics"]["mean_within_session_retention"] * 100 + 1.5,
            f"Avg {data['aggregate_metrics']['mean_within_session_retention']*100:.1f}%",
            color=BLUE, fontsize=8, ha="right")

    fig.tight_layout()
    save(fig, "fig1_retention_comparison.png")


# ── Figure 2 – Dialogue Coherence ────────────────────────────────────────────
def fig2_coherence(data):
    scenarios = data["scenario_results"]
    labels = [s["scenario_id"] for s in scenarios]
    within = [s["within_session_coherence"] * 100 for s in scenarios]
    cross = [s["cross_session_coherence"] * 100 for s in scenarios]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, within, width, label="Within-Session", color=GREEN, alpha=0.85)
    b2 = ax.bar(x + width / 2, cross, width, label="Cross-Session (Baseline)", color=ORANGE, alpha=0.85)

    ax.set_ylabel("Dialogue Coherence Score (%)")
    ax.set_title("Dialogue Coherence: Within-Session vs Cross-Session\n(Baseline: No Persistent Memory)")
    ax.set_xticks(x)
    ax.set_xticklabels([s["scenario_id"] for s in scenarios])
    ax.set_ylim(0, 110)
    ax.legend()

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=9, color=GREEN, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=9, color=ORANGE, fontweight="bold")

    fig.tight_layout()
    save(fig, "fig2_coherence_comparison.png")


# ── Figure 3 – Hallucination Rates ───────────────────────────────────────────
def fig3_hallucination(data):
    scenarios = data["scenario_results"]
    labels = [s["scenario_id"] for s in scenarios]
    s1_hall = [s["within_session_hallucination"] * 100 for s in scenarios]
    s2_hall = [s["cross_session_hallucination"] * 100 for s in scenarios]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, s1_hall, width, label="Session 1 (Within-Session)", color=LIGHT_BLUE, edgecolor=BLUE, alpha=0.9)
    b2 = ax.bar(x + width / 2, s2_hall, width, label="Session 2 (Cross-Session)", color=LIGHT_RED, edgecolor=RED, alpha=0.9)

    ax.set_ylabel("Hallucination Rate (%)")
    ax.set_title("Hallucination Frequency by Scenario and Session\n(Baseline Model)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 70)
    ax.legend()

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=9)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=9)

    avg_s1 = data["aggregate_metrics"]["mean_within_session_hallucination"] * 100
    avg_s2 = data["aggregate_metrics"]["mean_cross_session_hallucination"] * 100
    ax.axhline(y=avg_s2, color=RED, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(len(labels) - 0.5, avg_s2 + 1, f"Cross-session avg: {avg_s2:.1f}%",
            color=RED, fontsize=8, ha="right")

    fig.tight_layout()
    save(fig, "fig3_hallucination_rates.png")


# ── Figure 4 – Retention Decay Over Turns ────────────────────────────────────
def fig4_retention_decay(data):
    turn_data = data["aggregate_metrics"]["retention_by_turn_within_session"]
    turns = [t["turn"] for t in turn_data]
    retention = [t["retention"] * 100 for t in turn_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(turns, retention, marker="o", color=BLUE, linewidth=2.5, markersize=7, label="Within-Session Retention")
    ax.axhline(y=0, color=RED, linestyle="--", linewidth=2, alpha=0.8, label="Cross-Session Baseline (0%)")

    # Annotate each point
    for t, r in zip(turns, retention):
        ax.annotate(f"{r:.0f}%", (t, r), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color=BLUE)

    ax.fill_between(turns, retention, 0, alpha=0.12, color=BLUE)
    ax.set_xlabel("Conversation Turn")
    ax.set_ylabel("Fact Retention Accuracy (%)")
    ax.set_title("Retention Accuracy Decay Across Conversation Turns\n(Baseline: Within-Session; Cross-Session = 0%)")
    ax.set_xticks(turns)
    ax.set_ylim(-5, 115)
    ax.legend()

    # Annotate the cross-session line
    ax.text(10.2, 2, "Cross-session: 0%\n(all information lost)", color=RED, fontsize=9, ha="left")

    fig.tight_layout()
    save(fig, "fig4_retention_over_turns.png")


# ── Figure 5 – Radar / Spider Chart ──────────────────────────────────────────
def fig5_radar(data):
    m = data["aggregate_metrics"]

    categories = [
        "Within-Session\nRetention",
        "Cross-Session\nRetention",
        "Within-Session\nCoherence",
        "Cross-Session\nCoherence",
        "Memory\nRetrieval",
        "Non-Hallucination\n(Session 1)",
    ]

    values = [
        m["mean_within_session_retention"],
        m["mean_cross_session_retention"],
        m["mean_within_session_coherence"],
        m["mean_cross_session_coherence"],
        m["memory_retrieval_rate"],
        1.0 - m["mean_within_session_hallucination"],
    ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles_plot, values_plot, color=BLUE, linewidth=2)
    ax.fill(angles_plot, values_plot, color=BLUE, alpha=0.25)

    # Reference lines
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot(angles_plot, [r] * (N + 1), color="grey", linewidth=0.5, alpha=0.5)
        ax.text(angles[0], r, f"{r:.0%}", ha="center", va="bottom", fontsize=8, color="grey")

    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.set_title("Baseline Performance Profile\n(Llama 3.1 8B – No Persistent Memory)", pad=20, fontsize=13)

    # Annotate values
    for angle, val, cat in zip(angles, values, categories):
        ax.annotate(f"{val:.0%}", xy=(angle, val), fontsize=9, color=BLUE,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    fig.tight_layout()
    save(fig, "fig5_aggregate_radar.png")


# ── Figure 6 – Summary Table ─────────────────────────────────────────────────
def fig6_summary_table(data):
    m = data["aggregate_metrics"]
    hw = data["hardware_performance"]

    row_labels = [
        "Model",
        "Parameters",
        "GPU",
        "GPU VRAM Used",
        "Context Window",
        "Memory Architecture",
        "─── Evaluation Results ───",
        "Within-Session Retention",
        "Cross-Session Retention",
        "Within-Session Coherence",
        "Cross-Session Coherence",
        "Hallucination Rate (Sess.1)",
        "Hallucination Rate (Sess.2)",
        "Forgotten Info (Cross-Sess.)",
        "Memory Retrieval Rate",
        "─── Performance ───",
        "Avg Tokens / Second",
        "Avg Time-to-First-Token",
        "Model Load Time",
    ]

    row_values = [
        "Llama 3.1 8B (Base)",
        "8.03 billion",
        "NVIDIA RTX 3090 (24 GB)",
        f"{hw.get('peak_gpu_memory_gb', hw.get('gpu_vram_used_peak_gb', 'N/A'))} GB",
        "4,096 tokens (eval window)",
        "None (context window only)",
        "",
        f"{m['mean_within_session_retention']*100:.1f}%",
        f"{m['mean_cross_session_retention']*100:.1f}%",
        f"{m['mean_within_session_coherence']*100:.1f}%",
        f"{m['mean_cross_session_coherence']*100:.1f}%",
        f"{m['mean_within_session_hallucination']*100:.1f}%",
        f"{m['mean_cross_session_hallucination']*100:.1f}%",
        f"{m['mean_forgotten_cross_session']*100:.1f}%",
        f"{m['memory_retrieval_rate']*100:.1f}%",
        "",
        f"{hw['average_tokens_per_second']} tok/s",
        f"{hw['average_time_to_first_token_s']} s",
        f"{hw['model_load_time_s']} s",
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axis("off")

    col_labels = ["Metric", "Value (Baseline)"]
    table_data = list(zip(row_labels, row_values))

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.55)

    # Style the header
    for col in range(2):
        tbl[(0, col)].set_facecolor(BLUE)
        tbl[(0, col)].set_text_props(color="white", fontweight="bold")

    # Style section dividers and highlight key results
    key_result_rows = {8, 9, 10, 11, 12, 13, 14, 15}
    divider_rows = {7, 16}
    for row in range(1, len(table_data) + 1):
        for col in range(2):
            cell = tbl[(row, col)]
            label = row_labels[row - 1]
            if label.startswith("─"):
                cell.set_facecolor("#D5E8FA")
                cell.set_text_props(fontweight="bold", color=BLUE)
            elif row in key_result_rows:
                val = row_values[row - 1]
                if val.startswith("0.0") or val == "0%":
                    cell.set_facecolor("#FDEDEC")  # light red for 0% cross-session
                elif val.endswith("%") and float(val[:-1]) > 50:
                    cell.set_facecolor("#EBF5FB")  # light blue for good results
            if col == 0:
                cell.set_text_props(color="#2C3E50")
            elif col == 1:
                cell.set_text_props(fontweight="bold")

    ax.set_title("Baseline System Summary – Llama 3.1 8B (No Persistent Memory)",
                 fontsize=13, fontweight="bold", pad=10, color="#2C3E50")
    fig.tight_layout()
    save(fig, "fig6_performance_table.png")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("Generating poster charts from baseline results…")
    print(f"  Input : {RESULTS_JSON}")
    print(f"  Output: {CHARTS_DIR}/\n")

    data = load_results()

    fig1_retention(data)
    fig2_coherence(data)
    fig3_hallucination(data)
    fig4_retention_decay(data)
    fig5_radar(data)
    fig6_summary_table(data)

    print("\nAll charts generated successfully.")
    print("These files are ready to be included in your A1 poster.")


if __name__ == "__main__":
    main()
