"""
Poster figure generator for the baseline dialogue system evaluation.

Produces six publication-quality figures saved as PNG files in this directory:

  fig1_scenario_retention.png  – Retention accuracy per scenario (bar chart)
  fig2_outcome_distribution.png– Overall outcome breakdown (pie chart)
  fig3_coherence_decay.png     – Coherence drift over filler turns (line chart)
  fig4_within_vs_cross.png     – Within-session vs cross-session accuracy (grouped bar)
  fig5_hallucination_rate.png  – Hallucination rate per scenario (bar chart)
  fig6_summary_table.png       – Summary metrics table (rendered as figure)

Usage
-----
    python poster_figures/generate_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = os.path.join(os.path.dirname(__file__), "..")
RESULTS_JSON = os.path.join(ROOT, "results", "baseline_summary.json")
FULL_JSON    = os.path.join(ROOT, "results", "baseline_results.json")
OUT_DIR      = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Colour palette (Newcastle University brand colours + accessible palette)
# ---------------------------------------------------------------------------
NC_PURPLE  = "#6B2D8B"    # Newcastle University purple
NC_TEAL    = "#00B5B0"    # teal accent
NC_RED     = "#C8102E"    # red accent
NC_AMBER   = "#F5A623"    # amber
NC_GREY    = "#6D6E71"    # neutral grey
NC_LGREY   = "#D9D9D9"    # light grey
CORRECT_C  = NC_TEAL
FORGOTTEN_C= NC_AMBER
HALLUC_C   = NC_RED

SCENARIO_LABELS = [
    "S1: Cross-Session\nAmnesia",
    "S2: Context\nOverflow",
    "S3: Multi-Session\nCoherence",
    "S4: Long-Session\nDrift",
    "S5: Rapid Fact\nIntroduction",
]


def load_data():
    with open(RESULTS_JSON, encoding="utf-8") as f:
        summary = json.load(f)
    with open(FULL_JSON, encoding="utf-8") as f:
        full = json.load(f)
    return summary, full


def _save(fig, name: str, dpi: int = 200) -> str:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}")
    return path


# ===========================================================================
# Figure 1 – Retention accuracy per scenario
# ===========================================================================

def fig1_scenario_retention(summary: dict) -> None:
    scenarios = summary["per_scenario"]
    names     = SCENARIO_LABELS
    acc       = [s["retention_accuracy"] * 100 for s in scenarios]
    ci_lo = [s.get("retention_accuracy_ci_95", (0, 1))[0] * 100 for s in scenarios]
    ci_hi = [s.get("retention_accuracy_ci_95", (0, 1))[1] * 100 for s in scenarios]
    err_lo = [a - lo for a, lo in zip(acc, ci_lo)]
    err_hi = [hi - a  for a, hi in zip(acc, ci_hi)]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, acc, color=NC_PURPLE, width=0.55, zorder=3,
                  label="Retention Accuracy")
    ax.errorbar(x, acc,
                yerr=[err_lo, err_hi],
                fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Retention Accuracy (%)", fontsize=11)
    ax.set_title("Baseline Retention Accuracy per Evaluation Scenario\n"
                 "(with 95 % Wilson confidence intervals)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.axhline(y=summary["overall"]["mean_retention_accuracy"] * 100,
               color=NC_RED, linestyle="--", linewidth=1.5,
               label=f"Overall mean ({summary['overall']['mean_retention_accuracy']*100:.1f}%)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    _save(fig, "fig1_scenario_retention.png")


# ===========================================================================
# Figure 2 – Overall outcome distribution (pie / donut)
# ===========================================================================

def fig2_outcome_distribution(full: dict) -> None:
    counts = {"Correct": 0, "Forgotten": 0, "Hallucination": 0}
    for scenario in full["scenarios"]:
        for probe in scenario["probe_results"]:
            key = probe["outcome"].capitalize()
            if key == "Hallucination":
                key = "Hallucination"
            counts[key] += 1

    labels  = list(counts.keys())
    values  = list(counts.values())
    colours = [CORRECT_C, FORGOTTEN_C, HALLUC_C]
    explode = [0.03, 0.03, 0.07]

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colours, explode=explode,
        autopct="%1.1f%%", startangle=140, pctdistance=0.75,
        wedgeprops=dict(linewidth=2, edgecolor="white"),
        textprops=dict(fontsize=11),
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
        at.set_color("white")

    total = sum(values)
    ax.text(0, 0, f"n={total}\nprobes", ha="center", va="center",
            fontsize=11, fontweight="bold", color=NC_GREY)

    ax.set_title("Overall Recall Outcome Distribution – Baseline System\n"
                 "(all scenarios combined)", fontsize=12, fontweight="bold")

    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colours, labels)]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=10)

    _save(fig, "fig2_outcome_distribution.png")


# ===========================================================================
# Figure 3 – Coherence decay over filler turns (Scenario 4)
# ===========================================================================

def fig3_coherence_decay(full: dict) -> None:
    sc4 = next(s for s in full["scenarios"] if s["scenario_id"] == 4)
    decay = sc4["decay_curve"]
    if not decay:
        print("  [skip] No decay curve data for Scenario 4.")
        return

    turns  = [p["filler_turns"] for p in decay]
    acc    = [p["accuracy"] * 100 for p in decay]
    labels = [p["label"] for p in decay]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(turns, acc, color=NC_PURPLE, linewidth=2.5, marker="o",
            markersize=8, zorder=3, label="Recall accuracy")
    ax.fill_between(turns, acc, alpha=0.15, color=NC_PURPLE)

    for t, a, lbl in zip(turns, acc, labels):
        ax.annotate(f"{a:.0f}%", (t, a),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, fontweight="bold", color=NC_PURPLE)

    ax.set_xlabel("Number of Filler Turns After Fact Introduction", fontsize=11)
    ax.set_ylabel("Recall Accuracy (%)", fontsize=11)
    ax.set_title("Baseline Coherence Drift: Recall Accuracy vs. Conversation Length\n"
                 "(Scenario 4 – 1,024-token context window)", fontsize=12, fontweight="bold")
    ax.set_ylim(-5, 115)
    ax.set_xticks(turns)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.axhline(y=50, color=NC_GREY, linestyle=":", linewidth=1,
               label="50 % threshold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    _save(fig, "fig3_coherence_decay.png")


# ===========================================================================
# Figure 4 – Within-session vs cross-session retention (grouped bar)
# ===========================================================================

def fig4_within_vs_cross(summary: dict) -> None:
    scenarios = summary["per_scenario"]
    within = []
    cross  = []
    names  = []
    for s, lbl in zip(scenarios, SCENARIO_LABELS):
        w = s.get("within_session_retention")
        c = s.get("cross_session_retention")
        if w is not None or c is not None:
            names.append(lbl)
            within.append((w or 0) * 100)
            cross.append((c or 0) * 100)

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_w = ax.bar(x - width / 2, within, width, label="Within-session",
                    color=NC_TEAL, zorder=3)
    bars_c = ax.bar(x + width / 2, cross,  width, label="Cross-session",
                    color=NC_RED,  zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("Retention Accuracy (%)", fontsize=11)
    ax.set_title("Within-Session vs. Cross-Session Retention – Baseline System",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 120)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    for bar in bars_w:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=NC_TEAL)
    for bar in bars_c:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=NC_RED)

    _save(fig, "fig4_within_vs_cross.png")


# ===========================================================================
# Figure 5 – Hallucination & forgotten rates per scenario (stacked bar)
# ===========================================================================

def fig5_outcome_breakdown(summary: dict) -> None:
    scenarios = summary["per_scenario"]
    names     = SCENARIO_LABELS
    correct   = [s["retention_accuracy"] * 100 for s in scenarios]
    forgotten = [s["forgotten_rate"]     * 100 for s in scenarios]
    hallucin  = [s["hallucination_rate"] * 100 for s in scenarios]

    x     = np.arange(len(names))
    width = 0.55

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_c = ax.bar(x, correct,  width, label="Correct",       color=CORRECT_C,   zorder=3)
    bar_f = ax.bar(x, forgotten,width, bottom=correct,        label="Forgotten",  color=FORGOTTEN_C, zorder=3)
    bar_h = ax.bar(x, hallucin, width,
                   bottom=[c + f for c, f in zip(correct, forgotten)],
                   label="Hallucination", color=HALLUC_C, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Percentage of Recall Probes (%)", fontsize=11)
    ax.set_title("Recall Outcome Breakdown per Scenario – Baseline System",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    _save(fig, "fig5_outcome_breakdown.png")


# ===========================================================================
# Figure 6 – Summary metrics table
# ===========================================================================

def fig6_summary_table(summary: dict) -> None:
    overall   = summary["overall"]
    scenarios = summary["per_scenario"]

    rows = []
    for s, lbl in zip(scenarios, SCENARIO_LABELS):
        w = s.get("within_session_retention")
        c = s.get("cross_session_retention")
        rows.append([
            lbl.replace("\n", " "),
            f"{s['retention_accuracy']*100:.0f}%",
            f"{s['forgotten_rate']*100:.0f}%",
            f"{s['hallucination_rate']*100:.0f}%",
            f"{s['coherence_score']*100:.0f}%",
            f"{w*100:.0f}%" if w is not None else "—",
            f"{c*100:.0f}%" if c is not None else "—",
        ])

    rows.append([
        "OVERALL (mean)",
        f"{overall['mean_retention_accuracy']*100:.1f}%",
        f"{overall['mean_forgotten_rate']*100:.1f}%",
        f"{overall['mean_hallucination_rate']*100:.1f}%",
        f"{overall['mean_coherence_score']*100:.1f}%",
        (f"{overall['mean_within_session_retention']*100:.1f}%"
         if overall.get("mean_within_session_retention") else "—"),
        (f"{overall['mean_cross_session_retention']*100:.1f}%"
         if overall.get("mean_cross_session_retention") else "—"),
    ])

    col_labels = [
        "Scenario",
        "Retention\nAccuracy",
        "Forgotten\nRate",
        "Hallucin.\nRate",
        "Coherence\nScore",
        "Within-\nSession",
        "Cross-\nSession",
    ]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.0)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor(NC_PURPLE)
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Style overall row
    for j in range(len(col_labels)):
        table[(len(rows), j)].set_facecolor(NC_LGREY)
        table[(len(rows), j)].set_text_props(fontweight="bold")

    # Alternate row colouring
    for i in range(1, len(rows)):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#F7F0FB")

    ax.set_title("Baseline System – Evaluation Metrics Summary",
                 fontsize=12, fontweight="bold", pad=10)

    _save(fig, "fig6_summary_table.png")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("Generating poster figures…")
    summary, full = load_data()

    fig1_scenario_retention(summary)
    fig2_outcome_distribution(full)
    fig3_coherence_decay(full)
    fig4_within_vs_cross(summary)
    fig5_outcome_breakdown(summary)
    fig6_summary_table(summary)

    print(f"\nAll figures saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
