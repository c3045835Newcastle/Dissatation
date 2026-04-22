#!/usr/bin/env bash
# =============================================================================
# run_full_evaluation.sh
# =============================================================================
# ONE-COMMAND evaluation runner for the CSC3094 dissertation.
#
# Runs the complete pre vs post evaluation suite on your RX 9060 XT system,
# saves all result files, and prints a ready-to-copy comparison table.
#
# Requirements
# ────────────
#   • Python 3.10+ with a Hugging Face-authenticated environment
#   • ROCm 6.2 + PyTorch 2.4.0+rocm6.2 (or CUDA equivalent)
#   • ~16 GB VRAM (FP16) — or use --precision int4 for 5 GB VRAM
#   • ~8 GB system RAM when using FP16 + sentence-transformers
#
# Usage
# ─────
#   Full evaluation (both systems, FP16):
#     bash run_full_evaluation.sh
#
#   Quick smoke-test (INT4, reduced scenarios, ~30 min):
#     bash run_full_evaluation.sh --quick --precision int4
#
#   Post system only (e.g. re-run after code change):
#     bash run_full_evaluation.sh --system post
#
#   Dry-run (tests pipeline without loading the model):
#     bash run_full_evaluation.sh --dry-run
#
# Output
# ──────
#   results/pre_dialogue_quality.json   — pre-system dialogue quality
#   results/post_dialogue_quality.json  — post-system dialogue quality
#   results/pre_post_comparison.json    — delta table
#   (Console prints a formatted comparison table.)
# =============================================================================

set -euo pipefail

# ── Parse arguments ───────────────────────────────────────────────────────────
PRECISION="fp16"
SYSTEM="both"
QUICK=""
DRY_RUN=""
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --precision) PRECISION="$2"; shift 2 ;;
        --system)    SYSTEM="$2";    shift 2 ;;
        --quick)     QUICK="--quick"; shift ;;
        --dry-run)   DRY_RUN="--dry-run"; shift ;;
        --verbose)   VERBOSE="--verbose"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

header() { echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"; }
info()   { echo -e "${YELLOW}▶  $1${NC}"; }
ok()     { echo -e "${GREEN}✓  $1${NC}"; }

header
echo -e "${CYAN}  CSC3094 Dissertation — Full Evaluation Suite${NC}"
header
echo ""
info "Precision : $PRECISION"
info "System    : $SYSTEM"
info "Quick     : ${QUICK:-full}"
info "Dry-run   : ${DRY_RUN:-no}"
echo ""

# ── 1. Install / verify dependencies ─────────────────────────────────────────
info "Step 1/4 — Installing dependencies …"

pip install -q -r requirements.txt

# Optional: bitsandbytes for INT4/INT8
if [[ "$PRECISION" != "fp16" ]]; then
    pip install -q -r requirements-optional.txt
fi

# Memory-system dependencies
pip install -q -r requirements-memory.txt

ok "Dependencies ready."
echo ""

# ── 2. Verify HuggingFace login ───────────────────────────────────────────────
info "Step 2/4 — Checking HuggingFace authentication …"

if ! python -c "from huggingface_hub import whoami; whoami()" 2>/dev/null; then
    echo ""
    echo "  ⚠  Not logged in to HuggingFace."
    echo "  Run:  huggingface-cli login"
    echo "  Then re-run this script."
    if [[ -z "$DRY_RUN" ]]; then
        exit 1
    else
        echo "  Dry-run mode: continuing anyway."
    fi
else
    ok "HuggingFace authenticated."
fi
echo ""

# ── 3. Run the evaluation pipeline ───────────────────────────────────────────
info "Step 3/4 — Running evaluation pipeline …"
echo ""
echo "  This will take approximately:"
echo "  • Full FP16 evaluation (~25 scenarios × 3 runs × 2 systems): 3–6 hours"
echo "  • Quick INT4 evaluation (--quick --precision int4):           30–60 min"
echo "  • Dry-run (--dry-run):                                        <1 min"
echo ""

mkdir -p results

python evaluate_pre_post.py \
    --system    "$SYSTEM" \
    --precision "$PRECISION" \
    --output    results \
    ${QUICK} \
    ${DRY_RUN} \
    ${VERBOSE}

ok "Evaluation complete."
echo ""

# ── 4. Print result file locations ───────────────────────────────────────────
info "Step 4/4 — Result files written:"
echo ""

for f in results/pre_dialogue_quality.json \
          results/post_dialogue_quality.json \
          results/pre_post_comparison.json; do
    if [[ -f "$f" ]]; then
        echo "  ✓  $f"
    fi
done

echo ""
header
echo -e "${CYAN}  Next steps:${NC}"
echo ""
echo "  1.  Open results/pre_post_comparison.json to copy the delta values."
echo "  2.  Fill in the placeholder cells in EVALUATION_RESULTS.md."
echo "  3.  Commit results:  git add results/ && git commit -m 'Add evaluation results'"
echo ""
echo "  For hardware benchmarks (tokens/sec, VRAM, TTFT):"
echo "    python benchmark.py --save results/"
echo ""
header
