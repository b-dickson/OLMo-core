#!/bin/bash
#
# Attention Scaling Law Experiment Launcher
#
# Launches all training runs for the attention mechanism scaling law experiment.
# Total runs: 60 training + 24 test = 84 runs
#
# Usage:
#   # Launch all runs for all attention types
#   ./scripts/launch_attention_scaling.sh
#
#   # Launch only specific attention type
#   ATTENTION_TYPES="full" ./scripts/launch_attention_scaling.sh
#
#   # Launch only training sizes (skip test sizes)
#   SKIP_TEST=1 ./scripts/launch_attention_scaling.sh
#
#   # Dry run (print commands without executing)
#   DRY_RUN=1 ./scripts/launch_attention_scaling.sh
#

set -e

# Configuration
SCRIPT="src/scripts/train/ladder/attention_scaling_ladder.py"
CLUSTER="${CLUSTER:-ai2/augusta}"
WORKSPACE="${WORKSPACE:-ai2/oe-t-ladder}"
BUDGET="${BUDGET:-ai2/oe-base}"

# Model sizes
TRAIN_SIZES=("60M" "100M" "190M" "370M" "760M")
TEST_SIZES=("600M" "1B")

# Chinchilla multiples
MULTIPLES=("0.5" "1" "2" "4")

# Attention types (can be overridden via env var)
# Available types: full, sliding, gated_deltanet, deltanet, mamba2, rwkv7
ATTENTION_TYPES="${ATTENTION_TYPES:-full sliding gated_deltanet}"

# Flags
DRY_RUN="${DRY_RUN:-0}"
SKIP_TEST="${SKIP_TEST:-0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if script exists
if [ ! -f "$SCRIPT" ]; then
    log_error "Script not found: $SCRIPT"
    log_error "Please run from the repository root directory"
    exit 1
fi

# Count total runs
TOTAL_RUNS=0
for attn in $ATTENTION_TYPES; do
    for size in "${TRAIN_SIZES[@]}"; do
        for mult in "${MULTIPLES[@]}"; do
            ((TOTAL_RUNS++))
        done
    done
    if [ "$SKIP_TEST" != "1" ]; then
        for size in "${TEST_SIZES[@]}"; do
            for mult in "${MULTIPLES[@]}"; do
                ((TOTAL_RUNS++))
            done
        done
    fi
done

log_info "Attention Scaling Law Experiment Launcher"
log_info "=========================================="
log_info "Cluster: $CLUSTER"
log_info "Attention types: $ATTENTION_TYPES"
log_info "Training sizes: ${TRAIN_SIZES[*]}"
if [ "$SKIP_TEST" != "1" ]; then
    log_info "Test sizes: ${TEST_SIZES[*]}"
fi
log_info "Chinchilla multiples: ${MULTIPLES[*]}"
log_info "Total runs to launch: $TOTAL_RUNS"
echo ""

if [ "$DRY_RUN" = "1" ]; then
    log_warn "DRY RUN MODE - Commands will be printed but not executed"
    echo ""
fi

# Launch counter
LAUNCHED=0
FAILED=0

launch_run() {
    local attn="$1"
    local size="$2"
    local mult="$3"
    local name="attn-scaling-${attn}"

    local cmd="python $SCRIPT launch \
        --name=$name \
        --size=$size \
        --attention-type=$attn \
        --chinchilla-multiple=$mult \
        --cluster=$CLUSTER \
        --workspace=$WORKSPACE \
        --budget=$BUDGET \
        --no-follow"

    if [ "$DRY_RUN" = "1" ]; then
        echo "$cmd"
        ((LAUNCHED++))
    else
        log_info "Launching: $attn $size ${mult}x Chinchilla"
        if eval "$cmd"; then
            ((LAUNCHED++))
        else
            log_error "Failed to launch: $attn $size ${mult}x"
            ((FAILED++))
        fi
    fi
}

# Launch training runs
log_info "Launching training runs..."
for attn in $ATTENTION_TYPES; do
    log_info "=== Attention type: $attn ==="
    for size in "${TRAIN_SIZES[@]}"; do
        for mult in "${MULTIPLES[@]}"; do
            launch_run "$attn" "$size" "$mult"
        done
    done
done

# Launch test runs (for validation)
if [ "$SKIP_TEST" != "1" ]; then
    echo ""
    log_info "Launching test/validation runs..."
    for attn in $ATTENTION_TYPES; do
        log_info "=== Attention type: $attn (test) ==="
        for size in "${TEST_SIZES[@]}"; do
            for mult in "${MULTIPLES[@]}"; do
                launch_run "$attn" "$size" "$mult"
            done
        done
    done
fi

# Summary
echo ""
log_info "=========================================="
log_info "Launch Summary"
log_info "=========================================="
log_info "Successfully launched: $LAUNCHED"
if [ "$FAILED" -gt 0 ]; then
    log_error "Failed: $FAILED"
fi

if [ "$DRY_RUN" != "1" ]; then
    echo ""
    log_info "To check status of all runs:"
    for attn in $ATTENTION_TYPES; do
        echo "  python $SCRIPT status --name=attn-scaling-${attn}"
    done

    echo ""
    log_info "To download metrics after completion:"
    for attn in $ATTENTION_TYPES; do
        echo "  python $SCRIPT metrics-all --name=attn-scaling-${attn} --output-dir=./results"
    done

    echo ""
    log_info "To fit scaling laws after downloading metrics:"
    echo "  python scripts/analysis/fit_scaling_laws.py --metrics-dir=./results --output-dir=./analysis"
fi
