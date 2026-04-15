#!/bin/bash
# =============================================================================
# run_all.sh — Full reproducible pipeline for ShilmanDB
#
# Builds, tests, generates data, trains ML models, benchmarks against SQLite,
# and produces visualization charts. One command to reproduce everything.
#
# Usage:
#   bash scripts/run_all.sh           # run full pipeline from project root
#   ./scripts/run_all.sh              # same (script cd's to project root)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate Python venv
source .venv/bin/activate

PYTHON=".venv/bin/python"
TPCH_DATA_DIR="bench/tpch_data"
RESULTS_DIR="bench/results"

# ML artifacts
JOIN_MODEL_DIR="ml/join_optimizer/models"
EVICTION_MODEL_DIR="ml/eviction_policy/models"
JOIN_TRAINING_CSV="$TPCH_DATA_DIR/join_training_data.csv"
ZIPFIAN_TRACE="$TPCH_DATA_DIR/zipfian_access_trace.csv"
EVICTION_TRAINING_NPZ="$TPCH_DATA_DIR/eviction_training_data.npz"

# Temp DB files (process-isolated to avoid clobbering on parallel runs)
VERIFY_DB=$(mktemp /tmp/shilmandb_verify_XXXXXX.db)
trap 'rm -f "$VERIFY_DB"' EXIT

# Track timing
PIPELINE_START=$(date +%s)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

step_header() {
    local step="$1"
    local desc="$2"
    echo ""
    echo "========================================================================"
    echo "=== Step ${step}: ${desc}"
    echo "========================================================================"
    echo ""
}

elapsed_since() {
    local start="$1"
    local now
    now=$(date +%s)
    local secs=$(( now - start ))
    local mins=$(( secs / 60 ))
    local rem=$(( secs % 60 ))
    if [ "$mins" -gt 0 ]; then
        echo "${mins}m ${rem}s"
    else
        echo "${secs}s"
    fi
}

check_file() {
    local path="$1"
    local desc="$2"
    if [ ! -f "$path" ]; then
        echo "ERROR: ${desc} not found at ${path}"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Step 1: Build (Debug, no ML)
# ---------------------------------------------------------------------------
step_header 1 "Build (Debug, no ML)"

STEP_START=$(date +%s)
bash scripts/build.sh Debug OFF
echo "Step 1 completed in $(elapsed_since "$STEP_START")"

# ---------------------------------------------------------------------------
# Step 2: Run unit tests
# ---------------------------------------------------------------------------
step_header 2 "Run unit tests"

STEP_START=$(date +%s)
(cd build && ctest --output-on-failure)
echo "Step 2 completed in $(elapsed_since "$STEP_START")"

# ---------------------------------------------------------------------------
# Step 3: Generate TPC-H data
# ---------------------------------------------------------------------------
step_header 3 "Generate TPC-H data"

STEP_START=$(date +%s)

# Skip if all scale factors already present
ALL_DATA_PRESENT=true
for sf in 0.01 0.1 1.0; do
    if [ ! -f "$TPCH_DATA_DIR/sf${sf}/lineitem.tbl" ]; then
        ALL_DATA_PRESENT=false
        break
    fi
done

if [ "$ALL_DATA_PRESENT" = true ]; then
    echo "TPC-H data already exists for all scale factors — skipping generation."
else
    bash bench/generate_tpch_data.sh
fi

echo "Step 3 completed in $(elapsed_since "$STEP_START")"

# ---------------------------------------------------------------------------
# Step 4: Load + verify queries (non-ML, SF=0.01)
# ---------------------------------------------------------------------------
step_header 4 "Load + verify queries (non-ML, SF=0.01)"

STEP_START=$(date +%s)

./build/bench/load_tpch \
    --sf 0.01 \
    --db-file "$VERIFY_DB" \
    --data-dir "$TPCH_DATA_DIR/sf0.01/"

echo "Step 4 completed in $(elapsed_since "$STEP_START")"

# ---------------------------------------------------------------------------
# Step 5: Generate ML training data
# ---------------------------------------------------------------------------
step_header 5 "Generate ML training data"

STEP_START=$(date +%s)

echo "--- 5a: Join optimizer training data ---"
echo ""
$PYTHON ml/join_optimizer/generate_training_data.py --perturbations 80
check_file "$JOIN_TRAINING_CSV" "Join training data CSV"
echo ""

echo "--- 5b: Eviction policy — Zipfian access trace ---"
echo ""
$PYTHON ml/eviction_policy/generate_zipfian_trace.py \
    --num-accesses 1000000 \
    --output "$ZIPFIAN_TRACE"
check_file "$ZIPFIAN_TRACE" "Zipfian access trace"
echo ""

echo "--- 5c: Eviction policy — reuse distances + training data ---"
echo ""
$PYTHON ml/eviction_policy/compute_reuse_distances.py \
    --trace "$ZIPFIAN_TRACE" \
    --output "$EVICTION_TRAINING_NPZ"
check_file "$EVICTION_TRAINING_NPZ" "Eviction training data NPZ"

echo ""
echo "Step 5 completed in $(elapsed_since "$STEP_START")"

# ---------------------------------------------------------------------------
# Step 6: Train models
# ---------------------------------------------------------------------------
step_header 6 "Train ML models"

STEP_START=$(date +%s)

echo "--- 6a: Train join optimizer (wide architecture, cosine LR) ---"
echo ""
$PYTHON ml/join_optimizer/train_join_model.py \
    --model-variant wide \
    --lr 1e-3 \
    --batch-size 64 \
    --epochs 300 \
    --lr-schedule cosine \
    --patience 50
echo ""

echo "--- 6b: Train eviction policy ---"
echo ""
$PYTHON ml/eviction_policy/train_eviction_model.py \
    --smooth-alpha 0 \
    --init-scale 1.0 \
    --lr 5e-4 \
    --epochs 200

echo ""
echo "Step 6 completed in $(elapsed_since "$STEP_START")"

# ---------------------------------------------------------------------------
# Step 7: Export models to TorchScript
# ---------------------------------------------------------------------------
step_header 7 "Export models to TorchScript"

STEP_START=$(date +%s)

echo "--- 7a: Export join optimizer model ---"
echo ""
$PYTHON ml/join_optimizer/export_model.py
check_file "$JOIN_MODEL_DIR/join_order_model.pt" "Join order TorchScript model"
echo ""

echo "--- 7b: Export eviction policy model ---"
echo ""
$PYTHON ml/eviction_policy/export_model.py
check_file "$EVICTION_MODEL_DIR/eviction_model.pt" "Eviction TorchScript model"

echo ""
echo "Step 7 completed in $(elapsed_since "$STEP_START")"

# ---------------------------------------------------------------------------
# Step 8: Evaluate models (gate metrics)
# ---------------------------------------------------------------------------
step_header 8 "Evaluate models (gate metrics)"

STEP_START=$(date +%s)

echo "--- 8a: Join optimizer evaluation ---"
echo ""
$PYTHON ml/join_optimizer/evaluate_join_model.py
echo ""

echo "--- 8b: Eviction policy evaluation ---"
echo ""
$PYTHON ml/eviction_policy/evaluate_eviction_model.py

echo ""
echo "Step 8 completed in $(elapsed_since "$STEP_START")"

# ---------------------------------------------------------------------------
# Step 9: Setup LibTorch + rebuild with ML (Release)
# ---------------------------------------------------------------------------
step_header 9 "Setup LibTorch + rebuild (Release, ML enabled)"

STEP_START=$(date +%s)

echo "--- 9a: Setup LibTorch ---"
echo ""
bash scripts/setup_libtorch.sh
echo ""

echo "--- 9b: Build (Release, ML=ON) ---"
echo ""
bash scripts/build.sh Release ON

echo ""
echo "Step 9 completed in $(elapsed_since "$STEP_START")"

# ---------------------------------------------------------------------------
# Step 10: Run full benchmarks
# ---------------------------------------------------------------------------
step_header 10 "Run full benchmarks (ShilmanDB vs SQLite)"

STEP_START=$(date +%s)

$PYTHON bench/run_benchmarks.py --config all --sf all

echo ""
echo "Step 10 completed in $(elapsed_since "$STEP_START")"

# ---------------------------------------------------------------------------
# Step 11: Generate charts
# ---------------------------------------------------------------------------
step_header 11 "Generate visualization charts"

STEP_START=$(date +%s)

$PYTHON bench/plot_results.py --results-dir "$RESULTS_DIR"

echo ""
echo "Step 11 completed in $(elapsed_since "$STEP_START")"

# ---------------------------------------------------------------------------
# Step 12: Summary
# ---------------------------------------------------------------------------
step_header 12 "Pipeline summary"

TOTAL_ELAPSED=$(elapsed_since "$PIPELINE_START")

echo "Total pipeline time: $TOTAL_ELAPSED"
echo ""
echo "Artifacts produced:"
echo "  Build:"
echo "    build/                          — Release build with ML"
echo ""
echo "  TPC-H data:"
echo "    $TPCH_DATA_DIR/sf0.01/          — Scale factor 0.01"
echo "    $TPCH_DATA_DIR/sf0.1/           — Scale factor 0.1"
echo "    $TPCH_DATA_DIR/sf1.0/           — Scale factor 1.0"
echo ""
echo "  ML training data:"
echo "    $JOIN_TRAINING_CSV"
echo "    $ZIPFIAN_TRACE"
echo "    $EVICTION_TRAINING_NPZ"
echo ""
echo "  ML models:"

for f in "$JOIN_MODEL_DIR"/join_order_model.pt \
         "$JOIN_MODEL_DIR"/join_order_model_best.pt \
         "$JOIN_MODEL_DIR"/feature_stats.pt \
         "$EVICTION_MODEL_DIR"/eviction_model.pt \
         "$EVICTION_MODEL_DIR"/eviction_model_best.pt \
         "$EVICTION_MODEL_DIR"/model_config.pt; do
    if [ -f "$f" ]; then
        size=$(du -h "$f" | cut -f1 | xargs)
        echo "    $f ($size)"
    else
        echo "    $f (missing)"
    fi
done

echo ""
echo "  Benchmark results:"
echo "    $RESULTS_DIR/latencies.csv      — Merged latency data"

for chart in query_latency.png join_order_accuracy.png hit_rate.png; do
    path="$RESULTS_DIR/$chart"
    if [ -f "$path" ]; then
        echo "    $path"
    fi
done

echo ""
echo "Pipeline complete."
