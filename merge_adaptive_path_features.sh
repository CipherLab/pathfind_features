#!/bin/bash

# Script to merge adaptive targets with path features
# Combines the adaptive target discovery with path feature engineering

set -e  # Exit on any error

# Default values
EXPERIMENT_NAME=${1:-"adaptive_path_experiment"}
MAX_NEW_FEATURES=${2:-8}
SMOKE_MODE=${3:-false}

echo "=== MERGING ADAPTIVE TARGETS WITH PATH FEATURES ==="
echo "Experiment: $EXPERIMENT_NAME"
echo "Max new features: $MAX_NEW_FEATURES"
echo "Smoke mode: $SMOKE_MODE"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Create experiment directory
EXPERIMENT_DIR="pipeline_runs/$EXPERIMENT_NAME"
mkdir -p "$EXPERIMENT_DIR"

echo "Running pipeline to merge adaptive targets with path features..."

# Build smoke mode arguments
SMOKE_ARGS=""
if [ "$SMOKE_MODE" = "true" ]; then
    SMOKE_ARGS="--smoke-mode --smoke-max-eras 60 --smoke-row-limit 150000 --smoke-feature-limit 300"
fi

# Run the pipeline with pathfinding enabled
python run_pipeline.py run \
    --input-data v5.0/train.parquet \
    --features-json v5.0/features.json \
    --experiment-name "$EXPERIMENT_NAME" \
    --max-new-features "$MAX_NEW_FEATURES" \
    --pretty \
    $SMOKE_ARGS

echo ""
echo "=== MERGE COMPLETE ==="
echo "Generated artifacts:"
echo "- $EXPERIMENT_DIR/01_adaptive_targets.parquet (adaptive targets)"
echo "- $EXPERIMENT_DIR/02_discovered_relationships.json (path relationships)"
echo "- $EXPERIMENT_DIR/03_enhanced_features.parquet (merged features)"
echo "- $EXPERIMENT_DIR/new_feature_names.json (path feature names)"
echo "- $EXPERIMENT_DIR/features.json (merged feature list)"

echo ""
echo "Next steps:"
echo "1. Run './train_adaptive_path_model.sh $EXPERIMENT_NAME' to train the model"
echo "2. Run './benchmark_adaptive_path.sh $EXPERIMENT_NAME' to evaluate performance"