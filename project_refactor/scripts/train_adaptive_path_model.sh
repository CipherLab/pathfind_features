#!/bin/bash

# Script to train model with merged adaptive targets and path features

set -e  # Exit on any error

# Default values
EXPERIMENT_NAME=${1:-"adaptive_path_experiment"}
MODEL_TYPE=${2:-"experimental"}  # experimental, adaptive_only, or control
NUM_LEAVES=${3:-64}
LEARNING_RATE=${4:-0.05}

echo "=== TRAINING $MODEL_TYPE MODEL ==="
echo "Experiment: $EXPERIMENT_NAME"
echo "Model type: $MODEL_TYPE"
echo "Num leaves: $NUM_LEAVES"
echo "Learning rate: $LEARNING_RATE"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

EXPERIMENT_DIR="pipeline_runs/$EXPERIMENT_NAME"

# Check if experiment directory exists
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Experiment directory $EXPERIMENT_DIR does not exist!"
    echo "Run './merge_adaptive_path_features.sh $EXPERIMENT_NAME' first."
    exit 1
fi

echo "Training $MODEL_TYPE model..."

if [ "$MODEL_TYPE" = "experimental" ]; then
    # Train experimental model with both original and path features
    python train_experimental_model_chunked.py \
        --train-data "$EXPERIMENT_DIR/03_features_train.parquet" \
        --validation-data "$EXPERIMENT_DIR/03_features_validation.parquet" \
        --new-feature-names "$EXPERIMENT_DIR/new_feature_names.json" \
        --output-model "$EXPERIMENT_DIR/${MODEL_TYPE}_model.pkl" \
        --features-json "$EXPERIMENT_DIR/features.json" \
        --num-leaves "$NUM_LEAVES" \
        --learning-rate "$LEARNING_RATE"

elif [ "$MODEL_TYPE" = "adaptive_only" ]; then
    # Train adaptive-only model (original features + adaptive targets)
    python train_experimental_model_chunked.py \
        --train-data "$EXPERIMENT_DIR/01_adaptive_targets_train.parquet" \
        --validation-data "$EXPERIMENT_DIR/01_adaptive_targets_validation.parquet" \
        --output-model "$EXPERIMENT_DIR/${MODEL_TYPE}_model.pkl" \
        --num-leaves "$NUM_LEAVES" \
        --learning-rate "$LEARNING_RATE"

elif [ "$MODEL_TYPE" = "control" ]; then
    # Train control model (original features + original targets)
    python train_control_model_chunked.py \
        --train-data v5.0/train.parquet \
        --validation-data v5.0/validation.parquet \
        --output-model "$EXPERIMENT_DIR/${MODEL_TYPE}_model.pkl" \
        --features-json v5.0/features.json \
        --num-leaves "$NUM_LEAVES" \
        --learning-rate "$LEARNING_RATE"

else
    echo "Error: Unknown model type '$MODEL_TYPE'"
    echo "Supported types: experimental, adaptive_only, control"
    exit 1
fi

echo ""
echo "=== TRAINING COMPLETE ==="
echo "Generated files:"
echo "- $EXPERIMENT_DIR/${MODEL_TYPE}_model.pkl"
echo "- $EXPERIMENT_DIR/${MODEL_TYPE}_model.json"

echo ""
echo "Next step:"
echo "Run './benchmark_adaptive_path.sh $EXPERIMENT_NAME $MODEL_TYPE' to evaluate performance"
