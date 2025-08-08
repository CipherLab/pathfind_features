#!/bin/bash

# --- Configuration ---

# Stop on first error
set -e

# Base directory for data and artifacts
BASE_DIR="/home/mat/compression_test"
DATA_DIR="$BASE_DIR/v5.0"
ARTIFACT_DIR="$BASE_DIR/test_artifacts"
LOG_DIR="$BASE_DIR/logs"

# Python environment and script paths
PYTHON_ENV="$BASE_DIR/python_scripts/.venv/bin/python"
TRAIN_SCRIPT="$BASE_DIR/python_scripts/train_gpu_ranker.py"

# --- Control Model: Standard Features ---

CONTROL_FEATURES=$(jq -r '.feature_sets.medium | join(",")' "$DATA_DIR/features.json")

echo "--- 1. Training Control Model ---"
$PYTHON_ENV $TRAIN_SCRIPT \
    --train-data "$DATA_DIR/train.parquet" \
    --validation-data "$DATA_DIR/validation.parquet" \
    --feature-cols "$CONTROL_FEATURES" \
    --target-col "target" \
    --era-col "era" \
    --output-model-path "$ARTIFACT_DIR/control_model.lgb" \
    --feature-map-file "$ARTIFACT_DIR/control_features.json" \
    --log-file "$LOG_DIR/control_training.log" \
    --batch-size 50000

echo "--- Control Model Training Complete ---"

# --- Experimental Model: Relationship Features ---

# Combine original and new features
EXPERIMENTAL_FEATURES=$(jq -r '(.original_features + .relationship_features) | unique | join(",")' "$ARTIFACT_DIR/train_with_bootstrap_features_feature_mapping.json")

echo "--- 2. Training Experimental Model (with checkpointing) ---"
$PYTHON_ENV $TRAIN_SCRIPT \
    --train-data "$ARTIFACT_DIR/train_with_adaptive_target.parquet" \
    --validation-data "$ARTIFACT_DIR/validation_with_adaptive_target.parquet" \
    --feature-cols "$EXPERIMENTAL_FEATURES" \
    --target-col "adaptive_target" \
    --era-col "era" \
    --output-model-path "$ARTIFACT_DIR/experimental_model.lgb" \
    --feature-map-file "$ARTIFACT_DIR/experimental_features.json" \
    --log-file "$LOG_DIR/experimental_training.log" \
    --batch-size 50000 \
    --resume-from-checkpoint \
    --checkpoint-freq 10

echo "--- Experimental Model Training Complete ---"

echo "✅ Experiment Finished!"
