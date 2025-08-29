#!/bin/bash

# Single CLI script for training and validation.

set -e  # Exit on any error

# --- Configuration ---
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# --- Functions ---

log() {
    echo "$1" | tee -a "$LOG_FILE"
}

setup_logging() {
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"
    log "Starting run at $TIMESTAMP"
}

activate_venv() {
    log "Activating virtual environment..."
    # The venv is now in the dump folder, so we need to go up one level
    source ../dump/.venv/bin/activate
}

train_models() {
    log "--- Starting Model Training ---"

    log "Training Control Model..."
    python src/training/train_control_model_chunked.py \
        --train-data dump/v5.0/train.parquet \
        --validation-data dump/v5.0/validation.parquet \
        --output-model dump/pipeline_runs/my_experiment/control_model.pkl \
        --features-json dump/v5.0/features.json | tee -a "$LOG_FILE"

    log "Training Adaptive-Only Model..."
    python src/training/train_experimental_model_chunked.py \
        --train-data dump/pipeline_runs/my_experiment/01_adaptive_targets_train.parquet \
        --validation-data dump/pipeline_runs/my_experiment/01_adaptive_targets_validation.parquet \
        --output-model dump/pipeline_runs/my_experiment/adaptive_only_model.pkl | tee -a "$LOG_FILE"

    log "Training Experimental Model..."
    python src/training/train_experimental_model_chunked.py \
        --train-data dump/pipeline_runs/my_experiment/03_features_train.parquet \
        --validation-data dump/pipeline_runs/my_experiment/03_features_validation.parquet \
        --new-feature-names dump/pipeline_runs/my_experiment/new_feature_names.json \
        --output-model dump/pipeline_runs/my_experiment/experimental_model.pkl \
        --features-json dump/v5.0/features.json | tee -a "$LOG_FILE"

    log "--- Model Training Finished ---"
}

run_validation() {
    log "--- Starting Validation ---"
    python src/analysis/validation_framework.py | tee -a "$LOG_FILE"
    log "--- Validation Finished ---"
}

# --- Main Script ---

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [train|validate]"
    exit 1
fi

COMMAND=$1

setup_logging
activate_venv

case "$COMMAND" in
    train)
        train_models
        ;;
    validate)
        run_validation
        ;;
    *)
        log "Invalid command: $COMMAND"
        echo "Usage: $0 [train|validate]"
        exit 1
        ;;
esac

log "Run finished at $(date +"%Y%m%d_%H%M%S")"