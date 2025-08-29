#!/bin/bash

# Single CLI script for running the pipeline and validation.

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

run_pipeline() {
    log "--- Starting Pipeline ---"
    python pipeline/run_pipeline.py run \
        --input-data ../dump/v5.0/train.parquet \
        --features-json ../dump/v5.0/features.json \
        --experiment-name "my_experiment" \
        --max-new-features 30 \
        --force
    log "--- Pipeline Finished ---"
}

run_validation() {
    log "--- Starting Validation ---"
    python pipeline/validation_framework.py | tee -a "$LOG_FILE"
    log "--- Validation Finished ---"
}

# --- Main Script ---

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [pipeline|validate]"
    exit 1
fi

COMMAND=$1

setup_logging

case "$COMMAND" in
    pipeline)
        run_pipeline
        ;; 
    validate)
        run_validation
        ;; 
    *)
        log "Invalid command: $COMMAND"
        echo "Usage: $0 [pipeline|validate]"
        exit 1
        ;; 
esac

log "Run finished at $(date +"%Y%m%d_%H%M%S")"