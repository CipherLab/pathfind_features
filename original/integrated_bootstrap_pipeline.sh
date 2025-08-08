#!/bin/bash

set -e  # Exit on any error

# Configuration
INPUT_DATA="/home/mat/compression_test/v5.0/train.parquet"
ARTIFACTS_DIR="/home/mat/compression_test/artifacts"
LOGS_DIR="logs"
PYTHON_EXEC="./python_scripts/.venv/bin/python"

# Output file
FINAL_TRAINING_DATA="${ARTIFACTS_DIR}/train_with_bootstrap_features.parquet"

# Create directories
mkdir -p "${ARTIFACTS_DIR}"
mkdir -p "${LOGS_DIR}"

echo "=============================================="
echo "UNIFIED BOOTSTRAP FEATURE CREATION PIPELINE"
echo "=============================================="
echo "Input Data: ${INPUT_DATA}"
echo "Output Dir: ${ARTIFACTS_DIR}"

# ==============================================================================
# == QUICK TUNING RUN (DEFAULT)                                             ==
# ==============================================================================
# Use this for rapid parameter iteration. It runs on a small subset of recent
# eras and skips the time-consuming walk-forward analysis.

echo "Running in QUICK TUNE mode..."

./python_scripts/.venv/bin/python python_scripts/create_bootstrap_features.py \
  --input-data /home/mat/compression_test/v5.0/train.parquet \
  --output-data /home/mat/compression_test/artifacts/train_with_bootstrap_features.parquet \
  --feature-map-file /home/mat/compression_test/v5.0/features.json \
  --max-features 80 \
  --max-new-features 8 \
  --run-sanity-check \
  --log-file logs/fixed_bootstrap.log \
  --yolo-mode \
  --quick-tune \
  --max-tuning-eras 30 \
  --skip-walk-forward \
  --cache-dir "cache/bootstrap_cache" # Add this line
  # --force-recache # Uncomment to force a re-run of Stage 1

# ==============================================================================
# == FULL VALIDATION RUN (COMMENTED OUT)                                    ==
# ==============================================================================
# This is the full, methodologically pure run. Use it when you are confident
# in your parameters and want to generate the final feature set.

# echo "Running in FULL VALIDATION mode..."

# ./python_scripts/.venv/bin/python python_scripts/create_bootstrap_features.py \
#   --input-data /home/mat/compression_test/v5.0/train.parquet \
#   --output-data /home/mat/compression_test/artifacts/train_with_bootstrap_features.parquet \
#   --feature-map-file /home/mat/compression_test/v5.0/features.json \
#   --max-features 80 \
#   --max-new-features 8 \
#   --run-sanity-check \
#   --log-file logs/fixed_bootstrap.log \
#   --yolo-mode


echo "✅ Pipeline complete!"
echo "   Final training data created at: ${FINAL_TRAINING_DATA}"

echo "=============================================="
echo "🎉 UNIFIED BOOTSTRAP PIPELINE SUCCESS!"
echo "Ready for model training with enhanced data!"
echo "=============================================="
