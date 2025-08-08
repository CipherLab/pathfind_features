#!/bin/bash

set -e  # Exit on any error

# Configuration
INPUT_DATA="/home/mat/compression_test/artifacts/adaptive_target_data.parquet"
ARTIFACTS_DIR="/home/mat/compression_test/artifacts"
LOGS_DIR="logs"
PYTHON_EXEC="./.venv/bin/python"
CACHE_DIR="cache/pathfinding_cache"

# Output files
PATHFINDING_OUTPUT="${ARTIFACTS_DIR}/train_with_pathfinding_features.parquet"
PATHFINDING_RESULTS="${ARTIFACTS_DIR}/pathfinding_discovery_results.json"

# Create directories
mkdir -p "${ARTIFACTS_DIR}"
mkdir -p "${LOGS_DIR}"
mkdir -p "${CACHE_DIR}"

echo "=============================================="
echo "PATHFINDING DISCOVERY ON ADAPTIVE TARGETS"
echo "=============================================="
echo "Input Data: ${INPUT_DATA}"
echo "Output: ${PATHFINDING_OUTPUT}"
echo "Cache: ${CACHE_DIR}"
echo ""

# ==============================================================================
# == STEP 1: RUN PATHFINDING DISCOVERY ON ADAPTIVE TARGET DATA              ==
# ==============================================================================
# This runs the creative pathfinding discovery algorithm on our pre-computed
# adaptive targets to find feature relationships that predict the intelligent
# target combinations discovered in the first step.

echo "🚀 Starting Pathfinding Discovery on Adaptive Targets..."
echo "   This will find feature relationships that predict the optimized targets"
echo ""

# Run the bootstrap pipeline with our adaptive target data as input
# This will use the existing adaptive_target column and run pathfinding discovery

./.venv/bin/python python_scripts/create_bootstrap_features.py \
  --input-data "${INPUT_DATA}" \
  --output-data "${PATHFINDING_OUTPUT}" \
  --feature-map-file /home/mat/compression_test/v5.0/features.json \
  --max-features 80 \
  --max-new-features 35 \
  --run-sanity-check \
  --log-file "${LOGS_DIR}/pathfinding_discovery.log" \
  --cache-dir "${CACHE_DIR}" \
  --yolo-mode

echo ""
echo "✅ Pathfinding Discovery Complete!"
echo "   Output data: ${PATHFINDING_OUTPUT}"
echo "   Results file: ${PATHFINDING_RESULTS}"
echo ""

# ==============================================================================
# == STEP 2: ANALYZE RESULTS                                                ==
# ==============================================================================
echo "📊 Analyzing Discovery Results..."
echo ""

# Show summary of discovered relationships
if [ -f "${PATHFINDING_RESULTS}" ]; then
    echo "Discovered Relationships Summary:"
    cat "${PATHFINDING_RESULTS}" | jq '.relationship_discovery.relationships_found | length' 2>/dev/null || echo "No relationships found"
    echo ""
    
    echo "Top 5 Relationship Strengths:"
    cat "${PATHFINDING_RESULTS}" | jq '.relationship_discovery.relationships_found[:5] | .[] | "\(.feature1) <-> \(.feature2): \(.strength)"' 2>/dev/null || echo "No data available"
    echo ""
fi

# Show feature importance
if [ -f "${PATHFINDING_RESULTS}" ]; then
    echo "Feature Importance (Top 10):"
    cat "${PATHFINDING_RESULTS}" | jq '.relationship_discovery.feature_importance | to_entries | sort_by(-.value) | .[:10] | .[] | "\(.key): \(.value)"' 2>/dev/null || echo "No importance data available"
    echo ""
fi

echo "=============================================="
echo "🎉 PATHFINDING DISCOVERY COMPLETE!"
echo "Ready for model training with pathfinding features!"
echo "=============================================="
