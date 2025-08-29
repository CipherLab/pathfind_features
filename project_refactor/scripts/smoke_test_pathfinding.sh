#!/bin/bash

# Smoke test for pathfinding to estimate performance and runtime
# Uses very small parameters to complete quickly

set -e  # Exit on any error

echo "Activating virtual environment..."
source /home/mat/Downloads/pathfind_features/.venv/bin/activate

echo "=== PATHFINDING SMOKE TEST ==="
echo "Running with minimal parameters to estimate performance..."
echo ""

# Record start time
start_time=$(date +%s)

# Run pathfinding with smoke test parameters
cd /home/mat/Downloads/pathfind_features
PYTHONPATH=/home/mat/Downloads/pathfind_features /home/mat/Downloads/pathfind_features/.venv/bin/python bootstrap_pipeline/steps/step_02_pathfinding.py \
    --input-file pipeline_runs/my_experiment/01_adaptive_targets_train.parquet \
    --target-col adaptive_target \
    --output-relationships-file pipeline_runs/my_experiment/02_relationships_smoke_test.json \
    --debug \
    --debug-every-rows 1000 \
    --cache-dir cache/pathfinding_cache \
    --run-sanity-check \
    --pf-feature-cap 4 \
    --n-paths 4 \
    --max-path-length 3 \
    --top-k 10 \
    --batch-size 5000 \
    --last-n-eras 5 \
    --row-limit 50000

# Record end time and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=== SMOKE TEST RESULTS ==="
echo "Duration: $duration seconds ($(($duration / 60)) minutes $(($duration % 60)) seconds)"
echo ""

# Analyze results
/home/mat/Downloads/pathfind_features/.venv/bin/python -c "
import json
import os

# Load smoke test results
if os.path.exists('pipeline_runs/my_experiment/02_relationships_smoke_test.json'):
    with open('pipeline_runs/my_experiment/02_relationships_smoke_test.json', 'r') as f:
        relationships = json.load(f)
    
    print(f'Relationships found: {len(relationships)}')
    if relationships:
        strengths = [r['strength'] for r in relationships]
        print(f'Strength range: {min(strengths):.3f} - {max(strengths):.3f}')
        print(f'Average strength: {sum(strengths)/len(strengths):.3f}')
        
        # Show top relationships
        print('\nTop relationships:')
        for i, rel in enumerate(relationships[:3]):
            print(f'  {i+1}. {rel[\"feature1\"][-10:]} x {rel[\"feature2\"][-10:]} (strength: {rel[\"strength\"]:.3f})')
else:
    print('No relationships file found')

# Load debug file if it exists
debug_file = 'pipeline_runs/my_experiment/02_relationships_smoke_test.json.debug.json'
if os.path.exists(debug_file):
    with open(debug_file, 'r') as f:
        debug = json.load(f)
    print(f'\nDebug info:')
    print(f'  Rows processed: {debug.get(\"rows_processed\", \"N/A\")}')
    print(f'  Matrix max strength: {debug.get(\"matrix_max\", \"N/A\")}')
    print(f'  Successful paths: {debug.get(\"successful_paths\", \"N/A\")}')
"

echo ""
echo "=== ESTIMATIONS FOR FULL RUN ==="

# Calculate estimates
rows_processed=50000
total_rows_train=2746270  # From logs
total_rows_val=200000     # From logs

# Estimate time for different scenarios
full_train_time=$((duration * total_rows_train / rows_processed))
full_val_time=$((duration * total_rows_val / rows_processed))

echo "Smoke test processed: $rows_processed rows"
echo "Full training data: $total_rows_train rows"
echo "Estimated full training time: $full_train_time seconds ($(($full_train_time / 60)) minutes)"
echo "Estimated full validation time: $full_val_time seconds ($(($full_val_time / 60)) minutes)"
echo ""

echo "=== RECOMMENDATIONS ==="
echo "For a meaningful test, consider:"
echo "  --pf-feature-cap 50-100 (instead of 4)"
echo "  --last-n-eras 50-100 (instead of 5)"
echo "  --row-limit 500000 (instead of 50000)"
echo "  --top-k 30-50 (instead of 10)"
echo ""
echo "This would give ~10-20x more data while still being fast to iterate."