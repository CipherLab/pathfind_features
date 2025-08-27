#!/bin/bash

# Script to demonstrate the pathfinding limitation issue

set -e  # Exit on any error

echo "Activating virtual environment..."
source /home/mat/Downloads/pathfind_features/.venv/bin/activate

echo "=== PATHFINDING LIMITATION ANALYSIS ==="
echo ""

/home/mat/Downloads/pathfind_features/.venv/bin/python -c "
import json

# Load the relationships to show the limitation
with open('pipeline_runs/my_experiment/02_relationships_nme9zy4kt_10.json', 'r') as f:
    relationships = json.load(f)

print(f'❌ ISSUE FOUND: Pathfinding was limited to only {len(relationships)} relationships')
print(f'   - All relationships have identical strength: {relationships[0][\"strength\"]}')
print(f'   - This suggests the algorithm found the first relationships it could, not the best ones')
print('')
print('🔍 ROOT CAUSE: The pathfinding used --pf-feature-cap 8')
print('   This limited the search to only 8 features instead of all 2376 available features')
print('')
print('📊 IMPACT:')
print('   - Control/Adaptive-Only models: 2,376 features each')
print('   - Experimental model: Only 6 engineered features')
print('   - 99.7% loss of information!')
print('')
print('💡 SOLUTION: Re-run pathfinding without the --pf-feature-cap limitation')
print('   to allow searching across all available features.')
"

echo ""
echo "=== RECOMMENDATION ==="
echo "To fix this issue, re-run the pathfinding step with:"
echo "  --pf-feature-cap 100  # or remove it entirely"
echo "  --top-k 50            # allow more relationships"
echo ""
echo "This will allow the algorithm to find meaningful relationships"
echo "across the full feature space instead of just 8 features."