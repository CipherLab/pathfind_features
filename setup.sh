#!/bin/bash
# Numerai Pipeline Setup Script
# Handles Git LFS setup and data file management

set -e

echo "🚀 Setting up Numerai Pipeline Environment"
echo "=========================================="

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS is not installed. Please install it first:"
    echo "   Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "   macOS: brew install git-lfs"
    echo "   Windows: Download from https://git-lfs.github.com/"
    exit 1
fi

# Initialize Git LFS
echo "📦 Initializing Git LFS..."
git lfs install
echo "✅ Git LFS initialized"

# Check for .gitattributes
if [ ! -f ".gitattributes" ]; then
    echo "❌ .gitattributes file not found. Creating one..."
    cat > .gitattributes << 'EOF'
# Git LFS tracking for large data files
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pq filter=lfs diff=lfs merge=lfs -text

# Large data files that should be tracked with LFS
data/*.parquet filter=lfs diff=lfs merge=lfs -text
data/*.pq filter=lfs diff=lfs merge=lfs -text

# Exclude test data from LFS (keep in regular Git)
test_data/* -filter -diff -merge text
EOF
    echo "✅ Created .gitattributes file"
else
    echo "✅ .gitattributes file found"
fi

# Create data directory structure
echo "📁 Setting up data directories..."
mkdir -p numerai_minimal/data
mkdir -p numerai_minimal/test_data

# Check for Numerai data
if [ -d "v5.0" ]; then
    echo "📊 Found Numerai v5.0 data directory"

    # Count data files
    parquet_count=$(find v5.0 -name "*.parquet" | wc -l)
    json_count=$(find v5.0 -name "*.json" | wc -l)

    echo "   Found $parquet_count Parquet files"
    echo "   Found $json_count JSON files"

    if [ $parquet_count -gt 0 ]; then
        echo "   Parquet files:"
        ls -lh v5.0/*.parquet 2>/dev/null || true
    fi

    if [ $json_count -gt 0 ]; then
        echo "   JSON files:"
        ls -lh v5.0/*.json 2>/dev/null || true
    fi

    # Ask about moving data
    read -p "Move data files to numerai_minimal/data/? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Moving data files..."
        mv v5.0/* numerai_minimal/data/ 2>/dev/null || true
        rmdir v5.0 2>/dev/null || true
        echo "✅ Data files moved to numerai_minimal/data/"
    fi
else
    echo "⚠️  No v5.0 data directory found"
    echo "   To use real Numerai data:"
    echo "   1. Download from Numerai (https://numer.ai/)"
    echo "   2. Place files in numerai_minimal/data/"
    echo "   3. Run: git lfs track \"numerai_minimal/data/*.parquet\""
fi

# Generate synthetic test data
echo "🔄 Generating synthetic test data..."
cd numerai_minimal

python3 -c "
import pandas as pd
import numpy as np
import json
import os

print('Generating synthetic test datasets...')

# Ensure the test_data directory exists in case the calling
# environment didn't create it beforehand. Without this check,
# attempting to write the parquet files below would raise an
# OSError when the directory is missing.
os.makedirs('test_data', exist_ok=True)

# Small dataset for unit tests
np.random.seed(42)
n_rows_small = 5000
data_small = {
    'era': np.repeat(range(1, 51), n_rows_small // 50),
}

for i in range(20):
    data_small[f'feature_{i}'] = np.random.randn(n_rows_small)

for i in range(5):
    data_small[f'target_{i}'] = np.random.randn(n_rows_small) * 0.1

df_small = pd.DataFrame(data_small)
df_small.to_parquet('test_data/train_small.parquet', index=False)
print(f'✅ Generated small test dataset: {n_rows_small} rows')

# Medium dataset for integration tests
n_rows_medium = 25000
data_medium = {
    'era': np.repeat(range(1, 126), n_rows_medium // 125),
}

for i in range(100):
    data_medium[f'feature_{i}'] = np.random.randn(n_rows_medium)

for i in range(10):
    data_medium[f'target_{i}'] = np.random.randn(n_rows_medium) * 0.05

df_medium = pd.DataFrame(data_medium)
df_medium.to_parquet('test_data/train_medium.parquet', index=False)
print(f'✅ Generated medium test dataset: {n_rows_medium} rows')

# Features configuration
features = {
    'feature_sets': {
        'small': [f'feature_{i}' for i in range(10)],
        'medium': [f'feature_{i}' for i in range(50)],
        'large': [f'feature_{i}' for i in range(100)]
    },
    'targets': [f'target_{i}' for i in range(10)]
}

with open('test_data/features.json', 'w') as f:
    json.dump(features, f, indent=2)

print('✅ Generated features.json configuration')
"

cd ..

# Git LFS status
echo "📊 Git LFS Status:"
git lfs status

# Summary
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo "✅ Git LFS configured"
echo "✅ Data directories created"
echo "✅ Synthetic test data generated"
echo ""
echo "Next steps:"
echo "1. Add real Numerai data to numerai_minimal/data/ (optional)"
echo "2. Track large files: git lfs track \"numerai_minimal/data/*.parquet\""
echo "3. Run tests: cd numerai_minimal && python -m pytest pipeline/tests/"
echo "4. For CI/CD: Push to GitHub with LFS support enabled"
