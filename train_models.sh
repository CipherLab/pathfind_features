#!/bin/bash

# Script to train the three models with updated format
# Control, Adaptive-only, and Experimental

set -e  # Exit on any error

echo "Activating virtual environment..."
source /home/mat/Downloads/pathfind_features/.venv/bin/activate

echo "Training Control Model..."
/home/mat/Downloads/pathfind_features/.venv/bin/python train_control_model_chunked.py \
    --train-data v5.0/train.parquet \
    --validation-data v5.0/validation.parquet \
    --output-model pipeline_runs/my_experiment/control_model.pkl \
    --features-json v5.0/features.json

echo "Training Adaptive-Only Model..."
/home/mat/Downloads/pathfind_features/.venv/bin/python train_experimental_model_chunked.py \
    --train-data pipeline_runs/my_experiment/01_adaptive_targets_train.parquet \
    --validation-data pipeline_runs/my_experiment/01_adaptive_targets_validation.parquet \
    --output-model pipeline_runs/my_experiment/adaptive_only_model.pkl

echo "Training Experimental Model..."
/home/mat/Downloads/pathfind_features/.venv/bin/python train_experimental_model_chunked.py \
    --train-data pipeline_runs/my_experiment/03_features_train.parquet \
    --validation-data pipeline_runs/my_experiment/03_features_validation.parquet \
    --new-feature-names pipeline_runs/my_experiment/new_feature_names.json \
    --output-model pipeline_runs/my_experiment/experimental_model.pkl \
    --features-json v5.0/features.json

echo "All models trained successfully!"
echo "Generated files:"
echo "- pipeline_runs/my_experiment/control_model.pkl"
echo "- pipeline_runs/my_experiment/control_model_features.json"
echo "- pipeline_runs/my_experiment/adaptive_only_model.pkl"
echo "- pipeline_runs/my_experiment/adaptive_only_model_features.json"
echo "- pipeline_runs/my_experiment/experimental_model.pkl"
echo "- pipeline_runs/my_experiment/experimental_model_features.json"
