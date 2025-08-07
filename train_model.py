# train_model.py

import argparse
import json
import logging
import os
import sys
import lightgbm as lgb
import pandas as pd

def setup_logging(log_file):
    """Initializes logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Train a model on data from a pipeline run.")
    parser.add_argument("--run-dir", required=True, help="Path to the pipeline run directory.")
    parser.add_argument("--validation-data", required=True, help="Path to the validation data.")
    parser.add_argument("--model-type", choices=["control", "experimental"], default="experimental", help="Model to train.")
    parser.add_argument("--output-dir", default="models", help="Directory to save the trained model.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, f"{args.model_type}_training.log"))

    run_summary_path = os.path.join(args.run_dir, "run_summary.json")
    with open(run_summary_path, 'r') as f:
        run_summary = json.load(f)

    if args.model_type == "experimental":
        train_data_path = run_summary["artifacts"]["enhanced_data"]
        target_col = "adaptive_target"
        with open(run_summary["artifacts"]["new_features_list"]) as f:
            new_features = json.load(f)
        with open(run_summary["parameters"]["features_json"]) as f:
            original_features = json.load(f)["feature_sets"]["medium"]
        feature_cols = original_features + new_features
    else: # control
        train_data_path = run_summary["parameters"]["input_data"]
        target_col = "target" # Assuming standard target for control
        with open(run_summary["parameters"]["features_json"]) as f:
            feature_cols = json.load(f)["feature_sets"]["medium"]
        train_df = pd.read_parquet(train_data_path, columns=feature_cols + [target_col, 'era'])

    validation_df = pd.read_parquet(args.validation_data)

    logging.info(f"Training {args.model_type} model...")
    logging.info(f"Training data: {train_data_path}")
    logging.info(f"Validation data: {args.validation_data}")
    logging.info(f"Target column: {target_col}")
    logging.info(f"Number of features: {len(feature_cols)}")

    # Simplified training logic
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = validation_df[feature_cols]
    y_val = validation_df[target_col]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(10)]
    )

    model_path = os.path.join(args.output_dir, f"{args.model_type}_model.lgb")
    model.save_model(model_path)
    logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
