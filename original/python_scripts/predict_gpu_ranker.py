
import argparse
import json
import logging
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import gc
import pyarrow.parquet as pq

class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logging(log_file, log_level):
    """Configures the logging for the script."""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            FlushingStreamHandler(sys.stdout)
        ]
    )

def manual_sigmoid(x):
    """ARCH-9000: A sigmoid function, because raw model outputs are for savages."""
    x = np.clip(x, -500, 500) # Avoid overflow
    return 1.0 / (1.0 + np.exp(-x))

def generate_predictions():
    """
    ARCH-9000: Welcome to the prediction party, take two.
    This time with more columns and the magic of sigmoid.
    """
    parser = argparse.ArgumentParser(description="Generate predictions using a trained LightGBM model.")
    parser.add_argument("--model-path", required=True, help="Path to the trained LightGBM model file.")
    parser.add_argument("--output-preds-path", required=True, help="Path to save the output predictions Parquet file.")
    parser.add_argument("--live-data", required=True, help="Path to the live data Parquet file to predict on.")
    parser.add_argument("--feature-map-file", required=True, help="Path to the JSON file containing the list of feature column names.")
    parser.add_argument("--log-file", required=True, help="Path to the log file.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g., DEBUG, INFO, WARNING).")
    parser.add_argument("--batch-size", type=int, default=50000, help="Batch size for processing data chunks.")
    parser.add_argument("--numerai-model-id", help="Optional Numerai model ID for submission context.")

    args = parser.parse_args()

    setup_logging(args.log_file, args.log_level)
    logging.info("--- Prediction Script Starting (Corrected Version) ---")
    logging.info(f"Loading model from: {args.model_path}")
    logging.info(f"Live data source: {args.live_data}")
    logging.info(f"Output predictions to: {args.output_preds_path}")

    try:
        logging.info("Attempting to load the majestic model...")
        model = lgb.Booster(model_file=args.model_path)
        logging.info("Model loaded. It exists. That's a good start.")

        logging.info(f"Loading feature map from: {args.feature_map_file}")
        with open(args.feature_map_file, 'r') as f:
            feature_columns = json.load(f)
        logging.info(f"Loaded {len(feature_columns)} features. Let's hope they're the right ones.")

        logging.info(f"Opening live data file: {args.live_data}")
        parquet_file = pq.ParquetFile(args.live_data)
        
        all_predictions = []
        
        total_rows = parquet_file.metadata.num_rows
        total_chunks = (total_rows // args.batch_size) + (1 if total_rows % args.batch_size > 0 else 0)
        logging.info(f"Calculated {total_chunks} total chunks to process with batch size {args.batch_size}.")

        for i, batch in enumerate(parquet_file.iter_batches(batch_size=args.batch_size)):
            logging.info(f"--- Processing Chunk {i+1}/{total_chunks} ---")
            chunk_df = batch.to_pandas()
            chunk_df.reset_index(inplace=True) # ARCH-9000: The ID is the index. Let's make it a column.

            # ARCH-9000: Let's not assume dtypes. Explicitly handle non-numeric 'era' values.
            if 'era' in chunk_df.columns:
                chunk_df['era'] = pd.to_numeric(chunk_df['era'], errors='coerce').fillna(0).astype(int)
            
            if chunk_df.empty:
                logging.warning("Chunk is empty, skipping.")
                continue

            missing_cols = set(feature_columns) - set(chunk_df.columns)
            if missing_cols:
                logging.warning(f"Missing columns in this chunk: {missing_cols}. Filling with 0.5.")
                for col in missing_cols:
                    chunk_df[col] = 0.5

            X_live = chunk_df[feature_columns]

            logging.info("Generating raw predictions for the chunk...")
            raw_predictions = model.predict(X_live)

            logging.info("Applying sigmoid transformation...")
            transformed_predictions = manual_sigmoid(raw_predictions)
            final_predictions = np.clip(transformed_predictions, 0.0, 1.0)
            
            chunk_preds_df = pd.DataFrame({'id': chunk_df['id'], 'prediction': final_predictions})
            all_predictions.append(chunk_preds_df)
            
            logging.info(f"Finished chunk {i+1}. Predictions generated: {len(chunk_preds_df)}")
            del chunk_df, X_live, raw_predictions, transformed_predictions, final_predictions, chunk_preds_df
            gc.collect()

        if not all_predictions:
            logging.error("No predictions were generated. The live data might have been empty or filtered out completely.")
            sys.exit(1)

        logging.info("Concatenating all chunk predictions...")
        final_predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        output_dir = os.path.dirname(args.output_preds_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        logging.info(f"Saving final predictions to {args.output_preds_path}...")
        final_predictions_df.to_parquet(args.output_preds_path, index=False)
        
        logging.info("--- Predictions successfully generated and saved. ---")
        logging.info(f"Total predictions: {len(final_predictions_df)}")
        logging.info(f"Prediction range: [{final_predictions_df['prediction'].min():.6f}, {final_predictions_df['prediction'].max():.6f}]")

    except Exception as e:
        logging.critical(f"An unhandled error occurred during prediction generation: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    generate_predictions()
