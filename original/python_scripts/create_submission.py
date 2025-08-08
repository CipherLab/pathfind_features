
import pandas as pd
import argparse
import os

def create_submission_file(predictions_path, output_path):
    """
    Converts a Parquet prediction file to a CSV submission file.
    """
    print(f"Reading predictions from: {predictions_path}")
    if not os.path.exists(predictions_path):
        print(f"❌ ERROR: Prediction file not found at {predictions_path}")
        return

    pred_df = pd.read_parquet(predictions_path)

    if 'id' not in pred_df.columns or 'prediction' not in pred_df.columns:
        print("❌ ERROR: The prediction file must contain 'id' and 'prediction' columns.")
        return

    # Ensure the columns are in the correct order
    submission_df = pred_df[['id', 'prediction']]

    print(f"Saving submission file to: {output_path}")
    submission_df.to_csv(output_path, index=False)
    print(f"✅ Submission file created successfully at {output_path}")
    print(f"Total rows: {len(submission_df)}")
    print(f"Sample:\n{submission_df.head()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a submission file from Parquet predictions.")
    parser.add_argument("--predictions-file", default="test_artifacts/live_experimental_predictions.parquet", help="Path to the input Parquet predictions file.")
    parser.add_argument("--output-file", default="submission.csv", help="Path for the output submission CSV file.")
    args = parser.parse_args()

    create_submission_file(args.predictions_file, args.output_file)
