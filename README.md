# The Bootstrap Feature Discovery Pipeline

This project implements a two-stage pipeline for feature discovery:
1.  **Target Bootstrap Discovery:** Finds optimal, era-specific target combinations.
2.  **Creative Pathfinding:** Discovers novel feature relationships that predict the optimized targets.

The entire workflow is managed by `run_pipeline.py`.

## Quickstart

**Prerequisites:**
```bash
pip install -r requirements.txt
```

### Step 1: Run the Full Pipeline

This command executes all discovery and feature engineering steps. Outputs are saved to a unique, timestamped directory in `pipeline_runs/`.

```bash
python run_pipeline.py run \
  --input-data "v5.0/features.parquet" \
  --features-json "v5.0/features.json" \
  --run-name "my_first_run" \
  --max-new-features 30
```
*   **`--force`**: Use this to re-run all steps, even if cached results exist.
*   **`--skip-walk-forward`**: Use for very fast tests where target optimization is not critical.

### Step 2: List Previous Runs

See a history of all your pipeline executions.

```bash
python run_pipeline.py list
```

### Step 3: Train Your Model

Use the new `train_model.py` script to train a model on the output of a pipeline run.

```bash
python train_model.py \
  --run-dir "pipeline_runs/run_..." \
  --validation-data "v5.0/validation.parquet" \
  --model-type experimental
```

### Step 4: Generate Predictions

Use the `generate_predictions.py` script to generate predictions from your trained model.

```bash
python generate_predictions.py \
  --model models/experimental_model.lgb \
  --data v5.0/live.parquet \
  --output predictions.csv
```

