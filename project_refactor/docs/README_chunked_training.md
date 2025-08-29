# Chunked training scripts

Because your Parquet files are large, use these scripts to avoid OOM:

Control model (baseline target):


- train_control_model_chunked.py

Inputs:

- v5.0/train.parquet
- v5.0/validation.parquet

Target:

- target

Experimental model (adaptive target + engineered features):


- train_experimental_model_chunked.py

Inputs:

- pipeline_runs/.../03_enhanced_features.parquet
- enhanced validation parquet

Target:

- adaptive_target

Notes:


- Iterates Parquet in chunks with PyArrow, converts batch to float32, and warm-starts LightGBM booster across chunks.
- Uses a fixed small validation sample (val_rows) for early stopping-like guidance across chunks.
- Models are saved as pickle with both booster and feature list.
Chunked training scripts

Because your Parquet files are large, use these scripts to avoid OOM:

Control model (baseline target):
- train_control_model_chunked.py
  - Input: v5.0/train.parquet, v5.0/validation.parquet
  - Target: target

Experimental model (adaptive target + engineered features):
- train_experimental_model_chunked.py
  - Input: pipeline_runs/.../03_enhanced_features.parquet and enhanced validation parquet
  - Target: adaptive_target

Notes:
- Iterates Parquet in chunks with PyArrow, converts batch to float32, and warm-starts LightGBM booster across chunks.
- Uses a fixed small validation sample (val_rows) for early stopping-like guidance across chunks.
- Models are saved as pickle with both booster and feature list.
