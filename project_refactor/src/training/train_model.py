"""
DEPRECATED: train_model.py

This script performs full-memory training and has been replaced by chunked
implementations to avoid excessive memory usage. Use `train_models.sh` or the
chunked training scripts:

- `train_control_model_chunked.py`
- `train_experimental_model_chunked.py`

This file now exits with a helpful message to prevent accidental runs.
"""
import sys

sys.exit("Deprecated: use train_models.sh or the chunked training scripts (train_control_model_chunked.py, train_experimental_model_chunked.py)")
