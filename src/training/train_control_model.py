"""
DEPRECATED: train_control_model.py

This file has been intentionally replaced with a small stub to avoid accidental
memory-heavy runs. Use the chunked training scripts or the orchestration
script `train_models.sh` instead:

- `train_control_model_chunked.py`
- `train_experimental_model_chunked.py`
- `train_models.sh`

Keeping a stub here makes the repository explicit about the preferred workflow
while preserving history in version control if removal is desired later.
"""
import sys

sys.exit("Deprecated: use train_models.sh or the chunked training scripts (train_control_model_chunked.py, train_experimental_model_chunked.py)")
