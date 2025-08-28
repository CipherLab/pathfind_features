"""
DEPRECATED: train_experimental_model.py

This file has been replaced with a stub to avoid accidental full-memory
training. Use the chunked versions or the orchestration script `train_models.sh`:

- `train_experimental_model_chunked.py`
- `train_control_model_chunked.py`
- `train_models.sh`

If you need to reintroduce the original non-chunked training behavior, revert
this change in version control.
"""
import sys

sys.exit("Deprecated: use train_models.sh or the chunked training scripts (train_experimental_model_chunked.py, train_control_model_chunked.py)")
