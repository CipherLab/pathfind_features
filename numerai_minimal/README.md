# Numerai Minimal CLI Project

This project has been simplified to its core components for a CLI-based workflow.

## 📁 Directory Structure

```
numerai_minimal/
├── pipeline/                     # All python scripts for the pipeline
│   ├── __init__.py
│   ├── performance.py
│   ├── feature_discovery.py
│   ├── metrics_utils.py
│   ├── target_discovery.py
│   ├── step_01_target_discovery.py
│   ├── step_02_motif_discovery.py
│   ├── step_02_pathfinding.py
│   ├── step_03_feature_engineering.py
│   ├── step_03_motif_engineering.py
│   ├── build_adaptive_target.py
│   ├── cache.py
│   ├── utils.py
│   ├── validation_framework.py
│   ├── apply_bootstrap_to_validation.py
│   ├── generate_predictions.py
│   ├── train_control_model_chunked.py
│   ├── train_experimental_model_chunked.py
│   └── run_pipeline.py
├── scripts/                      # Shell scripts for automation
│   └── run.sh
└── tests/                        # Test framework
```

## 🚀 How to Use

The main entry point for the project is the `scripts/run.sh` script. This script can be used to either run the pipeline or the validation framework.

### Run the Pipeline

```bash
./scripts/run.sh pipeline
```

### Run Validation

```bash
./scripts/run.sh validate
```
