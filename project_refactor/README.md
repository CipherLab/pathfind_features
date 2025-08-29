# Pathfind Features - Minimal CLI Project

This project has been simplified to its core components for a CLI-based workflow.

## 📁 Directory Structure

```
pathfind_features/
├── src/                          # Source code organized by function
│   ├── training/                 # Model training scripts
│   │   ├── train_control_model_chunked.py
│   │   └── train_experimental_model_chunked.py
│   ├── analysis/                 # Analysis and validation scripts
│   │   └── validation_framework.py
│   ├── data/                     # Data processing and feature engineering
│   │   └── apply_bootstrap_to_validation.py
│   └── utils/                    # Utility functions and helpers
│       └── run_pipeline.py
├── bootstrap_pipeline/           # Bootstrap pipeline for feature engineering
│   ├── analysis/
│   ├── bootstrap/
│   ├── steps/
│   └── utils/
├── scripts/                      # Shell scripts for automation
│   └── train_models.sh
├── tests/                        # Test framework
└── transforms/                   # Data transformation modules
```

## 🚀 How to Use

The main entry point for the project is the `scripts/train_models.sh` script. This script can be used to either train the models or run the validation framework.

### Train Models

```bash
./scripts/train_models.sh train
```

### Run Validation

```bash
./scripts/train_models.sh validate
```

## 🔧 Development Workflow

1.  **Modify the pipeline:** Edit the scripts in `bootstrap_pipeline/` to change the feature engineering process.
2.  **Train the models:** Run `./scripts/train_models.sh train` to train the models with the new features.
3.  **Validate the results:** Run `./scripts/train_models.sh validate` to evaluate the performance of the new models.