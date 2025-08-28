# Pathfind Features - Organized Project Structure

## 📁 **New Directory Structure**

```
pathfind_features/
├── src/                          # Source code organized by function
│   ├── training/                 # Model training scripts
│   │   ├── ensemble_training.py
│   │   ├── generate_predictions.py
│   │   ├── hyperparameter_tuning.py
│   │   ├── train_control_model.py
│   │   ├── train_control_model_chunked.py
│   │   ├── train_experimental_model.py
│   │   ├── train_experimental_model_chunked.py
│   │   └── train_model.py
│   ├── analysis/                 # Analysis and validation scripts
│   │   ├── analyze_golden_eras.py
│   │   ├── compare_model_performance.py
│   │   ├── compare_targets.py
│   │   ├── methodological_validation.py
│   │   ├── out_of_sample_testing.py
│   │   ├── robust_cross_validation.py
│   │   ├── sharpe_investigation.py
│   │   └── validation.py
│   ├── data/                     # Data processing and feature engineering
│   │   ├── apply_bootstrap_to_validation.py
│   │   ├── data_utils.py
│   │   ├── execute_transform.py
│   │   └── feature_engineering.py
│   └── utils/                    # Utility functions and helpers
│       ├── model_utils.py
│       ├── move_file.py
│       ├── run_pipeline.py
│       ├── search_utils.py
│       └── update_scripts.py
├── bootstrap_pipeline/           # Bootstrap pipeline (already organized)
│   ├── analysis/
│   ├── bootstrap/
│   ├── steps/
│   └── utils/
├── scripts/                      # Shell scripts and automation
│   ├── training/                 # Training-related scripts
│   ├── analysis/                 # Analysis scripts
│   ├── automation/               # API and app runners
│   ├── train_models.sh
│   ├── benchmark_adaptive_path.sh
│   ├── compare_benchmarks.sh
│   └── ...
├── api/                          # API server and endpoints
├── web/                          # Web interface
├── tests/                        # Test framework and output management
├── docs/                         # Documentation
├── transforms/                   # Data transformation modules
├── cache/                        # Cache directories
└── pipeline_runs/                # Pipeline execution outputs
```

## 🚀 **How to Use the New Structure**

### **Running Scripts**

**Old way:**
```bash
python train_control_model_chunked.py --train-data data.parquet --output-model model.pkl
```

**New way:**
```bash
python src/training/train_control_model_chunked.py --train-data data.parquet --output-model model.pkl
```

### **Importing Modules**

```python
# Import from organized structure
from src.training.ensemble_training import EnsembleModel
from src.analysis.validation import validate_model
from src.data.data_utils import load_features
from src.utils.model_utils import save_model

# Or import from bootstrap pipeline
from bootstrap_pipeline.steps.step_01_target_discovery import run
```

### **Running Shell Scripts**

```bash
# Training scripts
./scripts/train_models.sh

# Analysis scripts
./scripts/analyze_path_features.sh

# Automation scripts
./scripts/automation/run_api.sh
```

## 📊 **Output Organization**

All Python scripts now automatically organize their outputs in the `tests/` folder with timestamped subdirectories:

```
tests/
├── script_name_20250828_084733/
│   ├── logs.log              # All logging output
│   ├── model.pkl            # Saved models
│   ├── predictions.csv      # Generated predictions
│   ├── results.json         # Analysis results
│   └── ...                  # Other outputs
```

## 🛠 **Key Improvements**

### **1. Logical Organization**
- **Training**: All model training and prediction scripts
- **Analysis**: Validation, comparison, and analysis tools
- **Data**: Feature engineering and data processing
- **Utils**: Shared utilities and helper functions

### **2. Consistent Output Management**
- All scripts use the same output directory structure
- Automatic timestamping prevents file conflicts
- Centralized logging in `tests/` folder

### **3. Better Maintainability**
- Clear separation of concerns
- Easier to find and modify specific functionality
- Reduced clutter in root directory

### **4. Scalability**
- Easy to add new scripts in appropriate categories
- Clear patterns for new team members
- Better version control organization

## 📝 **Migration Notes**

### **For Existing Scripts:**
- Update import paths if they reference moved modules
- Scripts in `src/` maintain their original functionality
- All CLI interfaces remain the same

### **For New Development:**
- Place training scripts in `src/training/`
- Analysis scripts in `src/analysis/`
- Data processing in `src/data/`
- Utilities in `src/utils/`

## 🔧 **Development Workflow**

1. **Add new training script**: `src/training/new_trainer.py`
2. **Add analysis tool**: `src/analysis/new_analyzer.py`
3. **Add utility**: `src/utils/new_helper.py`
4. **Run with organized outputs**: All logs/results go to `tests/`

## 📚 **Documentation**

- `docs/README.md` - Main project documentation
- `docs/README_API_CLI.md` - API and CLI usage
- `docs/README_chunked_training.md` - Training documentation
- `docs/requirements.txt` - Python dependencies

## 🎯 **Benefits Achieved**

✅ **Clean root directory** - Only essential files remain  
✅ **Logical grouping** - Related functionality together  
✅ **Consistent outputs** - All results organized in `tests/`  
✅ **Better collaboration** - Clear structure for team members  
✅ **Easier maintenance** - Find and modify code quickly  
✅ **Future-proof** - Easy to extend and add new features  

---

**The project is now well-organized and ready for efficient development and collaboration!** 🎉
