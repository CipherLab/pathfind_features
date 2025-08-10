from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PY = str((ROOT/".venv/bin/python") if (ROOT/".venv/bin/python").exists() else sys.executable)


def apply_to_validation(input_data: str, era_weights: str, relationships_file: Optional[str], output_data: str,
                        max_new_features: int = 40, row_limit: Optional[int] = None) -> int:
    args = [PY, str(ROOT/"apply_bootstrap_to_validation.py"),
            "--input-data", input_data,
            "--era-weights", era_weights,
            "--output-data", output_data,
            "--max-new-features", str(max_new_features)]
    if relationships_file:
        args += ["--relationships-file", relationships_file]
    if row_limit is not None:
        args += ["--row-limit", str(row_limit)]
    return subprocess.call(args, cwd=str(ROOT))


def generate_predictions(model_path: str, data_path: str, output_csv: str, batch_size: int = 100_000) -> int:
    args = [PY, str(ROOT/"generate_predictions.py"),
            "--model", model_path,
            "--data", data_path,
            "--output", output_csv,
            "--batch-size", str(batch_size)]
    return subprocess.call(args, cwd=str(ROOT))


def compare_models(control_csv: str, experimental_csv: str, validation_data: str, output_analysis: str,
                   target_col: str = "target", experimental_target_col: str = "adaptive_target") -> int:
    args = [PY, str(ROOT/"compare_model_performance.py"),
            "--control-predictions", control_csv,
            "--experimental-predictions", experimental_csv,
            "--validation-data", validation_data,
            "--output-analysis", output_analysis,
            "--target-col", target_col,
            "--experimental-target-col", experimental_target_col]
    return subprocess.call(args, cwd=str(ROOT))

def list_transforms() -> List[Dict[str, str]]:
    transforms_dir = ROOT / "transforms"
    if not transforms_dir.is_dir():
        return []
    
    transforms = []
    for filename in os.listdir(transforms_dir):
        if filename.endswith(".py"):
            with open(transforms_dir / filename, "r") as f:
                content = f.read()
            transforms.append({
                "name": filename.replace(".py", "").replace("_", " ").title(),
                "script": content,
            })
    return transforms

def execute_transform(input_data: str, transform_script: str, output_data: str) -> int:
    # The transform_script is the actual python code, so we need to save it to a temporary file
    # and pass the file path to the execute_transform.py script.
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(transform_script)
        script_path = f.name

    args = [PY, str(ROOT/"execute_transform.py"),
            "--input-data", input_data,
            "--transform-script", script_path,
            "--output-data", output_data]
    
    try:
        return subprocess.call(args, cwd=str(ROOT))
    finally:
        os.remove(script_path)
