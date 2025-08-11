from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict
import os
import tempfile
import json

ROOT = Path(__file__).resolve().parents[1]
PY = str((ROOT/".venv/bin/python") if (ROOT/".venv/bin/python").exists() else sys.executable)


def run_step_target_discovery(
    input_file: str, 
    features_json_file: str, 
    output_file: str, 
    discovery_file: str, 
    skip_walk_forward: bool = False,
    max_eras: int | None = None, 
    row_limit: int | None = None, 
    target_limit: int | None = None
) -> dict:
    """
    Runs the Target Bootstrap Discovery stage (step_01_target_discovery.py).
    """
    args = [
        PY, 
        str(ROOT / "bootstrap_pipeline" / "steps" / "step_01_target_discovery.py"),
        "--input-file", input_file,
        "--features-json-file", features_json_file,
        "--output-file", output_file,
        "--discovery-file", discovery_file,
    ]
    if skip_walk_forward:
        args.append("--skip-walk-forward")
    if max_eras is not None:
        args.extend(["--max-eras", str(max_eras)])
    if row_limit is not None:
        args.extend(["--row-limit", str(row_limit)])
    if target_limit is not None:
        args.extend(["--target-limit", str(target_limit)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    result = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, env=env)
    
    return {
        "code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


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

def execute_transform(input_data: str, transform_script: str, output_data: str) -> dict:
    # The transform_script is the actual python code, so we need to save it to a temporary file
    # and pass the file path to the execute_transform.py script.
    
    print("--- Transform Script ---")
    print(transform_script)
    print("------------------------")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(transform_script)
        script_path = f.name

    args = [PY, str(ROOT/"execute_transform.py"),
            "--input-data", input_data,
            "--transform-script", script_path,
            "--output-data", output_data]
    
    try:
        result = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True)
        return {
            "code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    finally:
        os.remove(script_path)

def move_file(source: str, destination: str) -> int:
    args = [PY, str(ROOT/"move_file.py"),
            "--source", source,
            "--destination", destination]
    return subprocess.call(args, cwd=str(ROOT))


def derive_features_json(input_parquet: str, output_json: str) -> int:
    """Create a minimal features.json from a parquet by listing feature_* columns as feature_sets.medium.

    Includes any feature-like columns (prefix 'feature'). Does not attempt to infer metadata beyond names.
    """
    try:
        import pyarrow.parquet as pq
    except Exception:
        return 2
    try:
        pf = pq.ParquetFile(str(ROOT / input_parquet) if not str(input_parquet).startswith('/') else input_parquet)
        names = list(pf.schema.names)
        features = [c for c in names if str(c).startswith('feature')]
        data = {"feature_sets": {"medium": features}}
        out_path = (ROOT / output_json) if not str(output_json).startswith('/') else Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return 0
    except Exception:
        return 1