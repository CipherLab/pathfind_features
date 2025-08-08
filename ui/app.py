import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "pipeline_runs"
V50_DIR = ROOT / "v5.0"


@dataclass
class RunPaths:
    base: Path
    control_model: Path
    experimental_model: Path
    control_preds: Path
    control_preds_on_enh: Path
    experimental_preds: Path
    validation_enhanced: Path
    report: Path
    target_discovery: Path
    relationships: Path
    enhanced_train: Path


def get_run_paths(run_dir: Path) -> RunPaths:
    return RunPaths(
        base=run_dir,
        control_model=run_dir / "models" / "control_lgbm.pkl",
        experimental_model=run_dir / "models" / "experimental_lgbm.pkl",
        control_preds=run_dir / "control_predictions.csv",
        control_preds_on_enh=run_dir / "control_predictions_on_enhanced.csv",
        experimental_preds=run_dir / "experimental_predictions.csv",
        validation_enhanced=run_dir / "validation_enhanced.parquet",
        report=run_dir / "performance_report.txt",
        target_discovery=run_dir / "01_target_discovery.json",
        relationships=run_dir / "02_discovered_relationships.json",
        enhanced_train=run_dir / "03_enhanced_features.parquet",
    )


def list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], key=lambda p: p.name)


def run_cmd(args: list[str], cwd: Optional[Path] = None) -> int:
    st.code(" ".join(args))
    proc = subprocess.run(args, cwd=str(cwd or ROOT))
    return proc.returncode


def read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def main():
    st.set_page_config(page_title="Pathfind Orchestrator", layout="wide")
    st.title("Pathfind Features: Orchestrator UI")

    runs = list_runs()
    run_names = [p.name for p in runs]
    sel = st.sidebar.selectbox("Select run folder", run_names, index=len(run_names) - 1 if run_names else 0)
    run_dir = (RUNS_DIR / sel) if sel else (RUNS_DIR / "run_latest")
    paths = get_run_paths(run_dir)

    st.sidebar.markdown("### Default data")
    train_path = st.sidebar.text_input("Train parquet", str(V50_DIR / "train.parquet"))
    valid_path = st.sidebar.text_input("Validation parquet", str(V50_DIR / "validation.parquet"))
    features_json = st.sidebar.text_input("features.json (optional)", str(V50_DIR / "features.json"))

    st.sidebar.markdown("### Shortcuts")
    st.sidebar.write(f"Run dir: {paths.base}")
    if st.sidebar.button("Open logs.log"):
        st.session_state["show_logs"] = True

    tab_dash, tab_control, tab_valid, tab_experiment, tab_compare = st.tabs(
        ["Dashboard", "Control", "Validation Enh", "Experimental", "Compare"]
    )

    with tab_dash:
        st.subheader("Run overview")
        cols = st.columns(3)
        cols[0].write("Control model")
        cols[0].write(paths.control_model.exists())
        cols[1].write("Experimental model")
        cols[1].write(paths.experimental_model.exists())
        cols[2].write("Enhanced validation")
        cols[2].write(paths.validation_enhanced.exists())

        st.subheader("Performance report")
        if paths.report.exists():
            st.text(read_text_if_exists(paths.report))
        else:
            st.info("No performance report yet. Generate predictions and run comparison.")

        if st.session_state.get("show_logs"):
            log_path = run_dir / "logs.log"
            st.subheader("logs.log")
            if log_path.exists():
                st.code(read_text_if_exists(log_path), language="text")
            else:
                st.info("logs.log not found in selected run.")

    with tab_control:
        st.subheader("Train control (chunked)")
        out_model = st.text_input("Output control model", str(paths.control_model))
        if st.button("Train control"):
            code = run_cmd([
                "python", "train_control_model_chunked.py",
                "--train-data", train_path,
                "--validation-data", valid_path,
                "--target-col", "target",
                "--features-json", features_json,
                "--output-model", out_model,
            ])
            if code == 0:
                st.success("Control training completed")
            else:
                st.error(f"Control training failed with exit code {code}")

        st.divider()
        st.subheader("Predict control")
        ctrl_out = st.text_input("Control predictions output", str(paths.control_preds))
        ctrl_data = st.text_input("Data for control predictions", valid_path)
        if st.button("Generate control predictions"):
            code = run_cmd([
                "python", "generate_predictions.py",
                "--model", out_model,
                "--data", ctrl_data,
                "--output", ctrl_out,
            ])
            if code == 0:
                st.success("Control predictions generated")
            else:
                st.error(f"Control predictions failed with exit code {code}")

    with tab_valid:
        st.subheader("Enhance validation (adaptive target + path features)")
        weights = st.text_input("Era weights (01_target_discovery.json)", str(paths.target_discovery))
        relationships = st.text_input("Relationships (02_discovered_relationships.json)", str(paths.relationships))
        out_valid = st.text_input("Output enhanced validation", str(paths.validation_enhanced))
        max_new = st.number_input("Max new features", value=40, min_value=0, max_value=1000)
        row_limit = st.number_input("Row limit (optional)", value=0, min_value=0, step=50000)
        if st.button("Build enhanced validation"):
            args = [
                "python", "apply_bootstrap_to_validation.py",
                "--input-data", valid_path,
                "--era-weights", weights,
                "--output-data", out_valid,
                "--max-new-features", str(max_new),
            ]
            if relationships:
                args += ["--relationships-file", relationships]
            if row_limit and int(row_limit) > 0:
                args += ["--row-limit", str(int(row_limit))]
            code = run_cmd(args)
            if code == 0:
                st.success("Enhanced validation built")
            else:
                st.error(f"Validation enhancement failed with exit code {code}")

    with tab_experiment:
        st.subheader("Train experimental (full or small)")
        enh_train = st.text_input("Enhanced train parquet (03_enhanced_features.parquet)", str(paths.enhanced_train))
        enh_valid = st.text_input("Enhanced validation parquet", str(paths.validation_enhanced))
        exp_model = st.text_input("Output experimental model", str(paths.experimental_model))
        estimators = st.number_input("Estimators", value=300, min_value=10, max_value=5000)
        num_leaves = st.number_input("Num leaves", value=64, min_value=8, max_value=4096)
        lr = st.number_input("Learning rate", value=0.05, min_value=0.001, max_value=1.0, step=0.01)
        if st.button("Train experimental"):
            code = run_cmd([
                "python", "train_experimental_model.py",
                "--train-data", enh_train,
                "--validation-data", enh_valid,
                "--target-col", "adaptive_target",
                "--output-model", exp_model,
                "--estimators", str(int(estimators)),
                "--num-leaves", str(int(num_leaves)),
                "--learning-rate", str(float(lr)),
            ])
            if code == 0:
                st.success("Experimental training completed")
            else:
                st.error(f"Experimental training failed with exit code {code}")

        st.divider()
        st.subheader("Predict experimental")
        exp_out = st.text_input("Experimental predictions output", str(paths.experimental_preds))
        pred_data = st.text_input("Data for experimental predictions", str(paths.validation_enhanced))
        if st.button("Generate experimental predictions"):
            code = run_cmd([
                "python", "generate_predictions.py",
                "--model", exp_model,
                "--data", pred_data,
                "--output", exp_out,
            ])
            if code == 0:
                st.success("Experimental predictions generated")
            else:
                st.error(f"Experimental predictions failed with exit code {code}")

    with tab_compare:
        st.subheader("Compare performance")
        ctrl_preds = st.text_input("Control predictions CSV", str(paths.control_preds_on_enh))
        exp_preds = st.text_input("Experimental predictions CSV", str(paths.experimental_preds))
        val_for_compare = st.text_input("Validation parquet for comparison", str(paths.validation_enhanced))
        report_out = st.text_input("Report output path", str(paths.report))
        target_col = st.text_input("Target column (control)", "target")
        exp_target_col = st.text_input("Target column (experimental)", "adaptive_target")
        if st.button("Run comparison"):
            code = run_cmd([
                "python", "compare_model_performance.py",
                "--control-predictions", ctrl_preds,
                "--experimental-predictions", exp_preds,
                "--validation-data", val_for_compare,
                "--output-analysis", report_out,
                "--target-col", target_col,
                "--experimental-target-col", exp_target_col,
            ])
            if code == 0:
                st.success("Comparison completed")
                st.text(read_text_if_exists(Path(report_out)))
            else:
                st.error(f"Comparison failed with exit code {code}")


if __name__ == "__main__":
    main()
