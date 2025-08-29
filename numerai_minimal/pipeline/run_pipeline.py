# run_pipeline.py

import argparse
import json
import logging
import os
import sys
from datetime import datetime
import time
import shutil
from pathlib import Path

import step_01_target_discovery
import step_02_pathfinding
import step_03_feature_engineering
import performance as analysis
from cache import compute_hash, stage_cache_lookup, stage_cache_store, materialize_cached_artifacts
import random
import numpy as np
from typing import List

def setup_logging(log_file):
    """Initializes logging to both file and console for a specific run."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def _resolve_smoke_value(args, attr_name, default_if_smoke):
    """Return user provided sampling value or a default if smoke-mode is enabled.
    If smoke-mode is off, returns None (meaning 'no limit') unless user explicitly set the flag.
    """
    explicit = getattr(args, attr_name)
    if explicit is not None:
        return explicit
    if getattr(args, 'smoke_mode', False):
        return default_if_smoke
    return None

def _write_merged_features_json(run_dir: str, base_features_json: str, new_features_list_path: str) -> str:
    """Create a features.json in run_dir by merging base features with newly engineered features.
    - Reads base_features_json (expects feature_sets.medium).
    - Appends any names from new_features_list_path that aren't already present.
    - Writes to run_dir/features.json and returns the path.
    Fallback: if base can't be parsed, writes a minimal features.json with the new features only.
    """
    out_path = os.path.join(run_dir, "features.json")
    try:
        base = {}
        with open(base_features_json, 'r') as f:
            base = json.load(f)
        with open(new_features_list_path, 'r') as f:
            new_names: List[str] = json.load(f)
        # Ensure structure
        if not isinstance(base, dict):
            base = {}
        fs = base.setdefault('feature_sets', {}) if isinstance(base, dict) else {}
        medium = fs.get('medium') if isinstance(fs, dict) else None
        if not isinstance(medium, list):
            medium = []
            if isinstance(fs, dict):
                fs['medium'] = medium
        # Normalize to names
        def name_of(x):
            if isinstance(x, str):
                return x
            if isinstance(x, dict) and 'name' in x:
                return str(x['name'])
            return None
        existing_names = []
        for item in medium:
            nm = name_of(item)
            if nm is not None:
                existing_names.append(nm)
        # Append new ones
        for nm in new_names:
            if nm not in existing_names:
                medium.append(nm)
                existing_names.append(nm)
        with open(out_path, 'w') as f:
            json.dump(base, f, indent=2)
        return out_path
    except Exception:
        # Fallback minimal
        try:
            with open(new_features_list_path, 'r') as f:
                new_names = json.load(f)
            minimal = {"feature_sets": {"medium": list(new_names or [])}}
            with open(out_path, 'w') as f:
                json.dump(minimal, f, indent=2)
            return out_path
        except Exception:
            return out_path


def main():
    # Clean up __pycache__ directories
    for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
        if '__pycache__' in dirs:
            shutil.rmtree(os.path.join(root, '__pycache__'))

    parser = argparse.ArgumentParser(description="Managed Bootstrap Feature Discovery Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- RUN Command ---
    p_run = subparsers.add_parser("run", help="Execute the discovery and feature engineering pipeline.")
    p_run.add_argument("--input-data", required=True, help="Path to the initial train.parquet file.")
    p_run.add_argument("--features-json", required=True, help="Path to the features.json file.")
    p_run.add_argument("--experiment-name", default=None, help="Name of the experiment. If provided, creates a directory with this name under pipeline_runs/. Otherwise, a timestamped directory is created.")
    p_run.add_argument("--force", action="store_true", help="Force re-execution of all steps, ignoring cache.")
    p_run.add_argument("--skip-walk-forward", action="store_true", help="Use equal weights for targets (for quick tests).")
    p_run.add_argument("--max-new-features", type=int, default=20, help="Number of new relationship features to create.")
    p_run.add_argument("--yolo-mode", action="store_true", help="Trust the original results, create 40+ features.")
    p_run.add_argument("--pf-debug", action="store_true", help="Enable detailed pathfinding debug logging and write a .debug.json summary.")
    p_run.add_argument("--pf-debug-every-rows", type=int, default=10000, help="Emit pathfinding debug stats every N processed rows (when --pf-debug is set).")
    p_run.add_argument("--pretty", action="store_true", help="Print a formatted run summary table at the end.")
    p_run.add_argument("--no-color", action="store_true", help="Disable ANSI colors in pretty output.")
    p_run.add_argument("--disable-pathfinding", action="store_true", help="Skip Stage 2 (pathfinding) and Stage 3 (feature engineering).")
    # Reuse artifacts
    p_run.add_argument("--stage1-from", dest="stage1_from", default=None, help="Path to a previous run directory to reuse Stage 1 artifacts (01_adaptive_targets.parquet and 01_target_discovery.json).")
    p_run.add_argument("--stage2-from", dest="stage2_from", default=None, help="Path to a previous run directory to reuse Stage 2 artifacts (02_discovered_relationships.json).")
    # Smoke / sampling controls (added for faster iterative testing)
    p_run.add_argument("--smoke-mode", action="store_true", help="Enable fast sampling mode (limits eras, rows, targets, and features for a quicker end-to-end test).")
    p_run.add_argument("--smoke-max-eras", type=int, default=None, help="Maximum number of eras to process (overrides in smoke-mode).")
    p_run.add_argument("--smoke-row-limit", type=int, default=None, help="Row limit across all batches per stage (overrides in smoke-mode).")
    p_run.add_argument("--smoke-target-limit", type=int, default=None, help="Limit number of target columns (prefix 'target') used in Stage 1.")
    p_run.add_argument("--smoke-feature-limit", type=int, default=None, help="Limit number of feature columns (prefix 'feature') used in Stage 2 pathfinding.")
    p_run.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")


    # --- LIST Command ---
    p_list = subparsers.add_parser("list", help="List all previous pipeline runs.")

    # --- ANALYZE Command ---
    p_analyze = subparsers.add_parser("analyze", help="Analyze the results of a previous run.")
    p_analyze.add_argument("--run-dir", required=True, help="Path to the pipeline run directory (e.g., pipeline_runs/run_...).")
    p_analyze.add_argument("--validation-data", required=True, help="Path to validation data with true targets.")
    p_analyze.add_argument("--control-predictions", required=True, help="Path to baseline model predictions for comparison.")

    args = parser.parse_args()

    if args.command == "list":
        list_runs()
        return

    if args.command == "analyze":
        analyze_run(args)
        return

    # --- Pipeline Execution Logic (for 'run' command) ---
    # 1. Create Run Directory
    if args.experiment_name:
        run_dir = os.path.join("pipeline_runs", args.experiment_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("pipeline_runs", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 2. Setup Logging
    setup_logging(os.path.join(run_dir, "logs.log"))
    
    logging.info(f" Starting new pipeline run in: {run_dir}")

    # 3. Initialize Run Summary
    run_summary = {
        "run_id": os.path.basename(run_dir),
        "start_time": datetime.now().isoformat(),
        "status": "RUNNING",
        "parameters": vars(args),
        "steps": {},
        "artifacts": {}
    }
    save_summary(run_summary, run_dir)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        # --- STAGE 1: Target Bootstrap Discovery ---
        stage1_output = os.path.join(run_dir, "01_adaptive_targets.parquet")
        stage1_discovery = os.path.join(run_dir, "01_target_discovery.json")
        stage1_cache_key = compute_hash({
            'stage': 'target_discovery',
            'input': os.path.abspath(args.input_data),
            'features_json': os.path.abspath(args.features_json),
            'skip_walk_forward': args.skip_walk_forward,
            'smoke': args.smoke_mode,
            'limits': {
                'max_eras': _resolve_smoke_value(args, 'smoke_max_eras', 60),
                'row_limit': _resolve_smoke_value(args, 'smoke_row_limit', 100_000),
                'target_limit': _resolve_smoke_value(args, 'smoke_target_limit', 8)
            },
            'seed': args.seed
        })
        if args.stage1_from:
            # Reuse Stage 1 artifacts from provided run directory
            logging.info("=== STAGE 1: REUSED FROM PROVIDED RUN ===")
            try:
                s1_src_adapt = os.path.join(args.stage1_from, "01_adaptive_targets.parquet")
                s1_src_json = os.path.join(args.stage1_from, "01_target_discovery.json")
                if not (os.path.exists(s1_src_adapt) and os.path.exists(s1_src_json)):
                    raise FileNotFoundError("Provided --stage1-from does not contain required Stage 1 artifacts")
                shutil.copy2(s1_src_adapt, stage1_output)
                shutil.copy2(s1_src_json, stage1_discovery)
                update_summary_step(run_summary, "target_discovery", 0, {"adaptive_targets": stage1_output, "discovery_json": stage1_discovery}, status="CACHED")
                save_summary(run_summary, run_dir)
            except Exception as e:
                logging.critical(f"Failed to reuse Stage 1 from {args.stage1_from}: {e}")
                raise
        elif not os.path.exists(stage1_output) or args.force:
            cached_dir, meta = stage_cache_lookup('stage1', stage1_cache_key)
            if cached_dir and not args.force:
                logging.info("=== STAGE 1: RESTORED FROM CACHE ===")
                materialize_cached_artifacts(cached_dir, {
                    'adaptive_targets': stage1_output,
                    'discovery_json': stage1_discovery
                }, run_dir)
                update_summary_step(run_summary, "target_discovery", 0, {"adaptive_targets": stage1_output, "discovery_json": stage1_discovery}, status="CACHED")
            else:
                logging.info("="*20 + " STAGE 1: Target Bootstrap Discovery " + "="*20)
                start_time = time.time()
                step_01_target_discovery.run(
                    input_file=args.input_data,
                    features_json_file=args.features_json,
                    output_file=stage1_output,
                    discovery_file=stage1_discovery,
                    skip_walk_forward=args.skip_walk_forward,
                    max_eras=_resolve_smoke_value(args, 'smoke_max_eras', default_if_smoke=60),
                    row_limit=_resolve_smoke_value(args, 'smoke_row_limit', default_if_smoke=100_000),
                    target_limit=_resolve_smoke_value(args, 'smoke_target_limit', default_if_smoke=8)
                )
                duration = time.time() - start_time
                update_summary_step(run_summary, "target_discovery", duration, {"adaptive_targets": stage1_output, "discovery_json": stage1_discovery})
                stage_cache_store('stage1', stage1_cache_key, {
                    'adaptive_targets': stage1_output,
                    'discovery_json': stage1_discovery
                }, {'seed': args.seed})
                save_summary(run_summary, run_dir)
        else:
            logging.info("="*20 + " STAGE 1: SKIPPED (Cached) " + "="*20)
            update_summary_step(run_summary, "target_discovery", 0, {"adaptive_targets": stage1_output, "discovery_json": stage1_discovery}, status="CACHED")
            save_summary(run_summary, run_dir)

        # Predefine output paths for later stages
        stage2_output = os.path.join(run_dir, "02_discovered_relationships.json")
        stage3_output = os.path.join(run_dir, "03_enhanced_features.parquet")

        # --- STAGE 2: Creative Pathfinding Discovery ---
        if not args.disable_pathfinding:
            stage2_cache_key = compute_hash({
                'stage': 'pathfinding',
                'input': stage1_output,
                'yolo': args.yolo_mode,
                'limits': {
                    'feature_limit': _resolve_smoke_value(args, 'smoke_feature_limit', 300),
                    'row_limit': _resolve_smoke_value(args, 'smoke_row_limit', 100_000)
                },
                'seed': args.seed
            })
            if args.stage2_from:
                logging.info("=== STAGE 2: REUSED FROM PROVIDED RUN ===")
                try:
                    s2_src = os.path.join(args.stage2_from, "02_discovered_relationships.json")
                    if not os.path.exists(s2_src):
                        raise FileNotFoundError("Provided --stage2-from does not contain 02_discovered_relationships.json")
                    shutil.copy2(s2_src, stage2_output)
                    update_summary_step(run_summary, "pathfinding", 0, {"relationships_json": stage2_output}, status="CACHED")
                    save_summary(run_summary, run_dir)
                except Exception as e:
                    logging.critical(f"Failed to reuse Stage 2 from {args.stage2_from}: {e}")
                    raise
            elif not os.path.exists(stage2_output) or args.force:
                cached_dir, meta = stage_cache_lookup('stage2', stage2_cache_key)
                if cached_dir and not args.force:
                    logging.info("=== STAGE 2: RESTORED FROM CACHE ===")
                    materialize_cached_artifacts(cached_dir, {'relationships_json': stage2_output}, run_dir)
                    update_summary_step(run_summary, "pathfinding", 0, {"relationships_json": stage2_output}, status="CACHED")
                else:
                    logging.info("="*20 + " STAGE 2: Creative Pathfinding Discovery " + "="*20)
                    start_time = time.time()
                    step_02_pathfinding.run(
                        input_file=stage1_output,
                        target_col="adaptive_target",
                        output_relationships_file=stage2_output,
                        yolo_mode=args.yolo_mode,
                        feature_limit=_resolve_smoke_value(args, 'smoke_feature_limit', default_if_smoke=300),
                        row_limit=_resolve_smoke_value(args, 'smoke_row_limit', default_if_smoke=100_000),
                        debug=args.pf_debug,
                        debug_every_rows=getattr(args, 'pf_debug_every_rows', 10000),
                    )
                    duration = time.time() - start_time
                    update_summary_step(run_summary, "pathfinding", duration, {"relationships_json": stage2_output})
                    stage_cache_store('stage2', stage2_cache_key, {'relationships_json': stage2_output}, {'seed': args.seed})
                    save_summary(run_summary, run_dir)
            else:
                logging.info("="*20 + " STAGE 2: SKIPPED (Cached) " + "="*20)
                update_summary_step(run_summary, "pathfinding", 0, {"relationships_json": stage2_output}, status="CACHED")
                save_summary(run_summary, run_dir)
        else:
            logging.info("="*20 + " STAGE 2: DISABLED " + "="*20)
            update_summary_step(run_summary, "pathfinding", 0, {}, status="DISABLED")

        # --- STAGE 3: Feature Engineering ---
        if not args.disable_pathfinding:
            stage3_cache_key = compute_hash({
                'stage': 'feature_engineering',
                'relationships': stage2_output,
                'max_new_features': args.max_new_features,
                'limits': {
                    'row_limit': _resolve_smoke_value(args, 'smoke_row_limit', 100_000)
                },
                'seed': args.seed
            })
            if not os.path.exists(stage3_output) or args.force:
                cached_dir, meta = stage_cache_lookup('stage3', stage3_cache_key)
                if cached_dir and not args.force:
                    logging.info("=== STAGE 3: RESTORED FROM CACHE ===")
                    materialize_cached_artifacts(cached_dir, {'enhanced_data': stage3_output, 'new_features_list': os.path.join(run_dir, 'new_feature_names.json')}, run_dir)
                    # Ensure merged features.json exists when restored from cache
                    merged_fjson = _write_merged_features_json(run_dir, args.features_json, os.path.join(run_dir, 'new_feature_names.json'))
                    update_summary_step(run_summary, "feature_engineering", 0, {"enhanced_data": stage3_output, "new_features_list": os.path.join(run_dir, 'new_feature_names.json'), "features_json": merged_fjson}, status="CACHED")
                else:
                    logging.info("="*20 + " STAGE 3: Feature Engineering " + "="*20)
                    start_time = time.time()
                    step_03_feature_engineering.run(
                        input_file=stage1_output,
                        relationships_file=stage2_output,
                        output_file=stage3_output,
                        max_features=args.max_new_features,
                        yolo_mode=args.yolo_mode,
                        row_limit=_resolve_smoke_value(args, 'smoke_row_limit', default_if_smoke=100_000)
                    )
                    duration = time.time() - start_time
                    merged_fjson = _write_merged_features_json(run_dir, args.features_json, os.path.join(run_dir, 'new_feature_names.json'))
                    update_summary_step(run_summary, "feature_engineering", duration, {"enhanced_data": stage3_output, "new_features_list": os.path.join(run_dir, 'new_feature_names.json'), "features_json": merged_fjson})
                    stage_cache_store('stage3', stage3_cache_key, {'enhanced_data': stage3_output, 'new_features_list': os.path.join(run_dir, 'new_feature_names.json'), 'features_json': merged_fjson}, {'seed': args.seed})
                    save_summary(run_summary, run_dir)
            else:
                logging.info("="*20 + " STAGE 3: SKIPPED (Cached) " + "="*20)
                # Attempt to fill in merged features.json when skipping
                merged_fjson = _write_merged_features_json(run_dir, args.features_json, os.path.join(run_dir, 'new_feature_names.json'))
                update_summary_step(run_summary, "feature_engineering", 0, {"enhanced_data": stage3_output, "new_features_list": os.path.join(run_dir, 'new_feature_names.json'), "features_json": merged_fjson}, status="CACHED")
                save_summary(run_summary, run_dir)
        else:
            logging.info("="*20 + " STAGE 3: DISABLED " + "="*20)
            update_summary_step(run_summary, "feature_engineering", 0, {}, status="DISABLED")

        logging.info("✅ Pipeline run completed successfully!")
        run_summary["status"] = "SUCCESS"
        run_summary["end_time"] = datetime.now().isoformat()

    except Exception as e:
        logging.critical(f"❌ Pipeline run failed: {e}", exc_info=True)
        run_summary["status"] = "FAILED"
        run_summary["error"] = str(e)
    
    save_summary(run_summary, run_dir)
    logging.info(f" Final run summary saved to {os.path.join(run_dir, 'run_summary.json')}")

    if args.command == "run" and getattr(args, "pretty", False):
        try:
            print_pretty_summary(run_summary, color=not args.no_color)
        except Exception as _e:  # noqa: F841
            logging.warning("Pretty summary failed; falling back to plain output")

# --- Helper Functions for the Conductor ---

def save_summary(summary_data, run_dir):
    """Saves the run summary dictionary to a JSON file."""
    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=4)

def update_summary_step(summary_data, step_name, duration_secs, artifacts, status="SUCCESS"):
    """Updates the summary with the results of a single step."""
    summary_data["steps"][step_name] = {
        "status": status,
        "duration_seconds": round(duration_secs, 2),
        "artifacts": artifacts
    }
    summary_data["artifacts"].update(artifacts)


def print_pretty_summary(summary, color=True):
    """Pretty-print a concise run summary without extra dependencies."""
    def c(code):
        return f"\033[{code}m" if color else ""
    RESET = c(0)
    BOLD = c(1)
    GREEN = c(32)
    RED = c(31)
    YELLOW = c(33)
    CYAN = c(36)

    status = summary.get("status", "UNKNOWN")
    status_color = GREEN if status == "SUCCESS" else (RED if status == "FAILED" else YELLOW)
    print(f"{BOLD}{CYAN}Pipeline Run:{RESET} {summary.get('run_id','')}  Status: {status_color}{status}{RESET}")
    print(f"Started : {summary.get('start_time','')}  Ended: {summary.get('end_time','-')}")
    if 'error' in summary:
        print(f"{RED}Error: {summary['error']}{RESET}")

    # Steps table
    headers = ["Step", "Status", "Duration(s)", "Artifacts"]
    rows = []
    for step, info in summary.get('steps', {}).items():
        step_status = info.get('status','')
        sc = GREEN if step_status == 'SUCCESS' else (YELLOW if step_status == 'CACHED' else RED)
        arts = ", ".join(sorted(info.get('artifacts', {}).keys())[:4])
        if len(info.get('artifacts', {})) > 4:
            arts += ", …"
        rows.append([step, f"{sc}{step_status}{RESET}", str(info.get('duration_seconds','')), arts])

    col_widths = [max(len(str(r[i])) for r in ([headers]+rows)) for i in range(len(headers))]
    sep = "+" + "+".join("-"*(w+2) for w in col_widths) + "+"
    def fmt_row(r):
        return "| " + " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(r))) + " |"

    print(sep)
    print(fmt_row(headers))
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print(sep)

    # Key artifact shortcuts
    art = summary.get('artifacts', {})
    print(f"Key Artifacts:")
    for label in ["adaptive_targets", "relationships_json", "enhanced_data", "new_features_list", "features_json"]:
        if label in art:
            print(f"  - {label}: {art[label]}")


def list_runs():
    """Lists all available runs in the pipeline_runs directory."""
    runs_dir = "pipeline_runs"
    if not os.path.isdir(runs_dir):
        print("The 'pipeline_runs' directory does not exist. No runs to list.")
        return
    print("="*20 + " Available Pipeline Runs " + "="*20)
    run_dirs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))])
    if not run_dirs:
        print("No runs found.")
        return
    for run in run_dirs:
        summary_path = os.path.join(runs_dir, run, "run_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            status = summary.get("status", "UNKNOWN")
            print(f"- {run:<30} | Status: {status}")
        else:
            print(f"- {run:<30} | Status: INCOMPLETE (no summary)")

def analyze_run(args):
    """Analyzes the results of a previous run."""
    print("="*20 + " Analyzing Run " + "="*20)
    print(f"Run Directory: {args.run_dir}")
    analysis.run_analysis(
        run_dir=args.run_dir,
        validation_data=args.validation_data,
        control_predictions=args.control_predictions
    )


if __name__ == "__main__":
    main()
