#!/usr/bin/env python3
"""
Script to update Python files to use the new tests output directory structure.
This script helps automate the process of modifying multiple files.
"""

import os
import re
from pathlib import Path

def update_file_imports(file_path):
    """Add the tests import to a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Check if import is already there
    if 'from tests import' in content:
        print(f"✓ {file_path} already has tests import")
        return False

    # Find the last import line
    lines = content.split('\n')
    last_import_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            last_import_idx = i
        elif line.strip() and not line.startswith('#'):
            break

    if last_import_idx >= 0:
        # Insert after the last import
        lines.insert(last_import_idx + 1, "from tests import setup_script_output, get_output_path, initialize_script_output, add_output_dir_arguments")
        new_content = '\n'.join(lines)

        with open(file_path, 'w') as f:
            f.write(new_content)

        print(f"✓ Updated imports in {file_path}")
        return True
    else:
        print(f"✗ Could not find import section in {file_path}")
        return False

def find_python_files():
    """Find all Python files that need updating."""
    files_to_update = []

    # Root directory Python files
    root_py_files = [
        'analyze_golden_eras.py',
        'apply_bootstrap_to_validation.py',
        'compare_model_performance.py',
        'compare_targets.py',
        'data_utils.py',
        'feature_engineering.py',
        'model_utils.py',
        'move_file.py',
        'run_pipeline.py',
        'search_utils.py',
        'train_experimental_model_chunked.py',
        'train_experimental_model.py',
        'train_control_model.py',
        'validation.py'
    ]

    for file in root_py_files:
        if os.path.exists(file):
            files_to_update.append(file)

    # Bootstrap pipeline files
    bootstrap_files = [
        'bootstrap_pipeline/analysis/performance.py',
        'bootstrap_pipeline/bootstrap/feature_discovery.py',
        'bootstrap_pipeline/bootstrap/metrics_utils.py',
        'bootstrap_pipeline/bootstrap/target_discovery.py',
        'bootstrap_pipeline/steps/step_02_motif_discovery.py',
        'bootstrap_pipeline/steps/step_02_pathfinding.py',
        'bootstrap_pipeline/steps/step_03_feature_engineering.py',
        'bootstrap_pipeline/steps/step_03_motif_engineering.py',
        'bootstrap_pipeline/utils/build_adaptive_target.py',
        'bootstrap_pipeline/utils/cache.py',
        'bootstrap_pipeline/utils/io.py'
    ]

    for file in bootstrap_files:
        if os.path.exists(file):
            files_to_update.append(file)

    return files_to_update

def main():
    print("Finding Python files to update...")
    files = find_python_files()

    print(f"Found {len(files)} files to update:")
    for file in files:
        print(f"  - {file}")

    print("\nUpdating imports...")
    updated_count = 0
    for file_path in files:
        if update_file_imports(file_path):
            updated_count += 1

    print(f"\nUpdated {updated_count} files with tests imports.")
    print("\nNote: You'll still need to manually update the main() functions")
    print("in each file to use the new output directory structure.")

if __name__ == '__main__':
    main()
