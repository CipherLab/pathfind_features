# tests/__init__.py
"""
Test utilities and output management for Python scripts.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import argparse


def setup_script_output(script_name: str, base_dir: str = "tests") -> Path:
    """
    Set up output directory structure for a script run.

    Args:
        script_name: Name of the script (without .py extension)
        base_dir: Base directory for outputs (default: "tests")

    Returns:
        Path to the script's output directory for this run
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory path
    output_dir = Path(base_dir) / f"{script_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def setup_logging_with_output_dir(output_dir: Path, log_filename: str = "script.log") -> None:
    """
    Set up logging to write to the specified output directory.

    Args:
        output_dir: Directory to write log files to
        log_filename: Name of the log file (default: "script.log")
    """
    # Remove existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_file = output_dir / log_filename

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_output_path(output_dir: Path, filename: str) -> Path:
    """
    Get a full path for an output file in the script's output directory.

    Args:
        output_dir: The script's output directory
        filename: Name of the output file

    Returns:
        Full path to the output file
    """
    return output_dir / filename


def add_output_dir_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add standard output directory arguments to an argument parser.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        Modified ArgumentParser
    """
    parser.add_argument(
        '--output-dir',
        help='Custom output directory (overrides automatic timestamped directory)'
    )
    parser.add_argument(
        '--base-output-dir',
        default='tests',
        help='Base directory for outputs (default: tests)'
    )
    return parser


def initialize_script_output(script_name: str, args=None) -> Path:
    """
    Initialize output directory for a script, either from args or automatically.

    Args:
        script_name: Name of the script
        args: Parsed arguments (if None, uses automatic directory)

    Returns:
        Path to the output directory
    """
    if args and hasattr(args, 'output_dir') and args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = args.base_output_dir if args and hasattr(args, 'base_output_dir') else 'tests'
        output_dir = setup_script_output(script_name, base_dir)

    return output_dir
