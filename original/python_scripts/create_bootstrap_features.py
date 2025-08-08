#!/usr/bin/env python3
"""
Entry point for the Fixed Unified Bootstrap Discovery Pipeline.

This script now delegates to the refactored pipeline modules for better separation of concerns.

To run, execute from the project root as a module:
python -m python_scripts.create_bootstrap_features --input-data <path> --output-data <path>
"""

import sys
import os

# Add the current directory to Python path so we can import bootstrap modules
sys.path.insert(0, os.path.dirname(__file__))

from bootstrap.pipeline import main
from bootstrap.utils import setup_logging

if __name__ == "__main__":
    # Basic logging setup until the full setup is initialized in the pipeline
    setup_logging("logs/fixed_unified_bootstrap.log", "INFO")
    main()
