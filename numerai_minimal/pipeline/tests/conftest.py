import os
import sys
from pathlib import Path

# Ensure project root and original/ are on sys.path for imports in tests
ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "original"
for p in [ROOT, ORIG]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Optionally, set environment variables used by some modules
os.environ.setdefault("PYTHONHASHSEED", "0")
