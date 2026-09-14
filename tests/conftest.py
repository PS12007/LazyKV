import sys
from pathlib import Path

# Tests import `harness` and `scripts` helpers from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
