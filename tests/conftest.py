import sys
from pathlib import Path

# The analysis modules live in code/ and import each other by name.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
