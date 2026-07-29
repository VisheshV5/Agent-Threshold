"""Makes the top-level demo/ package importable from tests.

demo/ isn't part of the installed threshold package (it's example
code, not library code), so it needs the repo root on sys.path
explicitly -- threshold itself is already importable via the editable
install.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
