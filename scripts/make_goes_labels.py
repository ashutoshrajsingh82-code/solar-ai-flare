#!/usr/bin/env python3
"""Root alias executing backend/scripts/make_goes_labels.py."""
import sys
from pathlib import Path

backend_script = Path(__file__).resolve().parent.parent / "backend" / "scripts" / "make_goes_labels.py"
if not backend_script.exists():
    backend_script = Path(__file__).resolve().parent / "backend" / "scripts" / "make_goes_labels.py"

import runpy
runpy.run_path(str(backend_script), run_name="__main__")
