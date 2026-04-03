# -----------------------------------------------------------------------------
# Bead Analyzer
#
# (c) 2026 Arnas Technologies, LLC
# Developed by Nathan O'Connor, PhD, MS
#
# Licensed under the MIT License.
# For consulting or custom development: nathan@arnastech.com
# -----------------------------------------------------------------------------

#!/usr/bin/env python3
"""Launcher for Bead Analyzer GUI (used by PyInstaller)."""
from bead_analyzer.gui import main

if __name__ == "__main__":
    exit(main())
