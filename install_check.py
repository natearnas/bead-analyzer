#!/usr/bin/env python
"""Pre-install compatibility check for bead-analyzer.

Run this script before installing to verify that your Python environment
meets the minimum requirements and that dependencies can be resolved.

Usage:
    python install_check.py
"""

import subprocess
import sys


MIN_PYTHON = (3, 9)
MIN_PIP = (21, 0)

CORE_DEPS = [
    "numpy>=1.20",
    "scipy>=1.7",
    "matplotlib>=3.5",
    "tifffile>=2022.2",
    "pandas>=1.3",
    "imagecodecs>=2022.1",
    "psutil>=5.8",
    "customtkinter>=5.2",
]


def check_python_version():
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
    status = "OK" if ok else "FAIL"
    print(f"[{status}] Python version: {v.major}.{v.minor}.{v.micro}"
          f" (requires >={MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
    return ok


def check_pip_version():
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "--version"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        # e.g. "pip 23.1.2 from ..."
        version_str = out.strip().split()[1]
        parts = tuple(int(x) for x in version_str.split(".")[:2])
        ok = parts >= MIN_PIP
        status = "OK" if ok else "WARN"
        msg = f"[{status}] pip version: {version_str}"
        if not ok:
            msg += f" (recommend >={MIN_PIP[0]}.{MIN_PIP[1]}; run: python -m pip install --upgrade pip)"
        print(msg)
        return ok
    except Exception as exc:
        print(f"[FAIL] Could not determine pip version: {exc}")
        return False


def dry_run_install():
    print("\nResolving dependencies (dry-run) ...")
    cmd = [
        sys.executable, "-m", "pip", "install", "--dry-run",
        "--no-build-isolation",
    ] + CORE_DEPS
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            print("[OK] All core dependencies can be resolved.")
            return True
        else:
            print("[FAIL] Dependency resolution failed:")
            # Show only the error lines, not the full output
            for line in result.stderr.splitlines():
                if line.strip():
                    print(f"  {line}")
            return False
    except subprocess.TimeoutExpired:
        print("[WARN] Dependency check timed out (slow network?).")
        return False
    except Exception as exc:
        print(f"[FAIL] Could not run pip dry-run: {exc}")
        return False


def main():
    print("=" * 50)
    print("bead-analyzer install check")
    print("=" * 50)
    print()

    py_ok = check_python_version()
    pip_ok = check_pip_version()
    print()

    if not py_ok:
        print("Python version too old. Please upgrade to 3.9 or later.")
        sys.exit(1)

    dep_ok = dry_run_install()
    print()

    if py_ok and pip_ok and dep_ok:
        print("All checks passed. You can install with:")
        print("  pip install -e .")
        print("  pip install -e \".[stardist]\"   # optional")
        print("  pip install -e \".[cellpose]\"   # optional")
    else:
        print("Some checks failed. Review the output above before installing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
