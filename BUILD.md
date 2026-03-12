# Building & Installing

## Requirements

- **Python:** >=3.9 (tested on 3.9, 3.10, 3.11)
- **pip:** >=21.0 recommended

### Pre-install Check

Before installing, you can verify your environment is ready:

```bash
python install_check.py
```

This checks Python version, pip version, and dry-runs dependency resolution.

## Install From Source

```bash
# Base install (Manual, Blob, Trackpy detection)
pip install -e .

# With StarDist support
pip install -e ".[stardist]"

# With Cellpose support
pip install -e ".[cellpose]"

# Development (adds pytest)
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

Tests also run automatically via [GitHub Actions](/.github/workflows/test.yml)
on every push/PR to `main` (Python 3.9–3.11, Ubuntu + Windows).

## Building the Windows Executable

### Prerequisites

```bash
pip install pyinstaller
pip install -e .
```

### Build GUI executable

```bash
pyinstaller build_exe.spec
```

The executable will be in `dist/bead-analyzer-gui.exe`.

### Notes

- The base build includes manual mode only. For StarDist or Cellpose, install those extras first, then rebuild.
- The executable is large (~200-400 MB) due to numpy, scipy, matplotlib.
- Antivirus may flag new executables; you may need to add an exception.
