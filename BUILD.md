# Building the Windows Executable

## Prerequisites

```bash
pip install pyinstaller
pip install -e .
```

## Build GUI executable

```bash
pyinstaller build_exe.spec
```

The executable will be in `dist/bead-analyzer-gui.exe`.

## Notes

- The base build includes manual mode only. For StarDist or Cellpose, install those extras first, then rebuild.
- The executable is large (~200-400 MB) due to numpy, scipy, matplotlib.
- Antivirus may flag new executables; you may need to add an exception.
