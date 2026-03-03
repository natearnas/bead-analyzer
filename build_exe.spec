# PyInstaller spec for Bead Analyzer GUI (Windows)
# Build: pyinstaller build_exe.spec

import sys

a = Analysis(
    ['run_gui.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'customtkinter',
        'PIL',
        'numpy',
        'scipy',
        'matplotlib',
        'tifffile',
        'pandas',
        'imagecodecs',
        'bead_analyzer',
        'bead_analyzer.core',
        'bead_analyzer.detectors',
        'bead_analyzer.analysis',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='bead-analyzer-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window for GUI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
