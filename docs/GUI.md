# GUI Documentation

## Launching

```bash
bead-analyzer-gui
```

Or: `python -m bead_analyzer.gui`

The GUI window title shows the version number (e.g., "Bead Analyzer v1.0.0").

## Layout Overview

The GUI is organized into logical sections from top to bottom:

### 1. File Selection
- **Input file** – Browse for TIFF/OME-TIFF stack
- **Output directory** – Where to save results (default: same directory as input file)

### 2. Experimental Parameters
- **Scale XY** (µm/pixel) – XY pixel size from your microscope metadata
- **Scale Z** (µm/pixel) – Z step size from your microscope metadata
- **Channel** – Channel index for multi-channel stacks (0-based, automatically detects number of channels)

### 3. Bead Detection
Five radio buttons (default: Blob):
- **Manual** – Right-click on beads interactively
- **Blob** – Classical Gaussian smooth + local maxima (recommended default)
- **Trackpy** – Bandpass filter + centroid detection (best for gradients/low-NA)
- **StarDist** – Pretrained neural network (best for beads ~15+ px diameter)
- **Cellpose** – Custom-trained model (best for beads ~15+ px diameter)

### 4. Core Analysis Controls (Always Available)

#### Fitting Method (Radio Buttons)
- **1D Gaussian** – Standard 1D Gaussian fitting per axis (fast, sub-pixel precision)
- **3D Gaussian** *(slower)* – Full 3D Gaussian fit to entire bead volume (more accurate for asymmetric PSFs)
- **Both** – Run both 1D and 3D fits
- **No fit** *(peak width only)* – Use only prominence-based peak width (fastest, no fitting)

**Robust fit (Huber loss)** – Checkbox (default: ON). Downweights outlier pixels in Gaussian fitting tails. Recommended for beads clipped by mask edges or contaminated by nearby beads.

#### Background Subtraction
- **Subtract background** – Interactive ROI selection for global background subtraction
- **Local background** – Use annulus-based local background instead of global minimum (recommended for light-sheet)

#### Quality & Output
- **Save bead diagnostics** – Save per-bead diagnostic plots to `bead_diagnostics/` folder
- **Auto-reject low QA** – Automatically exclude beads failing QA thresholds
- **Box size (px)** – Pixels around bead center for Z-profile averaging (default: 15)
- **Beads to avg (0=all)** – Number of beads for average bead (0 = all, N = closest to median Z-FWHM, default: 20)
- **QA min SNR** – Minimum signal-to-noise ratio (default: 3.0)
- **QA min symmetry** – Minimum Z-profile symmetry 0-1 scale (default: 0.6)

### 5. Detection Options (Enabled for Blob/Trackpy/StarDist only)
- **Review detection overlay** – Show detected points before processing (press 'y' to accept, 'n' to abort)
- **Blob fallback (StarDist)** – Fall back to blob detector if StarDist finds 0 beads (StarDist mode only)

### 6. Cellpose Options (Enabled for Cellpose mode only)
- **Model file** – Path to trained Cellpose model (or set `FWHM_CELLPOSE_MODEL` environment variable)
- **Native 3D** – Run full 3D Cellpose segmentation on entire stack (instead of 2D MIP)
- **Skip review** – Skip the interactive mask review step
- **Anisotropy (z/xy)** – Z/XY spacing ratio for 3D Cellpose (e.g., 2.0/0.26 = 7.7)
- **Min size (px)** – Minimum mask area in pixels (default: 3; increase to filter noise)
- **Flow threshold** – Cellpose flow error threshold (default: 0.4; lower = stricter filtering)

### 7. Run Button
- **Run Analysis** – Start analysis in background thread
- **Status** – Shows current progress (e.g., "Loading image...", "Detecting beads...", "Done")

## GUI Features

### Persistent Settings
The GUI automatically saves your settings to `~/.bead_analyzer_last_settings.json` and restores them when you relaunch. This includes:
- Last input file path
- Output directory
- All parameter values (scales, mode, fitting options, QA thresholds, etc.)

A copy is also saved to `<output_dir>/bead_analyzer_settings.json` for reproducibility.

### Smart UI Updates
- **Mode-dependent controls**: When you change detection mode, irrelevant options are automatically disabled/grayed out
  - Detection Options section is only active for Blob/Trackpy/StarDist
  - Cellpose Options section is only active for Cellpose mode
  - Blob fallback checkbox is only active in StarDist mode
- **Channel detection**: When you select an input file, the GUI automatically detects the number of channels and updates the channel dropdown

## Typical Workflow

### First-Time Setup
1. Click **Browse** next to "Input file" and select your 3D bead TIFF stack
2. Enter **Scale XY** and **Scale Z** from your microscope metadata
   - Check your microscope software (e.g., NIS-Elements, ZEN, FluoView)
   - Or use ImageJ: `Image → Show Info` to see pixel dimensions
3. Choose **Bead detection** (start with Blob for most cases)
4. Choose **Fitting method** (1D Gaussian is a good default, or Both for complete analysis)
5. Adjust **QA thresholds** if needed (defaults are usually fine)
6. Click **Run Analysis**

### Subsequent Uses
Settings are automatically restored from your last session, so you only need to:
1. Browse to new input file
2. Verify scales are correct (if switching microscopes)
3. Click Run Analysis

## Detection Mode Guidance

- **Blob**: Recommended default for most bead slides, especially tiny beads (< 10 px diameter)
- **Trackpy**: Good for low-NA data, nonuniform background gradients, or dim beads
- **StarDist**: Best when beads are relatively large (~15+ px diameter) and well-separated
- **Cellpose**: Best for dense/overlapping bead fields when beads are ~15+ px diameter (requires trained model)
- **Manual**: Use when automatic detection fails, or for targeted analysis of specific beads

## Tips & Best Practices

### Getting Good Results
- **Use Gaussian fitting** (`1D Gaussian` or `Both`) for publication-quality measurements
- **Enable "Auto-reject low QA"** to automatically filter poor-quality beads
- **Enable "Save bead diagnostics"** when troubleshooting to inspect individual bead measurements
- **Use "Local background"** for light-sheet data with varying out-of-focus haze
- **Keep "Robust fit" ON** (default) unless you have very clean, noise-free data

### Quick workflow
1. Click **Browse** to select your 3D bead stack
2. Set scale_xy and scale_z from your microscope metadata
3. Choose mode (Manual opens interactive windows for bead selection)
4. Choose fitting mode (1D / 3D / Both / No fit)
5. Check other options as needed
6. Click **Analyze beads**

### Common Adjustments
- **Too few beads detected?** Try Review detection overlay, then lower blob threshold or adjust trackpy parameters via CLI
- **Too many false detections?** Increase QA thresholds or enable Auto-reject
- **Beads at image edges causing problems?** Enable "Robust fit" to handle edge clipping
- **Want faster processing?** Use "No fit" mode (prominence-based width only)

## Cellpose-Specific Setup

Cellpose mode requires a trained model. Two ways to provide it:

### Option 1: Environment Variable (Recommended)
Set before launching the GUI:
- **Windows**: `set FWHM_CELLPOSE_MODEL=C:\path\to\cellpose_bead_model`
- **Unix/Mac**: `export FWHM_CELLPOSE_MODEL=/path/to/cellpose_bead_model`

Then launch: `bead-analyzer-gui`

### Option 2: Browse Button
Click "Browse" in the Cellpose options section and select your model file.

### Training a Cellpose Model
See main README "Cellpose Workflow" section for training instructions using `scripts/annotate_beads.py` and `scripts/train_cellpose.py`.

## Interactive Windows

Some modes open matplotlib windows for user input:

### Manual Mode
- **Bead selection window**: Right-click on each bead center, press Escape when done, close window to continue

### Background Subtraction
- **Background ROI window**: Right-click and drag to draw rectangle over background region, close window to continue

### Detection Review (if enabled)
- **Review window**: Press 'y' to accept detected points, 'n' to abort and retry with different settings

### Cellpose Review (if not skipped)
- **Mask overlay window**: Press 'y' to proceed with detected masks, 'n' to abort

## Troubleshooting

### GUI Issues
- **"Select a valid input file"** – Use Browse button to pick a valid TIFF file
- **"Scale XY and Z must be numbers"** – Enter numeric values (e.g., 0.26, not "0.26 µm")
- **Window is too tall for my screen** – Minimum height is 1030 px; use a larger monitor or reduce OS scaling
- **Settings not persisting** – Check write permissions for your home directory (`~/.bead_analyzer_last_settings.json`)

### Analysis Issues
- **"No beads detected"** – See main README Troubleshooting section, or try Manual mode
- **All FWHM values are None** – See main README Troubleshooting section
- **Cellpose model not found** – Verify path in Browse field or check `FWHM_CELLPOSE_MODEL` environment variable
- **Analysis hangs / no progress** – Check terminal/console for error messages; matplotlib windows may need to be closed

### When to Use CLI Instead
The CLI offers more control for advanced users:
- Batch processing multiple files
- Scripting and automation
- Fine-tuning detection parameters (blob_sigma, trackpy_diameter, stardist thresholds, etc.)
- Custom prominence thresholds, detrending, z_range filtering
- Loading coordinates from points_file

See [CLI.md](CLI.md) for full parameter reference.
