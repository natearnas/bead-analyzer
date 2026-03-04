# GUI Documentation

## Launching

```bash
bead-analyzer-gui
```

Or: `python -m bead_analyzer.gui`

## Layout

1. **Input file** – Browse for TIFF/OME-TIFF
2. **Output directory** – Where to save results (default: same as input)
3. **Experimental parameters**
   - Scale XY (µm/pixel)
   - Scale Z (µm/pixel)
   - Channel (0-based index for multi-channel stacks)
4. **Detection mode** – Manual, Blob, Trackpy, StarDist, or Cellpose (default: Blob)
5. **Analysis options** (always available)
   - **Fitting mode (radio buttons):**
     - `1D Gaussian` – Standard 1D Gaussian fitting per axis
     - `3D Gaussian` *(slower)* – Full 3D Gaussian fit to bead volume
     - `Both` – Run 1D and 3D fits
     - `No fit` *(peak width only)* – Use prominence-based peak width without Gaussian fitting
   - Subtract background – Interactive background ROI selection
   - Save bead diagnostics – Save per-bead diagnostic plots
   - Auto-reject low QA – Automatically reject beads with poor SNR/symmetry
   - Local background – Use annulus-based local background instead of global minimum
   - Robust fit (Huber loss, default ON) – Downweight outliers in Gaussian fitting tails
   - Box size, Beads for avg, QA thresholds (SNR, symmetry)
6. **Detection options** (enabled for Blob, Trackpy, StarDist modes)
   - Review detection overlay – Show detected points before processing
   - Blob fallback (StarDist) – Use local-max bead detector when StarDist fails (StarDist only)
7. **Cellpose options** (enabled for Cellpose mode only)
   - Model file – Select the model file (or set `FWHM_CELLPOSE_MODEL`)
   - Native 3D – Run full 3D Cellpose segmentation
   - Skip review – Skip the mask review step
   - Anisotropy (z/xy) – For anisotropic stacks (e.g. 1.0/0.51 = 1.96)
   - Min size (px) – Minimum mask area in pixels (default 3; set higher to filter small noise detections)
   - Flow threshold – Cellpose flow error threshold (default 0.4; lower = stricter)
8. **Run** – Start analysis

## Detection Mode Guidance

- **Blob**: Recommended default for most bead slides, especially tiny beads.
- **Trackpy**: Good for low-NA data or nonuniform background gradients.
- **StarDist / Cellpose**: Best when beads are relatively large in the image (about 15+ pixels in diameter).

## Workflow

1. Click **Browse** to select your 3D bead stack
2. Set scale_xy and scale_z from your microscope metadata
3. Choose mode (Manual opens interactive windows for bead selection)
4. Choose fitting mode (1D / 3D / Both / No fit)
5. Check other options as needed
6. Click **Run Analysis**

## Cellpose Mode

For Cellpose, set the environment variable before launching:

- Windows: `set FWHM_CELLPOSE_MODEL=C:\path\to\cellpose_bead_model`
- Unix: `export FWHM_CELLPOSE_MODEL=/path/to/cellpose_bead_model`

The GUI will show an error if the model path is not set or invalid.

## Troubleshooting

- **"Select a valid input file"** – Use Browse to pick a TIFF file
- **"Scale XY and Z must be numbers"** – Enter numeric values
- **"FWHM_CELLPOSE_MODEL"** – Set the env var for Cellpose mode
- **Matplotlib windows** – For manual/background selection, matplotlib windows will open; close them to continue
