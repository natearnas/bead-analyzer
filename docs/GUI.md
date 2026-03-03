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
   - NA (numerical aperture)
   - Fluorophore (e.g. FITC, Texas Red)
   - Channel (0-based index for multi-channel stacks)
4. **Detection mode** – Manual, StarDist, or Cellpose
5. **Cellpose model** – Select the model file (or set `FWHM_CELLPOSE_MODEL`)
6. **Options**
   - Fit Gaussian (1D) – Standard 1D Gaussian fitting per axis
   - Fit 3D Gaussian – Full 3D Gaussian fit (more accurate but slower)
   - Subtract background – Interactive background ROI selection
   - Save bead diagnostics – Save per-bead diagnostic plots
   - Auto-reject low QA – Automatically reject beads with poor SNR/symmetry
   - Review StarDist detection – Show detection overlay before processing
   - Blob fallback (StarDist) – Use local-max bead detector when StarDist fails
   - Cellpose native 3D – Run full 3D Cellpose segmentation
   - Cellpose anisotropy (z/xy) – For anisotropic stacks (e.g. 1.0/0.51 = 1.96)
   - Cellpose min size (px) – Minimum mask area in pixels (default 3; set higher to filter small noise detections)
   - Flow threshold – Cellpose flow error threshold (default 0.4; lower = stricter)
   - Skip Cellpose review – Skip the mask review step
   - Local background – Use annulus-based local background instead of global minimum
   - Robust fit (Huber loss) – Downweight outliers in Gaussian fitting tails
   - Box size, QA thresholds (SNR, symmetry)
7. **Run** – Start analysis

## Workflow

1. Click **Browse** to select your 3D bead stack
2. Set scale_xy and scale_z from your microscope metadata
3. Optionally enter NA and fluorophore for report metadata
4. Choose mode (Manual opens interactive windows for bead selection)
5. Check options as needed
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
