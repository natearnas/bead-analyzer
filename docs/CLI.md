# CLI Documentation

## Usage

```bash
bead-analyzer INPUT.tif --mode MODE --scale_xy X --scale_z Z [options]
```

Or: `python -m bead_analyzer.cli INPUT.tif ...`

## Required Arguments

| Argument | Description |
|----------|-------------|
| `input_file` | Path to 3D or 4D TIFF stack |
| `--scale_xy` | XY pixel size (µm/pixel) |
| `--scale_z` | Z pixel size (µm/pixel) |

## Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | blob | `manual`, `blob`, `trackpy`, `stardist`, or `cellpose` |
| `--channel` | 0 | Channel index for 4D stacks |
| `--box_size` | 15 | Box size for Z-profile (pixels) |
| `--line_length` | 5.0 | Line length for XY FWHM (µm) |
| `--z_smooth` | 1.0 (manual) | Gaussian sigma for Z smoothing |
| `--detrend` | False | Linear detrending of profiles |
| `--subtract_background` | False | Interactive background ROI |
| `--fit_gaussian` | False | Gaussian fit FWHM |
| `--fit_window` | 20 | Window for Gaussian fit (pixels) |
| `--gamma` | 1.0 | Display gamma |
| `--output_dir` | (input dir) | Output directory |
| `--no_plots` | False | Suppress plots |
| `--upsample_factor` | 4 | Upsampling for average bead |
| `--na` | - | Numerical aperture (metadata) |
| `--fluorophore` | - | Fluorophore name (metadata) |
| `--prominence_rel` | 0.1 | Relative prominence (fraction of signal range) |
| `--prominence_min` | - | Absolute prominence threshold (overrides relative) |
| `--review_detection` | False | Review detected points before analysis (manual mode excluded) |
| `--stardist_model` | 2D_versatile_fluo | StarDist pretrained model name (best for beads ~15+ px diameter) |
| `--stardist_prob_thresh` | 0.6 | StarDist probability threshold |
| `--stardist_nms_thresh` | 0.3 | StarDist NMS threshold |
| `--use_blob_fallback` | False | Fallback to local-max blob detector if StarDist fails |
| `--blob_sigma` | 1.2 | Blob detector Gaussian sigma (pixels) |
| `--blob_threshold_rel` | 0.2 | Blob detector threshold as signal-range fraction |
| `--blob_min_distance` | 5 | Blob detector minimum peak spacing (pixels) |
| `--stardist_n_tiles` | - | Tile grid for large images, e.g. `4 4` (prevents GPU OOM) |
| `--trackpy_diameter` | 5 | trackpy feature diameter in pixels (use odd values; start near bead size) |
| `--trackpy_minmass` | 5000 | trackpy minimum integrated brightness |
| `--trackpy_separation` | - | trackpy minimum feature separation (pixels) |
| `--cellpose_model` | - | Cellpose model path (best for beads ~15+ px diameter) |
| `--skip_cellpose_review` | False | Skip Cellpose detection review overlay |
| `--qa_min_snr` | 3.0 | QA: minimum Z-profile SNR |
| `--qa_min_symmetry` | 0.6 | QA: minimum Z-profile symmetry (0-1) |
| `--qa_auto_reject` | False | Automatically reject beads failing QA thresholds |
| `--fit_3d` | False | Fit 3D Gaussian to each bead (more accurate but slower) |
| `--save_diagnostics` | False | Save per-bead diagnostic plots to `bead_diagnostics/` |
| `--local_background` | False | Local annulus background subtraction instead of global minimum |
| `--robust_fit` | False | Robust Gaussian fitting (soft-L1 / Huber-like loss) |
| `--num_beads_avg` | 20 | Beads for average (nearest to median Z-FWHM; 0 = all beads) |

## Mode-Specific Options

### Manual

- `--points_file` – Load coordinates from CSV/TXT (skip interactive click)
- `--smooth_xy` – Sigma for XY profile smoothing

### Blob

- `--points_file` – Override detection with pre-defined points
- `--max_z_fwhm` – Reject beads with Z-FWHM above this (µm)
- `--blob_sigma`, `--blob_threshold_rel`, `--blob_min_distance` – Blob detector tuning

### Trackpy

- `--points_file` – Override detection with pre-defined points
- `--max_z_fwhm` – Reject beads with Z-FWHM above this (µm)
- `--trackpy_diameter` – Expected bead diameter in pixels (odd integer)
- `--trackpy_minmass`, `--trackpy_separation` – Brightness and spacing constraints

### StarDist

- `--points_file` – Override detection with pre-defined points
- `--max_z_fwhm` – Reject beads with Z-FWHM above this (µm)
- Best used when beads span approximately 15+ pixels in diameter.

### Cellpose

- `--cellpose_model` – Path to custom model (or set `FWHM_CELLPOSE_MODEL` env var)
- `--cellpose_diameter` – Expected bead diameter (µm)
- `--detection_gauss_sigma` – Pre-smoothing for detection
- `--cellpose_cellprob` – Cell probability threshold
- `--cellpose_min_size` – Minimum mask area in pixels (default 3; Cellpose default of 15 is too large for beads)
- `--cellpose_flow_threshold` – Flow error threshold (default 0.4; lower = stricter filtering)
- `--cellpose_do_3d` – Native Cellpose 3D segmentation on full stack
- `--anisotropy` – Z/XY spacing ratio for 3D Cellpose (e.g. 1.0/0.51 = 1.96)
- Best used when beads span approximately 15+ pixels in diameter.
- `--num_beads_avg` – Beads for average (near median FWHM, 0 = all beads)
- `--z_range` – Z range `min max` (pixels)
- `--z_analysis_margin` – Margin around peak for Z analysis
- `--reject_outliers` – MAD multiplier for outlier rejection
- `--max_z_fwhm` – Reject beads with Z-FWHM above this

## Examples

```bash
# Blob (default) with background subtraction and Gaussian fit
bead-analyzer beads.tif --mode blob --scale_xy 0.26 --scale_z 2 \
  --subtract_background --fit_gaussian --na 1.4 --fluorophore "FITC"

# Trackpy for gradient backgrounds / low-NA data
bead-analyzer beads.tif --mode trackpy --scale_xy 0.51 --scale_z 1.0 \
  --trackpy_diameter 5 --trackpy_minmass 2000 --fit_gaussian

# StarDist (best for larger beads, ~15+ px)
bead-analyzer beads.tif --mode stardist --scale_xy 0.26 --scale_z 2 \
  --fit_gaussian --max_z_fwhm 15

# Cellpose (best for larger beads, ~15+ px)
bead-analyzer beads.tif --mode cellpose --scale_xy 0.26 --scale_z 2 \
  --cellpose_model ./models/cellpose_bead_model --fit_gaussian \
  --z_smooth 0.75 --reject_outliers 3

# With 3D Gaussian fitting and auto QA rejection
bead-analyzer beads.tif --mode stardist --scale_xy 0.26 --scale_z 2 \
  --fit_gaussian --fit_3d --qa_auto_reject --qa_min_snr 5

# Save per-bead diagnostic plots for manual review
bead-analyzer beads.tif --mode manual --scale_xy 0.26 --scale_z 2 \
  --fit_gaussian --fit_3d --save_diagnostics --output_dir ./analysis_output

# Cellpose native 3D detection for anisotropic stack
bead-analyzer beads.tif --mode cellpose --scale_xy 0.51 --scale_z 1.0 \
  --cellpose_model ./models/cellpose_bead_model --cellpose_do_3d --anisotropy 1.96

# Robust fitting with local background (recommended for light-sheet with OOF haze)
bead-analyzer beads.tif --mode stardist --scale_xy 0.51 --scale_z 1.0 \
  --fit_gaussian --robust_fit --local_background
```

## Technical Notes

See the [README](../README.md#how-the-measurement-works) for a full walkthrough
of the measurement pipeline (center refinement, profile extraction, background
strategy, fitting methods, and QA).

Key points for option selection:

- **`--local_background`**: Use for light-sheet data where out-of-focus haze
  varies across the field.  Estimates background from an annular ring around
  each bead rather than the global profile minimum.
- **`--robust_fit`**: Use when beads may be clipped by segmentation mask edges
  or contaminated by nearby beads.  Switches to soft-L1 loss (Huber-like) to
  downweight outlier pixels.
- **`--fit_3d`**: Fits a full 3D Gaussian to the bead volume.  More accurate
  than independent 1D fits, especially for asymmetric PSFs.
- All Gaussian fits enforce parameter bounds (amplitude > 0, sigma >= 0.3 px)
  to prevent degenerate results.
