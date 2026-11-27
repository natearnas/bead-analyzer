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
| `--box_size` | 15 | Box size for Z-profile (pixels); the Z profile is the mean over this XY box at each slice |
| `--center_mode` | peak | XY center refinement mode: `peak`, `centroid`, `radial`, `edge` |
| `--line_length` | 5.0 | Line length for XY FWHM (µm) |
| `--z_smooth` | 1.0 (manual) | Gaussian sigma for Z smoothing |
| `--detrend` | False | Linear detrending of profiles |
| `--subtract_background` | False | Interactive background ROI |
| `--fit_gaussian` | False | Gaussian fit FWHM |
| `--fit_window` | 20 | Minimum window for Gaussian fit (pixels); widened automatically for broad peaks |
| `--gamma` | 1.0 | Display gamma |
| `--output_dir` | (input dir) | Output directory |
| `--no_plots` | False | Suppress plots |
| `--upsample_factor` | 4 | Upsampling for average bead |
| `--na` | - | Detection numerical aperture (recorded; used numerically by `--lightsheet`) |
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
| `--num_beads_avg` | 20 | Beads for average (nearest to median Z-FWHM; minimum 1) |
| `--sample_fraction` | 100 | Analyze this % of detected beads (1-100); random uniform sample; CSV/diagnostics only include analyzed beads |

## Light-Sheet Report And PSF Export

| Option | Default | Description |
|--------|---------|-------------|
| `--lightsheet` | off | Write the light-sheet report: sheet thickness per bead, pooled, and across the field |
| `--na` | None | Detection numerical aperture; with `--emission_nm` enables the box-size and theory checks |
| `--emission_nm` | None | Emission wavelength (nm) |
| `--excitation_nm` | None | Excitation wavelength (nm); compares a fitted Rayleigh range with an ideal Gaussian beam |
| `--ri` | 1.0 | Refractive index of the immersion/sample medium |
| `--bead_diameter` | None | Bead diameter (µm); removed from the sheet thickness in quadrature |
| `--sheet_axis` | auto | Image axis along which the sheet propagates: `x`, `y` or `auto` |
| `--scan_mode` | step | `step` (stage at rest during exposure) or `continuous` (stage moved one Z step per exposure) |
| `--export_psf` | off | Write `*_psf.tif` / `*_psf.json`: averaged bead at native voxel size, centered, unit sum |

The report works from the raw stack and the beads accepted by the main analysis, so detection and QA options apply to it too. Outputs: `*_lightsheet_summary.txt`, `*_lightsheet_beads.csv`, `*_lightsheet_report.json` and, unless `--no_plots`, `*_lightsheet_report.png`.

Columns of `*_lightsheet_beads.csv` (one row per bead):

| Column | Meaning |
|--------|---------|
| `id`, `x_um`, `y_um`, `z_um` | Bead id from the main analysis and its position; `z_um` is the sub-slice center of the box-integrated profile |
| `fwhm_z_box_um`, `fwhm_z_box_se_um` | Gaussian FWHM of the box-integrated Z profile and its standard error |
| `samples_per_fwhm` | Z slices per FWHM of that profile |
| `sheet_fwhm_um`, `sheet_fwhm_se_um` | Sheet thickness at the bead: the box-profile FWHM with bead size and scan motion removed |
| `sheet_width_1e2_um` | The same as a 1/e^2 full width (1.70 x FWHM) |
| `fwhm_z_onaxis_um`, `fwhm_z_onaxis_se_um` | FWHM of the Z profile through the bead center only: the system's on-axis axial resolution |
| `focus_offset_um` | Z of sharpest focus (on-axis peak) minus Z of the sheet center (box-profile peak) |
| `profile_misfit` | RMS residual of the Gaussian fit to the box profile, as a fraction of its amplitude |
| `sampling` | `ok` (3 or more slices per FWHM), `coarse` (2 to 3), `undersampled` (below 2) or `fit_failed` |
| `saturated`, `truncated`, `crowded` | Reasons a bead is set aside: a saturated pixel, a profile cut off by the stack, a neighbor inside the box at a similar depth |
| `used` | Whether the bead entered the per-bead statistics and the field fit |

## Mode-Specific Options

## Center Refinement Guidance

- `--center_mode peak` (default): best for tiny PSF-like beads where the brightest voxel is the true center.
- `--center_mode centroid`: good for filled large beads with noise/asymmetry.
- `--center_mode radial`: ring-friendly gradient-symmetry centering on the local slab.
- `--center_mode edge`: edge/gradient-symmetry centering with edge-weighted Z-plane selection; best first choice for hollow/ring-like beads.

## Recommended Starting Settings By Bead Type

- **Small sub-resolution beads**: Blob mode, `--box_size 7-15`, `--center_mode peak`.
- **Large filled beads**: Trackpy mode with larger `--trackpy_diameter`/`--trackpy_separation`, `--box_size ~51`, `--center_mode centroid`.
- **Large hollow/annular beads**: Trackpy mode with larger `--trackpy_diameter`/`--trackpy_separation`, `--box_size ~61`, `--center_mode edge`.

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
- GUI note: these Trackpy controls are also available under **Show Advanced Options → Trackpy options**.

### StarDist

- `--points_file` – Override detection with pre-defined points
- `--max_z_fwhm` – Reject beads with Z-FWHM above this (µm)
- Best used when beads span approximately 15+ pixels in diameter.
- On native Windows, StarDist can be CPU-heavy (TensorFlow backend), so runtime and RAM use may be higher than Blob/Trackpy.
- If you specifically need TensorFlow GPU workflows, Linux/WSL2 is commonly smoother than native Windows.

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
- For this bead workflow, use a bead-specific custom-trained model; generic biological pretrained models may underperform on tiny PSF beads.
- `--num_beads_avg` – Beads for average (near median FWHM, minimum 1)
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

# Light-sheet report with optics check and a measured PSF for deconvolution
bead-analyzer beads.tif --scale_xy 0.51 --scale_z 1.0 --fit_gaussian \
  --lightsheet --na 0.14 --emission_nm 525 --excitation_nm 488 --ri 1.0 \
  --bead_diameter 0.5 --sheet_axis x --export_psf

# Large annular beads: ring-friendly centering
bead-analyzer beads.tif --mode trackpy --scale_xy 0.0645 --scale_z 0.16 \
  --trackpy_diameter 39 --trackpy_separation 41 --box_size 61 --center_mode edge
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
