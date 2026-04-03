# Bead Analyzer

Analyze images of microspheres for characterizing light microscopy systems. Measures FWHM, PSF shape, SNR, and symmetry from fluorescent beads in entire 3D image stacks or lateral regions within them. Supports manual bead selection, classical blob/trackpy detection, StarDist automatic detection, and Cellpose with custom-trained models.

## Quick Start

### Installation

Choose the method that best fits your workflow:

#### A. Recommended: Using Conda or Virtual Environments

To prevent conflicts with other imaging tools, we highly recommend using a fresh environment:

```bash
# Create and activate a clean environment
conda create -n bead-env python=3.10
conda activate bead-env

# Install the analyzer
pip install .
```

For optional detection backends:

```bash
pip install ".[stardist]"   # StarDist support
pip install ".[cellpose]"   # Cellpose support
```

#### B. For Researchers (No Git Required)

If you do not use Git, follow these steps to get running in minutes:

1. **Download:** Click the green **"<> Code"** button above and select **"Download ZIP"**.
2. **Extract:** Unzip the folder to your analysis directory (e.g., `D:\Analysis\bead-analyzer`).
3. **Verify & Install:** Open a terminal (PowerShell or Terminal.app), navigate to that folder, and run:

```bash
python install_check.py  # Run the built-in environment checker
pip install .
```

### Running the App

```bash
# Launch the Graphical Interface
bead-analyzer-gui

# Use the Command Line (default mode = blob)
bead-analyzer image.tif --scale_xy 0.26 --scale_z 2.0
```

## CLI Reference

The command-line interface supports five detection modes. Required arguments: input file, `--scale_xy`, and `--scale_z`.

```bash
# Show all options
bead-analyzer --help

# Blob: classical detector (recommended default for most bead slides)
bead-analyzer beads.tif --mode blob --scale_xy 0.26 --scale_z 2 --fit_gaussian

# Trackpy: robust under intensity gradients / low-NA backgrounds
bead-analyzer beads.tif --mode trackpy --scale_xy 0.26 --scale_z 2 --fit_gaussian

# StarDist: neural network (best when beads are ~15+ px diameter)
bead-analyzer beads.tif --mode stardist --scale_xy 0.26 --scale_z 2 --fit_gaussian --max_z_fwhm 15

# Cellpose: custom model (best when beads are ~15+ px diameter)
bead-analyzer beads.tif --mode cellpose --scale_xy 0.26 --scale_z 2 --cellpose_model path/to/model --fit_gaussian
```

Add `--na 1.4 --fluorophore "FITC"` to record experimental metadata in the summary. Use `--output_dir ./results` to write outputs to a different folder. See [docs/CLI.md](docs/CLI.md) for full reference.

## Detection Modes

| Mode | Description | Dependencies |
|------|-------------|--------------|
| **Manual** | Right-click on beads in MIP | Base only |
| **Blob** | Classical Gaussian smooth + local maxima | Base only |
| **Trackpy** | Bandpass filter + sub-pixel centroid detection | trackpy |
| **StarDist** | Automatic detection (pretrained neural network) | stardist, csbdeep |
| **Cellpose** | Custom model, requires training | cellpose, torch |

### Detection Strategy Notes

- **Blob path**: recommended default for most bead slides, especially tiny beads.
- **Trackpy path**: preferred when backgrounds have gradients or low-NA blur.
- **StarDist/Cellpose path**: best when beads span roughly 15+ pixels in diameter.
- **Cellpose 3D**: for anisotropic z-stacks, use `--cellpose_do_3d --anisotropy (z_spacing/xy_spacing)`.
- **Center refinement for large annular beads**: try `--center_mode edge` first; use `--center_mode radial` as fallback and `--center_mode centroid` for filled but noisy beads.
- **Trackpy tuning for large beads**: increase `trackpy_diameter` and `trackpy_separation`; these are available in GUI Advanced options and CLI.

### Detector Caveats And Platform Notes

- **StarDist on native Windows**: often CPU-heavy in practical setups because StarDist depends on TensorFlow. This can increase runtime and RAM usage compared with classical detectors. TensorFlow GPU workflows are generally smoother on Linux/WSL2 than native Windows.
- **Cellpose in this pipeline**: intended for bead-specific custom models. Generic biological pretrained models (for cells/nuclei) often underperform on tiny PSF beads.
- **Practical default for beads**: start with Blob or Trackpy for most datasets, then use StarDist/Cellpose for dense, overlapping, or otherwise difficult fields where classical detectors struggle.

### AI Methods Not Yet Integrated

For bead-center localization, some labs report strong results with keypoint/heatmap spot detectors (2D) and heatmap-regression or U-Net-style models (3D). These are not currently integrated in this repository; current AI options are StarDist and Cellpose.

## Outputs

- `*_FWHM_data.csv` -- per-bead FWHM measurements (all modes)
- `*_FWHM_summary.txt` -- mean +/- std report
- `*_bead_quality.csv` -- QA metrics (SNR, symmetry)
- `*_average_bead_stack.tif` -- upsampled average bead (all modes)
- `*_average_bead_plot.png` -- XY/XZ/YZ projections of average bead
- `*_summary_figure.png` -- publication-quality figure: average bead projections with scale bars + mean profiles with Gaussian fit overlay
- `*_detection_overview.png` -- MIP with bead locations (green=accepted, red=rejected)
- `*_FWHM_heatmap.png` -- 3x3 spatial FWHM variation across the field (all modes)
- `*_rejected_beads.csv` -- beads filtered by QA (if `--qa_auto_reject`)
- `bead_diagnostics/` -- per-bead diagnostic plots with fit overlays on all axes (if `--save_diagnostics`)
- (Cellpose) `*_every_bead_log.csv`

## How The Measurement Works

Understanding the pipeline helps you choose the right options.

### 1. Bead Detection

A 2D maximum-intensity projection (MIP) is computed from the Z-stack. Beads are located on the MIP by one of the five methods listed above.

StarDist still has a **blob fallback** (`--use_blob_fallback`) when you want to keep StarDist as the primary detector but recover from sparse/failed detections.

### 2. Center Refinement

Each detected bead center is refined after detection. You can select the strategy with `--center_mode`:

- **`peak`** (default): local 3D intensity-peak recentering + sub-pixel parabolic refinement.
- **`centroid`**: intensity-weighted centroid on the local XY plane near peak Z.
- **`radial`**: gradient-symmetry-weighted center (ring-friendly for hollow/annular beads).
- **`edge`**: edge/gradient-symmetry center with edge-weighted Z-plane selection for hollow/ring-like beads.

For PSF-like sub-resolution spots, keep `peak`. For resolved hollow-looking beads, `edge` is the best first choice; `radial` remains a simpler ring-friendly fallback.

### 3. Profile Extraction

Three 1D intensity profiles (Z, X, and Y) are drawn through the refined center using bilinear interpolation (`scipy.ndimage.map_coordinates`, `order=1`) for lateral axes.

The line length (`--line_length`, default 5 um) should be at least 3x the expected FWHM so the profile tails reach background. The box size (`--box_size`, default 15 px) controls how many XY pixels are averaged for the Z profile -- larger boxes improve Z-profile SNR but blur if beads are close together.

### 4. Background Subtraction

Bead Analyzer supports three baseline correction strategies:

| Strategy | Flag | What it does | When to use |
|----------|------|-------------|-------------|
| **Global minimum** | (default) | `profile - min(profile)` | Clean samples, uniform background |
| **Detrend** | `--detrend` | `scipy.signal.detrend` (removes linear slope) | Profiles with a tilted baseline |
| **Local annulus** | `--local_background` | Subtracts the median intensity in a ring around the bead | Light-sheet data with spatially varying out-of-focus haze |

**Note:** The Gaussian fit model (`A*exp(...) + C`) independently estimates a constant background `C`, so the fitted FWHM is somewhat tolerant of baseline errors regardless of which strategy you choose. The prominence method also measures half-maximum relative to the peak base, not absolute zero.

### 5. FWHM Measurement

Both **Prominence** (robust to background offsets) and **Gaussian Fit** (parametric sub-pixel precision) methods are computed for analysis.

| Method | What it does | Strengths |
|--------|-------------|-----------|
| **Prominence** | Finds the peak, measures width at half the peak prominence | Fast, no assumptions about shape, works on noisy profiles |
| **Gaussian fit** | Fits `A * exp(-0.5*((x-mu)/sigma)^2) + C`, computes FWHM = 2*sqrt(2 ln 2) * sigma | Sub-pixel precision, gives a parametric model, reports fit quality |

Enable Gaussian fitting with `--fit_gaussian`.

### 6. Fitting Options

| Option | Flag | What it does |
|--------|------|-------------|
| **1D Gaussian** | `--fit_gaussian` | Fits 1D Gaussians to each axis profile independently |
| **3D Gaussian** | `--fit_3d` | Fits a single 3D Gaussian to the entire bead volume -- more accurate for asymmetric PSFs because it uses all voxels simultaneously |
| **Robust fit** | `--robust_fit` | Switches the optimizer to use soft-L1 loss (a smooth Huber-like function) instead of least-squares. This **downweights outlier pixels** in the profile tails -- useful when beads are clipped by the edge of a segmentation mask, when a neighboring bead contaminates the tail, or when the PSF is slightly asymmetric. |

All Gaussian fits (1D and 3D) enforce **parameter bounds**: amplitude > 0, sigma >= 0.3 px, center within the data window. This prevents degenerate fits (negative width, negative amplitude) that could silently corrupt results.

### 7. Quality Assurance

After fitting, each bead receives two QA scores:

- **SNR**: peak intensity above noise (estimated from the Z-profile baseline). Default threshold: 3.0.
- **Symmetry**: how symmetric the Z-profile is around its peak (1.0 = perfect). Default threshold: 0.6.

With `--qa_auto_reject`, beads below these thresholds are automatically removed from the summary statistics and written to a separate `*_rejected_beads.csv`.

### 8. Diagnostics

Use `--save_diagnostics` to write a PNG per bead to `bead_diagnostics/` showing:
- Z/X/Y profiles with FWHM markers
- XY/XZ/YZ projections through the bead center
- All numeric results (FWHM, QA scores, fit residuals)

This is invaluable for verifying that the pipeline is measuring what you expect.

## Recommended Options By Use Case

### Small sub-resolution beads (PSF-like spots)
```bash
bead-analyzer beads.tif --mode blob --scale_xy 0.26 --scale_z 2 \
  --box_size 11 --center_mode peak --fit_gaussian --qa_auto_reject
```
- Start with Blob (or Trackpy if background gradients are strong).
- Keep crop small (`--box_size` ~7-15 px) and center mode `peak`.

### Large filled beads (resolved, non-hollow)
```bash
bead-analyzer beads.tif --mode trackpy --scale_xy 0.0645 --scale_z 0.16 \
  --trackpy_diameter 31 --trackpy_separation 33 --box_size 51 \
  --center_mode centroid --fit_gaussian
```
- Increase Trackpy diameter/separation to match bead size in pixels.
- Use `centroid` when beads are broad/filled and `peak` drifts to local hot spots.

### Large hollow/annular beads
```bash
bead-analyzer beads.tif --mode trackpy --scale_xy 0.0645 --scale_z 0.16 \
  --trackpy_diameter 39 --trackpy_separation 41 --box_size 61 \
  --center_mode edge --fit_gaussian
```
- Use `edge` first for ring-like beads (best geometric centering in many cases).
- If needed, compare with `radial` on the same detections.
- If over-splitting occurs, raise `trackpy_separation` further.

### Standard confocal bead slide (high SNR, uniform background)
```bash
bead-analyzer beads.tif --mode blob --scale_xy 0.26 --scale_z 2 \
  --fit_gaussian --fit_3d --qa_auto_reject
```

### Light-sheet (spatially varying background, anisotropic Z)
```bash
bead-analyzer beads.tif --mode trackpy --scale_xy 0.51 --scale_z 1.0 \
  --fit_gaussian --fit_3d --robust_fit --local_background \
  --qa_auto_reject --save_diagnostics
```

### Keep StarDist primary but allow classical fallback
```bash
bead-analyzer beads.tif --mode stardist --scale_xy 0.51 --scale_z 1.0 \
  --use_blob_fallback --fit_gaussian
```

### Dense field with custom Cellpose model
```bash
bead-analyzer beads.tif --mode cellpose --scale_xy 0.26 --scale_z 2 \
  --cellpose_model path/to/model --fit_gaussian --reject_outliers 3
```

## Documentation

- [CLI usage](docs/CLI.md)
- [GUI usage](docs/GUI.md)
- [Example commands](examples/example_commands.txt)
- [Building & installing](BUILD.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [CI workflow](.github/workflows/test.yml) -- automatic testing on Python 3.9-3.11 for Windows and Ubuntu

## Supported Formats

**File types:** TIFF, BigTIFF, and OME-TIFF image stacks (`.tif`, `.tiff`, `.ome.tif`). Proprietary formats (`.nd2`, `.czi`, `.lif`, etc.) should be exported to TIFF from your acquisition software before analysis.

**Large files:** For multi-gigabyte stacks, crop your data before analysis or ensure your system has sufficient RAM, as the entire stack is loaded into memory.

The tool expects a single-channel or multi-channel stack with dimensions **(Z, Y, X)** or **(C, Z, Y, X)**. Use `--channel` to select a channel from multi-channel data.

## Sample Data

Use any 3D TIFF stack of fluorescent beads (e.g. from your confocal or light-sheet microscope).

## Cellpose Workflow

1. Annotate: `python scripts/annotate_beads.py image.tif` -> creates `*_raw.tif`, `*_masks.tif`
2. Train: `python scripts/train_cellpose.py path/to/folder`
3. Set model: `set FWHM_CELLPOSE_MODEL=path/to/cellpose_bead_model` (Windows) or `export FWHM_CELLPOSE_MODEL=...` (Unix)
4. Analyze: `bead-analyzer image.tif --mode cellpose --scale_xy 0.26 --scale_z 2 --cellpose_model path/to/model`

## Development & Reproducibility

### Testing

```bash
pip install -e ".[dev]"
ruff check .
pytest tests/ -v
```

Continuous Integration (CI) runs automatically on every push to `main` via [GitHub Actions](.github/workflows/test.yml), testing across Python 3.9-3.11 on both Ubuntu and Windows.

### Pre-install Check

If you are on an older or unfamiliar system, run the install checker first:

```bash
python install_check.py
```

This verifies your Python version (>=3.9), pip version, and dry-runs dependency resolution to catch conflicts before they happen.

### Reproducibility

- **Python:** >=3.9 (tested on 3.9, 3.10, 3.11)
- Current tested minimums in this repo:
  - `cellpose>=4.0.8`
  - `stardist>=0.9.2`
  - `csbdeep>=0.7.4`

## Consulting & Collaboration

This project is developed and maintained by Nathan O'Connor, PhD, MS at Arnas Technologies, LLC. I offer professional services for labs and companies requiring tailored, end-to-end solutions, specializing in bridging the gap from sample to reports and figures.

### How I Can Support Your Research

- **Custom Image Analysis:** Development of bespoke pipelines for complex datasets, including deconvolution, destriping, and 3D registration (e.g., Allen Brain Atlas mapping).
- **End-to-End Workflow Design:** Automation of your entire data lifecycle, from microscope-specific acquisition to publication-ready figures and statistical reports.
- **Grant Inclusion:** I am available for inclusion in NIH/NSF and private foundation grants as a Key Personnel or Consultant to lead neuroinformatics and imaging core initiatives.
- **Hardware Integration:** Custom software interfaces for specialized optical setups (e.g., multi-camera light-sheet systems, optical rectification).

### Contact

If you are interested in discussing a collaboration or professional engagement, please reach out via:

- **Email:** nathan@arnastech.com
- **Website:** [www.arnastechnologies.com](https://www.arnastechnologies.com)

## Acknowledgments

This tool optionally integrates the following packages. If you use one of
these detection modes in published work, please cite the original authors:

- **Trackpy** -- Allan, D. B., Caswell, T., Keim, N. C., van der Wel, C. M., & Verweij, R. W. *soft-matter/trackpy*. Zenodo. [DOI: 10.5281/zenodo.4682814](https://doi.org/10.5281/zenodo.4682814)
- **StarDist** -- Schmidt, U., Weigert, M., Broaddus, C., & Myers, G. *Cell Detection with Star-Convex Polygons.* MICCAI 2018. [DOI: 10.1007/978-3-030-00934-2_30](https://doi.org/10.1007/978-3-030-00934-2_30)
- **Cellpose** -- Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. *Cellpose: a generalist algorithm for cellular segmentation.* Nature Methods 18, 100-106 (2021). [DOI: 10.1038/s41592-020-01018-x](https://doi.org/10.1038/s41592-020-01018-x)

## License

MIT
