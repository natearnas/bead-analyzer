# Bead Analyzer

Analyze images of microspheres for characterizing light microscopy systems. Measures FWHM, PSF shape, SNR, and symmetry from fluorescent beads in 3D image stacks. Supports manual bead selection, StarDist automatic detection, and Cellpose with custom-trained models.

## Quick Start

```bash
# Install
pip install -e .
# For StarDist: pip install -e ".[stardist]"
# For Cellpose: pip install -e ".[cellpose]"

# GUI
bead-analyzer-gui

# CLI (manual mode)
bead-analyzer image.tif --mode manual --scale_xy 0.26 --scale_z 2.0
```

## CLI

The command-line interface supports all three detection modes. Required arguments: input file, `--scale_xy`, and `--scale_z`.

```bash
# Show all options
bead-analyzer --help

# Manual: interactive bead selection (opens matplotlib windows)
bead-analyzer beads.tif --mode manual --scale_xy 0.26 --scale_z 2 --fit_gaussian

# StarDist: automatic detection (no training needed)
bead-analyzer beads.tif --mode stardist --scale_xy 0.26 --scale_z 2 --fit_gaussian --max_z_fwhm 15

# Cellpose: custom model (requires training first)
bead-analyzer beads.tif --mode cellpose --scale_xy 0.26 --scale_z 2 --cellpose_model path/to/model --fit_gaussian
```

Add `--na 1.4 --fluorophore "FITC"` to record experimental metadata in the summary. Use `--output_dir ./results` to write outputs to a different folder. See [docs/CLI.md](docs/CLI.md) for full reference.

## Detection Modes

| Mode | Description | Dependencies |
|------|-------------|--------------|
| **Manual** | Right-click on beads in MIP | Base only |
| **StarDist** | Automatic detection (pretrained) | stardist, csbdeep |
| **Cellpose** | Custom model, requires training | cellpose, torch |

### Detection Strategy Notes

- **StarDist path**: default is `2D_versatile_fluo` on the MIP, with optional classical blob fallback for sparse bead fields.
- **Cellpose path**: default is 2D MIP inference; optional native 3D inference is available with anisotropy support.
- **Best practice**: for anisotropic z-stacks, use Cellpose 3D with `--anisotropy (z_spacing/xy_spacing)`.

## Outputs

- `*_FWHM_data.csv` or `*_FWHM_districts.csv` – per-bead FWHM
- `*_FWHM_summary.txt` – mean ± std
- `*_bead_quality.csv` – QA metrics (SNR, symmetry)
- `*_average_bead_stack.tif` – upsampled average bead
- `*_average_bead_plot.png` – XY/XZ/YZ projections
- `*_rejected_beads.csv` – beads filtered by QA (if `--qa_auto_reject`)
- `bead_diagnostics/` – per-bead diagnostic plots (if `--save_diagnostics`)
- (Cellpose) `*_every_bead_log.csv`, `*_FWHM_heatmap.png`

## How The Measurement Works

Understanding the pipeline helps you choose the right options.

### 1. Bead Detection

A 2D maximum-intensity projection (MIP) is computed from the Z-stack. Beads
are located on the MIP by one of three methods:

| Method | How it works | When to use |
|--------|-------------|-------------|
| **Manual** | You right-click on beads | Few beads, or unusual samples |
| **StarDist** | Pretrained neural network (`2D_versatile_fluo`) finds star-convex objects | Most bead slides; no training needed |
| **Cellpose** | Custom-trained model segments bead masks | Dense or irregular fields where StarDist struggles |

StarDist has a **blob fallback** (`--use_blob_fallback`): if the neural network
finds zero beads (e.g. because beads are too dim or too sparse for the pretrained
model), it falls back to a classical Gaussian-smooth + local-maximum detector.

### 2. Center Refinement

Each detected bead center is refined to **sub-pixel precision** using parabolic
interpolation around the 3D intensity peak.  This matters because at coarse
pixel sizes (e.g. 0.51 µm/px), a 500 nm bead spans only ~3 pixels — snapping
to the nearest integer pixel shifts the center by up to 0.5 px, which is ~17%
of the bead FWHM.

### 3. Profile Extraction

Three 1D intensity profiles are drawn through the refined center:

- **Z profile**: average intensity in a small box around the center, per Z-plane
- **X profile**: a line across the best-focus plane, sampled with bilinear
  interpolation (`scipy.ndimage.map_coordinates`, `order=1`)
- **Y profile**: same, perpendicular axis

The line length (`--line_length`, default 5 µm) should be at least 3× the
expected FWHM so the profile tails reach background.  The box size
(`--box_size`, default 15 px) controls how many XY pixels are averaged for
the Z profile — larger boxes improve Z-profile SNR but blur if beads are close
together.

### 4. Background Subtraction

Before measuring the width, each profile needs its baseline set to zero.
Three strategies are available:

| Strategy | Flag | What it does | When to use |
|----------|------|-------------|-------------|
| **Global minimum** | (default) | `profile - min(profile)` | Clean samples, uniform background |
| **Detrend** | `--detrend` | `scipy.signal.detrend` (removes linear slope) | Profiles with a tilted baseline |
| **Local annulus** | `--local_background` | Subtracts the median intensity in a ring around the bead | Light-sheet data with spatially varying out-of-focus haze |

The **local annulus** method estimates background from a ring whose inner radius
equals the box half-size and outer radius extends a few more pixels.  This is
the most accurate option when different beads sit on different local background
levels (common in light-sheet microscopy where out-of-focus fluorescence varies
across the field).

**Note:** The Gaussian fit model (`A*exp(...) + C`) independently estimates a
constant background `C`, so the fitted FWHM is somewhat tolerant of baseline
errors regardless of which strategy you choose.  The prominence method also
measures half-maximum relative to the peak base, not absolute zero.

### 5. FWHM Measurement

Two methods are always computed for the Z axis, and optionally for X/Y:

| Method | What it does | Strengths |
|--------|-------------|-----------|
| **Prominence** | Finds the peak, measures width at half the peak prominence | Fast, no assumptions about shape, works on noisy profiles |
| **Gaussian fit** | Fits `A * exp(-0.5*((x-µ)/σ)²) + C`, computes FWHM = 2√(2 ln 2) × σ | Sub-pixel precision, gives a parametric model, reports fit quality |

Enable Gaussian fitting with `--fit_gaussian`.

### 6. Fitting Options

| Option | Flag | What it does |
|--------|------|-------------|
| **1D Gaussian** | `--fit_gaussian` | Fits 1D Gaussians to each axis profile independently |
| **3D Gaussian** | `--fit_3d` | Fits a single 3D Gaussian to the entire bead volume — more accurate for asymmetric PSFs because it uses all voxels simultaneously |
| **Robust fit** | `--robust_fit` | Switches the optimizer to use soft-L1 loss (a smooth Huber-like function) instead of least-squares. This **downweights outlier pixels** in the profile tails — useful when beads are clipped by the edge of a segmentation mask, when a neighboring bead contaminates the tail, or when the PSF is slightly asymmetric. Without this, a single bright outlier pixel in the tail can pull the Gaussian wider than reality. |

All Gaussian fits (1D and 3D) enforce **parameter bounds**: amplitude > 0,
sigma ≥ 0.3 px, center within the data window.  This prevents degenerate fits
(negative width, negative amplitude) that could silently corrupt results.

### 7. Quality Assurance

After fitting, each bead gets two QA scores:

- **SNR**: peak intensity above noise (estimated from the Z-profile baseline).
  Default threshold: 3.0.
- **Symmetry**: how symmetric the Z-profile is around its peak (1.0 = perfect).
  Default threshold: 0.6.

With `--qa_auto_reject`, beads below these thresholds are automatically removed
from the summary statistics and written to a separate `*_rejected_beads.csv`.

### 8. Diagnostics

`--save_diagnostics` writes a PNG per bead to `bead_diagnostics/` showing:
- Z/X/Y profiles with FWHM markers
- XY/XZ/YZ projections through the bead center
- All numeric results (FWHM, QA scores, fit residuals)

This is invaluable for verifying that the pipeline is measuring what you expect.

## Recommended Options By Use Case

### Standard confocal bead slide (high SNR, uniform background)
```bash
bead-analyzer beads.tif --mode stardist --scale_xy 0.26 --scale_z 2 \
  --fit_gaussian --fit_3d --qa_auto_reject
```

### Light-sheet (spatially varying background, anisotropic Z)
```bash
bead-analyzer beads.tif --mode stardist --scale_xy 0.51 --scale_z 1.0 \
  --fit_gaussian --fit_3d --robust_fit --local_background \
  --qa_auto_reject --save_diagnostics
```

### Sparse beads where StarDist finds nothing
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
- [Building Windows executable](BUILD.md)

## Sample Data

Use any 3D TIFF stack of fluorescent beads (e.g. from your confocal or light-sheet microscope). The tool expects a single-channel or multi-channel stack with dimensions (Z, Y, X) or (C, Z, Y, X).

## Cellpose Workflow

1. Annotate: `python scripts/annotate_beads.py image.tif` → creates `*_raw.tif`, `*_masks.tif`
2. Train: `python scripts/train_cellpose.py path/to/folder`
3. Set model: `set FWHM_CELLPOSE_MODEL=path/to/cellpose_bead_model` (Windows) or `export FWHM_CELLPOSE_MODEL=...` (Unix)
4. Analyze: `bead-analyzer image.tif --mode cellpose --scale_xy 0.26 --scale_z 2 --cellpose_model path/to/model`

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## Reproducibility And Versions

- Current tested minimums in this repo:
  - `cellpose>=4.0.8`
  - `stardist>=0.9.2`
  - `csbdeep>=0.7.4`
- Install extras:
  - `pip install -e ".[stardist]"`
  - `pip install -e ".[cellpose]"`

## Consulting & Collaboration

This project is developed and maintained by Nathan O'Connor, PhD, MS,Arnas Technologies, LLC.

While this software is provided as a tool for the community, I offer professional services for labs and companies requiring tailored, end-to-end solutions. I specialize in bridging the gap from sample to reports and figures, ensuring that your raw imaging data is transformed into actionable scientific insight.

### How I Can Support Your Research

- **Custom Image Analysis:** Development of bespoke pipelines for complex datasets, including deconvolution, destriping, and 3D registration (e.g., Allen Brain Atlas mapping).
- **End-to-End Workflow Design:** Automation of your entire data lifecycle, from microscope-specific acquisition to publication-ready figures and statistical reports.
- **Grant Inclusion:** I am available for inclusion in NIH/NSF and private foundation grants as a Key Personnel or Consultant to lead neuroinformatics and imaging core initiatives.
- **Hardware Integration:** Custom software interfaces for specialized optical setups (e.g., multi-camera light-sheet systems, optical rectification).

### Contact

If you are interested in discussing a collaboration or professional engagement, please reach out via:

- **Email:** nate@arnastechnologies.com
- **Website:** [www.arnastechnologies.com](https://www.arnastechnologies.com)

## License

MIT
