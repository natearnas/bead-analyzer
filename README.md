# Bead Analyzer

Analyze images of microspheres for characterizing light microscopy systems. Measures FWHM, PSF shape, SNR, and symmetry from fluorescent beads in 3D image stacks. Supports manual bead selection, classical blob/trackpy detection, StarDist automatic detection, and Cellpose with custom-trained models.

## Quick Start

```bash
# Install
pip install -e .
# For StarDist: pip install -e ".[stardist]"
# For Cellpose: pip install -e ".[cellpose]"

# Verify installation
bead-analyzer --help

# GUI
bead-analyzer-gui

# CLI (default mode = blob)
bead-analyzer image.tif --scale_xy 0.26 --scale_z 2.0

# Test with your own data
# Get pixel sizes from your microscope metadata (ImageJ: Image → Show Info)
# For Nikon: check image properties or .nd2 metadata
# For Zeiss: check .czi metadata or LSM properties
# For Olympus: check .oib/.oir metadata
```

## CLI

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

## Outputs

- `*_FWHM_data.csv` – per-bead FWHM measurements (all modes)
- `*_FWHM_summary.txt` – mean ± std report
- `*_bead_quality.csv` – QA metrics (SNR, symmetry)
- `*_average_bead_stack.tif` – upsampled average bead (all modes)
- `*_average_bead_plot.png` – XY/XZ/YZ projections of average bead
- `*_summary_figure.png` – publication-quality figure: average bead projections with scale bars + mean profiles with Gaussian fit overlay
- `*_detection_overview.png` – MIP with bead locations (green=accepted, red=rejected)
- `*_FWHM_heatmap.png` – 3x3 spatial FWHM variation across the field (all modes)
- `*_rejected_beads.csv` – beads filtered by QA (if `--qa_auto_reject`)
- `bead_diagnostics/` – per-bead diagnostic plots with fit overlays on all axes (if `--save_diagnostics`)
- (Cellpose) `*_every_bead_log.csv`

## How The Measurement Works

Understanding the pipeline helps you choose the right options.

### 1. Bead Detection

A 2D maximum-intensity projection (MIP) is computed from the Z-stack. Beads
are located on the MIP by one of five methods:

| Method | How it works | When to use |
|--------|-------------|-------------|
| **Manual** | You right-click on beads | Few beads, or unusual samples |
| **Blob** | Gaussian smoothing + local maxima | Default for small fluorescent beads |
| **Trackpy** | Bandpass filtering + centroid localization | Low-NA / background gradients |
| **StarDist** | Pretrained neural network (`2D_versatile_fluo`) finds star-convex objects | Beads roughly 15+ px diameter |
| **Cellpose** | Custom-trained model segments bead masks | Dense/overlapping fields, beads roughly 15+ px diameter |

StarDist still has a **blob fallback** (`--use_blob_fallback`) when you want to
keep StarDist as the primary detector but recover from sparse/failed detections.

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

Why prominence (instead of simple max-peak half-height):

- **Max-peak half-height** uses half of the absolute peak value. If baseline/background is non-zero, this biases width estimates.
- **Prominence half-height** uses half of `(peak - base)` and measures relative to the local peak base, so it is much more robust to background offsets and sloped baselines.
- **Gaussian fit** gives a third independent estimate based on fitted sigma and is useful when you want sub-pixel parametric widths.

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

## Parameter Quick Reference

This table summarizes all key parameters. For full details, see [docs/CLI.md](docs/CLI.md).

### Required (All Modes)
| Parameter | What it does | Example |
|-----------|-------------|---------|
| `input_file` | Path to 3D TIFF stack | `beads.tif` |
| `--scale_xy` | XY pixel size in µm/pixel (from microscope metadata) | `0.26` |
| `--scale_z` | Z step size in µm/pixel (from microscope metadata) | `2.0` |

### Detection Mode
| Parameter | What it does | Default |
|-----------|-------------|---------|
| `--mode` | Detection method: `manual`, `blob`, `trackpy`, `stardist`, `cellpose` | `blob` |

### Measurement Settings
| Parameter | What it does | Default | When to adjust |
|-----------|-------------|---------|----------------|
| `--box_size` | Pixels around bead center for Z-profile averaging | 15 | Larger → more averaging (better SNR), but requires sparse beads |
| `--line_length` | Length of X/Y profile lines in µm | 5.0 | Should be ≥3× expected FWHM so tails reach background |
| `--fit_gaussian` | Enable 1D Gaussian fitting for sub-pixel FWHM precision | Off | Enable for publication-quality measurements |
| `--fit_3d` | Enable 3D Gaussian fitting (slower, more accurate) | Off | Use for asymmetric PSFs or when you need maximum accuracy |
| `--robust_fit` | Use soft-L1 loss to downweight outliers in fits | Off | Use when beads are clipped by mask edges or have nearby contamination |

### Background Correction
| Parameter | What it does | Default | When to use |
|-----------|-------------|---------|-------------|
| `--subtract_background` | Interactive ROI selection for global background subtraction | Off | Use for images with uniform elevated background |
| `--detrend` | Remove linear slope from profiles | Off | Use when profiles have tilted baselines |
| `--local_background` | Subtract median intensity from annular ring around each bead | Off | **Use for light-sheet** data with spatially varying out-of-focus haze |

### Quality Assurance
| Parameter | What it does | Default | When to adjust |
|-----------|-------------|---------|----------------|
| `--qa_min_snr` | Minimum acceptable signal-to-noise ratio | 3.0 | Increase for stricter filtering (e.g., 5.0) |
| `--qa_min_symmetry` | Minimum Z-profile symmetry (0-1, 1=perfect) | 0.6 | Lower if beads are asymmetric; raise for stricter QA |
| `--qa_auto_reject` | Automatically exclude beads failing QA thresholds | Off | Enable to auto-filter low-quality beads |
| `--reject_outliers` | MAD multiplier for outlier rejection (Cellpose mode) | None | Use 3.0 to remove statistical outliers from summary |
| `--max_z_fwhm` | Reject beads with Z-FWHM above this value (µm) | None | Filter out-of-focus or aggregated beads |

### Output Options
| Parameter | What it does | Default |
|-----------|-------------|---------|
| `--output_dir` | Directory for results | Same as input file |
| `--save_diagnostics` | Save per-bead diagnostic plots to `bead_diagnostics/` | Off |
| `--num_beads_avg` | Number of beads to average (0 = all, N = closest to median Z-FWHM) | 20 |
| `--no_plots` | Suppress all plots | Off |

### Detection-Specific Parameters

#### Blob Mode
| Parameter | What it does | Default |
|-----------|-------------|---------|
| `--blob_sigma` | Gaussian smoothing sigma (pixels) | 1.2 |
| `--blob_threshold_rel` | Detection threshold as fraction of intensity range | 0.2 |
| `--blob_min_distance` | Minimum spacing between beads (pixels) | 5 |

#### Trackpy Mode
| Parameter | What it does | Default |
|-----------|-------------|---------|
| `--trackpy_diameter` | Expected bead diameter in pixels (use odd values) | 5 |
| `--trackpy_minmass` | Minimum integrated brightness (lower = accept dimmer beads) | 5000 |
| `--trackpy_separation` | Minimum feature separation in pixels | diameter + 1 |

#### StarDist Mode
| Parameter | What it does | Default |
|-----------|-------------|---------|
| `--stardist_model` | Pretrained model name | `2D_versatile_fluo` |
| `--stardist_prob_thresh` | Probability threshold for detection | 0.6 |
| `--stardist_nms_thresh` | Non-maximum suppression threshold | 0.3 |
| `--use_blob_fallback` | Fall back to blob detector if StarDist finds 0 beads | Off |
| `--stardist_n_tiles` | Tile grid for large images (e.g., `4 4`) to prevent GPU OOM | None |

#### Cellpose Mode
| Parameter | What it does | Default |
|-----------|-------------|---------|
| `--cellpose_model` | Path to trained Cellpose model (or set `FWHM_CELLPOSE_MODEL` env var) | None |
| `--cellpose_diameter` | Expected bead diameter in µm (None = use model default) | None |
| `--cellpose_do_3d` | Use native Cellpose 3D segmentation on full stack | Off |
| `--anisotropy` | Z/XY spacing ratio for 3D Cellpose (e.g., 2.0/0.26 = 7.7) | None |
| `--cellpose_min_size` | Minimum mask area in pixels | 3 |
| `--cellpose_flow_threshold` | Flow error threshold (lower = stricter) | 0.4 |
| `--skip_cellpose_review` | Skip interactive mask review step | Off |

### Other Options
| Parameter | What it does |
|-----------|-------------|
| `--review_detection` | Review detected points before analysis (blob/trackpy/stardist) |
| `--points_file` | Load bead coordinates from CSV/TXT (overrides automatic detection) |
| `--channel` | Channel index for multi-channel stacks (0-based) |
| `--gamma` | Display gamma for visualization |
| `--na` | Numerical aperture (metadata only, recorded in summary) |
| `--fluorophore` | Fluorophore name (metadata only, recorded in summary) |

## Troubleshooting

### "No beads detected" or "Found 0 beads"

**Blob/StarDist mode:**
- Try `--use_blob_fallback` (StarDist only)
- Lower `--blob_threshold_rel` (try 0.1 or 0.05)
- Adjust `--blob_sigma` (try 0.8 for small beads, 2.0 for large beads)
- Check that your image is a 3D stack (not a single 2D plane)
- Verify `--scale_xy` and `--scale_z` are in µm/pixel (not nm or other units)

**Trackpy mode:**
- Lower `--trackpy_minmass` (try 1000 or 500 for dim beads)
- Adjust `--trackpy_diameter` to match bead size in pixels (must be odd)
- Check that beads span at least 3-5 pixels in XY

**Cellpose mode:**
- Lower `--cellpose_min_size` (try 1 or 2 for tiny beads)
- Increase `--cellpose_flow_threshold` (try 0.6 or 0.8 for less strict filtering)
- Check that model path is correct and model file exists
- Verify beads span ~15+ pixels (Cellpose works best on larger features)

### "FWHM values are too large" or "Z-FWHM is 10+ µm"

- **Check pixel scales**: Are `--scale_xy` and `--scale_z` correct? Common error: using nm instead of µm (divide by 1000)
- **Filter large values**: Use `--max_z_fwhm 5` to exclude out-of-focus or aggregated beads
- **Check focus**: Beads may be out of focus or clumped together
- **Try manual mode**: Right-click on a single well-focused bead to verify measurements

### "Gaussian fit failed" or "All FWHM values are None"

- Profiles may be too noisy or flat
- Try `--fit_gaussian` with `--robust_fit` to handle outliers
- Increase `--box_size` for better SNR in Z-profiles
- Check that `--prominence_rel 0.1` is appropriate (lower if profiles have low contrast)

### "Beads are rejected by QA filter"

- Lower `--qa_min_snr` (try 2.0 for dim beads)
- Lower `--qa_min_symmetry` (try 0.4 for asymmetric beads)
- Disable auto-rejection and inspect `*_bead_quality.csv` to see QA scores
- Use `--save_diagnostics` to visually inspect rejected beads in `bead_diagnostics/`

### "GUI doesn't show Cellpose model field"

- Set environment variable before launching GUI:
  - Windows: `set FWHM_CELLPOSE_MODEL=C:\path\to\model`
  - Unix/Mac: `export FWHM_CELLPOSE_MODEL=/path/to/model`
- Or use the "Browse" button in Cellpose options section

### "Trackpy/StarDist/Cellpose not available"

Install the optional dependency:
```bash
pip install trackpy              # for trackpy
pip install -e ".[stardist]"     # for StarDist
pip install -e ".[cellpose]"     # for Cellpose
```

### "Memory error" or "CUDA out of memory" (StarDist/Cellpose)

- **StarDist**: Use `--stardist_n_tiles 4 4` to process image in tiles
- **Cellpose**: Reduce image size or use CPU mode (GPU disabled automatically if CUDA unavailable)
- Close other applications to free RAM
- Consider using blob or trackpy mode for large images

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
