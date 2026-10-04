# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.2.3] - 2026-10-04

### Changed (measurement-affecting)
- **X/Y FWHM values change.** Lateral line profiles were sampled at `N/(N-1)` px spacing but treated as 1 px, where `N` is the number of line samples (`line_length / scale_xy`, made odd). They are now sampled at exactly 1 px, with cubic-spline instead of bilinear interpolation. X/Y FWHM (prominence and 1D Gaussian) rises by about 2-5% at `N = 19` (0.26 um/px with the default 5 um line; spots 3.5-7 px wide) and about 7-10% at `N = 9` (0.51 um/px; spots 3.5-5 px wide). Z and 3D-fit values are not affected. Re-analyze rather than rescale older results, because the change depends on spot size.
- 1D Gaussian fits widen their window for broad peaks. With the fixed 20-sample window, peaks wider than about 16 samples read low (16% at 20 samples per FWHM, 29% at 24); narrower peaks are unchanged.
- The Z-profile SNR excludes a width-scaled zone around the peak when estimating baseline noise. Broad peaks were under-reported (a true SNR of 20 read below the 3.0 gate at 16 samples per FWHM in a 41-slice stack) and could be rejected by `--qa_auto_reject`.

### Fixed
- `--robust_fit` now scales the soft-L1 loss to the peak amplitude. On raw-count data the 1D fit returned no result for about half of the peaks 2 samples wide, and the 3D fit scattered about ten times more than the plain fit.
- `--config` can supply `scale_xy` and `scale_z`; previously the CLI exited because both were required on the command line.
- The average bead is built from the volumes of the selected beads by id (in Cellpose mode rejected beads shifted the pairing) and is centered on the central voxel instead of half a voxel off.
- Cellpose mode: beads at the exact image edge no longer break the 3x3 heatmap assignment.
- The 1D Gaussian fit no longer fails when `--fit_window` is more than four times the profile length.
- GUI: the average-bead field is labeled as a bead count, not a percentage.

### Documentation
- The README now states that the Z profile is a box average, what its FWHM means for light-sheet and widefield stacks, and when to use `--fit_3d` instead.

## [1.2.2] - 2026-06-12

### Fixed
- Prominence-based FWHM now measures only the contiguous region around the selected peak, preventing overestimation when a neighboring bead/peak lies within the same profile
- SNR quality metric no longer returns astronomically large values on flat/synthetic baselines (noise floor is now tied to signal amplitude rather than a fixed 1e-9)
- Default peak prominence is now derived from the profile's dynamic range instead of a fixed absolute value, making detection robust for unscaled (e.g. raw 16-bit) profiles

### Added
- Direct unit tests for the `analysis.py` math helpers: `_quality_metrics` (SNR), `_weighted_centroid_2d`, `_radial_center_2d`, and `_edge_midpoint_z`
- Regression test for prominence FWHM in the presence of a neighboring peak

## [1.2.1] - 2026-05-04

### Added
- `.zenodo.json` describing the deposit metadata (title, description, MIT license, author, keywords) for the Zenodo GitHub-release archival workflow
- DOI badge placeholder, Citation section, and Code Availability statement in the README
- CITATION.cff bumped to 1.2.1 in preparation for the first DOI-minting release

## [1.2.0] - 2026-04-03

### Added
- New `edge` center refinement mode in CLI and GUI for hollow/ring-like beads, with edge-weighted Z-plane selection
- Method-center diagnostic overlays (large crosshairs in XY/XZ/YZ and method-center Z marker) while preserving legacy reference lines
- Figure legend in per-bead diagnostics clarifying existing lines versus method-based center markers

### Changed
- Updated center-mode documentation across README, CLI docs, GUI docs, and in-app GUI docs guidance for annular beads
- Added Ruff linting configuration and dev dependency, and cleaned baseline lint issues in the repository

## [1.1.0] - 2026-04-01

### Added
- Configurable bead center refinement modes (`peak`, `centroid`, `radial`) across CLI and GUI
- Human-readable `*_run_settings.txt` output file for reproducible run settings
- Advanced GUI Trackpy options (`Diameter`, `Minmass`, `Separation`)
- New GUI use cases and docs guidance for small, large filled, and large hollow beads

### Changed
- Single-bead summary spread now reports `N/A` instead of `nan` when `n=1`
- Improved advanced GUI layout and docs-panel placement behavior on wider windows

## [1.0.0] - 2026-03-03

### Added
- Five detection modes: Manual, Blob, Trackpy, StarDist, Cellpose
- 1D and 3D Gaussian fitting with robust (soft-L1) option
- Quality assurance metrics (SNR, symmetry) with auto-reject
- Per-bead diagnostic plots
- Average bead stack and publication-quality summary figures
- Spatial FWHM heatmap
- CLI and GUI interfaces
- Cellpose custom model training pipeline
- Local annulus background subtraction for light-sheet data
- Sub-pixel center refinement via parabolic interpolation
