# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
