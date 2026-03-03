# Recommendation Memo: 500 nm Beads / 5x 0.14 NA / Light-Sheet

**Stack parameters:** 0.51 µm/px XY, 1.0 µm/px Z, 500 nm fluorescent beads

---

## 1. What You're Actually Measuring

At 5x/0.14 NA the diffraction-limited PSF is much larger than the bead:

| Quantity | Formula | Value |
|----------|---------|-------|
| Bead diameter | — | 0.5 µm |
| Lateral PSF FWHM | 0.51 × λ / NA | **1.82 µm** |
| Bead in XY pixels | 0.5 / 0.51 | 0.98 px (sub-pixel!) |
| PSF in XY pixels | 1.82 / 0.51 | **3.57 px** |
| Axial PSF FWHM (detection) | 2 × n × λ / NA² | ~51 µm (detection optics alone) |
| Axial resolution (practical) | Set by light-sheet thickness | Typically 3–10 µm |
| Z FWHM in pixels | 3–10 µm / 1.0 | **3–10 px** |

Because the bead is 3.6× smaller than the PSF, the measured FWHM equals the
system PSF directly — no deconvolution correction needed (the sub-resolution
regime where the bead acts as a delta function).

## 2. Sampling Concerns

### XY (3.57 px across the PSF)

This is right at the Nyquist limit. Consequences:

- **Sub-pixel centering matters.** Snapping to the nearest integer pixel shifts
  the center by up to 0.25 µm, which is ~14% of the expected FWHM. The
  parabolic refinement (now built-in) reduces this to <0.05 px.
- **Gaussian fitting has only ~7 data points across the peak.** The fit is
  feasible but not overdetermined — use `--robust_fit` to guard against single
  outlier pixels pulling the width.
- **The prominence method will be noisy.** With so few samples across the
  peak, interpolated half-max crossings jump by full pixels. Trust the Gaussian
  fit FWHM over the prominence FWHM for your lateral measurements.

### Z (3–10 px across the PSF)

Z sampling depends on the effective light-sheet thickness. At 1.0 µm/px with a
typical 3–5 µm sheet, you have 3–5 pixels across the axial FWHM — similar
sampling concerns as XY. The 3D Gaussian fit (`--fit_3d`) is the most accurate
option here because it uses all voxels simultaneously rather than independent
1D slices.

## 3. Background Strategy

Light-sheet microscopes produce spatially varying out-of-focus haze from the
illumination sheet tails. This means different beads sit on different local
background levels.

**Recommendation: Use `--local_background`.**

The global minimum subtraction (default) works if background is uniform. For
light-sheet data it is not. The annulus-based local background estimator
measures the median in a ring around each bead, giving each bead its own
baseline. The Gaussian model's `C` parameter also absorbs residual background,
so the combination of annulus + fit is robust.

## 4. Detection Strategy

### StarDist (recommended starting point)

`2D_versatile_fluo` was trained on fluorescence microscopy images of cells and
nuclei, not sub-resolution point sources. At 5x/0.14 NA your beads are dim
~4 px spots on the MIP, which is unlike the training data.

**Expect:** StarDist may detect most beads at default thresholds but might miss
dim ones or hallucinate detections on background fluctuations.

**Mitigation:** Use `--use_blob_fallback` so if StarDist finds zero beads, the
classical Gaussian-smooth + local-maximum detector takes over. Also use
`--review_detection` for your first runs to visually verify what was found.

### Cellpose with custom model

Training a custom Cellpose model on your specific bead images would give the
best detection accuracy, especially for dense fields or fields with varying
bead brightness. However:

- You need annotated training data (the `annotate_beads.py` script helps)
- For sparse, well-separated beads, the marginal improvement over
  StarDist + blob fallback + QA gating is small
- The measurement accuracy is dominated by the fitting step, not detection

**Verdict: Not worth training a custom model unless you have >100 beads per
FOV, beads that overlap, or StarDist+blob consistently misses >20% of your
beads.** The QA gating (`--qa_auto_reject`) catches the worst detection
errors anyway.

**Important Cellpose parameter:** If you do use Cellpose, note that the default
`min_size=15` (pixels area) will silently discard beads smaller than ~4.4 px
diameter. At 0.51 µm/px your 500 nm beads have ~0.98 px diameter and ~12 px²
PSF area, right at the boundary.  Use `--cellpose_min_size 3` (the new default
in this tool) to avoid losing beads.

### Manual

Always valid for small bead counts (<20). Use `--points_file` to save and
reload coordinates for reproducibility.

## 5. Fitting Strategy

| Option | Recommendation | Why |
|--------|---------------|-----|
| `--fit_gaussian` | **Yes, always** | The Gaussian fit gives sub-pixel FWHM precision; prominence is too coarse at 3.5 px sampling |
| `--fit_3d` | **Yes** | Uses all voxels instead of three 1D slices; more robust for your low-sample-count regime |
| `--robust_fit` | **Yes** | With only ~7 px across the peak, one bad pixel (hot pixel, cosmic ray, neighboring bead tail) can shift the fit by >10%. Soft-L1 loss limits the damage. |

## 6. QA Settings

| Parameter | Suggested Value | Rationale |
|-----------|----------------|-----------|
| `--qa_auto_reject` | On | Automatically drop bad beads |
| `--qa_min_snr` | 3.0 (default) | Good starting point; raise to 5 if you see noisy fits |
| `--qa_min_symmetry` | 0.5 (lower than default 0.6) | At 3.5 px sampling, even good beads will show mild asymmetry from pixelation |

## 7. Recommended Command

### First run (with diagnostics for verification)

```bash
bead-analyzer beads.tif --mode stardist --scale_xy 0.51 --scale_z 1.0 \
  --fit_gaussian --fit_3d --robust_fit --local_background \
  --use_blob_fallback --review_detection \
  --qa_auto_reject --qa_min_symmetry 0.5 \
  --save_diagnostics --na 0.14 --fluorophore "500nm bead" \
  --output_dir ./results_500nm
```

Check `results_500nm/bead_diagnostics/` — verify that:
1. XY profiles show a clean Gaussian with the fit overlaid
2. Z profiles reach baseline on both sides of the peak
3. The reported FWHM_XY is in the range 1.5–2.5 µm (consistent with 0.14 NA)
4. Rejected beads make sense (genuinely bad, not good beads thrown out)

### Production run (after verification)

```bash
bead-analyzer beads.tif --mode stardist --scale_xy 0.51 --scale_z 1.0 \
  --fit_gaussian --fit_3d --robust_fit --local_background \
  --use_blob_fallback \
  --qa_auto_reject --qa_min_symmetry 0.5 \
  --no_plots --na 0.14 --fluorophore "500nm bead" \
  --output_dir ./results_500nm
```

## 8. Expected Results

| Axis | Expected FWHM | Notes |
|------|---------------|-------|
| X | 1.5–2.2 µm | Lateral PSF; may differ from Y if alignment is imperfect |
| Y | 1.5–2.2 µm | Should be close to X |
| Z | 3–10 µm | Dominated by light-sheet thickness, not detection NA |

If measured FWHM_XY is significantly larger than ~2 µm, check:
- Is the bead in focus? (verify Z profile has a clear peak)
- Is the sample mounted flat? (tilted coverslip blurs MIP-based detection)
- Are beads aggregated? (two beads read as one wide "bead")

If measured FWHM_Z is >15 µm, the light-sheet may be poorly aligned or the
`--box_size` may be too large (averaging over adjacent beads).

## 9. Do You Need to Retrain StarDist/Cellpose?

**Short answer: No, not yet.**

The current pipeline with `StarDist + blob fallback + QA gating` is sufficient
for PSF characterization of well-separated beads. The measurement precision is
limited by the 3.5 px sampling, not by detection accuracy.

**Retrain only if:**
- You have very dense fields (>100 beads, many touching/overlapping)
- StarDist + blob consistently miss >20% of visible beads
- You want fully automated batch processing with zero manual review

In that case, use Cellpose (not StarDist retraining) because:
- Cellpose's training pipeline is simpler (just mask annotations)
- Cellpose handles variable-size objects better
- The `annotate_beads.py` script in this repo generates training data directly

## 10. Summary

| Decision | Choice | Confidence |
|----------|--------|------------|
| Detection method | StarDist + blob fallback | High — simple, sufficient for sparse beads |
| Retrain ML model? | No | High — not the bottleneck |
| Background subtraction | `--local_background` | High — essential for light-sheet |
| Fitting | 1D + 3D Gaussian, robust | High — maximizes precision at coarse sampling |
| QA gating | On, symmetry threshold 0.5 | Medium — may need tuning after first run |
| Diagnostics | On for first run | High — verify before trusting |
