"""Tests for bead_analyzer.analysis math helpers."""

import math

import numpy as np

from bead_analyzer.analysis import (
    _edge_midpoint_z,
    _quality_metrics,
    _radial_center_2d,
    _weighted_centroid_2d,
)
from bead_analyzer.core import calculate_fwhm_prominence, gaussian_func

# ---------------------------------------------------------------------------
# _quality_metrics
# ---------------------------------------------------------------------------

def _symmetric_gaussian_profile(n=51, peak_idx=25, sigma=5.0, amp=100.0, baseline=10.0):
    """Helper: symmetric Gaussian Z-profile on a low-noise baseline."""
    x = np.arange(n, dtype=np.float64)
    return amp * np.exp(-0.5 * ((x - peak_idx) / sigma) ** 2) + baseline


def test_quality_metrics_high_snr_symmetric():
    """Clean symmetric Gaussian yields high SNR and symmetry close to 1."""
    profile = _symmetric_gaussian_profile()
    snr, symmetry, qa_flag = _quality_metrics(profile, peak_idx=25)
    assert snr is not None and math.isfinite(snr)
    assert snr > 10.0, f"Expected high SNR, got {snr}"
    assert symmetry is not None
    assert symmetry > 0.9, f"Expected symmetry close to 1, got {symmetry}"


def test_quality_metrics_asymmetric_lower_symmetry():
    """Asymmetric profile (one side scaled) yields clearly lower symmetry."""
    x = np.arange(51, dtype=np.float64)
    peak_idx = 25
    sigma = 5.0
    # Left lobe at normal amplitude; right lobe at half amplitude
    left = 100.0 * np.exp(-0.5 * ((x[:peak_idx] - peak_idx) / sigma) ** 2) + 10.0
    right = 50.0 * np.exp(-0.5 * ((x[peak_idx + 1:] - peak_idx) / sigma) ** 2) + 10.0
    profile = np.concatenate([left, [110.0], right])
    _, symmetry_asym, _ = _quality_metrics(profile, peak_idx=peak_idx)

    sym_profile = _symmetric_gaussian_profile()
    _, symmetry_sym, _ = _quality_metrics(sym_profile, peak_idx=25)

    assert symmetry_asym is not None
    assert symmetry_sym is not None
    assert symmetry_asym < symmetry_sym, (
        f"Asymmetric symmetry ({symmetry_asym:.3f}) should be below symmetric ({symmetry_sym:.3f})"
    )


def test_quality_metrics_snr_finite_flat_baseline():
    """Flat baseline (MAD=0) must not produce an absurdly large SNR (regression for noise-floor fix)."""
    # Constant baseline with a single raised peak
    profile = np.full(30, 5.0)
    profile[15] = 105.0  # peak = 105, baseline = 5, signal_amp = 100
    snr, _, _ = _quality_metrics(profile, peak_idx=15)
    assert snr is not None and math.isfinite(snr)
    # With noise = max(0, 100*1e-3, 1e-9) = 0.1, SNR = 100 / 0.1 = 1000
    # Still finite and not > 1e6
    assert snr < 1e6, f"SNR should be finite and bounded, got {snr}"


def test_quality_metrics_too_short_returns_none():
    """Profile shorter than 5 returns (None, None, None)."""
    assert _quality_metrics(np.array([1, 2, 3]), peak_idx=1) == (None, None, None)


# ---------------------------------------------------------------------------
# _weighted_centroid_2d
# ---------------------------------------------------------------------------

def _gaussian_blob(cx, cy, sigma=2.0, size=21):
    """Create a 2D Gaussian blob centered at (cx, cy) within a size×size patch."""
    yy, xx = np.indices((size, size), dtype=np.float64)
    return np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / sigma ** 2))


def test_weighted_centroid_2d_known_location():
    """Centroid of a Gaussian blob returns location close to its center."""
    size = 21
    true_cx_local, true_cy_local = 10.5, 10.5  # sub-pixel center in local frame
    x_offset, y_offset = 5.0, 7.0
    img = _gaussian_blob(true_cx_local, true_cy_local, size=size)
    cx, cy = _weighted_centroid_2d(img, x_offset, y_offset)
    assert abs(cx - (true_cx_local + x_offset)) < 0.5, f"cx={cx}, expected~{true_cx_local + x_offset}"
    assert abs(cy - (true_cy_local + y_offset)) < 0.5, f"cy={cy}, expected~{true_cy_local + y_offset}"


def test_weighted_centroid_2d_zero_image_fallback():
    """All-zero image falls back to geometric center offset by x_offset/y_offset."""
    img = np.zeros((10, 10), dtype=np.float32)
    cx, cy = _weighted_centroid_2d(img, x_offset=3.0, y_offset=4.0)
    assert abs(cx - (3.0 + 4.5)) < 0.1
    assert abs(cy - (4.0 + 4.5)) < 0.1


# ---------------------------------------------------------------------------
# _radial_center_2d
# ---------------------------------------------------------------------------

def test_radial_center_2d_hollow_ring():
    """Hollow ring returns a center close to the ring's geometric center."""
    size = 41
    cx_true, cy_true = 20.0, 20.0
    radius = 8.0
    yy, xx = np.indices((size, size), dtype=np.float64)
    dist = np.sqrt((xx - cx_true) ** 2 + (yy - cy_true) ** 2)
    # Thin bright ring
    img = np.exp(-0.5 * ((dist - radius) / 1.5) ** 2).astype(np.float32)
    cx, cy = _radial_center_2d(img, x_offset=0.0, y_offset=0.0)
    assert abs(cx - cx_true) < 1.5, f"cx={cx}, expected~{cx_true}"
    assert abs(cy - cy_true) < 1.5, f"cy={cy}, expected~{cy_true}"


# ---------------------------------------------------------------------------
# _edge_midpoint_z
# ---------------------------------------------------------------------------

def test_edge_midpoint_z_symmetric_edges():
    """Profile with two symmetric step edges returns Z near the midpoint."""
    n = 40
    profile = np.zeros(n, dtype=np.float32)
    # Rising edge at index ~8, falling edge at index ~31 → midpoint ≈ 19.5
    profile[8:32] = 1.0
    # Smooth transitions
    profile[6] = 0.1
    profile[7] = 0.6
    profile[32] = 0.6
    profile[33] = 0.1
    midpoint_expected = (8 + 31) / 2.0  # ≈ 19.5
    result = _edge_midpoint_z(profile, default_z=20)
    assert abs(result - midpoint_expected) < 3, f"result={result}, expected~{midpoint_expected}"


def test_edge_midpoint_z_too_short_returns_default():
    """Profile shorter than 7 returns the default_z."""
    profile = np.array([0, 1, 1, 1, 0], dtype=np.float32)
    assert _edge_midpoint_z(profile, default_z=2) == 2


# ---------------------------------------------------------------------------
# Prominence FWHM — neighbor regression (Fix #2)
# ---------------------------------------------------------------------------

def test_fwhm_prominence_not_inflated_by_neighbor():
    """FWHM of primary peak is not inflated by a secondary neighboring peak."""
    sigma = 4.0
    fwhm_expected = 2 * np.sqrt(2 * np.log(2)) * sigma
    x = np.arange(200)
    # Primary peak at x=70, secondary (smaller but still above primary half-max) at x=130
    primary = gaussian_func(x, 10.0, 70.0, sigma, 0.0)
    secondary = gaussian_func(x, 7.0, 130.0, sigma, 0.0)  # amplitude 7 > half-max of primary (5)
    profile = primary + secondary
    result = calculate_fwhm_prominence(profile, scale_factor=1.0)
    assert result is not None, "Expected a valid FWHM result"
    # The FWHM should match the primary peak's true FWHM within 10%
    assert np.isclose(result['fwhm_px'], fwhm_expected, rtol=0.10), (
        f"fwhm_px={result['fwhm_px']:.2f}, expected~{fwhm_expected:.2f} — "
        "neighbor may be inflating the FWHM"
    )
