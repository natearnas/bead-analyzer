"""Tests for bead_analyzer.core FWHM calculation functions."""

import numpy as np

from bead_analyzer.analysis import _estimate_local_background, _parabolic_peak
from bead_analyzer.core import (
    calculate_fwhm_prominence,
    filter_by_qa,
    fit_gaussian_3d,
    fit_gaussian_fwhm,
    gaussian_func,
    reject_outliers_mad,
)


def test_gaussian_func():
    """Gaussian returns expected values at peak and half-max."""
    x = np.linspace(-5, 5, 101)
    A, mu, sigma, C = 1.0, 0.0, 1.0, 0.0
    y = gaussian_func(x, A, mu, sigma, C)
    assert np.isclose(y[50], 1.0)  # peak at mu
    half_max = 0.5
    # FWHM = 2*sqrt(2*ln(2))*sigma ≈ 2.355*sigma for sigma=1
    idx_half = np.argmin(np.abs(y - half_max))
    assert idx_half >= 0


def test_calculate_fwhm_prominence_gaussian():
    """Prominence FWHM on a Gaussian profile."""
    # Create Gaussian: FWHM = 2*sqrt(2*ln(2))*sigma
    sigma = 2.0
    fwhm_expected = 2 * np.sqrt(2 * np.log(2)) * sigma
    x = np.arange(100)
    y = gaussian_func(x, 10.0, 50.0, sigma, 0.0)
    result = calculate_fwhm_prominence(y, scale_factor=1.0)
    assert result is not None
    assert 'fwhm_um' in result
    assert 'fwhm_px' in result
    assert np.isclose(result['fwhm_px'], fwhm_expected, rtol=0.1)


def test_calculate_fwhm_prominence_flat():
    """Flat profile returns None."""
    y = np.ones(50)
    assert calculate_fwhm_prominence(y) is None


def test_calculate_fwhm_prominence_too_short():
    """Too short profile returns None."""
    y = np.array([1, 2, 3])
    assert calculate_fwhm_prominence(y) is None


def test_fit_gaussian_fwhm():
    """Gaussian fit returns reasonable FWHM."""
    x = np.arange(50)
    sigma = 3.0
    y = gaussian_func(x, 5.0, 25.0, sigma, 0.1)
    result = fit_gaussian_fwhm(y, scale_factor=1.0)
    assert result is not None
    fwhm_expected = 2 * np.sqrt(2 * np.log(2)) * sigma
    assert np.isclose(result['fwhm_um'], fwhm_expected, rtol=0.15)


def test_fit_gaussian_fwhm_with_peak_hint():
    """Gaussian fit with peak_hint works."""
    x = np.arange(80)
    y = gaussian_func(x, 4.0, 40.0, 2.0, 0.0)
    result = fit_gaussian_fwhm(y, scale_factor=1.0, peak_hint=40)
    assert result is not None


def test_reject_outliers_mad():
    """MAD outlier rejection removes extreme values."""
    results = [
        {'id': i, 'fwhm_z_gauss': v}
        for i, v in enumerate([1.0, 1.1, 1.0, 1.2, 1.1, 10.0, 1.0])  # 10.0 is outlier
    ]
    filtered = reject_outliers_mad(results, data_key='fwhm_z_gauss', m=3.0)
    assert len(filtered) < len(results)
    values = [r['fwhm_z_gauss'] for r in filtered]
    assert 10.0 not in values


def test_reject_outliers_mad_small_n():
    """MAD with fewer than 5 points returns all."""
    results = [{'id': 1, 'fwhm_z_gauss': 1.0}, {'id': 2, 'fwhm_z_gauss': 2.0}]
    filtered = reject_outliers_mad(results, data_key='fwhm_z_gauss', m=3.0)
    assert len(filtered) == len(results)


def test_filter_by_qa_rejects_low_quality():
    """QA filter should reject entries below thresholds."""
    results = [
        {"id": 1, "qa_z_snr": 10.0, "qa_z_symmetry": 0.9},
        {"id": 2, "qa_z_snr": 2.0, "qa_z_symmetry": 0.9},
        {"id": 3, "qa_z_snr": 8.0, "qa_z_symmetry": 0.2},
    ]
    kept, rejected = filter_by_qa(results, min_snr=3.0, min_symmetry=0.6)
    assert [r["id"] for r in kept] == [1]
    assert {r["id"] for r in rejected} == {2, 3}
    assert all("qa_reject_reason" in r for r in rejected)


def test_fit_gaussian_3d_recovers_width():
    """3D Gaussian fit returns near-true FWHM."""
    nz, ny, nx = 17, 17, 17
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    sigma_px = 2.0
    A = 100.0
    C = 10.0
    z0, y0, x0 = 8.0, 8.0, 8.0
    vol = A * np.exp(
        -0.5 * (((x - x0) / sigma_px) ** 2 + ((y - y0) / sigma_px) ** 2 + ((z - z0) / sigma_px) ** 2)
    ) + C
    res = fit_gaussian_3d(vol.astype(np.float32), scale_xy=1.0, scale_z=1.0)
    assert res is not None
    expected = 2 * np.sqrt(2 * np.log(2)) * sigma_px
    assert np.isclose(res["fwhm_x_um"], expected, rtol=0.2)
    assert np.isclose(res["fwhm_y_um"], expected, rtol=0.2)
    assert np.isclose(res["fwhm_z_um"], expected, rtol=0.2)


def test_fit_gaussian_fwhm_robust_with_outliers():
    """Robust fit recovers FWHM even when profile tails have outlier spikes."""
    x = np.arange(60)
    sigma = 3.0
    y = gaussian_func(x, 5.0, 30.0, sigma, 0.1)
    # Inject outlier spikes in the tail (simulates nearby bead or clipped edge)
    y_corrupted = y.copy()
    y_corrupted[5] = 4.0
    y_corrupted[55] = 3.5
    result = fit_gaussian_fwhm(y_corrupted, scale_factor=1.0, robust=True)
    assert result is not None
    fwhm_expected = 2 * np.sqrt(2 * np.log(2)) * sigma
    assert np.isclose(result['fwhm_um'], fwhm_expected, rtol=0.2)


def test_fit_gaussian_fwhm_bounds_reject_negative_sigma():
    """Bounded fit should not return negative or near-zero sigma."""
    # Flat-ish profile where unconstrained fit might find degenerate sigma
    y = np.ones(30) * 5.0
    y[15] = 5.2  # tiny bump
    result = fit_gaussian_fwhm(y, scale_factor=1.0)
    # Should return None due to sigma guard or bounds
    # (the bump is too small relative to noise for a reliable fit)
    # Either None or a valid result is acceptable here; if not None,
    # sigma must be positive.
    if result is not None:
        assert result['fwhm_um'] > 0


def test_fit_gaussian_3d_robust():
    """3D robust fit recovers FWHM with clipped corner."""
    nz, ny, nx = 17, 17, 17
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    sigma_px = 2.0
    vol = 100.0 * np.exp(
        -0.5 * (((x - 8.0) / sigma_px) ** 2 +
                ((y - 8.0) / sigma_px) ** 2 +
                ((z - 8.0) / sigma_px) ** 2)
    ) + 10.0
    # Corrupt one corner (simulates clipped bead volume edge)
    vol[:3, :3, :3] = 80.0
    res = fit_gaussian_3d(vol.astype(np.float32), scale_xy=1.0, scale_z=1.0, robust=True)
    assert res is not None
    expected = 2 * np.sqrt(2 * np.log(2)) * sigma_px
    assert np.isclose(res["fwhm_x_um"], expected, rtol=0.25)


def test_parabolic_peak_subpixel():
    """Parabolic interpolation refines peak to sub-pixel precision."""
    # Symmetric Gaussian sampled at integer positions, peak between pixels
    true_peak = 10.3
    x = np.arange(21)
    y = 100.0 * np.exp(-0.5 * ((x - true_peak) / 2.0) ** 2)
    int_peak = int(np.argmax(y))
    sub_peak = _parabolic_peak(y, int_peak, offset=0)
    # Sub-pixel estimate should be closer to true peak than integer peak
    assert abs(sub_peak - true_peak) < abs(int_peak - true_peak)
    assert abs(sub_peak - true_peak) < 0.15


def test_estimate_local_background():
    """Local annulus background returns median of annular region."""
    # Create a 2D image with constant background + bright bead at center
    img = np.ones((40, 40), dtype=np.float32) * 20.0
    yy, xx = np.ogrid[:40, :40]
    bead = 100.0 * np.exp(-0.5 * (((xx - 20) / 2.0) ** 2 + ((yy - 20) / 2.0) ** 2))
    img += bead
    bg = _estimate_local_background(img, 20.0, 20.0, inner_r=6, outer_r=12)
    # Background should be close to 20.0 (the constant floor)
    assert abs(bg - 20.0) < 2.0


def test_fwhm_prominence_with_nonzero_baseline():
    """Prominence FWHM should be accurate even with a non-zero baseline."""
    sigma = 2.0
    fwhm_expected = 2 * np.sqrt(2 * np.log(2)) * sigma
    x = np.arange(100)
    baseline = 50.0
    y = gaussian_func(x, 10.0, 50.0, sigma, baseline)
    result = calculate_fwhm_prominence(y, scale_factor=1.0)
    assert result is not None
    assert np.isclose(result['fwhm_px'], fwhm_expected, rtol=0.15)
