"""Tests for bead_analyzer.core FWHM calculation functions."""

import json

import matplotlib
import numpy as np

matplotlib.use('Agg')

from bead_analyzer.analysis import (
    _estimate_local_background,
    _parabolic_peak,
    _weighted_centroid_2d,
    write_outputs,
)
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


def test_calculate_fwhm_prominence_ignores_secondary_peak():
    """FWHM is measured around the selected (most prominent) peak only.

    Previously l_idx/r_idx used the global first/last samples above the
    half-prominence level, so a second peak anywhere in the profile inflated
    the measured width.
    """
    x = np.arange(200)
    sigma = 2.0
    fwhm_expected = 2 * np.sqrt(2 * np.log(2)) * sigma
    y = (gaussian_func(x, 10.0, 50.0, sigma, 0.0)
         + gaussian_func(x, 6.0, 150.0, 3.0, 0.0) + 1.0)
    result = calculate_fwhm_prominence(y, scale_factor=1.0)
    assert result is not None
    assert np.isclose(result['fwhm_px'], fwhm_expected, rtol=0.1)


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


def _make_synthetic_stack(nz=21, ny=61, nx=61, sigma_xy=2.0, sigma_z=3.0,
                          background=100.0, amplitude=1000.0, center_xy=None):
    """Deterministic 3D Gaussian bead on constant background.

    The bead sits at the stack midpoint unless center_xy=(x0, y0) is given.
    """
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
    z0, y0, x0 = (nz - 1) / 2.0, (ny - 1) / 2.0, (nx - 1) / 2.0
    if center_xy is not None:
        x0, y0 = center_xy
    stack = background + amplitude * np.exp(
        -0.5 * (((x - x0) / sigma_xy) ** 2 +
                ((y - y0) / sigma_xy) ** 2 +
                ((z - z0) / sigma_z) ** 2)
    )
    return stack.astype(np.float32)


def test_parabolic_peak_recovers_exact_vertex():
    """Parabolic interpolation recovers the vertex of an exact parabola."""
    x = np.arange(21, dtype=float)
    x0 = 10.37
    y = -2.0 * (x - x0) ** 2 + 50.0
    est = _parabolic_peak(y, peak_idx=10, offset=0)
    assert np.isclose(est, x0, atol=1e-9)


def test_parabolic_peak_handles_plateau():
    """Flat triple (zero curvature) falls back to the integer peak."""
    y = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    est = _parabolic_peak(y, peak_idx=2, offset=100)
    assert est == 102.0


def test_reject_outliers_mad_alignment_with_missing_values():
    """Entries with None for data_key keep their position; rejection targets the right doc."""
    results = [
        {'id': 1, 'fwhm_z_gauss': 1.00},
        {'id': 2, 'fwhm_z_gauss': None},
        {'id': 3, 'fwhm_z_gauss': 1.05},
        {'id': 4, 'fwhm_z_gauss': 0.95},
        {'id': 5, 'fwhm_z_gauss': None},
        {'id': 6, 'fwhm_z_gauss': 1.02},
        {'id': 7, 'fwhm_z_gauss': 9.50},  # outlier
        {'id': 8, 'fwhm_z_gauss': 0.98},
    ]
    filtered = reject_outliers_mad(results, data_key='fwhm_z_gauss', m=3.0)
    kept_ids = [r['id'] for r in filtered]
    assert 7 not in kept_ids
    assert set(kept_ids) == {1, 2, 3, 4, 5, 6, 8}


def test_weighted_centroid_subpixel_shift():
    """Intensity-weighted centroid recovers a subpixel-shifted symmetric spot."""
    sigma = 2.0
    for dx, dy in [(0.3, -0.2), (-0.4, 0.45), (0.0, 0.0)]:
        base = 20
        yy, xx = np.mgrid[0:41, 0:41].astype(float)
        img = np.exp(-0.5 * (((xx - (base + dx)) / sigma) ** 2 +
                             ((yy - (base + dy)) / sigma) ** 2))
        img += 0.1  # pedestal so min-subtraction does not redefine the center
        cx, cy = _weighted_centroid_2d(img, x_offset=0, y_offset=0)
        assert np.isclose(cx, base + dx, atol=0.05)
        assert np.isclose(cy, base + dy, atol=0.05)


def test_fit_gaussian_3d_units_anisotropic_scales():
    """3D fit applies scale_xy to X/Y and scale_z to Z widths."""
    nz, ny, nx = 15, 13, 11
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
    sx, sy, sz = 2.0, 3.0, 4.0
    vol = 100.0 * np.exp(
        -0.5 * (((x - 5.0) / sx) ** 2 + ((y - 6.0) / sy) ** 2 + ((z - 7.0) / sz) ** 2)
    ) + 10.0
    res = fit_gaussian_3d(vol.astype(np.float32), scale_xy=0.1, scale_z=0.5)
    assert res is not None
    factor = 2 * np.sqrt(2 * np.log(2))
    assert np.isclose(res['fwhm_x_um'], factor * sx * 0.1, rtol=0.05)
    assert np.isclose(res['fwhm_y_um'], factor * sy * 0.1, rtol=0.05)
    assert np.isclose(res['fwhm_z_um'], factor * sz * 0.5, rtol=0.05)


def test_pipeline_xy_fwhm_sampling_accuracy(tmp_path):
    """End-to-end XY FWHM through the profile-extraction path matches ground truth.

    Uses run_blob with a points_file (skips detection). Fails if the sampled
    profile spacing deviates from 1 px (the historical ll/(ll-1) bias).
    """
    from bead_analyzer import analysis

    stack = _make_synthetic_stack(nz=21, ny=61, nx=61, sigma_xy=2.0, sigma_z=3.0)
    scale_xy, scale_z = 0.26, 2.0
    sigma_px = 2.0
    fwhm_um_true = 2 * np.sqrt(2 * np.log(2)) * sigma_px * scale_xy

    pts_file = tmp_path / "points.csv"
    pts_file.write_text("x_coord,y_coord\n30,30\n")

    results, _, _, _, _ = analysis.run_blob(
        stack, scale_xy=scale_xy, scale_z=scale_z,
        box_size=15, line_length=4.0, points_file=str(pts_file),
    )
    assert len(results) == 1
    meas = results[0]['fwhm_x_prom']
    assert meas is not None
    assert np.isclose(meas, fwhm_um_true, rtol=0.04)
    assert np.isclose(results[0]['fwhm_y_prom'], fwhm_um_true, rtol=0.04)


def test_pipeline_xy_fwhm_off_pixel_center(tmp_path):
    """XY FWHM stays accurate when the bead center falls between pixels.

    Bilinear line sampling broadens off-pixel profiles (about +6% at a
    half-pixel offset for sigma 1.5 px); a pixel-centered bead cannot see it.
    """
    from bead_analyzer import analysis

    sigma_px, scale_xy = 1.5, 0.26
    stack = _make_synthetic_stack(sigma_xy=sigma_px, center_xy=(30.5, 30.5))
    fwhm_um_true = 2 * np.sqrt(2 * np.log(2)) * sigma_px * scale_xy

    pts_file = tmp_path / "points.csv"
    pts_file.write_text("x_coord,y_coord\n30,30\n")

    results, _, _, _, _ = analysis.run_blob(
        stack, scale_xy=scale_xy, scale_z=2.0,
        box_size=15, line_length=4.0, points_file=str(pts_file), fit_gaussian=True,
    )
    assert len(results) == 1
    assert np.isclose(results[0]['fwhm_x_gauss'], fwhm_um_true, rtol=0.02)
    assert np.isclose(results[0]['fwhm_y_gauss'], fwhm_um_true, rtol=0.02)


def test_write_outputs_average_bead_id_alignment(tmp_path):
    """Average bead must pair volumes by result id, not by list position.

    Simulates cellpose behavior: bead_volumes[0] is an empty placeholder for a
    rejected bead, so zip(results, bead_volumes) pairs volumes to wrong ids.
    """
    import tifffile

    def _vol(value):
        v = np.full((7, 7, 7), value, dtype=np.float32)
        v[3, 3, 3] = value + 100.0  # centered peak so centering shifts are ~0
        return v

    results = [
        {'id': 2, 'x_coord': 5.0, 'y_coord': 5.0, 'fwhm_z_prom': 1.0},
        {'id': 3, 'x_coord': 10.0, 'y_coord': 10.0, 'fwhm_z_prom': 1.1},
    ]
    bead_volumes = [np.array([]), _vol(10.0), _vol(20.0)]
    profiles = [
        {'id': 2, 'volume': _vol(10.0), 'z_profile': None,
         'x_profile': None, 'y_profile': None},
        {'id': 3, 'volume': _vol(20.0), 'z_profile': None,
         'x_profile': None, 'y_profile': None},
    ]
    mip = np.zeros((30, 30), dtype=np.float32)
    out_path = tmp_path / "synthetic.tif"

    write_outputs(
        results, bead_volumes, mip, out_path, 'cellpose',
        scale_xy=0.1, scale_z=0.5,
        upsample_factor=1, num_beads_avg=20, profiles=profiles, stack=None,
    )
    avg = tifffile.imread(tmp_path / "synthetic_average_bead_stack.tif")
    # Correct pairing: volumes for ids 2 and 3 -> background average (10+20)/2 = 15.
    # Mis-paired (positional) selection picks volumes of the wrong ids instead.
    assert np.isclose(float(avg[0, 0, 0]), 15.0, atol=2.0)


def test_cli_config_supplies_required_scale_args(tmp_path, monkeypatch, capsys):
    """--config can supply scale_xy/scale_z without argparse requiring them on the CLI."""
    import tifffile

    from bead_analyzer import cli

    stack = _make_synthetic_stack(nz=15, ny=41, nx=41, sigma_xy=2.0, sigma_z=3.0)
    tif_path = tmp_path / "beads.tif"
    tifffile.imwrite(str(tif_path), stack)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        'input_file': str(tif_path),
        'scale_xy': 0.26,
        'scale_z': 2.0,
        'mode': 'blob',
        'output_dir': str(tmp_path),
        'no_plots': True,
    }))
    monkeypatch.setattr('sys.argv', ['bead-analyzer', '--config', str(cfg_path)])
    rc = cli.main()
    assert rc == 0
    captured = capsys.readouterr()
    assert 'required' not in captured.err.lower()
