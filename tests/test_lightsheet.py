"""Tests for the light-sheet report, the synthetic bead generator and the PSF export.

Every test measures a simulated stack whose sheet thickness is known.
"""

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from bead_analyzer import analysis, lightsheet, optics, psf_export
from bead_analyzer.core import fit_gaussian_fwhm
from bead_analyzer.synthetic import make_bead_field

# Low-NA, large-pixel geometry (cleared-tissue style): cheap to simulate.
LOW_NA = dict(na=0.1, scale_xy=2.0, emission_nm=525.0, ri=1.0, sheet_fwhm_um=7.0)


def _measure(tmp_path, sim, box_size=9, **report_args):
    """Simulate a bead field, run the bead analysis at the true positions, build the report."""
    stack, truth = make_bead_field(**sim)
    points = tmp_path / "points.csv"
    points.write_text("x_coord,y_coord\n" + "".join(
        f"{int(round(x))},{int(round(y))}\n" for x, y in zip(truth['x'], truth['y'])))
    results = analysis.run_blob(
        stack, scale_xy=truth['scale_xy'], scale_z=truth['scale_z'], box_size=box_size,
        line_length=10 * truth['scale_xy'], fit_gaussian=True, points_file=str(points),
    )[0]
    args = dict(na=truth['na'], emission_nm=truth['emission_nm'], ri=truth['ri'], n_boot=100)
    args.update(report_args)
    report = lightsheet.analyze(stack, results, truth['scale_xy'], truth['scale_z'], box_size=box_size, **args)
    return stack, truth, results, report


def test_synthetic_box_sum_follows_the_sheet():
    """The bead signal summed over a box that holds the defocus blur has the sheet's FWHM."""
    stack, truth = make_bead_field(shape=(41, 120, 120), n_beads=8, scale_z=2.0, photons=4e4,
                                   brightness_cv=0.0, background=0.0, read_noise=0.0, offset=0.0, seed=1, **LOW_NA)
    half = truth['patch_px']
    ratios, energy = [], []
    for x, y, z in zip(truth['x'], truth['y'], truth['z']):
        xi, yi = int(round(x)), int(round(y))
        profile = stack[:, yi - half:yi + half + 1, xi - half:xi + half + 1].sum(axis=(1, 2), dtype=np.float64)
        fit = fit_gaussian_fwhm(profile, truth['scale_z'], peak_hint=int(round(z)))
        ratios.append(fit['fwhm_um'] / 7.0)
        energy.append(fit['amplitude'] / 4e4)
    assert np.allclose(ratios, 1.0, atol=0.015)
    # A bead centered in the sheet delivers its photons to the box, apart from the
    # few percent in the outer diffraction rings.
    assert np.all((np.array(energy) > 0.95) & (np.array(energy) < 1.01))


def test_sheet_thickness_from_single_beads(tmp_path):
    """Fine Z sampling: the median of single-bead fits recovers a uniform sheet."""
    _, _, _, report = _measure(tmp_path, dict(shape=(41, 260, 260), n_beads=80, scale_z=2.0, photons=3000.0,
                                              background=20.0, seed=2, **LOW_NA))
    s = report['summary']
    assert s['sampling'] == 'ok' and s['method'] == 'per_bead_median'
    assert np.isclose(s['sheet_fwhm_um'], 7.0, rtol=0.02)
    assert s['sheet_fwhm_ci_um'][0] <= 7.0 <= s['sheet_fwhm_ci_um'][1]
    assert np.isclose(s['sheet_width_1e2_um'], s['sheet_fwhm_um'] * optics.E2_WIDTH_PER_FWHM)
    assert np.isclose(report['pooled']['fwhm_um'], 7.0, rtol=0.02)
    assert report['field']['verdict'] == 'flat'
    assert report['optics']['box_ok']


def test_sheet_thickness_at_coarse_z_uses_shared_fit(tmp_path):
    """1.4 slices per FWHM: single-bead fits are not used; one shared-width fit recovers the sheet."""
    _, _, _, report = _measure(tmp_path, dict(shape=(31, 360, 360), n_beads=120, scale_z=5.0, photons=3000.0,
                                              background=20.0, seed=3, **LOW_NA))
    s = report['summary']
    assert s['sampling'] == 'undersampled' and s['method'] == 'pooled_fit'
    assert s['samples_per_fwhm'] < lightsheet.MIN_SAMPLES_SINGLE
    assert not report['beads']['used'].any()
    assert np.isclose(s['sheet_fwhm_um'], 7.0, rtol=0.03)
    assert report['field']['verdict'] == 'flat'
    assert any("too coarse" in note for note in report['notes'])


@pytest.mark.parametrize("axis", ["x", "y"])
def test_waist_fit_recovers_the_beam(tmp_path, axis):
    """A focused sheet: waist thickness, waist position and Rayleigh range come back, axis found automatically."""
    _, truth, _, report = _measure(tmp_path, dict(shape=(41, 420, 420), n_beads=220, scale_z=2.0, rayleigh_um=250.0,
                                                  sheet_axis=axis, min_separation_px=14, seed=4, **LOW_NA),
                                   excitation_nm=488.0)
    field = report['field']
    assert field['verdict'] == 'waist'
    assert field['axis'] == axis and field['axis_source'] == 'auto'
    assert np.isclose(field['waist_fwhm_um'], 7.0, rtol=0.03)
    assert abs(field['waist_pos_um'] - truth['waist_pos_um']) < 15.0
    assert np.isclose(field['rayleigh_um'], 250.0, rtol=0.08)
    assert field['rayleigh_ci_um'][0] <= field['rayleigh_um'] <= field['rayleigh_ci_um'][1]
    assert np.isclose(field['ideal_rayleigh_um'], optics.rayleigh_range_um(field['waist_fwhm_um'], 0.488))
    assert report['summary']['sheet_fwhm_waist_um'] == field['waist_fwhm_um']


def test_bead_size_is_removed(tmp_path):
    """3 µm beads broaden a 7 µm sheet profile by about 2.5%; the correction takes it out."""
    sim = dict(shape=(41, 200, 200), n_beads=40, scale_z=2.0, bead_diameter_um=3.0, seed=5, **LOW_NA)
    _, _, _, raw = _measure(tmp_path, sim)
    _, _, _, corrected = _measure(tmp_path, sim, bead_diameter_um=3.0)
    assert raw['summary']['sheet_fwhm_um'] > 7.0 * 1.015
    assert np.isclose(corrected['summary']['sheet_fwhm_um'], 7.0, rtol=0.01)


def test_continuous_scan_is_removed(tmp_path):
    """Stage motion during the exposure broadens the profile; scan_mode='continuous' removes it."""
    sim = dict(shape=(31, 200, 200), n_beads=40, scale_z=3.0, scan_mode='continuous', seed=6, **LOW_NA)
    _, _, _, raw = _measure(tmp_path, sim)
    _, _, _, corrected = _measure(tmp_path, sim, scan_mode='continuous')
    assert raw['summary']['sheet_fwhm_um'] > 7.0 * 1.025
    assert np.isclose(corrected['summary']['sheet_fwhm_um'], 7.0, rtol=0.015)
    assert np.isclose(corrected['pooled']['fwhm_um'], 7.0, rtol=0.015)


def test_box_grows_to_hold_the_defocused_image(tmp_path):
    """At higher NA a small box clips the defocused bead and reads thin; the report enlarges it."""
    sim = dict(shape=(41, 150, 150), n_beads=10, na=0.5, ri=1.33, emission_nm=525.0, sheet_fwhm_um=4.0,
               scale_xy=0.33, scale_z=0.8, seed=7)
    stack, truth, results, report = _measure(tmp_path, sim, box_size=7)
    small_box = lightsheet.measure_beads(stack, results, 0.33, 0.8, 7)[0]
    assert np.median(small_box['sheet_fwhm_um']) < 4.0 * 0.97
    assert report['summary']['box_px_used'] > 7
    assert report['optics']['box_ok']
    assert np.isclose(report['summary']['sheet_fwhm_um'], 4.0, rtol=0.02)
    # The on-axis width is the system's axial resolution: sheet and detection combined.
    assert np.isclose(report['optics']['onaxis_ratio'], 1.0, atol=0.05)
    assert report['optics']['onaxis_axial_measured_um'] < 0.9 * report['summary']['sheet_fwhm_um']


def test_fit_field_verdicts():
    """Constant, too few points, and a thickness trend whose waist lies outside the field."""
    rng = np.random.default_rng(0)
    u = np.linspace(0.0, 1000.0, 120)
    flat = lightsheet.fit_field(u, 7.0 + rng.normal(0, 0.2, u.size), rng, n_boot=50)
    assert flat['verdict'] == 'flat'
    assert np.isclose(flat['mean_fwhm_um'], 7.0, atol=0.1)

    few = lightsheet.fit_field(u[:8], np.full(8, 7.0), rng, n_boot=50)
    assert few['verdict'] == 'insufficient'

    trend = optics.sheet_fwhm_at(u, 7.0, -400.0, 300.0) * (1 + rng.normal(0, 0.02, u.size))
    gradient = lightsheet.fit_field(u, trend, rng, n_boot=50)
    assert gradient['verdict'] == 'gradient'
    assert gradient['edge_fwhm_um'][1] > gradient['edge_fwhm_um'][0]

    # Far beyond the waist the thickness grows linearly: the fit gives the rate
    # and the point it extrapolates to.
    far = optics.sheet_fwhm_at(u, 7.0, -2000.0, 100.0) * (1 + rng.normal(0, 0.02, u.size))
    cone = lightsheet.fit_field(u, far, rng, n_boot=50)
    assert cone['verdict'] == 'gradient' and cone['far_field']
    assert np.isclose(cone['divergence_um_per_um'], 7.0 / 100.0, rtol=0.05)
    assert np.isclose(cone['waist_pos_um'], -2000.0, rtol=0.1)
    assert not gradient['far_field']

    # A handful of wild outliers must not turn a flat field into a waist.
    wild = 7.0 + rng.normal(0, 0.2, u.size)
    wild[::17] *= 1.6
    assert lightsheet.fit_field(u, wild, rng, n_boot=50)['verdict'] == 'flat'


def test_report_files_and_psf_export(tmp_path):
    """The report writes its files; the exported PSF is centered, non-negative and sums to one."""
    stack, truth, results, report = _measure(tmp_path, dict(shape=(41, 220, 220), n_beads=50, scale_z=2.0,
                                                            photons=5000.0, seed=8, **LOW_NA))
    out = tmp_path / "beads.tif"
    paths = lightsheet.write_outputs(report, out)
    assert {'beads', 'summary', 'json', 'figure'} <= set(paths)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths.values())
    saved = json.loads(paths['json'].read_text(encoding='utf-8'))
    assert np.isclose(saved['summary']['sheet_fwhm_um'], report['summary']['sheet_fwhm_um'])
    assert "Sheet thickness" in paths['summary'].read_text(encoding='utf-8')

    psf, info = psf_export.build_psf(stack, results, truth['scale_xy'], truth['scale_z'], box_size=9)
    assert psf.dtype == np.float32 and all(n % 2 == 1 for n in psf.shape)
    assert np.isclose(float(psf.sum()), 1.0, atol=1e-5) and float(psf.min()) >= 0.0
    center = tuple(n // 2 for n in psf.shape)
    assert np.unravel_index(int(np.argmax(psf)), psf.shape) == center
    assert info['z_registration'] == 'sub-slice' and info['n_beads'] > 20
    # Along Z the on-axis PSF is the sheet narrowed slightly by detection.
    assert 0.9 * 7.0 < info['fwhm_um']['z'] < 1.03 * 7.0

    written = psf_export.export_psf(stack, results, truth['scale_xy'], truth['scale_z'], 9, out)
    assert written['psf'].exists() and written['json'].exists()


def test_cli_lightsheet_and_psf(tmp_path, monkeypatch, capsys):
    """--lightsheet and --export_psf run from the command line and leave the main outputs in place."""
    import tifffile

    from bead_analyzer import cli

    stack, truth = make_bead_field(shape=(31, 200, 200), n_beads=40, scale_z=2.0, seed=9, **LOW_NA)
    tif = tmp_path / "beads.tif"
    tifffile.imwrite(str(tif), stack)
    monkeypatch.setattr('sys.argv', [
        'bead-analyzer', str(tif), '--scale_xy', '2.0', '--scale_z', '2.0', '--box_size', '9',
        '--line_length', '20', '--blob_min_distance', '8', '--fit_gaussian', '--no_plots',
        '--lightsheet', '--na', '0.1', '--emission_nm', '525', '--ri', '1.0', '--export_psf',
    ])
    assert cli.main() == 0
    assert "Light-Sheet Report" in capsys.readouterr().out
    for suffix in ("_FWHM_data.csv", "_lightsheet_beads.csv", "_lightsheet_summary.txt",
                   "_lightsheet_report.json", "_psf.tif", "_psf.json"):
        assert (tmp_path / f"beads{suffix}").exists(), suffix
    report = json.loads((tmp_path / "beads_lightsheet_report.json").read_text(encoding='utf-8'))
    assert np.isclose(report['summary']['sheet_fwhm_um'], 7.0, rtol=0.03)


def test_crowding_is_three_dimensional():
    """Two beads stacked in XY but far apart in Z do not spoil each other's profile."""
    stack = np.zeros((60, 40, 40), dtype=np.float32) + 100.0
    z = np.arange(60, dtype=np.float32)
    for z0 in (15.0, 45.0):
        stack[:, 18:23, 18:23] += 1000.0 * np.exp(-0.5 * ((z - z0) / 2.0) ** 2)[:, None, None]
    results = [{'id': 1, 'x_coord': 20.0, 'y_coord': 20.0, 'z_coord': 15},
               {'id': 2, 'x_coord': 20.0, 'y_coord': 20.0, 'z_coord': 45}]
    anywhere = lightsheet.measure_beads(stack, results, 1.0, 1.0, 9)[0]
    nearby = lightsheet.measure_beads(stack, results, 1.0, 1.0, 9, z_half_window=14, z_reach=12.0)[0]
    assert anywhere['crowded'].all()
    assert not nearby['crowded'].any() and not nearby['truncated'].any()
    expected = 2.0 * optics.FWHM_PER_SIGMA
    assert np.allclose(nearby['sheet_fwhm_um'], expected, rtol=0.01)
    assert np.allclose(nearby['z_um'], [15.0, 45.0], atol=0.05)
    assert np.allclose(nearby['focus_offset_um'], 0.0, atol=0.05)


def test_report_without_beads_or_optics():
    """No beads gives an empty report; missing optics skips the theory check instead of failing."""
    stack = np.zeros((11, 32, 32), dtype=np.uint16)
    empty = lightsheet.analyze(stack, [], 1.0, 1.0)
    assert empty['summary']['sheet_fwhm_um'] is None and len(empty['beads']) == 0
    assert "No beads" in lightsheet.format_summary(empty) or empty['notes']
