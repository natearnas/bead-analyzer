# -----------------------------------------------------------------------------
# Bead Analyzer
#
# (c) 2026 Arnas Technologies, LLC
# Developed by Nathan O'Connor, PhD, MS
#
# Licensed under the MIT License.
# For consulting or custom development: nathan@arnastech.com
# -----------------------------------------------------------------------------

"""
Light-sheet report: illumination-sheet thickness from bead Z profiles.

With widefield (camera) detection, defocus spreads a bead's light sideways but
does not remove it. The bead signal summed over an XY box that holds the
defocused image therefore follows the illumination profile alone, and the FWHM
of that Z profile is the sheet thickness at the bead. This module measures it
per bead, pools beads when Z sampling is too coarse for single-bead fits, and
fits the thickness across the field.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import spline_filter1d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.special import erf

from . import __version__, optics
from .core import fit_gaussian_fwhm

# Z samples per FWHM: single-bead values are reported from 2, and trusted from 3.
MIN_SAMPLES_SINGLE = 2.0
GOOD_SAMPLES = 3.0
MIN_POINTS_FIELD = 15
MIN_BINS_FIELD = 6
MAX_BOX_PX = 101
# The waist model must beat a constant by this much (BIC) and change the
# thickness by at least 5% across the field before a waist is reported.
DELTA_BIC_MIN = 10.0
MIN_VARIATION = 0.05


def _mad_sigma(values):
    values = np.asarray(values, dtype=float)
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


def _onaxis_profile(stack, x_c, y_c, k0, k1, pad=6):
    """Intensity at the sub-pixel bead center in each slice k0..k1-1.

    Interpolates within each slice only (cubic B-spline); Z samples are never mixed.
    """
    ny, nx = stack.shape[1:]
    xi, yi = int(np.floor(x_c)), int(np.floor(y_c))
    y0, y1 = max(0, yi - pad), min(ny, yi + pad + 2)
    x0, x1 = max(0, xi - pad), min(nx, xi + pad + 2)
    patch = np.asarray(stack[k0:k1, y0:y1, x0:x1], dtype=np.float64)
    iy, ix = yi - y0, xi - x0
    if iy < 1 or ix < 1 or iy + 2 >= patch.shape[1] or ix + 2 >= patch.shape[2]:
        return patch[:, int(np.clip(round(y_c) - y0, 0, patch.shape[1] - 1)),
                     int(np.clip(round(x_c) - x0, 0, patch.shape[2] - 1))]
    coef = spline_filter1d(spline_filter1d(patch, order=3, axis=1), order=3, axis=2)

    def weights(t):
        return np.array([(1 - t) ** 3, 3 * t ** 3 - 6 * t ** 2 + 4,
                         -3 * t ** 3 + 3 * t ** 2 + 3 * t + 1, t ** 3]) / 6.0

    block = coef[:, iy - 1:iy + 3, ix - 1:ix + 3]
    return np.einsum('kyx,y,x->k', block, weights(y_c - yi), weights(x_c - xi))


def measure_beads(stack, results, scale_xy, scale_z, box_size, scan_mode='step', bead_diameter_um=None,
                  z_half_window=None, z_reach=None):
    """Measure the box-integrated and on-axis Z profiles of every bead.

    ``z_half_window`` limits each profile to that many slices either side of the
    bead (default: the whole stack). ``z_reach`` makes the crowding test
    three-dimensional: a neighbor inside the box only counts if it is also
    within that many slices in Z (default: at any depth).

    Returns (table, z_profiles, peaks): a DataFrame with one row per bead, the
    box-averaged Z profile of each bead, and the index of the sample nearest
    its fitted center within that profile.
    """
    nz, ny, nx = stack.shape
    half_box = int(box_size) // 2
    sat_level = np.iinfo(stack.dtype).max if np.issubdtype(stack.dtype, np.integer) else None
    xy = np.array([[float(r['x_coord']), float(r['y_coord'])] for r in results], dtype=float).reshape(-1, 2)
    z_all = np.array([float(r['z_coord']) for r in results])
    crowded = np.zeros(len(xy), dtype=bool)
    if len(xy) > 1:
        for i, near in enumerate(cKDTree(xy).query_ball_point(xy, r=float(box_size))):
            others = [j for j in near if j != i]
            # In a 3D bead sample most neighbors in XY sit at other depths and do
            # not touch this bead's profile.
            crowded[i] = bool(others) and (z_reach is None or bool(np.any(np.abs(z_all[others] - z_all[i]) < z_reach)))

    rows, z_profiles, peaks = [], [], []
    for r, (x_c, y_c), is_crowded in zip(results, xy, crowded):
        z_pk = int(np.clip(int(r['z_coord']), 0, nz - 1))
        xi, yi = int(round(x_c)), int(round(y_c))
        y1, y2 = max(0, yi - half_box), min(ny, yi + half_box + 1)
        x1, x2 = max(0, xi - half_box), min(nx, xi + half_box + 1)
        z_lo = 0 if z_half_window is None else max(0, z_pk - int(z_half_window))
        z_hi = nz if z_half_window is None else min(nz, z_pk + int(z_half_window) + 1)
        z_prof = np.mean(stack[z_lo:z_hi, y1:y2, x1:x2], axis=(1, 2), dtype=np.float64)
        row = {
            'id': r.get('id'), 'x_um': x_c * scale_xy, 'y_um': y_c * scale_xy, 'z_um': np.nan,
            'fwhm_z_box_um': np.nan, 'fwhm_z_box_se_um': np.nan, 'samples_per_fwhm': np.nan,
            'sheet_fwhm_um': np.nan, 'sheet_fwhm_se_um': np.nan, 'sheet_width_1e2_um': np.nan,
            'fwhm_z_onaxis_um': np.nan, 'fwhm_z_onaxis_se_um': np.nan, 'focus_offset_um': np.nan,
            'profile_misfit': np.nan,
            'sampling': 'fit_failed', 'saturated': False, 'truncated': False,
            'crowded': bool(is_crowded), 'used': False,
        }
        fit = fit_gaussian_fwhm(z_prof, scale_z, peak_hint=z_pk - z_lo)
        if fit is None:
            rows.append(row)
            z_profiles.append(z_prof)
            peaks.append(z_pk - z_lo)
            continue
        fwhm, samples = fit['fwhm_um'], fit['fwhm_px']
        center = fit['center_px'] + z_lo
        k_c = int(np.clip(round(center), 0, nz - 1))
        reach = int(np.ceil(1.5 * samples))
        # Cut off by the end of the stack, or by the end of the profile window.
        row['truncated'] = bool(k_c - reach < z_lo or k_c + reach > z_hi - 1)
        row['profile_misfit'] = fit['rmse'] / fit['amplitude'] if fit['amplitude'] > 0 else np.nan
        if sat_level is not None:
            core_vol = stack[max(0, k_c - 2):k_c + 3, y1:y2, x1:x2]
            row['saturated'] = bool(core_vol.size and core_vol.max() >= sat_level)

        sheet = fwhm
        if scan_mode == 'continuous':
            sheet = optics.remove_scan_slab(sheet, scale_z)
        sheet = optics.remove_bead_size(sheet, bead_diameter_um)
        row.update({
            'z_um': center * scale_z, 'fwhm_z_box_um': fwhm, 'fwhm_z_box_se_um': fit['fwhm_se_um'],
            'samples_per_fwhm': samples, 'sheet_fwhm_um': sheet,
            'sheet_fwhm_se_um': fit['fwhm_se_um'] * fwhm / sheet if np.isfinite(sheet) and sheet > 0 else np.nan,
            'sheet_width_1e2_um': sheet * optics.E2_WIDTH_PER_FWHM,
            'sampling': 'ok' if samples >= GOOD_SAMPLES else ('coarse' if samples >= MIN_SAMPLES_SINGLE else 'undersampled'),
        })

        hw = max(10, int(np.ceil(3 * samples)))
        k0, k1 = max(0, k_c - hw), min(nz, k_c + hw + 1)
        on_axis = _onaxis_profile(stack, x_c, y_c, k0, k1)
        # The sharpest slice can sit away from the sheet center, so look for it.
        a, b = max(0, k_c - k0 - reach), min(len(on_axis), k_c - k0 + reach + 1)
        fit_on = fit_gaussian_fwhm(on_axis, scale_z, peak_hint=a + int(np.argmax(on_axis[a:b])))
        if fit_on is not None:
            row['fwhm_z_onaxis_um'] = fit_on['fwhm_um']
            row['fwhm_z_onaxis_se_um'] = fit_on['fwhm_se_um']
            # Sharpest focus (on-axis peak) relative to the sheet center (peak of the
            # box-integrated profile). Non-zero when the sheet is not in the focal plane.
            row['focus_offset_um'] = (fit_on['center_px'] + k0 - center) * scale_z
        rows.append(row)
        z_profiles.append(z_prof)
        peaks.append(k_c - z_lo)
    return pd.DataFrame(rows), z_profiles, np.array(peaks, dtype=int)


# --------------------------------------------------------------------------- pooled fit

def _profile_windows(z_profiles, peaks, half_window):
    """Stack every bead's Z profile in a window around its peak, scaled to unit amplitude."""
    n, m = len(z_profiles), 2 * half_window + 1
    values, valid = np.zeros((n, m)), np.zeros((n, m))
    for i, (prof, k) in enumerate(zip(z_profiles, peaks)):
        lo = k - half_window
        a, b = max(0, lo), min(len(prof), k + half_window + 1)
        if b - a < 4:
            continue
        seg = np.asarray(prof[a:b], dtype=float)
        amp = seg.max() - seg.min()
        if amp <= 0:
            continue
        values[i, a - lo:b - lo] = (seg - seg.min()) / amp
        valid[i, a - lo:b - lo] = 1.0
    return values, valid


def _rss_matrix(values, valid, sigmas, half_window, continuous):
    """Best residual sum of squares of each bead for each candidate width.

    For a given width every bead keeps its own amplitude, baseline (solved in
    closed form) and sub-slice center (searched on a grid), so the only
    quantity shared between beads is the width.
    """
    z = np.arange(-half_window, half_window + 1.0)
    centers = np.linspace(-1.0, 1.0, 41)
    delta = z[None, :] - centers[:, None]
    n_valid = valid.sum(axis=1)[:, None]
    wy = valid * values
    s_y = wy.sum(axis=1)[:, None]
    s_yy = (wy * values).sum(axis=1)[:, None]
    out = np.empty((values.shape[0], len(sigmas)))
    with np.errstate(divide='ignore', invalid='ignore'):
        for j, sigma in enumerate(sigmas):
            if continuous:
                root = sigma * np.sqrt(2.0)
                basis = erf((delta + 0.5) / root) - erf((delta - 0.5) / root)
            else:
                basis = np.exp(-0.5 * (delta / sigma) ** 2)
            s_g = valid @ basis.T
            s_gg = valid @ (basis * basis).T
            s_gy = wy @ basis.T
            det = n_valid * s_gg - s_g ** 2
            amp = np.where(det > 1e-12, (n_valid * s_gy - s_g * s_y) / det, 0.0)
            amp = np.clip(amp, 0.0, None)
            base = (s_y - amp * s_g) / np.maximum(n_valid, 1.0)
            rss = (s_yy - 2 * amp * s_gy - 2 * base * s_y + amp ** 2 * s_gg
                   + 2 * amp * base * s_g + base ** 2 * n_valid)
            out[:, j] = rss.min(axis=1)
    return out


def _grid_minimum(cost, sigmas):
    """Minimum of a cost curve on a log-spaced grid, refined with a parabola."""
    j = int(np.argmin(cost))
    if j == 0 or j == len(sigmas) - 1:
        return float(sigmas[j]), True
    y0, y1, y2 = cost[j - 1], cost[j], cost[j + 1]
    curv = y0 - 2 * y1 + y2
    shift = 0.5 * (y0 - y2) / curv if curv > 0 else 0.0
    step = np.log(sigmas[j + 1] / sigmas[j])
    return float(sigmas[j] * np.exp(shift * step)), False


def _pooled_estimate(rss, sigmas, rows, scale_z, rng, n_boot):
    """Shared-width estimate for a set of beads, with a bootstrap over beads."""
    rows = np.asarray(rows, dtype=int)
    if rows.size < 3:
        return None
    sub = rss[rows]
    sigma, at_edge = _grid_minimum(sub.sum(axis=0), sigmas)
    # Drop beads the shared width fits badly (neighbors in the window, debris).
    j = int(np.argmin(np.abs(np.log(sigmas / sigma))))
    cut = np.median(sub[:, j]) + 8.0 * _mad_sigma(sub[:, j])
    keep = sub[:, j] <= cut if cut > 0 else np.ones(rows.size, dtype=bool)
    if keep.sum() >= 3 and keep.sum() < rows.size:
        sub = sub[keep]
        sigma, at_edge = _grid_minimum(sub.sum(axis=0), sigmas)
    n = sub.shape[0]
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    boots = np.array([_grid_minimum(c, sigmas)[0] for c in counts @ sub])
    to_um = optics.FWHM_PER_SIGMA * scale_z
    return {
        'fwhm_um': sigma * to_um,
        'se_um': float(np.std(boots)) * to_um,
        'ci_um': [float(np.percentile(boots, 2.5)) * to_um, float(np.percentile(boots, 97.5)) * to_um],
        'n_beads': int(n), 'n_trimmed': int(rows.size - n), 'at_grid_edge': bool(at_edge),
    }


# --------------------------------------------------------------------------- field fit

def _waist_curve(p, u):
    return np.sqrt(p[0] ** 2 + (p[2] * (u - p[1])) ** 2)


def fit_field(u, fwhm, rng, n_boot=200, min_points=MIN_POINTS_FIELD):
    """Fit thickness against position: constant, or a Gaussian-beam waist.

    Returns a dict whose ``verdict`` is 'flat', 'waist', 'gradient' (thickness
    changes but the waist lies outside the field) or 'insufficient'.
    """
    u, fwhm = np.asarray(u, dtype=float), np.asarray(fwhm, dtype=float)
    ok = np.isfinite(u) & np.isfinite(fwhm)
    u, fwhm = u[ok], fwhm[ok]
    out = {'verdict': 'insufficient', 'n_points': int(u.size)}
    if u.size < min_points:
        return out
    u_min, u_max = float(u.min()), float(u.max())
    span = u_max - u_min
    if span <= 0:
        return out
    med = float(np.median(fwhm))
    scatter = _mad_sigma(fwhm) or 0.05 * med
    lower = [1e-3 * med, u_min - 3 * span, 0.0]
    upper = [3 * float(fwhm.max()), u_max + 3 * span, 100 * med / span]

    def solve(uu, ff, start, robust):
        kw = dict(loss='soft_l1', f_scale=scatter) if robust else {}
        return least_squares(lambda p: _waist_curve(p, uu) - ff, np.clip(start, lower, upper),
                             bounds=(lower, upper), **kw)

    best = None
    f_low = float(np.percentile(fwhm, 10))
    for u0 in np.linspace(u_min - 0.5 * span, u_max + 0.5 * span, 9):
        for zr in (0.25 * span, span, 4 * span):
            res = solve(u, fwhm, [f_low, u0, f_low / zr], robust=True)
            if best is None or res.cost < best.cost:
                best = res
    resid = fwhm - _waist_curve(best.x, u)
    spread = max(_mad_sigma(resid), 1e-9 * med)
    inl = np.abs(resid) <= 3.5 * spread
    if inl.sum() < min_points:
        inl = np.ones(u.size, dtype=bool)
    ui, fi = u[inl], fwhm[inl]
    n = int(inl.sum())
    fit = solve(ui, fi, best.x, robust=False)
    rss_waist = max(2.0 * fit.cost, 1e-300)
    rss_flat = max(float(np.sum((fi - fi.mean()) ** 2)), 1e-300)
    delta_bic = n * np.log(rss_flat / rss_waist) - 2.0 * np.log(n)
    f0, u0, slope = (float(v) for v in fit.x)
    grid = np.linspace(u_min, u_max, 201)
    curve = _waist_curve(fit.x, grid)
    variation = float(curve.max() / curve.min() - 1.0)

    out.update({
        'n_points': int(u.size), 'n_inliers': n, 'delta_bic': float(delta_bic),
        'field_um': [u_min, u_max], 'variation_fraction': variation,
        'mean_fwhm_um': float(fi.mean()), 'mean_se_um': float(fi.std(ddof=1) / np.sqrt(n)),
        'scatter_um': float(np.sqrt(rss_waist / n)),
        'edge_fwhm_um': [float(curve[0]), float(curve[-1])],
        'curve_u_um': grid.tolist(), 'curve_fwhm_um': curve.tolist(),
    })
    if delta_bic <= DELTA_BIC_MIN or variation < MIN_VARIATION:
        out['verdict'] = 'flat'
        out['curve_fwhm_um'] = np.full(grid.size, fi.mean()).tolist()
        return out

    inside = u_min + 0.02 * span < u0 < u_max - 0.02 * span
    out['verdict'] = 'waist' if inside else 'gradient'
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        b = solve(ui[pick], fi[pick], fit.x, robust=False).x
        boots.append((b[0], b[1], b[0] / b[2] if b[2] > 0 else np.inf))
    boots = np.array(boots)

    def interval(col):
        return [float(np.percentile(boots[:, col], 2.5)), float(np.percentile(boots[:, col], 97.5))]

    out.update({
        'waist_fwhm_um': f0, 'waist_fwhm_ci_um': interval(0),
        'waist_pos_um': u0, 'waist_pos_ci_um': interval(1),
        'rayleigh_um': f0 / slope if slope > 0 else float('inf'), 'rayleigh_ci_um': interval(2),
    })
    if not inside:
        # Far from its waist a beam widens linearly. The field then fixes the rate
        # of widening and the point the thickness extrapolates to, but not the
        # waist thickness itself.
        out['slope_um_per_um'] = float((curve[-1] - curve[0]) / span)
        out['far_field'] = bool(f0 < 0.5 * curve.min())
        if out['far_field']:
            out['divergence_um_per_um'] = slope
    return out


# --------------------------------------------------------------------------- report

def _lateral_measured(results, bead_diameter_um):
    """Median lateral FWHM from the main analysis (Gaussian fit when available)."""
    for stem in ('gauss', 'prom'):
        vals = []
        for r in results:
            for axis in ('x', 'y'):
                v = r.get(f'fwhm_{axis}_{stem}', r.get(f'fwhm_{axis}_{stem}_um'))
                if v is not None and np.isfinite(v):
                    vals.append(float(v))
        if vals:
            return float(optics.remove_bead_size(np.median(vals), bead_diameter_um)), stem
    return None, None


def analyze(stack, results, scale_xy, scale_z, box_size=15, na=None, emission_nm=None,
            excitation_nm=None, ri=None, bead_diameter_um=None, sheet_axis='auto',
            scan_mode='step', n_boot=200, seed=0, source=None):
    """Build the light-sheet report for an analyzed bead stack.

    Parameters
    ----------
    stack : ndarray (Z, Y, X)
        The raw stack, before any background subtraction.
    results : list of dict
        Accepted beads from the main analysis (``x_coord``, ``y_coord``, ``z_coord``).
    na, emission_nm, ri : float, optional
        Detection NA, emission wavelength and refractive index. With these the
        report checks the box size and compares the measurements with theory.
    excitation_nm : float, optional
        Used to compare a fitted Rayleigh range with an ideal Gaussian beam.
    bead_diameter_um : float, optional
        Removes the broadening of a uniformly labeled bead.
    sheet_axis : 'x', 'y' or 'auto'
        Image axis along which the sheet propagates.
    scan_mode : 'step' or 'continuous'
        'continuous' if the stage moved through one Z step during each exposure.

    Returns a dict with ``beads`` (DataFrame), ``summary``, ``pooled``, ``field``,
    ``optics`` and ``notes``.
    """
    if scan_mode not in ('step', 'continuous'):
        raise ValueError("scan_mode must be 'step' or 'continuous'")
    if sheet_axis not in ('x', 'y', 'auto'):
        raise ValueError("sheet_axis must be 'x', 'y' or 'auto'")
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError("stack must be a 3D (Z, Y, X) array")
    rng = np.random.default_rng(seed)
    notes = []
    have_optics = na is not None and emission_nm is not None
    ri_used = float(ri) if ri else 1.0
    if have_optics and not ri:
        notes.append("Refractive index not given; 1.0 assumed for the optics check.")
    lam = emission_nm / 1000.0 if emission_nm else None
    report = {
        'bead_analyzer_version': __version__,
        'source': str(source) if source is not None else None,
        'inputs': {
            'scale_xy_um': scale_xy, 'scale_z_um': scale_z, 'box_size_px': int(box_size), 'na': na,
            'emission_nm': emission_nm, 'excitation_nm': excitation_nm, 'ri': ri,
            'bead_diameter_um': bead_diameter_um, 'sheet_axis': sheet_axis, 'scan_mode': scan_mode,
        },
        'beads': pd.DataFrame(), 'summary': {'n_beads': len(results), 'n_used': 0, 'sheet_fwhm_um': None},
        'pooled': None, 'field': {'verdict': 'insufficient', 'n_points': 0}, 'optics': None, 'notes': notes,
    }
    if not results:
        notes.append("No beads to analyze.")
        return report

    def clean_mask(tab):
        return (tab['sampling'] != 'fit_failed') & ~tab['saturated'] & ~tab['truncated'] & ~tab['crowded']

    box_used = int(box_size)
    beads, z_profiles, peaks = measure_beads(stack, results, scale_xy, scale_z, box_used, scan_mode, bead_diameter_um)
    box_limit = int(min(MAX_BOX_PX, (min(stack.shape[1:]) - 1) // 2 * 2 + 1))
    density_limited = False
    for _ in range(4):
        # A box that clips the defocused image reads thin, which in turn asks
        # for a smaller box than is needed, so the size is re-checked after each pass.
        ok = clean_mask(beads)
        if not have_optics or not ok.any():
            break
        guess = float(np.nanmedian(beads.loc[ok, 'sheet_fwhm_um']))
        if not np.isfinite(guess):
            break
        need = 2 * int(np.ceil(optics.defocus_halfwidth_um(guess, na, lam, ri_used) / scale_xy)) + 1
        if need <= box_used or box_used >= box_limit:
            break
        slices = float(np.nanmedian(beads.loc[ok, 'samples_per_fwhm']))
        trial_box = min(need, box_limit)
        trial = measure_beads(stack, results, scale_xy, scale_z, trial_box, scan_mode, bead_diameter_um,
                              z_half_window=int(np.ceil(6 * slices)) + 20, z_reach=3.5 * slices)
        n_clean = int(clean_mask(trial[0]).sum())
        if n_clean < max(10, 0.02 * len(results)):
            notes.append(f"The optics call for a box of {trial_box} px, but only {n_clean} of {len(results)} beads are "
                         f"isolated at that size. The {box_used} px box was kept, so the thickness is a lower bound. "
                         "A sparser bead sample would remove this limit.")
            density_limited = True
            break
        box_used = trial_box
        beads, z_profiles, peaks = trial
    if box_used != int(box_size):
        why = "to hold more of the defocused bead image." if density_limited else "so that the box holds the defocused bead image."
        notes.append(f"Z profiles were integrated over a {box_used} px box instead of {int(box_size)} px, {why}")
    clean = clean_mask(beads).to_numpy()
    summary = report['summary']
    summary.update({
        'box_px_used': box_used,
        'n_fit_failed': int((beads['sampling'] == 'fit_failed').sum()),
        'n_saturated': int(beads['saturated'].sum()), 'n_truncated': int(beads['truncated'].sum()),
        'n_crowded': int(beads['crowded'].sum()),
    })
    if clean.sum() < 3:
        notes.append("Fewer than 3 isolated, unsaturated beads with a complete Z profile; no thickness reported.")
        report['beads'] = beads
        return report

    samples = float(np.median(beads.loc[clean, 'samples_per_fwhm']))
    sampling = 'ok' if samples >= GOOD_SAMPLES else ('coarse' if samples >= MIN_SAMPLES_SINGLE else 'undersampled')
    summary.update({'samples_per_fwhm': samples, 'sampling': sampling})

    # Pooled, shared-width fit over all clean beads.
    clean_rows = np.flatnonzero(clean)
    half_window = int(np.clip(np.ceil(3.0 * np.percentile(beads.loc[clean, 'samples_per_fwhm'], 90)), 4, 40))
    values, valid = _profile_windows([z_profiles[i] for i in clean_rows], peaks[clean_rows], half_window)
    sigmas = np.geomspace(0.2, half_window / 1.5, 120)
    rss = _rss_matrix(values, valid, sigmas, half_window, scan_mode == 'continuous')
    usable = np.flatnonzero(valid.sum(axis=1) >= 4)

    def pooled_sheet(rows):
        est = _pooled_estimate(rss, sigmas, rows, scale_z, rng, n_boot)
        if est is None:
            return None
        raw = est['fwhm_um']
        sheet = optics.remove_bead_size(raw, bead_diameter_um)
        scale = sheet / raw if np.isfinite(sheet) else np.nan
        est.update({'profile_fwhm_um': raw, 'fwhm_um': float(sheet), 'se_um': est['se_um'] * scale,
                    'ci_um': [est['ci_um'][0] * scale, est['ci_um'][1] * scale]})
        return est

    pooled = pooled_sheet(usable)
    report['pooled'] = pooled

    # Per-bead values, reported when single-bead fits are trustworthy.
    used = clean & (beads['sampling'] != 'undersampled').to_numpy() & np.isfinite(beads['sheet_fwhm_um'].to_numpy())
    beads['used'] = used
    per_bead = None
    if sampling != 'undersampled' and used.sum() >= 3:
        vals = beads.loc[used, 'sheet_fwhm_um'].to_numpy()
        boots = np.median(vals[rng.integers(0, vals.size, (n_boot, vals.size))], axis=1)
        per_bead = {'fwhm_um': float(np.median(vals)), 'n_beads': int(vals.size),
                    'ci_um': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    'bead_to_bead_sd_um': _mad_sigma(vals)}
    summary['per_bead'] = per_bead
    headline = per_bead if per_bead is not None else pooled
    if headline is not None and np.isfinite(headline['fwhm_um']):
        summary.update({
            'sheet_fwhm_um': headline['fwhm_um'], 'sheet_fwhm_ci_um': headline['ci_um'],
            'sheet_width_1e2_um': headline['fwhm_um'] * optics.E2_WIDTH_PER_FWHM,
            'method': 'per_bead_median' if per_bead is not None else 'pooled_fit',
        })
    summary['n_used'] = int(used.sum()) if per_bead is not None else (pooled['n_beads'] if pooled else 0)
    if per_bead is None and pooled:
        summary['n_poor_fit'] = pooled['n_trimmed']

    if sampling == 'undersampled':
        notes.append(f"Z sampling is {samples:.1f} slices per FWHM, too coarse for single-bead fits. "
                     "The thickness comes from one fit shared by all beads; per-bead values are not used.")
    elif sampling == 'coarse':
        notes.append(f"Z sampling is {samples:.1f} slices per FWHM; single-bead values are noisy. "
                     "A Z step of a third of the sheet thickness or finer is recommended.")
    offsets = beads.loc[used & np.isfinite(beads['focus_offset_um'].to_numpy()), 'focus_offset_um'].to_numpy()
    if offsets.size >= 3:
        summary['focus_offset_um'] = float(np.median(offsets))
        summary['focus_offset_spread_um'] = _mad_sigma(offsets)
        if summary.get('sheet_fwhm_um') and abs(summary['focus_offset_um']) > max(scale_z, 0.1 * summary['sheet_fwhm_um']):
            notes.append(f"The sharpest focus sits {summary['focus_offset_um']:+.1f} um from the sheet center "
                         "(median over beads): the sheet is not centered on the detection focal plane.")
    misfit = beads.loc[used & np.isfinite(beads['profile_misfit'].to_numpy()), 'profile_misfit'].to_numpy()
    if misfit.size >= 3:
        summary['profile_misfit'] = float(np.median(misfit))
        if summary['profile_misfit'] > 0.05:
            notes.append(f"The Z profiles depart from a Gaussian (median residual {100 * summary['profile_misfit']:.0f}% "
                         "of the peak). The sheet is not Gaussian here, so read the equivalent FWHM as approximate.")
    if summary['n_crowded'] > 0.3 * len(beads):
        notes.append(f"{summary['n_crowded']} of {len(beads)} beads have a neighbor inside the {box_used} px box and "
                     "were not used. A sparser bead sample would use more of them.")
    if bead_diameter_um and summary.get('sheet_fwhm_um') and bead_diameter_um > summary['sheet_fwhm_um']:
        notes.append("The bead diameter exceeds the sheet thickness; the bead-size correction is unreliable. "
                     "Use smaller beads.")
    if scan_mode == 'step':
        notes.append("Step-and-shoot acquisition assumed. Use --scan_mode continuous if the stage moved "
                     "during each exposure.")

    # Thickness across the field.
    axes = ['x', 'y'] if sheet_axis == 'auto' else [sheet_axis]
    fits = {}
    for axis in axes:
        u_all = beads[f'{axis}_um'].to_numpy()
        if sampling != 'undersampled':
            fits[axis] = fit_field(u_all[used], beads.loc[used, 'sheet_fwhm_um'].to_numpy(), rng, n_boot)
            fits[axis]['points'] = 'beads'
        else:
            u_clean = u_all[clean_rows][usable]
            n_bins = int(min(12, usable.size // 15))
            pts = []
            if n_bins >= MIN_BINS_FIELD:
                for chunk in np.array_split(np.argsort(u_clean), n_bins):
                    est = pooled_sheet(usable[chunk])
                    if est is not None and np.isfinite(est['fwhm_um']):
                        pts.append((float(np.median(u_clean[chunk])), est['fwhm_um'], est['se_um']))
            pts = np.array(pts).reshape(-1, 3)
            fits[axis] = fit_field(pts[:, 0], pts[:, 1], rng, n_boot, min_points=MIN_BINS_FIELD)
            fits[axis].update({'points': 'pooled_bins', 'bin_u_um': pts[:, 0].tolist(),
                               'bin_fwhm_um': pts[:, 1].tolist(), 'bin_se_um': pts[:, 2].tolist()})
    axis = max(fits, key=lambda a: fits[a].get('delta_bic', -np.inf))
    field = fits[axis]
    field['axis'] = axis
    field['axes_checked'] = list(fits)
    field['axis_source'] = 'auto' if sheet_axis == 'auto' else 'user'
    if sheet_axis == 'auto':
        field['delta_bic_by_axis'] = {a: fits[a].get('delta_bic') for a in fits}
        if field['verdict'] in ('waist', 'gradient'):
            notes.append(f"Propagation axis chosen automatically ({axis}). Declare it with --sheet_axis if known.")
    if field['verdict'] == 'waist':
        summary['sheet_fwhm_waist_um'] = field['waist_fwhm_um']
        if excitation_nm:
            ideal = float(optics.rayleigh_range_um(field['waist_fwhm_um'], excitation_nm / 1000.0, ri_used))
            field['ideal_rayleigh_um'] = ideal
            field['rayleigh_ratio'] = field['rayleigh_um'] / ideal
    elif field.get('far_field') and excitation_nm:
        field['ideal_waist_fwhm_um'] = float(optics.waist_fwhm_from_divergence(
            field['divergence_um_per_um'], excitation_nm / 1000.0, ri_used))
    report['field'] = field

    if have_optics:
        det_axial = float(optics.axial_fwhm_um(na, lam, ri_used))
        lateral_theory = float(optics.lateral_fwhm_um(na, lam))
        # A camera pixel averages over its width, which broadens a spot like a
        # uniform distribution: the same 0.462 w^2 term as a scan step.
        lateral_expected = float(np.sqrt(lateral_theory ** 2 + optics.SLAB_FWHM2_PER_STEP2 * scale_xy ** 2))
        lateral, lateral_method = _lateral_measured(results, bead_diameter_um)
        onaxis_vals = beads.loc[used & np.isfinite(beads['fwhm_z_onaxis_um'].to_numpy()), 'fwhm_z_onaxis_um']
        onaxis = None
        if len(onaxis_vals):
            onaxis = float(np.median(onaxis_vals))
            if scan_mode == 'continuous':
                onaxis = optics.remove_scan_slab(onaxis, scale_z)
            onaxis = float(optics.remove_bead_size(onaxis, bead_diameter_um))
            onaxis = onaxis if np.isfinite(onaxis) else None
        info = {
            'na': na, 'emission_nm': emission_nm, 'ri': ri_used,
            'lateral_theory_um': lateral_theory, 'lateral_expected_um': lateral_expected,
            'lateral_measured_um': lateral, 'lateral_method': lateral_method,
            'lateral_ratio': lateral / lateral_expected if lateral else None,
            'detection_axial_fwhm_um': det_axial, 'onaxis_axial_measured_um': onaxis,
        }
        sheet = summary.get('sheet_fwhm_um')
        if sheet:
            expected = float(optics.system_axial_fwhm_um(sheet, det_axial))
            need_px = 2 * int(np.ceil(optics.defocus_halfwidth_um(sheet, na, lam, ri_used) / scale_xy)) + 1
            info.update({'onaxis_axial_expected_um': expected,
                         'onaxis_ratio': onaxis / expected if onaxis else None,
                         'box_px_needed': need_px, 'box_ok': bool(box_used >= need_px)})
            if box_used < need_px and not density_limited:
                notes.append(f"The {box_used} px box does not hold the defocused bead image (needs {need_px} px); "
                             "the sheet thickness is underestimated.")
        report['optics'] = info
    else:
        notes.append("Give --na and --emission_nm (and --ri) to check the box size and compare with theory.")

    report['beads'] = beads
    return report


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def _fmt_ci(ci, digits=2):
    if not ci:
        return ""
    lo, hi = ci
    hi_txt = f"{hi:.{digits}f}" if np.isfinite(hi) else "unbounded"
    return f" (95% CI {lo:.{digits}f} to {hi_txt} um)"


def format_summary(report):
    """Plain-text summary of a light-sheet report."""
    s, field, opt, pooled = report['summary'], report['field'], report['optics'], report['pooled']
    lines = ["=" * 60, "--- Light-Sheet Report ---", "=" * 60]
    if report.get('source'):
        lines.append(f"Source: {Path(report['source']).name}")
    lines.append(f"Beads: {s['n_beads']} analyzed, {s['n_used']} used")
    dropped = [f"{s.get(k, 0)} {label}" for k, label in (('n_crowded', 'crowded'), ('n_truncated', 'cut off in Z'),
                                                         ('n_saturated', 'saturated'), ('n_fit_failed', 'fit failed'),
                                                         ('n_poor_fit', 'poorly fitted by the shared width'))
               if s.get(k)]
    if dropped:
        lines.append("  Not used: " + ", ".join(dropped))
    if s.get('samples_per_fwhm'):
        lines.append(f"Z sampling: {s['samples_per_fwhm']:.1f} slices per FWHM ({s['sampling']})")
    lines.append("")
    lines.append("Sheet thickness (FWHM of an equivalent Gaussian sheet)")
    varies = field['verdict'] in ('waist', 'gradient')
    if s.get('sheet_fwhm_um'):
        how = "median of single-bead fits" if s['method'] == 'per_bead_median' else "one fit shared by all beads"
        name = "Thickness:       "
        if varies:
            name = "Field median:    " if s['method'] == 'per_bead_median' else "Field average:   "
        lines.append(f"  {name} {s['sheet_fwhm_um']:.2f} um{_fmt_ci(s.get('sheet_fwhm_ci_um'))}  [{how}]")
        lines.append(f"  1/e^2 full width: {s['sheet_width_1e2_um']:.2f} um")
        if s.get('per_bead') and pooled and not varies:
            lines.append(f"  Shared-width fit: {pooled['fwhm_um']:.2f} um{_fmt_ci(pooled['ci_um'])} (cross-check)")
        if s.get('per_bead'):
            lines.append(f"  Bead-to-bead spread: {s['per_bead']['bead_to_bead_sd_um']:.2f} um")
        if s.get('focus_offset_um') is not None:
            lines.append(f"  Sharpest focus relative to sheet center: {s['focus_offset_um']:+.2f} um "
                         f"(spread {s['focus_offset_spread_um']:.2f} um)")
        if varies:
            lines.append("  The thickness varies across the field; see below.")
    else:
        lines.append("  Not available.")
    lines.append("")
    axis = field.get('axis')
    if field['verdict'] == 'insufficient':
        lines.append("Across the field: too few beads to fit the thickness against position.")
    else:
        src = "chosen automatically" if field.get('axis_source') == 'auto' else "user-declared"
        lo, hi = field['field_um']
        if field['verdict'] == 'flat':
            checked = " and ".join(field.get('axes_checked', [axis]))
            lines.append(f"Across the field (checked along {checked})")
            lines.append(f"  Uniform: {field['mean_fwhm_um']:.2f} +/- {field['mean_se_um']:.2f} um; "
                         "no waist detected within the field.")
        else:
            lines.append(f"Across the field (along {axis}, {src}; beads span {lo:.0f}-{hi:.0f} um)")
            if field['verdict'] == 'waist':
                lines.append(f"  Waist thickness: {field['waist_fwhm_um']:.2f} um{_fmt_ci(field['waist_fwhm_ci_um'])}")
                lines.append(f"  Waist position:  {field['waist_pos_um']:.0f} um along {axis}"
                             f"{_fmt_ci(field['waist_pos_ci_um'], 0)}")
                lines.append(f"  Rayleigh range:  {field['rayleigh_um']:.0f} um{_fmt_ci(field['rayleigh_ci_um'], 0)}")
                lines.append(f"  Field within sqrt(2) of the waist thickness: {2 * field['rayleigh_um']:.0f} um")
                if field.get('ideal_rayleigh_um'):
                    lines.append(f"  Ideal Gaussian beam of this waist: {field['ideal_rayleigh_um']:.0f} um "
                                 f"(measured / ideal = {field['rayleigh_ratio']:.2f})")
            else:
                lines.append("  Thickness changes across the field, but the waist lies outside it.")
            e0, e1 = field['edge_fwhm_um']
            lines.append(f"  Thickness at the field edges: {e0:.2f} um ({axis} = {lo:.0f}), {e1:.2f} um ({axis} = {hi:.0f})")
            if field['verdict'] == 'gradient':
                lines.append(f"  Change: {100 * field['slope_um_per_um']:+.2f} um per 100 um along {axis}")
                if field.get('far_field'):
                    apex = field['waist_pos_um']
                    beyond = lo - apex if apex < lo else apex - hi
                    lines.append(f"  The thickness extrapolates to a focus near {axis} = {apex:.0f} um"
                                 f"{_fmt_ci(field['waist_pos_ci_um'], 0)}, about {beyond:.0f} um beyond the beads.")
                    if field.get('ideal_waist_fwhm_um'):
                        lines.append(f"  An ideal Gaussian sheet widening at this rate has a waist of "
                                     f"{field['ideal_waist_fwhm_um']:.1f} um FWHM.")
    if opt:
        lines.append("")
        lines.append(f"Optics check (NA {opt['na']}, emission {opt['emission_nm']:g} nm, n {opt['ri']:g})")
        if opt.get('lateral_measured_um'):
            lines.append(f"  Lateral FWHM: measured {opt['lateral_measured_um']:.2f} um, expected "
                         f"{opt['lateral_expected_um']:.2f} um (diffraction limit {opt['lateral_theory_um']:.2f} um "
                         f"plus pixel size; ratio {opt['lateral_ratio']:.2f})")
        lines.append(f"  Detection axial FWHM alone: {opt['detection_axial_fwhm_um']:.1f} um")
        if opt.get('onaxis_axial_measured_um') and opt.get('onaxis_axial_expected_um'):
            lines.append(f"  On-axis axial FWHM: measured {opt['onaxis_axial_measured_um']:.2f} um, "
                         f"expected {opt['onaxis_axial_expected_um']:.2f} um from sheet and detection "
                         f"(ratio {opt['onaxis_ratio']:.2f})")
        if 'box_ok' in opt:
            state = "holds" if opt['box_ok'] else "does NOT hold"
            lines.append(f"  Box: {s['box_px_used']} px {state} the defocused bead image (needs {opt['box_px_needed']} px)")
    if report['notes']:
        lines.append("")
        lines.append("Notes")
        lines.extend(f"  - {n}" for n in report['notes'])
    return "\n".join(lines) + "\n"


def _save_figure(report, out_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    beads, field, s = report['beads'], report['field'], report['summary']
    used = beads[beads['used']] if len(beads) else beads
    fig = Figure(figsize=(15, 4.6))
    FigureCanvasAgg(fig)
    ax_map, ax_u, ax_hist = fig.subplots(1, 3)

    if len(used):
        sc = ax_map.scatter(used['x_um'], used['y_um'], c=used['sheet_fwhm_um'], s=14, cmap='viridis')
        fig.colorbar(sc, ax=ax_map, label="Sheet FWHM (um)")
    ax_map.set_xlabel("x (um)")
    ax_map.set_ylabel("y (um)")
    ax_map.set_title("Sheet thickness at each bead")
    ax_map.invert_yaxis()
    ax_map.set_aspect('equal', adjustable='datalim')

    axis = field.get('axis') or 'x'
    if len(used):
        ax_u.plot(used[f'{axis}_um'], used['sheet_fwhm_um'], '.', color='0.6', ms=4, label="beads")
    if field.get('points') == 'pooled_bins' and field.get('bin_u_um'):
        ax_u.errorbar(field['bin_u_um'], field['bin_fwhm_um'], yerr=field['bin_se_um'], fmt='o', color='C0',
                      capsize=3, label="shared-width fit per bin")
    if field.get('curve_u_um'):
        ax_u.plot(field['curve_u_um'], field['curve_fwhm_um'], '-', color='C3', lw=2, label=f"fit: {field['verdict']}")
    ax_u.set_xlabel(f"{axis} (um)")
    ax_u.set_ylabel("Sheet FWHM (um)")
    ax_u.set_title("Thickness across the field")
    ax_u.legend(fontsize=8)

    if len(used):
        ax_hist.hist(used['sheet_fwhm_um'].dropna(), bins=30, color='0.7')
    if s.get('sheet_fwhm_um'):
        what = "median" if s.get('method') == 'per_bead_median' else "shared fit"
        ax_hist.axvline(s['sheet_fwhm_um'], color='C3', lw=2, label=f"{what} {s['sheet_fwhm_um']:.2f} um")
        ax_hist.legend(fontsize=8)
    ax_hist.set_xlabel("Sheet FWHM (um)")
    ax_hist.set_ylabel("Beads")
    ax_hist.set_title("Single-bead values")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def write_outputs(report, file_path, no_plots=False):
    """Write the report next to ``file_path``: bead CSV, text summary, JSON and figure."""
    file_path = Path(file_path)
    stem = file_path.stem
    paths = {}
    beads = report['beads']
    if len(beads):
        paths['beads'] = file_path.with_name(f"{stem}_lightsheet_beads.csv")
        beads.to_csv(paths['beads'], index=False)
    paths['summary'] = file_path.with_name(f"{stem}_lightsheet_summary.txt")
    paths['summary'].write_text(format_summary(report), encoding='utf-8')
    paths['json'] = file_path.with_name(f"{stem}_lightsheet_report.json")
    payload = {k: v for k, v in report.items() if k != 'beads'}
    paths['json'].write_text(json.dumps(_json_safe(payload), indent=2), encoding='utf-8')
    if not no_plots and len(beads):
        paths['figure'] = file_path.with_name(f"{stem}_lightsheet_report.png")
        try:
            _save_figure(report, paths['figure'])
        except Exception as e:
            print(f"WARNING: Light-sheet figure could not be saved: {e}")
            paths.pop('figure')
    return paths
