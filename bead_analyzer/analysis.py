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
Unified FWHM analysis pipeline for manual, blob, trackpy, StarDist, and Cellpose modes.
"""

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import PowerNorm
from scipy import signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, shift, zoom
from scipy.signal import find_peaks

from . import detectors
from .core import (
    INTERACTIVE_PREVIEW_NOTE,
    RectangleDrawer,
    add_interaction_key,
    add_left_drag_pan,
    add_mousewheel_zoom,
    calculate_fwhm_prominence,
    filter_by_qa,
    fit_gaussian_3d,
    fit_gaussian_fwhm,
    gaussian_func,
    make_preview_image,
    preview_rect_to_full,
    preview_to_full_point,
    reject_outliers_mad,
)


def _figure_agg(figsize):
    """Create a Figure with Agg canvas. Safe to use from a background thread (no Tk)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig


def _savefig_close(fig, out_path, dpi=300, bbox_inches='tight'):
    """Save an Agg figure and promptly release its memory."""
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches)
    finally:
        try:
            fig.clf()
        except Exception:
            pass
        del fig
        gc.collect()


def _ensure_stack_3d(img, channel=0):
    """Return 3D stack (Z, Y, X)."""
    if img.ndim == 4:
        return img[channel]
    return img


def _subtract_background(stack, rect_coords):
    """Subtract mean of ROI from each plane."""
    if not rect_coords:
        return stack
    x1, y1, x2, y2 = rect_coords
    xs, xe = sorted([x1, x2])[0], sorted([x1, x2])[1]
    ys, ye = sorted([y1, y2])[0], sorted([y1, y2])[1]
    stack = stack.astype(np.float32)
    for z in range(stack.shape[0]):
        roi = stack[z, int(ys):int(ye), int(xs):int(xe)]
        if roi.size:
            stack[z] -= roi.mean()
    stack[stack < 0] = 0
    return stack


def _interactive_background_roi(mip, title="Right-click and drag to draw a background region, then close", **imshow_kw):
    """Show MIP, let user draw background ROI; return rect_coords or None. Run on main thread from GUI."""
    preview_mip, ds_factor = make_preview_image(mip)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(preview_mip, cmap='gray', aspect='equal', **imshow_kw)
    ax.set_title(title)
    ax.text(
        0.01, 0.01, INTERACTIVE_PREVIEW_NOTE,
        transform=ax.transAxes, fontsize=9, color='white',
        ha='left', va='bottom', bbox=dict(facecolor='black', alpha=0.55, pad=4)
    )
    add_mousewheel_zoom(ax)
    add_left_drag_pan(ax)
    add_interaction_key(fig, [
        "Left-click + drag: Pan",
        "Mouse wheel: Zoom in/out",
        "Right-click + drag: Draw ROI",
        "Close window: Continue",
    ])
    rd = RectangleDrawer(ax)
    plt.show()
    return preview_rect_to_full(rd.rect_coords, ds_factor, mip.shape)


def _get_display_norm(mip, gamma=1.0, vmin_pct=None, vmax_pct=None):
    """Return imshow kwargs for display."""
    I_min, I_max = float(np.min(mip)), float(np.max(mip))
    dyn = I_max - I_min
    vmin = I_min + (vmin_pct / 100.0) * dyn if vmin_pct is not None else None
    vmax = I_min + (100.0 - (vmax_pct or 0)) / 100.0 * dyn if vmax_pct is not None else np.percentile(mip, 99.8)
    if gamma != 1.0:
        return {'norm': PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)}
    return {'vmin': vmin, 'vmax': vmax}


def _quality_metrics(z_profile, peak_idx, min_snr=3.0, min_symmetry=0.6):
    """Compute simple QA metrics from the Z profile."""
    if z_profile is None or len(z_profile) < 5:
        return None, None, None
    peak_idx = int(peak_idx)
    # Baseline from regions away from peak
    left = z_profile[:max(0, peak_idx - 2)]
    right = z_profile[min(len(z_profile), peak_idx + 3):]
    baseline_vals = np.concatenate([left, right]) if len(left) + len(right) >= 3 else z_profile
    baseline_med = float(np.median(baseline_vals))
    baseline_mad = float(np.median(np.abs(baseline_vals - baseline_med)))
    noise = baseline_mad * 1.4826 + 1e-9
    snr = float((z_profile[peak_idx] - baseline_med) / noise)

    # Symmetry around peak
    left = z_profile[:peak_idx]
    right = z_profile[peak_idx + 1:]
    min_len = min(len(left), len(right))
    if min_len < 3:
        symmetry = None
    else:
        left_seg = left[-min_len:]
        right_seg = right[:min_len][::-1]
        l_min, l_max = float(np.min(left_seg)), float(np.max(left_seg))
        r_min, r_max = float(np.min(right_seg)), float(np.max(right_seg))
        if l_max == l_min or r_max == r_min:
            symmetry = None
        else:
            left_norm = (left_seg - l_min) / (l_max - l_min)
            right_norm = (right_seg - r_min) / (r_max - r_min)
            symmetry = float(1.0 - np.mean(np.abs(left_norm - right_norm)))
            symmetry = max(0.0, min(1.0, symmetry))
    qa_flag = None
    if symmetry is not None:
        qa_flag = (snr < min_snr) or (symmetry < min_symmetry)
    return snr, symmetry, qa_flag


def _estimate_local_background(slice_2d, x_c, y_c, inner_r, outer_r):
    """Estimate local background from an annular region around (x_c, y_c).

    Returns the median intensity in the annulus between inner_r and outer_r
    pixels from the center.  Falls back to the global minimum if the annulus
    contains fewer than 8 pixels (e.g. near image edges).
    """
    ny, nx = slice_2d.shape
    yy, xx = np.ogrid[:ny, :nx]
    dist_sq = (xx - x_c) ** 2 + (yy - y_c) ** 2
    mask = (dist_sq >= inner_r ** 2) & (dist_sq <= outer_r ** 2)
    vals = slice_2d[mask]
    if len(vals) < 8:
        return float(np.min(slice_2d))
    return float(np.median(vals))


def _parabolic_peak(values, peak_idx, offset):
    """Refine integer peak to sub-pixel using parabolic interpolation.

    Fits a parabola through the peak and its two neighbors.
    Returns the sub-pixel coordinate in the original array frame.
    """
    n = len(values)
    if peak_idx <= 0 or peak_idx >= n - 1:
        return float(offset + peak_idx)
    y_m1 = float(values[peak_idx - 1])
    y_0 = float(values[peak_idx])
    y_p1 = float(values[peak_idx + 1])
    denom = 2.0 * (y_m1 + y_p1 - 2.0 * y_0)
    if abs(denom) < 1e-12:
        return float(offset + peak_idx)
    shift = (y_m1 - y_p1) / denom
    return float(offset + peak_idx) + np.clip(shift, -0.5, 0.5)


def _weighted_centroid_2d(img_2d, x_offset, y_offset):
    """Return intensity-weighted centroid in global coordinates."""
    vals = np.asarray(img_2d, dtype=np.float32)
    if vals.size == 0:
        return float(x_offset), float(y_offset)
    w = vals - float(np.min(vals))
    w[w < 0] = 0
    w_sum = float(np.sum(w))
    if w_sum <= 1e-9:
        # Fallback: geometric center of the local patch.
        return float(x_offset + (vals.shape[1] - 1) / 2.0), float(y_offset + (vals.shape[0] - 1) / 2.0)
    yy, xx = np.indices(vals.shape, dtype=np.float32)
    cx = float(np.sum(w * xx) / w_sum) + float(x_offset)
    cy = float(np.sum(w * yy) / w_sum) + float(y_offset)
    return cx, cy


def _radial_center_2d(img_2d, x_offset, y_offset):
    """Estimate center from gradient-magnitude symmetry (ring-friendly)."""
    vals = np.asarray(img_2d, dtype=np.float32)
    if vals.size == 0:
        return float(x_offset), float(y_offset)
    sm = gaussian_filter(vals, sigma=1.0)
    gy, gx = np.gradient(sm)
    gmag = np.hypot(gx, gy).astype(np.float32, copy=False)
    return _weighted_centroid_2d(gmag, x_offset, y_offset)


def _edge_center_xy(search_vol, rel_z, x_offset, y_offset):
    """Estimate XY center from a gradient-weighted slab around rel_z."""
    z_lo = max(0, rel_z - 1)
    z_hi = min(search_vol.shape[0], rel_z + 2)
    slab = np.mean(search_vol[z_lo:z_hi], axis=0) if z_hi > z_lo else search_vol[rel_z]
    return _radial_center_2d(slab, x_offset, y_offset)


def _edge_midpoint_z(z_profile, default_z):
    """Estimate Z center from opposing edge transitions in a 1D profile."""
    prof = np.asarray(z_profile, dtype=np.float32)
    if prof.size < 7 or not np.all(np.isfinite(prof)):
        return int(default_z)
    sm = gaussian_filter1d(prof, sigma=1.0)
    dz = np.gradient(sm)
    edge_strength = np.abs(dz).astype(np.float32, copy=False)
    peak = int(np.argmax(sm))
    if peak <= 1 or peak >= edge_strength.size - 2:
        return int(default_z)
    left = int(np.argmax(edge_strength[:peak])) if peak > 0 else -1
    right_local = int(np.argmax(edge_strength[peak + 1:])) if peak + 1 < edge_strength.size else -1
    if left < 0 or right_local < 0:
        return int(default_z)
    right = right_local + peak + 1
    if right <= left + 1:
        return int(default_z)
    contrast = float(np.max(sm) - np.min(sm))
    edge_floor = max(1e-6, 0.03 * contrast)
    if edge_strength[left] < edge_floor or edge_strength[right] < edge_floor:
        return int(default_z)
    return int(np.clip(round((left + right) / 2.0), 0, prof.size - 1))


def _recenter_point(stack, x_c, y_c, half_box, center_mode='peak'):
    """Recenter point using selected strategy: peak, centroid, radial, or edge."""
    x_c_int, y_c_int = int(round(x_c)), int(round(y_c))
    y1_s = max(0, y_c_int - half_box)
    y2_s = min(stack.shape[1], y_c_int + half_box + 1)
    x1_s = max(0, x_c_int - half_box)
    x2_s = min(stack.shape[2], x_c_int + half_box + 1)
    z_prof = np.mean(stack[:, y1_s:y2_s, x1_s:x2_s], axis=(1, 2))
    approx_z = int(np.argmax(z_prof))
    z1_s = max(0, approx_z - half_box)
    z2_s = min(stack.shape[0], approx_z + half_box + 1)
    search_vol = stack[z1_s:z2_s, y1_s:y2_s, x1_s:x2_s]
    if search_vol.size == 0:
        return float(x_c_int), float(y_c_int)

    mode = str(center_mode or 'peak').lower()
    rel_z = int(np.clip(approx_z - z1_s, 0, max(0, search_vol.shape[0] - 1)))

    if mode == 'centroid':
        plane = search_vol[rel_z]
        return _weighted_centroid_2d(plane, x1_s, y1_s)

    if mode == 'radial':
        return _edge_center_xy(search_vol, rel_z, x1_s, y1_s)

    if mode == 'edge':
        sm3d = gaussian_filter(search_vol.astype(np.float32, copy=False), sigma=1.0)
        gz, gy, gx = np.gradient(sm3d)
        gmag3d = np.sqrt(gx * gx + gy * gy + gz * gz).astype(np.float32, copy=False)
        z_edge_prof = np.mean(gmag3d, axis=(1, 2))
        if z_edge_prof.size and np.all(np.isfinite(z_edge_prof)) and float(np.max(z_edge_prof) - np.min(z_edge_prof)) > 1e-6:
            rel_z_edge = int(np.argmax(z_edge_prof))
        else:
            rel_z_edge = rel_z
        return _edge_center_xy(search_vol, rel_z_edge, x1_s, y1_s)

    rel_z, rel_y, rel_x = np.unravel_index(np.argmax(search_vol), search_vol.shape)

    # Sub-pixel refinement along X and Y via parabolic interpolation
    # Extract 1D profiles through the peak voxel for each lateral axis
    x_line = search_vol[rel_z, rel_y, :]
    y_line = search_vol[rel_z, :, rel_x]
    sub_x = _parabolic_peak(x_line, rel_x, x1_s)
    sub_y = _parabolic_peak(y_line, rel_y, y1_s)
    return sub_x, sub_y


def _save_bead_diagnostic(bead_id, volume, z_profile, x_profile, y_profile,
                          scale_xy, scale_z, fwhm_result, output_dir,
                          fit_gaussian=False, fit_3d_result=None,
                          method_center_x_px=None, method_center_y_px=None,
                          method_center_z_profile_px=None, method_center_z_volume_px=None):
    """
    Generate and save a diagnostic plot for a single bead.

    Uses Agg backend so this can run from a worker thread without creating GUI
    windows or Tk icons. Shows Z/X/Y profiles with FWHM markers, XY/XZ/YZ
    projections, and optionally Gaussian fits.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    output_dir = Path(output_dir)
    diag_dir = output_dir / "bead_diagnostics"
    diag_dir.mkdir(exist_ok=True)

    fig = Figure(figsize=(14, 10))
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    z_ax = np.arange(len(z_profile)) * scale_z
    x_ax = np.arange(len(x_profile)) * scale_xy
    y_ax = np.arange(len(y_profile)) * scale_xy

    ax_z = fig.add_subplot(gs[0, 0])
    ax_z.plot(z_ax, z_profile, 'b-', lw=1.5, label='Z profile')
    if fwhm_result and 'fwhm_z_prom_um' in fwhm_result:
        fwhm_z = fwhm_result['fwhm_z_prom_um']
        peak_idx = np.argmax(z_profile)
        peaks, props = find_peaks(z_profile, prominence=0.1)
        if peaks.size:
            best = np.argmax(props['prominences'])
            half_max = z_profile[peaks[best]] - props['prominences'][best] / 2.0
        else:
            half_max = (np.max(z_profile) + np.min(z_profile)) / 2
        ax_z.axhline(half_max, color='r', ls='--', alpha=0.7, label=f'FWHM={fwhm_z:.2f}µm')
        ax_z.axvline(z_ax[peak_idx], color='g', ls=':', alpha=0.5)
    if method_center_z_profile_px is not None and len(z_profile):
        z_idx = int(np.clip(round(method_center_z_profile_px), 0, len(z_profile) - 1))
        ax_z.axvline(z_ax[z_idx], color='c', ls='-', lw=2.2, alpha=0.9, label='Method center (Z)')
    if fit_gaussian and fwhm_result and 'fwhm_z_gauss_um' in fwhm_result and fwhm_result['fwhm_z_gauss_um']:
        try:
            import warnings

            from scipy.optimize import OptimizeWarning, curve_fit
            pk = np.argmax(z_profile)
            hw = min(10, len(z_profile) // 2)
            xs = np.arange(max(0, pk - hw), min(len(z_profile), pk + hw + 1))
            ys = z_profile[xs]
            p0 = [ys.max() - ys.min(), pk, hw / 2, ys.min()]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(gaussian_func, xs, ys, p0=p0, maxfev=2000)
            fit_x = np.linspace(xs[0], xs[-1], 100)
            fit_y = gaussian_func(fit_x, *popt)
            ax_z.plot(fit_x * scale_z, fit_y, 'r-', lw=1, alpha=0.8, label='Gauss fit')
        except Exception:
            pass
    ax_z.set_xlabel('Z (µm)')
    ax_z.set_ylabel('Intensity')
    ax_z.set_title('Z Profile')
    ax_z.legend(loc='upper right', fontsize=8)

    ax_x = fig.add_subplot(gs[0, 1])
    ax_x.plot(x_ax, x_profile, 'g-', lw=1.5, label='X profile')
    if fwhm_result and 'fwhm_x_prom_um' in fwhm_result:
        fwhm_x = fwhm_result['fwhm_x_prom_um']
        peaks_x, props_x = find_peaks(x_profile, prominence=0.1)
        if peaks_x.size:
            best_x = np.argmax(props_x['prominences'])
            half_max = x_profile[peaks_x[best_x]] - props_x['prominences'][best_x] / 2.0
        else:
            half_max = (np.max(x_profile) + np.min(x_profile)) / 2
        ax_x.axhline(half_max, color='r', ls='--', alpha=0.7, label=f'FWHM={fwhm_x:.2f}µm')
    if fit_gaussian and fwhm_result and 'fwhm_x_gauss_um' in fwhm_result and fwhm_result['fwhm_x_gauss_um']:
        try:
            import warnings

            from scipy.optimize import OptimizeWarning, curve_fit
            pk = np.argmax(x_profile)
            hw = min(10, len(x_profile) // 2)
            xs = np.arange(max(0, pk - hw), min(len(x_profile), pk + hw + 1))
            ys = x_profile[xs]
            p0 = [ys.max() - ys.min(), pk, hw / 2, ys.min()]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(gaussian_func, xs, ys, p0=p0, maxfev=2000)
            fit_x = np.linspace(xs[0], xs[-1], 100)
            fit_y = gaussian_func(fit_x, *popt)
            ax_x.plot(fit_x * scale_xy, fit_y, 'r-', lw=1, alpha=0.8, label='Gauss fit')
        except Exception:
            pass
    ax_x.set_xlabel('X (µm)')
    ax_x.set_ylabel('Intensity')
    ax_x.set_title('X Profile')
    ax_x.legend(loc='upper right', fontsize=8)

    ax_y = fig.add_subplot(gs[0, 2])
    ax_y.plot(y_ax, y_profile, 'm-', lw=1.5, label='Y profile')
    if fwhm_result and 'fwhm_y_prom_um' in fwhm_result:
        fwhm_y = fwhm_result['fwhm_y_prom_um']
        peaks_y, props_y = find_peaks(y_profile, prominence=0.1)
        if peaks_y.size:
            best_y = np.argmax(props_y['prominences'])
            half_max = y_profile[peaks_y[best_y]] - props_y['prominences'][best_y] / 2.0
        else:
            half_max = (np.max(y_profile) + np.min(y_profile)) / 2
        ax_y.axhline(half_max, color='r', ls='--', alpha=0.7, label=f'FWHM={fwhm_y:.2f}µm')
    if fit_gaussian and fwhm_result and 'fwhm_y_gauss_um' in fwhm_result and fwhm_result['fwhm_y_gauss_um']:
        try:
            import warnings

            from scipy.optimize import OptimizeWarning, curve_fit
            pk = np.argmax(y_profile)
            hw = min(10, len(y_profile) // 2)
            xs = np.arange(max(0, pk - hw), min(len(y_profile), pk + hw + 1))
            ys = y_profile[xs]
            p0 = [ys.max() - ys.min(), pk, hw / 2, ys.min()]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(gaussian_func, xs, ys, p0=p0, maxfev=2000)
            fit_x = np.linspace(xs[0], xs[-1], 100)
            fit_y = gaussian_func(fit_x, *popt)
            ax_y.plot(fit_x * scale_xy, fit_y, 'r-', lw=1, alpha=0.8, label='Gauss fit')
        except Exception:
            pass
    ax_y.set_xlabel('Y (µm)')
    ax_y.set_ylabel('Intensity')
    ax_y.set_title('Y Profile')
    ax_y.legend(loc='upper right', fontsize=8)

    if volume is not None and volume.size > 0:
        nz, ny, nx = volume.shape
        z_mid, y_mid, x_mid = nz // 2, ny // 2, nx // 2
        mcx = None if method_center_x_px is None else float(np.clip(method_center_x_px, 0, max(0, nx - 1)))
        mcy = None if method_center_y_px is None else float(np.clip(method_center_y_px, 0, max(0, ny - 1)))
        mcz = None if method_center_z_volume_px is None else float(np.clip(method_center_z_volume_px, 0, max(0, nz - 1)))

        ax_xy = fig.add_subplot(gs[1, 0])
        ax_xy.imshow(volume[z_mid, :, :], cmap='viridis', aspect='equal')
        ax_xy.axhline(y_mid, color='r', ls='--', alpha=0.5)
        ax_xy.axvline(x_mid, color='r', ls='--', alpha=0.5)
        if mcx is not None and mcy is not None:
            ax_xy.axhline(mcy, color='c', lw=2.0, alpha=0.9)
            ax_xy.axvline(mcx, color='c', lw=2.0, alpha=0.9)
            ax_xy.plot(mcx, mcy, marker='+', color='c', markersize=12, markeredgewidth=2.0)
        ax_xy.set_title(f'XY (z={z_mid})')
        ax_xy.set_xlabel('X (px)')
        ax_xy.set_ylabel('Y (px)')

        ax_xz = fig.add_subplot(gs[1, 1])
        ax_xz.imshow(volume[:, y_mid, :], cmap='viridis', aspect=scale_z / scale_xy)
        ax_xz.axhline(z_mid, color='r', ls='--', alpha=0.5)
        ax_xz.axvline(x_mid, color='r', ls='--', alpha=0.5)
        if mcx is not None and mcz is not None:
            ax_xz.axhline(mcz, color='c', lw=2.0, alpha=0.9)
            ax_xz.axvline(mcx, color='c', lw=2.0, alpha=0.9)
            ax_xz.plot(mcx, mcz, marker='+', color='c', markersize=12, markeredgewidth=2.0)
        ax_xz.set_title(f'XZ (y={y_mid})')
        ax_xz.set_xlabel('X (px)')
        ax_xz.set_ylabel('Z (px)')

        ax_yz = fig.add_subplot(gs[1, 2])
        ax_yz.imshow(volume[:, :, x_mid], cmap='viridis', aspect=scale_z / scale_xy)
        ax_yz.axhline(z_mid, color='r', ls='--', alpha=0.5)
        ax_yz.axvline(y_mid, color='r', ls='--', alpha=0.5)
        if mcy is not None and mcz is not None:
            ax_yz.axhline(mcz, color='c', lw=2.0, alpha=0.9)
            ax_yz.axvline(mcy, color='c', lw=2.0, alpha=0.9)
            ax_yz.plot(mcy, mcz, marker='+', color='c', markersize=12, markeredgewidth=2.0)
        ax_yz.set_title(f'YZ (x={x_mid})')
        ax_yz.set_xlabel('Y (px)')
        ax_yz.set_ylabel('Z (px)')

    ax_info = fig.add_subplot(gs[2, :])
    ax_info.axis('off')

    info_lines = [f"Bead #{bead_id}"]
    if fwhm_result:
        for key in ['fwhm_x_prom_um', 'fwhm_y_prom_um', 'fwhm_z_prom_um']:
            if key in fwhm_result and fwhm_result[key]:
                axis = key.split('_')[1].upper()
                info_lines.append(f"FWHM-{axis} (prom): {fwhm_result[key]:.3f} µm")
        for key in ['fwhm_x_gauss_um', 'fwhm_y_gauss_um', 'fwhm_z_gauss_um']:
            if key in fwhm_result and fwhm_result[key]:
                axis = key.split('_')[1].upper()
                info_lines.append(f"FWHM-{axis} (gauss): {fwhm_result[key]:.3f} µm")
        if 'qa_z_snr' in fwhm_result and fwhm_result['qa_z_snr']:
            info_lines.append(f"QA SNR: {fwhm_result['qa_z_snr']:.1f}")
        if 'qa_z_symmetry' in fwhm_result and fwhm_result['qa_z_symmetry']:
            info_lines.append(f"QA Symmetry: {fwhm_result['qa_z_symmetry']:.2f}")
        if 'center_mode' in fwhm_result and fwhm_result['center_mode']:
            info_lines.append(f"Center mode: {fwhm_result['center_mode']}")

    if fit_3d_result:
        info_lines.append("")
        info_lines.append("3D Gaussian fit:")
        info_lines.append(f"  FWHM-X: {fit_3d_result['fwhm_x_um']:.3f} µm")
        info_lines.append(f"  FWHM-Y: {fit_3d_result['fwhm_y_um']:.3f} µm")
        info_lines.append(f"  FWHM-Z: {fit_3d_result['fwhm_z_um']:.3f} µm")
        info_lines.append(f"  Residual: {fit_3d_result['residual_norm']:.3f}")

    ax_info.text(0.02, 0.95, "\n".join(info_lines), transform=ax_info.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Figure-wide legend clarifying existing vs method-center markers.
    legend_handles = [
        Line2D([0], [0], color='r', ls='--', lw=1.8, label='Existing reference lines'),
        Line2D([0], [0], color='g', ls=':', lw=1.8, label='Existing Z-profile peak line'),
        Line2D([0], [0], color='c', ls='-', lw=2.2, marker='+', markersize=10, label='Method-based center'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=3, fontsize=8, frameon=True)

    out_path = diag_dir / f"bead_{bead_id:04d}_diagnostic.png"
    _savefig_close(fig, out_path, dpi=150, bbox_inches='tight')
    return out_path


def _filter_auxiliary_by_ids(results, bead_volumes, profiles):
    """Keep bead_volumes/profiles aligned with accepted result IDs."""
    if not results:
        return [], []
    by_id_volume = {}
    by_id_profile = {}
    for p in profiles:
        pid = p.get('id')
        if pid is not None:
            by_id_profile[pid] = p
            by_id_volume[pid] = p.get('volume', np.array([]))
    out_volumes = []
    out_profiles = []
    for r in results:
        rid = r.get('id')
        p = by_id_profile.get(rid)
        if p is not None:
            out_profiles.append(p)
            out_volumes.append(by_id_volume.get(rid, np.array([])))
    return out_volumes, out_profiles


def run_manual(stack, scale_xy, scale_z, box_size=21, line_length=5.0,
               z_smooth=1.0, detrend=False, subtract_background=False,
               fit_gaussian=False, fit_window=20, smooth_xy=None,
               vmin_pct=None, vmax_pct=None, gamma=1.0, points_file=None,
               na=None, fluorophore=None, prominence_rel=0.1,
               prominence_min=None, qa_min_snr=3.0, qa_min_symmetry=0.6,
               fit_3d=False, save_diagnostics=False, qa_auto_reject=False,
               output_dir=None, local_background=False, robust_fit=False,
               sample_fraction=100, **kwargs):
    """
    Manual mode: interactive click or load from points_file.
    Returns (results, bead_volumes, mip, profiles, rejected).
    """
    stack = _ensure_stack_3d(stack, kwargs.get('channel', 0))
    stack = stack.astype(np.float32)
    mip = np.max(stack, axis=0)
    imshow_kw = _get_display_norm(mip, gamma, vmin_pct, vmax_pct)

    if subtract_background:
        run_on_main = kwargs.get('run_on_main')
        if callable(run_on_main):
            rect_coords = run_on_main(_interactive_background_roi, mip, **imshow_kw)
        else:
            rect_coords = _interactive_background_roi(mip, **imshow_kw)
        if rect_coords:
            stack = _subtract_background(stack, rect_coords)
            mip = np.max(stack, axis=0)

    def _manual_pick_points(mip_img, imshow_kwargs):
        preview_mip, ds_factor = make_preview_image(mip_img)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(preview_mip, cmap='gray', **imshow_kwargs)
        ax.set_title("Press Escape when done, and close the window.")
        ax.text(
            0.01, 0.01, INTERACTIVE_PREVIEW_NOTE,
            transform=ax.transAxes, fontsize=9, color='white',
            ha='left', va='bottom', bbox=dict(facecolor='black', alpha=0.55, pad=4)
        )
        add_mousewheel_zoom(ax)
        add_left_drag_pan(ax)
        add_interaction_key(fig, [
            "Left-click + drag: Pan",
            "Mouse wheel: Zoom in/out",
            "Right-click: Add bead point",
            "Esc: Finish selection, then close window",
        ])
        preview_pts = detectors.get_points_manual(preview_mip, ax, fig)
        return [preview_to_full_point(x, y, ds_factor) for x, y in preview_pts]

    if points_file:
        pts = detectors.load_points_from_file(points_file)
    else:
        run_on_main = kwargs.get('run_on_main')
        if callable(run_on_main):
            pts = run_on_main(_manual_pick_points, mip, imshow_kw)
        else:
            pts = _manual_pick_points(mip, imshow_kw)

    if not pts:
        return [], [], mip, [], []

    status_cb = kwargs.get('status_callback')
    n_pts_orig = len(pts)
    if sample_fraction is not None and sample_fraction < 100:
        n_keep = max(1, int(n_pts_orig * (sample_fraction / 100)))
        rng = np.random.default_rng()
        idx = rng.choice(n_pts_orig, size=n_keep, replace=False)
        pts = [pts[i] for i in idx]
        if callable(status_cb):
            status_cb(f"Sampling {n_keep} of {n_pts_orig} beads...")
    n_pts = len(pts)
    if callable(status_cb):
        status_cb(f"Processing beads (0/{n_pts})...")
    half_box = box_size // 2
    center_mode = str(kwargs.get('center_mode', 'peak')).lower()
    if center_mode not in ('peak', 'centroid', 'radial', 'edge'):
        center_mode = 'peak'
    if callable(status_cb):
        status_cb(f"Center mode: {center_mode}")
    results = []
    bead_volumes = []
    profiles = []
    for i, (x_c_orig, y_c_orig) in enumerate(pts):
        if callable(status_cb):
            status_cb(f"Processing beads ({i + 1}/{n_pts})...")
        x_c, y_c = _recenter_point(stack, x_c_orig, y_c_orig, half_box, center_mode=center_mode)
        x_c_int, y_c_int = int(round(x_c)), int(round(y_c))
        y1, y2 = max(0, y_c_int - half_box), min(stack.shape[1], y_c_int + half_box + 1)
        x1, x2 = max(0, x_c_int - half_box), min(stack.shape[2], x_c_int + half_box + 1)
        z_profile_raw = np.mean(stack[:, y1:y2, x1:x2], axis=(1, 2))
        z_profile_sm = gaussian_filter(z_profile_raw, sigma=z_smooth) if z_smooth else z_profile_raw
        best_z = int(np.argmax(z_profile_sm))
        if center_mode in ('radial', 'edge'):
            best_z = _edge_midpoint_z(z_profile_raw, best_z)
        z_pr = signal.detrend(z_profile_raw) if detrend else z_profile_raw - np.min(z_profile_raw)
        fwhm_z = calculate_fwhm_prominence(
            z_pr, scale_z, prominence_min=prominence_min, prominence_rel=prominence_rel
        )
        if fwhm_z is None:
            continue
        fwhm_z_gauss = fit_gaussian_fwhm(
            z_pr, scale_z, fit_window, peak_hint=best_z, robust=robust_fit
        ) if fit_gaussian else None
        sl = stack[best_z]
        # Estimate local background from annulus if requested
        bg_level = 0.0
        if local_background:
            inner_r = max(3, half_box)
            outer_r = inner_r + max(3, half_box // 2)
            bg_level = _estimate_local_background(sl, x_c, y_c, inner_r, outer_r)
        line_len_px = max(5, int(line_length / scale_xy))
        if line_len_px % 2 == 0:
            line_len_px += 1
        half_line = line_len_px / 2.0
        fwhms = {}
        x_profile = None
        y_profile = None
        for axis in ['X', 'Y']:
            if axis == 'X':
                coords = np.vstack((np.full(line_len_px, y_c), np.linspace(x_c - half_line, x_c + half_line, line_len_px)))
            else:
                coords = np.vstack((np.linspace(y_c - half_line, y_c + half_line, line_len_px), np.full(line_len_px, x_c)))
            raw = map_coordinates(sl, coords, order=1)
            if detrend:
                proc = signal.detrend(raw)
            elif local_background:
                proc = raw - bg_level
                proc[proc < 0] = 0
            else:
                proc = raw - np.min(raw)
            if smooth_xy and smooth_xy > 0:
                proc = gaussian_filter(proc, sigma=smooth_xy)
                if detrend:
                    proc = signal.detrend(proc)
                elif not local_background:
                    proc = proc - np.min(proc)
            if axis == 'X':
                x_profile = proc
            else:
                y_profile = proc
            f = calculate_fwhm_prominence(
                proc, scale_xy, prominence_min=prominence_min, prominence_rel=prominence_rel
            )
            fwhms[f'fwhm_{axis.lower()}_prom_um'] = f['fwhm_um'] if f else None
            fg = fit_gaussian_fwhm(proc, scale_xy, fit_window, robust=robust_fit) if fit_gaussian else None
            fwhms[f'fwhm_{axis.lower()}_gauss_um'] = fg['fwhm_um'] if fg else None
        qa_snr, qa_sym, qa_flag = _quality_metrics(z_profile_raw, best_z, qa_min_snr, qa_min_symmetry)

        z1_f, z2_f = max(0, best_z - half_box), min(stack.shape[0], best_z + half_box + 1)
        volume = stack[z1_f:z2_f, y1:y2, x1:x2]

        fit_3d_result = None
        if fit_3d and volume.size > 0:
            fit_3d_result = fit_gaussian_3d(volume, scale_xy, scale_z, robust=robust_fit)

        if fwhms.get('fwhm_x_prom_um') and fwhms.get('fwhm_y_prom_um'):
            result = {
                'id': i + 1, 'x_coord': x_c, 'y_coord': y_c, 'z_coord': best_z,
                **fwhms,
                'fwhm_z_prom_um': fwhm_z['fwhm_um'],
                'fwhm_z_gauss_um': fwhm_z_gauss['fwhm_um'] if fwhm_z_gauss else None,
                'qa_z_snr': qa_snr,
                'qa_z_symmetry': qa_sym,
                'qa_flag': qa_flag,
            }
            if fit_3d_result:
                result['fwhm_x_3d_um'] = fit_3d_result['fwhm_x_um']
                result['fwhm_y_3d_um'] = fit_3d_result['fwhm_y_um']
                result['fwhm_z_3d_um'] = fit_3d_result['fwhm_z_um']
                result['fit_3d_residual'] = fit_3d_result['residual_norm']
            results.append(result)
            bead_volumes.append(volume)
            profiles.append({
                'id': i + 1,
                'z_profile': z_pr,
                'x_profile': x_profile,
                'y_profile': y_profile,
                'volume': volume,
                'fit_3d_result': fit_3d_result,
                'method_center_x_px': float(x_c - x1),
                'method_center_y_px': float(y_c - y1),
                'method_center_z_profile_px': float(best_z),
                'method_center_z_volume_px': float(best_z - z1_f),
            })

    rejected = []
    if callable(status_cb):
        status_cb("Preparing outputs...")
    if qa_auto_reject and results:
        results, rejected = filter_by_qa(results, qa_min_snr, qa_min_symmetry)
        bead_volumes, profiles = _filter_auxiliary_by_ids(results, bead_volumes, profiles)

    if save_diagnostics and output_dir and results:
        n_res = len(results)
        for idx, r in enumerate(results):
            if callable(status_cb) and (idx % 50 == 0 or idx == n_res - 1):
                status_cb(f"Saving diagnostics ({idx + 1}/{n_res})...")
            p = next((pr for pr in profiles if pr.get('id') == r.get('id')), None)
            if p is not None:
                fwhm_for_diag = dict(r)
                fwhm_for_diag['center_mode'] = center_mode
                _save_bead_diagnostic(
                    r['id'], p['volume'], p['z_profile'], p['x_profile'], p['y_profile'],
                    scale_xy, scale_z, fwhm_for_diag, output_dir, fit_gaussian, p.get('fit_3d_result'),
                    method_center_x_px=p.get('method_center_x_px'),
                    method_center_y_px=p.get('method_center_y_px'),
                    method_center_z_profile_px=p.get('method_center_z_profile_px'),
                    method_center_z_volume_px=p.get('method_center_z_volume_px'),
                )

    return results, bead_volumes, mip, profiles, rejected


def run_stardist(stack, scale_xy, scale_z, box_size=7, line_length=5.0,
                 z_smooth=None, detrend=False, subtract_background=False,
                 fit_gaussian=False, fit_window=20, max_z_fwhm=None,
                 points_file=None, na=None, fluorophore=None,
                 prominence_rel=0.1, prominence_min=None, review_detection=False,
                 qa_min_snr=3.0, qa_min_symmetry=0.6,
                 fit_3d=False, save_diagnostics=False, qa_auto_reject=False,
                 stardist_model='2D_versatile_fluo', stardist_prob_thresh=0.6,
                 stardist_nms_thresh=0.3, use_blob_fallback=False,
                 blob_sigma=1.2, blob_threshold_rel=0.2, blob_min_distance=5,
                 stardist_n_tiles=None,
                 use_trackpy=False, trackpy_diameter=5, trackpy_minmass=5000,
                 trackpy_separation=None,
                 detector_backend='stardist',
                 output_dir=None, local_background=False, robust_fit=False,
                 sample_fraction=100, **kwargs):
    """
    StarDist mode: automatic detection or points_file override.
    Returns (results, bead_volumes, mip, profiles, rejected).
    """
    stack = _ensure_stack_3d(stack, kwargs.get('channel', 0))
    stack = stack.astype(np.float32)
    mip = np.max(stack, axis=0)
    if subtract_background:
        run_on_main = kwargs.get('run_on_main')
        if callable(run_on_main):
            rect_coords = run_on_main(_interactive_background_roi, mip)
        else:
            rect_coords = _interactive_background_roi(mip)
        if rect_coords:
            stack = _subtract_background(stack, rect_coords)
            mip = np.max(stack, axis=0)
    if detector_backend == 'blob':
        pts = detectors.get_points_blob(
            mip,
            points_file=points_file,
            sigma=blob_sigma,
            threshold_rel=blob_threshold_rel,
            min_distance_px=blob_min_distance,
        )
    elif detector_backend == 'trackpy':
        pts = detectors.get_points_trackpy(
            mip,
            points_file=points_file,
            diameter=trackpy_diameter,
            minmass=trackpy_minmass,
            separation=trackpy_separation,
        )
    else:
        pts = detectors.get_points_stardist(
            mip,
            points_file=points_file,
            model_name=stardist_model,
            prob_thresh=stardist_prob_thresh,
            nms_thresh=stardist_nms_thresh,
            use_blob_fallback=use_blob_fallback,
            blob_sigma=blob_sigma,
            blob_threshold_rel=blob_threshold_rel,
            blob_min_distance=blob_min_distance,
            n_tiles=stardist_n_tiles,
            use_trackpy=use_trackpy,
            trackpy_diameter=trackpy_diameter,
            trackpy_minmass=trackpy_minmass,
            trackpy_separation=trackpy_separation,
        )
    if pts and review_detection:
        title = {
            'stardist': "StarDist Detection Review",
            'blob': "Blob Detection Review",
            'trackpy': "Trackpy Detection Review",
        }.get(detector_backend, "Detection Review")
        status_cb = kwargs.get('status_callback')
        if callable(status_cb):
            status_cb("Reviewing detection – press Y/N in the review window")
        if not detectors.review_detection_points(mip, pts, title=title):
            return [], [], mip, [], []
    if not pts:
        return [], [], mip, [], []
    status_cb = kwargs.get('status_callback')
    n_pts_orig = len(pts)
    if sample_fraction is not None and sample_fraction < 100:
        n_keep = max(1, int(n_pts_orig * (sample_fraction / 100)))
        rng = np.random.default_rng()
        idx = rng.choice(n_pts_orig, size=n_keep, replace=False)
        pts = [pts[i] for i in idx]
        if callable(status_cb):
            status_cb(f"Sampling {n_keep} of {n_pts_orig} beads...")
    n_pts = len(pts)
    if callable(status_cb):
        status_cb(f"Processing beads (0/{n_pts})...")
    half_box = box_size // 2
    center_mode = str(kwargs.get('center_mode', 'peak')).lower()
    if center_mode not in ('peak', 'centroid', 'radial', 'edge'):
        center_mode = 'peak'
    if callable(status_cb):
        status_cb(f"Center mode: {center_mode}")
    results = []
    bead_volumes = []
    profiles = []
    for i, (x_c_raw, y_c_raw) in enumerate(pts):
        if callable(status_cb):
            status_cb(f"Processing beads ({i + 1}/{n_pts})...")
        x_c, y_c = _recenter_point(stack, x_c_raw, y_c_raw, half_box, center_mode=center_mode)
        x_c_int, y_c_int = int(round(x_c)), int(round(y_c))
        y1, y2 = max(0, y_c_int - half_box), min(stack.shape[1], y_c_int + half_box + 1)
        x1, x2 = max(0, x_c_int - half_box), min(stack.shape[2], x_c_int + half_box + 1)
        z_raw = np.mean(stack[:, y1:y2, x1:x2], axis=(1, 2))
        z_sm = gaussian_filter(z_raw, sigma=z_smooth) if z_smooth else z_raw
        best_z = int(np.argmax(z_sm))
        z_pr = signal.detrend(z_raw) if detrend else z_raw - np.min(z_raw)
        fwhm_z = calculate_fwhm_prominence(
            z_pr, scale_z, prominence_min=prominence_min, prominence_rel=prominence_rel
        )
        if fwhm_z is None:
            continue
        if max_z_fwhm and fwhm_z['fwhm_um'] > max_z_fwhm:
            continue
        fwhm_z_gauss = fit_gaussian_fwhm(
            z_pr, scale_z, fit_window, peak_hint=best_z, robust=robust_fit
        ) if fit_gaussian else None
        sl = stack[best_z]
        bg_level = 0.0
        if local_background:
            inner_r = max(3, half_box)
            outer_r = inner_r + max(3, half_box // 2)
            bg_level = _estimate_local_background(sl, x_c, y_c, inner_r, outer_r)
        ll = max(5, int(line_length / scale_xy))
        if ll % 2 == 0:
            ll += 1
        hl = ll / 2.0
        fwhms = {}
        x_profile = None
        y_profile = None
        for axis in ['X', 'Y']:
            coords = np.vstack((np.full(ll, y_c), np.linspace(x_c - hl, x_c + hl, ll))) if axis == 'X' else np.vstack((np.linspace(y_c - hl, y_c + hl, ll), np.full(ll, x_c)))
            prof = map_coordinates(sl, coords, order=1)
            if detrend:
                pr = signal.detrend(prof)
            elif local_background:
                pr = prof - bg_level
                pr[pr < 0] = 0
            else:
                pr = prof - np.min(prof)
            if axis == 'X':
                x_profile = pr
            else:
                y_profile = pr
            f = calculate_fwhm_prominence(
                pr, scale_xy, prominence_min=prominence_min, prominence_rel=prominence_rel
            )
            fg = fit_gaussian_fwhm(pr, scale_xy, fit_window, robust=robust_fit) if fit_gaussian else None
            fwhms[f'fwhm_{axis.lower()}_prom'] = f['fwhm_um'] if f else None
            fwhms[f'fwhm_{axis.lower()}_gauss'] = fg['fwhm_um'] if fg else None
        qa_snr, qa_sym, qa_flag = _quality_metrics(z_raw, best_z, qa_min_snr, qa_min_symmetry)

        z1, z2 = max(0, best_z - half_box), min(stack.shape[0], best_z + half_box + 1)
        volume = stack[z1:z2, y1:y2, x1:x2]

        fit_3d_result = None
        if fit_3d and volume.size > 0:
            fit_3d_result = fit_gaussian_3d(volume, scale_xy, scale_z, robust=robust_fit)

        if fwhms.get('fwhm_x_prom') and fwhms.get('fwhm_y_prom'):
            result = {
                'id': i + 1, 'x_coord': x_c, 'y_coord': y_c, 'z_coord': best_z,
                'fwhm_x_prom': fwhms['fwhm_x_prom'], 'fwhm_y_prom': fwhms['fwhm_y_prom'],
                'fwhm_z_prom': fwhm_z['fwhm_um'],
                'fwhm_x_gauss': fwhms['fwhm_x_gauss'], 'fwhm_y_gauss': fwhms['fwhm_y_gauss'],
                'fwhm_z_gauss': fwhm_z_gauss['fwhm_um'] if fwhm_z_gauss else None,
                'qa_z_snr': qa_snr,
                'qa_z_symmetry': qa_sym,
                'qa_flag': qa_flag,
            }
            if fit_3d_result:
                result['fwhm_x_3d'] = fit_3d_result['fwhm_x_um']
                result['fwhm_y_3d'] = fit_3d_result['fwhm_y_um']
                result['fwhm_z_3d'] = fit_3d_result['fwhm_z_um']
                result['fit_3d_residual'] = fit_3d_result['residual_norm']
            results.append(result)
            bead_volumes.append(volume)
            profiles.append({
                'id': i + 1,
                'z_profile': z_pr,
                'x_profile': x_profile,
                'y_profile': y_profile,
                'volume': volume,
                'fit_3d_result': fit_3d_result,
                'method_center_x_px': float(x_c - x1),
                'method_center_y_px': float(y_c - y1),
                'method_center_z_profile_px': float(best_z),
                'method_center_z_volume_px': float(best_z - z1),
            })

    rejected = []
    if callable(status_cb):
        status_cb("Preparing outputs...")
    if qa_auto_reject and results:
        results, rejected = filter_by_qa(results, qa_min_snr, qa_min_symmetry)
        bead_volumes, profiles = _filter_auxiliary_by_ids(results, bead_volumes, profiles)

    if save_diagnostics and output_dir and results:
        n_res = len(results)
        for idx, r in enumerate(results):
            if callable(status_cb) and (idx % 50 == 0 or idx == n_res - 1):
                status_cb(f"Saving diagnostics ({idx + 1}/{n_res})...")
            p = next((pr for pr in profiles if pr.get('id') == r.get('id')), None)
            if p:
                fwhm_for_diag = {
                    'fwhm_x_prom_um': r.get('fwhm_x_prom'),
                    'fwhm_y_prom_um': r.get('fwhm_y_prom'),
                    'fwhm_z_prom_um': r.get('fwhm_z_prom'),
                    'fwhm_x_gauss_um': r.get('fwhm_x_gauss'),
                    'fwhm_y_gauss_um': r.get('fwhm_y_gauss'),
                    'fwhm_z_gauss_um': r.get('fwhm_z_gauss'),
                    'qa_z_snr': r.get('qa_z_snr'),
                    'qa_z_symmetry': r.get('qa_z_symmetry'),
                    'center_mode': center_mode,
                }
                _save_bead_diagnostic(
                    r['id'], p['volume'], p['z_profile'], p['x_profile'], p['y_profile'],
                    scale_xy, scale_z, fwhm_for_diag, output_dir, fit_gaussian, p.get('fit_3d_result'),
                    method_center_x_px=p.get('method_center_x_px'),
                    method_center_y_px=p.get('method_center_y_px'),
                    method_center_z_profile_px=p.get('method_center_z_profile_px'),
                    method_center_z_volume_px=p.get('method_center_z_volume_px'),
                )

    return results, bead_volumes, mip, profiles, rejected


def run_blob(*args, **kwargs):
    """Blob detector mode: classical Gaussian smooth + local maxima."""
    kwargs['detector_backend'] = 'blob'
    # Blob mode should not silently invoke StarDist fallback logic.
    kwargs['use_blob_fallback'] = False
    kwargs['use_trackpy'] = False
    return run_stardist(*args, **kwargs)


def run_trackpy(*args, **kwargs):
    """trackpy detector mode: bandpass + feature centroid detection."""
    kwargs['detector_backend'] = 'trackpy'
    kwargs['use_blob_fallback'] = False
    kwargs['use_trackpy'] = True
    return run_stardist(*args, **kwargs)


def run_cellpose(stack, scale_xy, scale_z, model_path, box_size=15, line_length=5.0,
                 z_smooth=None, detrend=False, subtract_background=False,
                 fit_gaussian=False, fit_window=20, max_z_fwhm=None,
                 cellpose_diameter=None, detection_gauss_sigma=None, cellpose_cellprob=0.0,
                 cellpose_do_3d=False, anisotropy=None,
                 cellpose_min_size=3, cellpose_flow_threshold=0.4,
                 z_range=None, z_analysis_margin=20, reject_outliers=None,
                 num_beads_avg=20, na=None, fluorophore=None,
                 prominence_rel=0.1, prominence_min=None,
                 skip_review=False, qa_min_snr=3.0, qa_min_symmetry=0.6,
                 fit_3d=False, save_diagnostics=False, qa_auto_reject=False,
                 output_dir=None, local_background=False, robust_fit=False,
                 sample_fraction=100, **kwargs):
    """
    Cellpose mode: custom model, tiled inference, review step.
    Returns (results, bead_volumes, mip, bead_log, profiles, rejected).
    """
    stack = _ensure_stack_3d(stack, kwargs.get('channel', 0))
    stack = stack.astype(np.float32)
    mip = np.max(stack, axis=0)
    if subtract_background:
        run_on_main = kwargs.get('run_on_main')
        norm = PowerNorm(gamma=kwargs.get('gamma', 1.0)) if kwargs.get('gamma', 1.0) != 1 else None
        roi_kw = {'norm': norm} if norm is not None else {}
        if callable(run_on_main):
            rect_coords = run_on_main(_interactive_background_roi, mip, **roi_kw)
        else:
            rect_coords = _interactive_background_roi(mip, **roi_kw)
        if rect_coords:
            stack = _subtract_background(stack, rect_coords)
            mip = np.max(stack, axis=0)
    if cellpose_do_3d:
        pts, masks = detectors.get_points_cellpose_3d(
            stack,
            model_path,
            diameter=cellpose_diameter,
            cellprob_threshold=cellpose_cellprob,
            anisotropy=anisotropy,
            min_size=cellpose_min_size,
            flow_threshold=cellpose_flow_threshold,
        )
    else:
        pts, masks = detectors.get_points_cellpose(
            mip, model_path, diameter=cellpose_diameter,
            gauss_sigma=detection_gauss_sigma, cellprob_threshold=cellpose_cellprob,
            min_size=cellpose_min_size, flow_threshold=cellpose_flow_threshold,
        )
    if not pts:
        return [], [], mip, [], [], []
    status_cb = kwargs.get('status_callback')
    review_mask = np.max(masks, axis=0) if (masks is not None and getattr(masks, "ndim", 0) == 3) else masks
    if not skip_review:
        if callable(status_cb):
            status_cb("Reviewing detection – press Y/N in the review window")
    if not skip_review and not detectors.review_detection_cellpose(mip, review_mask):
        return [], [], mip, [], [], []
    n_pts_orig = len(pts)
    if sample_fraction is not None and sample_fraction < 100:
        n_keep = max(1, int(n_pts_orig * (sample_fraction / 100)))
        rng = np.random.default_rng()
        idx = rng.choice(n_pts_orig, size=n_keep, replace=False)
        pts = [pts[i] for i in idx]
        if callable(status_cb):
            status_cb(f"Sampling {n_keep} of {n_pts_orig} beads...")
    img_h, img_w, z_d = stack.shape[1], stack.shape[2], stack.shape[0]
    z_search_min, z_search_max = (z_range[0], min(z_range[1], z_d)) if z_range else (0, z_d)
    half_box = box_size // 2
    center_mode = str(kwargs.get('center_mode', 'peak')).lower()
    if center_mode not in ('peak', 'centroid', 'radial', 'edge'):
        center_mode = 'peak'
    if callable(status_cb):
        status_cb(f"Center mode: {center_mode}")
    results = []
    bead_volumes = []
    bead_log = []
    profiles = []
    n_pts = len(pts)
    if callable(status_cb):
        status_cb(f"Processing beads (0/{n_pts})...")
    for i, p in enumerate(pts):
        if callable(status_cb):
            status_cb(f"Processing beads ({i + 1}/{n_pts})...")
        if len(p) == 3:
            x_c, y_c, z_hint = p
            z_hint = int(round(z_hint))
        else:
            x_c, y_c = p
            z_hint = None
        x_c, y_c = _recenter_point(stack, x_c, y_c, half_box, center_mode=center_mode)
        log_entry = {'id': i + 1, 'x_coord': x_c, 'y_coord': y_c, 'status': 'rejected', 'reason': ''}
        bead_volumes.append(np.array([]))
        x_i, y_i = int(round(x_c)), int(round(y_c))
        y1, y2 = max(0, y_i - half_box), min(img_h, y_i + half_box + 1)
        x1, x2 = max(0, x_i - half_box), min(img_w, x_i + half_box + 1)
        z_raw = np.mean(stack[:, y1:y2, x1:x2], axis=(1, 2))
        z_sm = gaussian_filter(z_raw, sigma=z_smooth) if z_smooth else z_raw
        prof_in_range = z_sm[z_search_min:z_search_max]
        if prof_in_range.size == 0:
            log_entry['reason'] = 'Z-range empty'
            bead_log.append(log_entry)
            continue
        peak_idx = int(np.argmax(prof_in_range))
        best_z = z_search_min + peak_idx
        if z_hint is not None:
            best_z = int(np.clip(z_hint, z_search_min, max(z_search_min, z_search_max - 1)))
        z_sub_start = max(0, best_z - z_analysis_margin)
        z_sub_end = min(z_d, best_z + z_analysis_margin + 1)
        z_for_analysis = z_raw[z_sub_start:z_sub_end]
        peak_idx_sub = best_z - z_sub_start
        z_pr = signal.detrend(z_for_analysis) if detrend else z_for_analysis - np.min(z_for_analysis)
        fwhm_z = calculate_fwhm_prominence(
            z_pr, scale_z, prominence_min=prominence_min, prominence_rel=prominence_rel
        )
        if fwhm_z is None:
            log_entry['reason'] = 'Z-FWHM failed'
            bead_log.append(log_entry)
            continue
        if max_z_fwhm and fwhm_z['fwhm_um'] > max_z_fwhm:
            log_entry['reason'] = f"Z-FWHM > {max_z_fwhm}"
            bead_log.append(log_entry)
            continue
        fwhm_z_gauss = fit_gaussian_fwhm(z_pr, scale_z, fit_window, peak_hint=peak_idx_sub, robust=robust_fit) if fit_gaussian else None
        sl = stack[best_z]
        bg_level = 0.0
        if local_background:
            inner_r = max(3, half_box)
            outer_r = inner_r + max(3, half_box // 2)
            bg_level = _estimate_local_background(sl, x_c, y_c, inner_r, outer_r)
        ll = max(5, int(line_length / scale_xy))
        if ll % 2 == 0:
            ll += 1
        hl = ll / 2.0
        fwhms = {}
        failed = False
        x_profile = None
        y_profile = None
        for axis in ['X', 'Y']:
            coords = np.vstack((np.full(ll, y_c), np.linspace(x_c - hl, x_c + hl, ll))) if axis == 'X' else np.vstack((np.linspace(y_c - hl, y_c + hl, ll), np.full(ll, x_c)))
            prof = map_coordinates(sl, coords, order=1)
            if detrend:
                pr = signal.detrend(prof)
            elif local_background:
                pr = prof - bg_level
                pr[pr < 0] = 0
            else:
                pr = prof - np.min(prof)
            if axis == 'X':
                x_profile = pr
            else:
                y_profile = pr
            f = calculate_fwhm_prominence(
                pr, scale_xy, prominence_min=prominence_min, prominence_rel=prominence_rel
            )
            fg = fit_gaussian_fwhm(pr, scale_xy, fit_window, robust=robust_fit) if fit_gaussian else None
            fwhms[f'fwhm_{axis.lower()}_prom'] = f['fwhm_um'] if f else None
            fwhms[f'fwhm_{axis.lower()}_gauss'] = fg['fwhm_um'] if fg else None
            if f is None:
                log_entry['reason'] = f'{axis}-FWHM failed'
                failed = True
                break
        if failed:
            bead_log.append(log_entry)
            continue
        log_entry['status'] = 'accepted'
        bead_log.append(log_entry)
        w3, h3 = img_w / 3.0, img_h / 3.0
        row, col = int(y_c // h3), int(x_c // w3)
        qa_snr, qa_sym, qa_flag = _quality_metrics(z_raw, best_z, qa_min_snr, qa_min_symmetry)

        zr = box_size
        z1, z2 = max(0, best_z - zr), min(z_d, best_z + zr + 1)
        volume = stack[z1:z2, y1:y2, x1:x2]
        bead_volumes[-1] = volume

        fit_3d_result = None
        if fit_3d and volume.size > 0:
            fit_3d_result = fit_gaussian_3d(volume, scale_xy, scale_z, robust=robust_fit)

        result = {
            'id': i + 1, 'x_coord': x_c, 'y_coord': y_c, 'z_coord': best_z, 'district': (row, col),
            'fwhm_x_prom': fwhms['fwhm_x_prom'], 'fwhm_y_prom': fwhms['fwhm_y_prom'], 'fwhm_z_prom': fwhm_z['fwhm_um'],
            'fwhm_x_gauss': fwhms['fwhm_x_gauss'], 'fwhm_y_gauss': fwhms['fwhm_y_gauss'], 'fwhm_z_gauss': fwhm_z_gauss['fwhm_um'] if fwhm_z_gauss else None,
            'qa_z_snr': qa_snr,
            'qa_z_symmetry': qa_sym,
            'qa_flag': qa_flag,
        }
        if fit_3d_result:
            result['fwhm_x_3d'] = fit_3d_result['fwhm_x_um']
            result['fwhm_y_3d'] = fit_3d_result['fwhm_y_um']
            result['fwhm_z_3d'] = fit_3d_result['fwhm_z_um']
            result['fit_3d_residual'] = fit_3d_result['residual_norm']
        results.append(result)
        profiles.append({
            'id': i + 1,
            'z_profile': z_pr,
            'x_profile': x_profile,
            'y_profile': y_profile,
            'volume': volume,
            'fit_3d_result': fit_3d_result,
            'method_center_x_px': float(x_c - x1),
            'method_center_y_px': float(y_c - y1),
            'method_center_z_profile_px': float(peak_idx_sub),
            'method_center_z_volume_px': float(best_z - z1),
        })

    if callable(status_cb):
        status_cb("Preparing outputs...")
    if reject_outliers and results:
        accepted_before = {r['id'] for r in results}
        results = reject_outliers_mad(results, data_key='fwhm_z_gauss', m=reject_outliers)
        rejected_ids = accepted_before - {r['id'] for r in results}
        for le in bead_log:
            if le['id'] in rejected_ids:
                le['status'] = 'rejected'
                le['reason'] = f"Outlier (MAD > {reject_outliers})"
        bead_volumes, profiles = _filter_auxiliary_by_ids(results, bead_volumes, profiles)

    rejected = []
    if qa_auto_reject and results:
        results, rejected = filter_by_qa(results, qa_min_snr, qa_min_symmetry)
        for r in rejected:
            for le in bead_log:
                if le['id'] == r['id']:
                    le['status'] = 'rejected'
                    le['reason'] = r.get('qa_reject_reason', 'QA failed')
        bead_volumes, profiles = _filter_auxiliary_by_ids(results, bead_volumes, profiles)

    if save_diagnostics and output_dir and results:
        n_res = len(results)
        for idx, r in enumerate(results):
            if callable(status_cb) and (idx % 50 == 0 or idx == n_res - 1):
                status_cb(f"Saving diagnostics ({idx + 1}/{n_res})...")
            p = next((pr for pr in profiles if pr['id'] == r['id']), None)
            if p:
                fwhm_for_diag = {
                    'fwhm_x_prom_um': r.get('fwhm_x_prom'),
                    'fwhm_y_prom_um': r.get('fwhm_y_prom'),
                    'fwhm_z_prom_um': r.get('fwhm_z_prom'),
                    'fwhm_x_gauss_um': r.get('fwhm_x_gauss'),
                    'fwhm_y_gauss_um': r.get('fwhm_y_gauss'),
                    'fwhm_z_gauss_um': r.get('fwhm_z_gauss'),
                    'qa_z_snr': r.get('qa_z_snr'),
                    'qa_z_symmetry': r.get('qa_z_symmetry'),
                    'center_mode': center_mode,
                }
                _save_bead_diagnostic(
                    r['id'], p['volume'], p['z_profile'], p['x_profile'], p['y_profile'],
                    scale_xy, scale_z, fwhm_for_diag, output_dir, fit_gaussian, p.get('fit_3d_result'),
                    method_center_x_px=p.get('method_center_x_px'),
                    method_center_y_px=p.get('method_center_y_px'),
                    method_center_z_profile_px=p.get('method_center_z_profile_px'),
                    method_center_z_volume_px=p.get('method_center_z_volume_px'),
                )

    return results, bead_volumes, mip, bead_log, profiles, rejected


def write_outputs(results, bead_volumes, mip, file_path, mode, scale_xy, scale_z,
                  upsample_factor=4, no_plots=False, num_beads_avg=20,
                  bead_log=None, na=None, fluorophore=None, gamma=1.0,
                  qa_min_snr=3.0, qa_min_symmetry=0.6, rejected=None,
                  stack=None,
                  profiles=None, center_mode='peak', run_settings=None):
    """Write CSV, TXT summary, average bead, heatmap, detection overview, and summary figure."""
    file_path = Path(file_path)
    extra_meta = []
    if na is not None:
        extra_meta.append(f"NA: {na}")
    if fluorophore:
        extra_meta.append(f"Fluorophore: {fluorophore}")
    extra_meta.append(f"Center mode: {center_mode}")

    # Save a human-readable run settings report alongside other outputs.
    settings_txt_path = file_path.with_name(f"{file_path.stem}_run_settings.txt")
    settings_lines = [
        "=" * 60,
        "--- Bead Analyzer Run Settings ---",
        "=" * 60,
        f"Source: {file_path.name}",
    ]
    settings_dict = dict(run_settings) if isinstance(run_settings, dict) else {}
    if not settings_dict:
        settings_dict = {
            'mode': mode,
            'scale_xy': scale_xy,
            'scale_z': scale_z,
            'gamma': gamma,
            'num_beads_avg': num_beads_avg,
            'qa_min_snr': qa_min_snr,
            'qa_min_symmetry': qa_min_symmetry,
            'center_mode': center_mode,
        }
    settings_dict.setdefault('mode', mode)
    settings_dict.setdefault('scale_xy', scale_xy)
    settings_dict.setdefault('scale_z', scale_z)
    settings_dict.setdefault('center_mode', center_mode)
    for k in sorted(settings_dict):
        settings_lines.append(f"{k}: {settings_dict[k]}")
    with open(settings_txt_path, 'w') as f:
        f.write("\n".join(settings_lines))
    print(f"Settings saved to: {settings_txt_path}")

    if mode == 'cellpose' and bead_log:
        log_df = pd.DataFrame(bead_log)
        log_path = file_path.with_name(f"{file_path.stem}_every_bead_log.csv")
        log_df.to_csv(log_path, index=False, float_format='%.2f')
        print(f"Bead log saved to: {log_path}")

    if rejected:
        rej_df = pd.DataFrame(rejected)
        rej_path = file_path.with_name(f"{file_path.stem}_rejected_beads.csv")
        rej_df.to_csv(rej_path, index=False, float_format='%.4f')
        print(f"Rejected beads saved to: {rej_path}")

    if not no_plots and mip is not None:
        try:
            _save_detection_overview(results, rejected, mip, file_path, gamma=gamma)
        except Exception as e:
            print(f"WARNING: Detection overview could not be saved: {e}")
        if stack is not None:
            try:
                _save_mip_views(stack, file_path, scale_xy, scale_z)
            except Exception as e:
                print(f"WARNING: MIP views figure could not be saved: {e}")

    if not results:
        print("No beads passed analysis. No reports written.")
        return

    df = pd.DataFrame(results)

    # Compute district coordinates for all modes (for heatmap)
    if 'district' in df.columns:
        df['district_row'] = df['district'].apply(lambda x: x[0])
        df['district_col'] = df['district'].apply(lambda x: x[1])
        df = df.drop(columns=['district'], errors='ignore')
    elif 'x_coord' in df.columns and 'y_coord' in df.columns and mip is not None:
        img_h, img_w = mip.shape[:2]
        h3, w3 = img_h / 3.0, img_w / 3.0
        df['district_row'] = (df['y_coord'] // h3).astype(int).clip(0, 2)
        df['district_col'] = (df['x_coord'] // w3).astype(int).clip(0, 2)

    csv_path = file_path.with_name(f"{file_path.stem}_FWHM_data.csv")
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"CSV saved to: {csv_path}")

    lines = ["=" * 60, "--- Bead Analyzer Summary Report ---", "=" * 60,
             f"Source: {file_path.name}", f"Total beads: {len(df)}"]
    if rejected:
        lines.append(f"QA rejected: {len(rejected)}")
    if extra_meta:
        lines.append("\n".join(extra_meta))
    for method, key_suffix in [('Prominence', 'prom'), ('Gaussian', 'gauss'), ('3D Gaussian', '3d')]:
        method_lines = []
        for axis in ['x', 'y', 'z']:
            candidates = [
                f'fwhm_{axis}_{key_suffix}_um',
                f'fwhm_{axis}_{key_suffix}',
            ]
            key = next((k for k in candidates if k in df.columns), None)
            if key and df[key].notna().any():
                vals = df[key].dropna()
                std_text = f"{vals.std():.3f}" if len(vals) > 1 else "N/A"
                method_lines.append(f"  Avg FWHM-{axis.upper()}: {vals.mean():.3f} ± {std_text} µm (n={len(vals)})")
        if method_lines:
            lines.append(f"\n--- {method} ---")
            lines.extend(method_lines)
    if 'fit_3d_residual' in df.columns and df['fit_3d_residual'].notna().any():
        resid = df['fit_3d_residual'].dropna()
        resid_std_text = f"{resid.std():.3f}" if len(resid) > 1 else "N/A"
        lines.append(f"\n3D Fit residual: {resid.mean():.3f} ± {resid_std_text}")
    if 'qa_z_snr' in df.columns or 'qa_z_symmetry' in df.columns:
        qa_df = df[['id']].copy()
        if 'qa_z_snr' in df.columns:
            qa_df['qa_z_snr'] = df['qa_z_snr']
        if 'qa_z_symmetry' in df.columns:
            qa_df['qa_z_symmetry'] = df['qa_z_symmetry']
        if 'qa_flag' in df.columns:
            qa_df['qa_flag'] = df['qa_flag']
        else:
            qa_df['qa_flag'] = (qa_df.get('qa_z_snr', 0) < qa_min_snr) | (qa_df.get('qa_z_symmetry', 1) < qa_min_symmetry)
        qa_path = file_path.with_name(f"{file_path.stem}_bead_quality.csv")
        qa_df.to_csv(qa_path, index=False)
        lines.append(f"\nQA: {qa_df['qa_flag'].sum()} beads flagged (snr<{qa_min_snr} or symmetry<{qa_min_symmetry}).")
    txt_path = file_path.with_name(f"{file_path.stem}_FWHM_summary.txt")
    with open(txt_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Summary saved to: {txt_path}")

    # --- Average bead: unified selection for all modes ---
    valid_vols = [v for v in bead_volumes if v is not None and getattr(v, "size", 0) > 0]
    fwhm_col = next((k for k in ['fwhm_z_prom_um', 'fwhm_z_prom'] if k in df.columns), None)
    if valid_vols and fwhm_col and df[fwhm_col].notna().any():
        if num_beads_avg > 0 and num_beads_avg < len(df):
            median_fwhm = df[fwhm_col].median()
            df_sel = df.copy()
            df_sel['_dist'] = (df_sel[fwhm_col] - median_fwhm).abs()
            sel = df_sel.nsmallest(num_beads_avg, '_dist')
            sel_ids = sel['id'].tolist()
        else:
            sel_ids = df['id'].tolist()
        volume_by_id = {
            int(r.get('id')): v
            for r, v in zip(results, bead_volumes)
            if v is not None and getattr(v, "size", 0) > 0
        }
        sel_vols = [volume_by_id[sid] for sid in sel_ids if sid in volume_by_id]
        if sel_vols:
            valid_vols = sel_vols
    avg_vol = None
    if valid_vols and not no_plots:
        try:
            avg_vol = _save_average_bead(valid_vols, file_path, scale_xy, scale_z, upsample_factor)
        except Exception as e:
            print(f"WARNING: Average bead outputs could not be saved: {e}")
            avg_vol = None

    # --- Heatmap: all modes ---
    fwhm_heatmap_col = next((k for k in ['fwhm_z_gauss_um', 'fwhm_z_gauss'] if k in df.columns), None)
    if fwhm_heatmap_col and 'district_row' in df.columns and not no_plots:
        try:
            _save_heatmap(results, mip, file_path, df, gamma=gamma, fwhm_col=fwhm_heatmap_col)
        except Exception as e:
            print(f"WARNING: Heatmap could not be saved: {e}")

    # --- Summary figure ---
    if avg_vol is not None and not no_plots and profiles:
        try:
            _save_summary_figure(avg_vol, profiles, file_path, scale_xy, scale_z, upsample_factor)
        except Exception as e:
            print(f"WARNING: Summary figure could not be saved: {e}")


def _save_average_bead(volumes, file_path, scale_xy, scale_z, upsample_factor=4):
    """Save upsampled average bead stack and plot. Returns the average volume."""
    centered = []
    for vol in volumes:
        up = zoom(vol, upsample_factor, order=3)
        peak = np.unravel_index(np.argmax(up), up.shape)
        center = np.array(up.shape) / 2.0
        centered.append(shift(up, center - peak, order=3, mode='constant', cval=0))
    max_d = np.max([v.shape for v in centered], axis=0)
    padded = [np.pad(v, [(0, max_d[i] - v.shape[i]) for i in range(3)], mode='constant', constant_values=0) for v in centered]
    avg = np.mean(padded, axis=0).astype(np.float32)
    new_xy = scale_xy / upsample_factor
    new_z = scale_z / upsample_factor
    out_tif = file_path.with_name(f"{file_path.stem}_average_bead_stack.tif")
    tifffile.imwrite(out_tif, avg, imagej=True, resolution=(1 / new_xy, 1 / new_xy), metadata={'unit': 'um', 'spacing': new_z})
    print(f"Average bead saved to: {out_tif}")
    fig = _figure_agg((15, 5))
    axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
    z_m, y_m, x_m = (d // 2 for d in avg.shape)
    axes[0].imshow(avg[z_m, :, :], cmap='viridis', aspect=1)
    axes[1].imshow(avg[:, y_m, :], cmap='viridis', aspect=new_z / new_xy)
    axes[2].imshow(avg[:, :, x_m], cmap='viridis', aspect=new_z / new_xy)
    axes[0].set_title('XY'); axes[1].set_title('XZ'); axes[2].set_title('YZ')
    fig.tight_layout()
    out_png = file_path.with_name(f"{file_path.stem}_average_bead_plot.png")
    _savefig_close(fig, out_png, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {out_png}")
    return avg


def _save_heatmap(results, mip, file_path, df, gamma=1.0, fwhm_col='fwhm_z_gauss'):
    """Save district heatmap for any mode."""
    if 'district_row' not in df.columns:
        return
    if fwhm_col not in df.columns or not df[fwhm_col].notna().any():
        return
    stats = df.groupby(['district_row', 'district_col'])[fwhm_col].agg(['mean', 'std', 'count']).reset_index()
    heatmap = np.full((3, 3), np.nan)
    for _, row in stats.iterrows():
        heatmap[int(row['district_row']), int(row['district_col'])] = row['mean']
    fig = _figure_agg((7, 6))
    ax = fig.add_subplot(111)
    valid = heatmap[~np.isnan(heatmap)]
    if len(valid) > 0:
        im = ax.imshow(heatmap, cmap='viridis', interpolation='nearest',
                       vmin=np.nanmin(valid), vmax=np.nanmax(valid))
        fig.colorbar(im, ax=ax, label='Mean FWHM (µm)')
    for _, row in stats.iterrows():
        r, c = int(row['district_row']), int(row['district_col'])
        ax.text(c, r,
                f"{row['mean']:.2f} ± {row['std']:.2f}\n(n={row['count']})",
                color='white', ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5))
    ax.set_title('District Mean FWHM Heatmap')
    ax.set_xlabel('District column')
    ax.set_ylabel('District row')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    out_path = file_path.with_name(f"{file_path.stem}_FWHM_heatmap.png")
    _savefig_close(fig, out_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {out_path}")


def _save_detection_overview(results, rejected, mip, file_path, gamma=1.0):
    """Save MIP with detected bead locations marked: green=accepted, red=rejected."""
    from matplotlib.colors import PowerNorm
    fig = _figure_agg((12, 10))
    ax = fig.add_subplot(111)
    norm = PowerNorm(gamma=gamma) if gamma != 1 else None
    ax.imshow(mip, cmap='gray', norm=norm)
    if results:
        xs = [r['x_coord'] for r in results]
        ys = [r['y_coord'] for r in results]
        ax.scatter(xs, ys, s=60, facecolors='none', edgecolors='lime', linewidths=1.5, label=f'Accepted ({len(results)})')
        for r in results:
            ax.annotate(str(r['id']), (r['x_coord'], r['y_coord']),
                        color='lime', fontsize=6, xytext=(4, 4), textcoords='offset points')
    if rejected:
        rxs = [r['x_coord'] for r in rejected if 'x_coord' in r]
        rys = [r['y_coord'] for r in rejected if 'y_coord' in r]
        if rxs:
            ax.scatter(rxs, rys, s=60, facecolors='none', edgecolors='red', linewidths=1.5, label=f'Rejected ({len(rxs)})')
            for r in rejected:
                if 'x_coord' in r:
                    ax.annotate(str(r.get('id', '')), (r['x_coord'], r['y_coord']),
                                color='red', fontsize=6, xytext=(4, 4), textcoords='offset points')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Detection Overview')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    out_path = file_path.with_name(f"{file_path.stem}_detection_overview.png")
    _savefig_close(fig, out_path, dpi=300, bbox_inches='tight')
    print(f"Detection overview saved to: {out_path}")


def _save_mip_views(stack, file_path, scale_xy, scale_z, xy_gamma=0.65):
    """Save XY/XZ/YZ maximum intensity projections in one physically scaled figure."""
    from matplotlib.colors import PowerNorm
    if stack is None or getattr(stack, "ndim", 0) != 3:
        return
    xy_mip = np.max(stack, axis=0)
    xz_mip = np.max(stack, axis=1)
    yz_mip = np.max(stack, axis=2)

    z_um = stack.shape[0] * scale_z
    y_um = stack.shape[1] * scale_xy
    x_um = stack.shape[2] * scale_xy

    fig = _figure_agg((16, 5))
    ax_xy = fig.add_subplot(1, 3, 1)
    ax_xz = fig.add_subplot(1, 3, 2)
    ax_yz = fig.add_subplot(1, 3, 3)

    norm_xy = PowerNorm(gamma=xy_gamma) if xy_gamma != 1 else None
    ax_xy.imshow(xy_mip, cmap='gray', extent=[0, x_um, y_um, 0], aspect='equal', norm=norm_xy)
    ax_xy.set_title('XY MIP')
    ax_xy.set_xlabel('X (µm)')
    ax_xy.set_ylabel('Y (µm)')

    ax_xz.imshow(xz_mip, cmap='gray', extent=[0, x_um, z_um, 0], aspect='equal')
    ax_xz.set_title('XZ MIP')
    ax_xz.set_xlabel('X (µm)')
    ax_xz.set_ylabel('Z (µm)')

    ax_yz.imshow(yz_mip, cmap='gray', extent=[0, y_um, z_um, 0], aspect='equal')
    ax_yz.set_title('YZ MIP')
    ax_yz.set_xlabel('Y (µm)')
    ax_yz.set_ylabel('Z (µm)')

    fig.tight_layout()
    out_path = file_path.with_name(f"{file_path.stem}_MIP_views.png")
    _savefig_close(fig, out_path, dpi=300, bbox_inches='tight')
    print(f"MIP views saved to: {out_path}")


def _add_scalebar(ax, scale_um_per_px, img_size_px, orientation='horizontal', color='white'):
    """Draw a scale bar on an axes. Picks a round bar length in µm."""
    fov_um = img_size_px * scale_um_per_px
    target = fov_um * 0.2
    candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    bar_um = min(candidates, key=lambda c: abs(c - target))
    bar_px = bar_um / scale_um_per_px
    margin = img_size_px * 0.05
    if orientation == 'horizontal':
        x0 = margin
        y0 = img_size_px - margin
        ax.plot([x0, x0 + bar_px], [y0, y0], color=color, lw=3)
        ax.text(x0 + bar_px / 2, y0 - margin * 0.5, f'{bar_um} µm',
                color=color, ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        x0 = margin
        y0 = img_size_px - margin
        ax.plot([x0, x0], [y0, y0 - bar_px], color=color, lw=3)
        ax.text(x0 + margin * 0.5, y0 - bar_px / 2, f'{bar_um} µm',
                color=color, ha='left', va='center', fontsize=8, fontweight='bold', rotation=90)


def _save_summary_figure(avg_vol, profiles, file_path, scale_xy, scale_z, upsample_factor=4):
    """Publication-quality summary: avg bead projections with scale bars + mean profiles with fits."""
    if avg_vol is None or not profiles:
        return

    new_xy = scale_xy / upsample_factor
    new_z = scale_z / upsample_factor
    nz, ny, nx = avg_vol.shape
    z_m, y_m, x_m = nz // 2, ny // 2, nx // 2

    fig = _figure_agg((16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Top row: projections with scale bars
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xy.imshow(avg_vol[z_m, :, :], cmap='inferno', aspect=1)
    ax_xy.set_title('Average Bead: XY')
    ax_xy.axis('off')
    _add_scalebar(ax_xy, new_xy, nx)

    ax_xz = fig.add_subplot(gs[0, 1])
    ax_xz.imshow(avg_vol[:, y_m, :], cmap='inferno', aspect=new_z / new_xy)
    ax_xz.set_title('Average Bead: XZ')
    ax_xz.axis('off')
    _add_scalebar(ax_xz, new_xy, nx)

    ax_yz = fig.add_subplot(gs[0, 2])
    ax_yz.imshow(avg_vol[:, :, x_m], cmap='inferno', aspect=new_z / new_xy)
    ax_yz.set_title('Average Bead: YZ')
    ax_yz.axis('off')
    _add_scalebar(ax_yz, new_xy, ny)

    # Bottom row: mean profiles with std shading and Gaussian fits
    profile_data = {'Z': [], 'X': [], 'Y': []}
    for p in profiles:
        if p.get('z_profile') is not None:
            profile_data['Z'].append(p['z_profile'])
        if p.get('x_profile') is not None:
            profile_data['X'].append(p['x_profile'])
        if p.get('y_profile') is not None:
            profile_data['Y'].append(p['y_profile'])

    axis_configs = [
        ('Z', scale_z, gs[1, 0], 'b'),
        ('X', scale_xy, gs[1, 1], 'g'),
        ('Y', scale_xy, gs[1, 2], 'm'),
    ]

    for axis_name, scale, gs_pos, color in axis_configs:
        ax = fig.add_subplot(gs_pos)
        profs = profile_data[axis_name]
        if not profs:
            ax.set_title(f'{axis_name} Profile (no data)')
            continue
        max_len = max(len(p) for p in profs)
        padded = np.full((len(profs), max_len), np.nan)
        for i, p in enumerate(profs):
            offset = (max_len - len(p)) // 2
            padded[i, offset:offset + len(p)] = p
        mean_prof = np.nanmean(padded, axis=0)
        std_prof = np.nanstd(padded, axis=0)
        x_um = np.arange(max_len) * scale

        ax.fill_between(x_um, mean_prof - std_prof, mean_prof + std_prof, alpha=0.25, color=color)
        ax.plot(x_um, mean_prof, color=color, lw=2, label=f'Mean (n={len(profs)})')

        valid = ~np.isnan(mean_prof)
        if np.sum(valid) >= 4:
            try:
                xs_fit = np.where(valid)[0]
                ys_fit = mean_prof[valid]
                pk = np.argmax(ys_fit)
                p0 = [ys_fit.max() - ys_fit.min(), xs_fit[pk], len(xs_fit) / 4, ys_fit.min()]
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(gaussian_func, xs_fit, ys_fit, p0=p0, maxfev=3000)
                fwhm_val = 2 * np.sqrt(2 * np.log(2)) * abs(popt[2]) * scale
                fit_xs = np.linspace(xs_fit[0], xs_fit[-1], 200)
                ax.plot(fit_xs * scale, gaussian_func(fit_xs, *popt), 'r-', lw=1.5, alpha=0.8,
                        label=f'Fit FWHM={fwhm_val:.2f} µm')
            except Exception:
                pass

        ax.set_xlabel(f'{axis_name} (µm)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title(f'{axis_name} Profile')
        ax.legend(loc='upper right', fontsize=8)

    out_path = file_path.with_name(f"{file_path.stem}_summary_figure.png")
    _savefig_close(fig, out_path, dpi=300, bbox_inches='tight')
    print(f"Summary figure saved to: {out_path}")
