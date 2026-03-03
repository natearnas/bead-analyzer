"""
Unified FWHM analysis pipeline for manual, StarDist, and Cellpose modes.
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import map_coordinates, zoom, shift, gaussian_filter
from scipy import signal
import tifffile
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from .core import (
    RectangleDrawer,
    calculate_fwhm_prominence,
    fit_gaussian_fwhm,
    fit_gaussian_3d,
    reject_outliers_mad,
    filter_by_qa,
    gaussian_func,
)
from . import detectors


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


def _recenter_point(stack, x_c, y_c, half_box):
    """Recenter point to local 3D maximum with sub-pixel refinement.

    First finds the integer voxel peak, then refines X, Y to sub-pixel
    precision using parabolic interpolation so that downstream
    map_coordinates calls are not limited by pixel-snapping error.
    """
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
                          fit_gaussian=False, fit_3d_result=None):
    """
    Generate and save a diagnostic plot for a single bead.
    
    Shows Z/X/Y profiles with FWHM markers, XY/XZ/YZ projections,
    and optionally Gaussian fits.
    """
    output_dir = Path(output_dir)
    diag_dir = output_dir / "bead_diagnostics"
    diag_dir.mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    z_ax = np.arange(len(z_profile)) * scale_z
    x_ax = np.arange(len(x_profile)) * scale_xy
    y_ax = np.arange(len(y_profile)) * scale_xy
    
    ax_z = fig.add_subplot(gs[0, 0])
    ax_z.plot(z_ax, z_profile, 'b-', lw=1.5, label='Z profile')
    if fwhm_result and 'fwhm_z_prom_um' in fwhm_result:
        fwhm_z = fwhm_result['fwhm_z_prom_um']
        peak_idx = np.argmax(z_profile)
        half_max = (np.max(z_profile) + np.min(z_profile)) / 2
        ax_z.axhline(half_max, color='r', ls='--', alpha=0.7, label=f'FWHM={fwhm_z:.2f}µm')
        ax_z.axvline(z_ax[peak_idx], color='g', ls=':', alpha=0.5)
    if fit_gaussian and fwhm_result and 'fwhm_z_gauss_um' in fwhm_result and fwhm_result['fwhm_z_gauss_um']:
        try:
            from scipy.optimize import curve_fit
            pk = np.argmax(z_profile)
            hw = min(10, len(z_profile) // 2)
            xs = np.arange(max(0, pk - hw), min(len(z_profile), pk + hw + 1))
            ys = z_profile[xs]
            p0 = [ys.max() - ys.min(), pk, hw / 2, ys.min()]
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
        half_max = (np.max(x_profile) + np.min(x_profile)) / 2
        ax_x.axhline(half_max, color='r', ls='--', alpha=0.7, label=f'FWHM={fwhm_x:.2f}µm')
    ax_x.set_xlabel('X (µm)')
    ax_x.set_ylabel('Intensity')
    ax_x.set_title('X Profile')
    ax_x.legend(loc='upper right', fontsize=8)
    
    ax_y = fig.add_subplot(gs[0, 2])
    ax_y.plot(y_ax, y_profile, 'm-', lw=1.5, label='Y profile')
    if fwhm_result and 'fwhm_y_prom_um' in fwhm_result:
        fwhm_y = fwhm_result['fwhm_y_prom_um']
        half_max = (np.max(y_profile) + np.min(y_profile)) / 2
        ax_y.axhline(half_max, color='r', ls='--', alpha=0.7, label=f'FWHM={fwhm_y:.2f}µm')
    ax_y.set_xlabel('Y (µm)')
    ax_y.set_ylabel('Intensity')
    ax_y.set_title('Y Profile')
    ax_y.legend(loc='upper right', fontsize=8)
    
    if volume is not None and volume.size > 0:
        nz, ny, nx = volume.shape
        z_mid, y_mid, x_mid = nz // 2, ny // 2, nx // 2
        
        ax_xy = fig.add_subplot(gs[1, 0])
        ax_xy.imshow(volume[z_mid, :, :], cmap='viridis', aspect='equal')
        ax_xy.axhline(y_mid, color='r', ls='--', alpha=0.5)
        ax_xy.axvline(x_mid, color='r', ls='--', alpha=0.5)
        ax_xy.set_title(f'XY (z={z_mid})')
        ax_xy.set_xlabel('X (px)')
        ax_xy.set_ylabel('Y (px)')
        
        ax_xz = fig.add_subplot(gs[1, 1])
        ax_xz.imshow(volume[:, y_mid, :], cmap='viridis', aspect=scale_z / scale_xy)
        ax_xz.axhline(z_mid, color='r', ls='--', alpha=0.5)
        ax_xz.axvline(x_mid, color='r', ls='--', alpha=0.5)
        ax_xz.set_title(f'XZ (y={y_mid})')
        ax_xz.set_xlabel('X (px)')
        ax_xz.set_ylabel('Z (px)')
        
        ax_yz = fig.add_subplot(gs[1, 2])
        ax_yz.imshow(volume[:, :, x_mid], cmap='viridis', aspect=scale_z / scale_xy)
        ax_yz.axhline(z_mid, color='r', ls='--', alpha=0.5)
        ax_yz.axvline(y_mid, color='r', ls='--', alpha=0.5)
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
    
    out_path = diag_dir / f"bead_{bead_id:04d}_diagnostic.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
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
               **kwargs):
    """
    Manual mode: interactive click or load from points_file.
    Returns (results, bead_volumes, mip, profiles, rejected).
    """
    stack = _ensure_stack_3d(stack, kwargs.get('channel', 0))
    stack = stack.astype(np.float32)
    mip = np.max(stack, axis=0)
    imshow_kw = _get_display_norm(mip, gamma, vmin_pct, vmax_pct)

    if subtract_background:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(mip, cmap='gray', aspect='equal', **imshow_kw)
        ax.set_title("Draw rectangle over background, then close")
        rd = RectangleDrawer(ax)
        plt.show()
        if rd.rect_coords:
            stack = _subtract_background(stack, rd.rect_coords)
            mip = np.max(stack, axis=0)

    if points_file:
        pts = detectors.load_points_from_file(points_file)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(mip, cmap='gray', **imshow_kw)
        ax.set_title("Right-click on beads. Press Escape when done.")
        pts = detectors.get_points_manual(mip, ax, fig)

    if not pts:
        return [], [], mip, [], []

    half_box = box_size // 2
    results = []
    bead_volumes = []
    profiles = []
    for i, (x_c_orig, y_c_orig) in enumerate(pts):
        x_c, y_c = _recenter_point(stack, x_c_orig, y_c_orig, half_box)
        x_c_int, y_c_int = int(round(x_c)), int(round(y_c))
        y1, y2 = max(0, y_c_int - half_box), min(stack.shape[1], y_c_int + half_box + 1)
        x1, x2 = max(0, x_c_int - half_box), min(stack.shape[2], x_c_int + half_box + 1)
        z_profile_raw = np.mean(stack[:, y1:y2, x1:x2], axis=(1, 2))
        z_profile_sm = gaussian_filter(z_profile_raw, sigma=z_smooth) if z_smooth else z_profile_raw
        best_z = int(np.argmax(z_profile_sm))
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
            })
    
    rejected = []
    if qa_auto_reject and results:
        results, rejected = filter_by_qa(results, qa_min_snr, qa_min_symmetry)
        bead_volumes, profiles = _filter_auxiliary_by_ids(results, bead_volumes, profiles)
    
    if save_diagnostics and output_dir and results:
        for r in results:
            p = next((pr for pr in profiles if pr.get('id') == r.get('id')), None)
            if p is not None:
                _save_bead_diagnostic(
                    r['id'], p['volume'], p['z_profile'], p['x_profile'], p['y_profile'],
                    scale_xy, scale_z, r, output_dir, fit_gaussian, p.get('fit_3d_result')
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
                 output_dir=None, local_background=False, robust_fit=False,
                 **kwargs):
    """
    StarDist mode: automatic detection or points_file override.
    Returns (results, bead_volumes, mip, profiles, rejected).
    """
    stack = _ensure_stack_3d(stack, kwargs.get('channel', 0))
    stack = stack.astype(np.float32)
    mip = np.max(stack, axis=0)
    if subtract_background:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(mip, cmap='gray')
        ax.set_title("Draw rectangle over background, then close")
        rd = RectangleDrawer(ax)
        plt.show()
        if rd.rect_coords:
            stack = _subtract_background(stack, rd.rect_coords)
            mip = np.max(stack, axis=0)
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
        if not detectors.review_detection_points(mip, pts, title="StarDist Detection Review"):
            return [], [], mip, [], []
    if not pts:
        return [], [], mip, [], []
    half_box = box_size // 2
    results = []
    bead_volumes = []
    profiles = []
    for i, (x_c_raw, y_c_raw) in enumerate(pts):
        x_c, y_c = _recenter_point(stack, x_c_raw, y_c_raw, half_box)
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
            })

    rejected = []
    if qa_auto_reject and results:
        results, rejected = filter_by_qa(results, qa_min_snr, qa_min_symmetry)
        bead_volumes, profiles = _filter_auxiliary_by_ids(results, bead_volumes, profiles)

    if save_diagnostics and output_dir and results:
        for r in results:
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
                }
                _save_bead_diagnostic(
                    r['id'], p['volume'], p['z_profile'], p['x_profile'], p['y_profile'],
                    scale_xy, scale_z, fwhm_for_diag, output_dir, fit_gaussian, p.get('fit_3d_result')
                )

    return results, bead_volumes, mip, profiles, rejected


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
                 **kwargs):
    """
    Cellpose mode: custom model, tiled inference, review step.
    Returns (results, bead_volumes, mip, bead_log, profiles, rejected).
    """
    stack = _ensure_stack_3d(stack, kwargs.get('channel', 0))
    stack = stack.astype(np.float32)
    mip = np.max(stack, axis=0)
    if subtract_background:
        fig, ax = plt.subplots(figsize=(10, 8))
        norm = PowerNorm(gamma=kwargs.get('gamma', 1.0)) if kwargs.get('gamma', 1.0) != 1 else None
        ax.imshow(mip, cmap='gray', aspect='equal', norm=norm)
        ax.set_title("Draw rectangle over background, then close")
        rd = RectangleDrawer(ax)
        plt.show()
        if rd.rect_coords:
            stack = _subtract_background(stack, rd.rect_coords)
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
    review_mask = np.max(masks, axis=0) if (masks is not None and getattr(masks, "ndim", 0) == 3) else masks
    if not skip_review and not detectors.review_detection_cellpose(mip, review_mask):
        return [], [], mip, [], [], []
    img_h, img_w, z_d = stack.shape[1], stack.shape[2], stack.shape[0]
    z_search_min, z_search_max = (z_range[0], min(z_range[1], z_d)) if z_range else (0, z_d)
    half_box = box_size // 2
    results = []
    bead_volumes = []
    bead_log = []
    profiles = []
    for i, p in enumerate(pts):
        if len(p) == 3:
            x_c, y_c, z_hint = p
            z_hint = int(round(z_hint))
        else:
            x_c, y_c = p
            z_hint = None
        x_c, y_c = _recenter_point(stack, x_c, y_c, half_box)
        log_entry = {'id': i + 1, 'x_coord': x_c, 'y_coord': y_c, 'status': 'rejected', 'reason': ''}
        bead_volumes.append(np.array([]))
        x_i, y_i = int(round(x_c)), int(round(y_c))
        x_c_int, y_c_int = x_i, y_i
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
        })
    
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
        for r in results:
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
                }
                _save_bead_diagnostic(
                    r['id'], p['volume'], p['z_profile'], p['x_profile'], p['y_profile'],
                    scale_xy, scale_z, fwhm_for_diag, output_dir, fit_gaussian, p.get('fit_3d_result')
                )
    
    return results, bead_volumes, mip, bead_log, profiles, rejected


def write_outputs(results, bead_volumes, mip, file_path, mode, scale_xy, scale_z,
                  upsample_factor=4, no_plots=False, num_beads_avg=20,
                  bead_log=None, na=None, fluorophore=None, gamma=1.0,
                  qa_min_snr=3.0, qa_min_symmetry=0.6, rejected=None):
    """Write CSV, TXT summary, average bead, and (for Cellpose) heatmap and bead log."""
    file_path = Path(file_path)
    extra_meta = []
    if na is not None:
        extra_meta.append(f"NA: {na}")
    if fluorophore:
        extra_meta.append(f"Fluorophore: {fluorophore}")

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

    if not results:
        print("No beads passed analysis. No reports written.")
        return

    df = pd.DataFrame(results)
    if mode == 'cellpose' and 'district' in df.columns:
        df['district_row'] = df['district'].apply(lambda x: x[0])
        df['district_col'] = df['district'].apply(lambda x: x[1])
        df = df.drop(columns=['district'], errors='ignore')
        csv_path = file_path.with_name(f"{file_path.stem}_FWHM_districts.csv")
    else:
        csv_path = file_path.with_name(f"{file_path.stem}_FWHM_data.csv")
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"CSV saved to: {csv_path}")

    lines = ["=" * 60, "--- 3D FWHM Summary Report ---", "=" * 60,
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
                method_lines.append(f"  Avg FWHM-{axis.upper()}: {vals.mean():.3f} ± {vals.std():.3f} µm (n={len(vals)})")
        if method_lines:
            lines.append(f"\n--- {method} ---")
            lines.extend(method_lines)
    if 'fit_3d_residual' in df.columns and df['fit_3d_residual'].notna().any():
        resid = df['fit_3d_residual'].dropna()
        lines.append(f"\n3D Fit residual: {resid.mean():.3f} ± {resid.std():.3f}")
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

    valid_vols = [v for v in bead_volumes if v.size > 0]
    if valid_vols and mode == 'cellpose' and 'fwhm_z_prom' in df.columns:
        median_fwhm = df['fwhm_z_prom'].median()
        df = df.copy()
        df['_dist'] = (df['fwhm_z_prom'] - median_fwhm).abs()
        n_sel = min(num_beads_avg, len(df))
        sel = df.nsmallest(n_sel, '_dist')
        sel_ids = sel['id'].tolist()
        volume_by_id = {
            int(r.get('id')): v
            for r, v in zip(results, bead_volumes)
            if v is not None and getattr(v, "size", 0) > 0
        }
        sel_vols = [volume_by_id[sid] for sid in sel_ids if sid in volume_by_id]
        valid_vols = sel_vols
    if valid_vols and not no_plots:
        _save_average_bead(valid_vols, file_path, scale_xy, scale_z, upsample_factor)
    if mode == 'cellpose' and 'fwhm_z_gauss' in df.columns and not no_plots:
        _save_heatmap(results, mip, file_path, df, gamma=gamma)


def _save_average_bead(volumes, file_path, scale_xy, scale_z, upsample_factor=4):
    """Save upsampled average bead stack and plot."""
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    z_m, y_m, x_m = (d // 2 for d in avg.shape)
    axes[0].imshow(avg[z_m, :, :], cmap='viridis', aspect=1)
    axes[1].imshow(avg[:, y_m, :], cmap='viridis', aspect=new_z / new_xy)
    axes[2].imshow(avg[:, :, x_m], cmap='viridis', aspect=new_z / new_xy)
    axes[0].set_title('XY'); axes[1].set_title('XZ'); axes[2].set_title('YZ')
    plt.tight_layout()
    out_png = file_path.with_name(f"{file_path.stem}_average_bead_plot.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {out_png}")


def _save_heatmap(results, mip, file_path, df, gamma=1.0):
    """Save district heatmap (Cellpose)."""
    if 'district_row' not in df.columns:
        return
    from matplotlib.colors import PowerNorm
    stats = df.groupby(['district_row', 'district_col'])['fwhm_z_gauss'].agg(['mean', 'std', 'count']).reset_index()
    heatmap = np.full((3, 3), np.nan)
    for _, row in stats.iterrows():
        heatmap[int(row['district_row']), int(row['district_col'])] = row['mean']
    fig, ax = plt.subplots(figsize=(12, 10))
    norm = PowerNorm(gamma=gamma) if gamma != 1 else None
    ax.imshow(mip, cmap='gray', norm=norm)
    valid = heatmap[~np.isnan(heatmap)]
    if len(valid) > 0:
        im = ax.imshow(heatmap, cmap='viridis', alpha=0.5, vmin=np.nanmin(valid), vmax=np.nanmax(valid))
        fig.colorbar(im, ax=ax, label='Mean FWHM (µm)')
    for _, row in stats.iterrows():
        r, c = int(row['district_row']), int(row['district_col'])
        ax.text(c * mip.shape[1] / 3 + mip.shape[1] / 6, r * mip.shape[0] / 3 + mip.shape[0] / 6,
                f"{row['mean']:.2f} ± {row['std']:.2f}\n(n={row['count']})",
                color='white', ha='center', va='center', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
    out_path = file_path.with_name(f"{file_path.stem}_FWHM_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to: {out_path}")
