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
Shared FWHM calculation logic and matplotlib-based UI helpers.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from matplotlib.widgets import RectangleSelector


class RectangleDrawer:
    """Helper class to draw a rectangle on a matplotlib axes (right-click and drag)."""

    def __init__(self, ax):
        self.ax = ax
        self.rect_coords = None
        self.rs = RectangleSelector(
            ax, self.onselect,
            button=[3], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )

    def onselect(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.rect_coords = (x1, y1, x2, y2)


class MultiPointClicker:
    """Helper class to collect right-click points on a matplotlib axes. Press Escape when done."""

    def __init__(self, ax):
        self.ax = ax
        self.points = []
        self.cid_mouse = ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = ax.figure.canvas.mpl_connect('key_press_event', self.on_key)
        print("\n--- Bead Selection ---")
        print("Right-click on the center of each bead you want to analyze.")
        print("Press the Escape key when you are finished.")

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 3:
            self.points.append((event.xdata, event.ydata))
            self.ax.plot(event.xdata, event.ydata, 'm+', markersize=12, markeredgewidth=2)
            self.ax.figure.canvas.draw()

    def on_key(self, event):
        if event.key == 'escape':
            print(f"\nFinished. {len(self.points)} points collected. Close the figure to continue.")
            self.ax.figure.canvas.mpl_disconnect(self.cid_mouse)
            self.ax.figure.canvas.mpl_disconnect(self.cid_key)


def calculate_fwhm_prominence(profile, scale_factor=1.0, prominence_min=None, prominence_rel=None):
    """
    Calculate FWHM using prominence method (half-max at peak - prominence/2).
    Returns dict with fwhm_um, fwhm_px, prominence, or None if failed.
    """
    if profile is None or len(profile) < 5:
        return None
    try:
        if prominence_rel is not None:
            amp = float(np.max(profile) - np.min(profile))
            if amp <= 0:
                return None
            prominence_min = amp * float(prominence_rel)
        if prominence_min is None:
            prominence_min = 0.1
        peaks, props = find_peaks(profile, prominence=prominence_min)
        if not peaks.size:
            return None
        i = np.argmax(props['prominences'])
        pk = peaks[i]
        h = profile[pk] - props['prominences'][i] / 2.0
        above = np.where(profile > h)[0]
        if len(above) < 2:
            return None
        l_idx, r_idx = above[0], above[-1]
        if l_idx == 0 or r_idx >= len(profile) - 1:
            return None

        def cross(a, b):
            if a < 0 or b < 0 or a >= len(profile) or b >= len(profile):
                return float(a) if 0 <= a < len(profile) else float(b)
            v0, v1 = profile[a], profile[b]
            return a + (h - v0) * (b - a) / (v1 - v0) if v0 != v1 else float(a)

        left_crossing = cross(l_idx - 1, l_idx)
        right_crossing = cross(r_idx, r_idx + 1)
        fwhm_px = right_crossing - left_crossing
        fwhm_um = fwhm_px * scale_factor
        return {
            'fwhm_um': fwhm_um,
            'fwhm_px': fwhm_px,
            'prominence': props['prominences'][i],
        }
    except (RuntimeWarning, ValueError):
        return None


def gaussian_func(x, A, mu, sigma, C):
    """Gaussian: A * exp(-0.5*((x-mu)/sigma)^2) + C"""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + C


def fit_gaussian_fwhm(profile, scale_factor=1.0, window_size=20, peak_hint=None,
                      robust=False):
    """
    Fit a Gaussian to the profile and return FWHM in µm.

    Parameters
    ----------
    profile : array-like
        1-D intensity profile.
    scale_factor : float
        Pixel-to-µm conversion factor.
    window_size : int
        Window around the peak used for fitting.
    peak_hint : int or None
        Index hint for the peak location.
    robust : bool
        If True, use bounded Trust-Region Reflective with soft-L1 loss
        (Huber-like) to downweight outliers from clipped edges or nearby beads.

    Returns
    -------
    dict or None
        Dictionary with fwhm_um, or None if fitting failed.
    """
    if profile is None or len(profile) < 4:
        return None
    try:
        pk = peak_hint if peak_hint is not None else np.argmax(profile)
        hw = window_size // 2
        xs = np.arange(max(0, pk - hw), min(len(profile), pk + hw + 1))
        ys = profile[xs]
        if len(ys) < 4:
            return None
        A_init = float(ys.max() - ys.min())
        if A_init <= 0:
            return None
        sigma_init = max(0.8, window_size / 4.0)
        p0 = [A_init, float(pk), sigma_init, float(ys.min())]

        if robust:
            bounds_lo = [0.0, float(xs[0]), 0.3, float(ys.min()) - 0.5 * A_init]
            bounds_hi = [A_init * 5.0, float(xs[-1]), float(len(profile)), float(ys.max())]
            popt, _ = curve_fit(
                gaussian_func, xs, ys, p0=p0,
                bounds=(bounds_lo, bounds_hi),
                method='trf', loss='soft_l1', maxfev=5000,
            )
        else:
            bounds_lo = [0.0, float(xs[0]), 0.3, float(ys.min()) - 0.5 * A_init]
            bounds_hi = [A_init * 5.0, float(xs[-1]), float(len(profile)), float(ys.max())]
            popt, _ = curve_fit(
                gaussian_func, xs, ys, p0=p0,
                bounds=(bounds_lo, bounds_hi),
                maxfev=5000,
            )

        sigma_fit = abs(popt[2])
        if sigma_fit < 0.25 or not np.isfinite(sigma_fit):
            return None
        fwhm_px = 2 * np.sqrt(2 * np.log(2)) * sigma_fit
        fwhm_um = fwhm_px * scale_factor
        return {'fwhm_um': fwhm_um}
    except Exception:
        return None


def reject_outliers_mad(results, data_key='fwhm_z_gauss', m=3.0):
    """Reject outliers using Median Absolute Deviation. Returns filtered list."""
    if not results:
        return []
    data = np.array([r[data_key] for r in results if r.get(data_key) is not None])
    if len(data) < 5:
        return results
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    if mad == 0:
        return results
    z = 0.6745 * np.abs(data - med) / mad
    out, idx = [], 0
    for r in results:
        if r.get(data_key) is not None:
            if z[idx] < m:
                out.append(r)
            idx += 1
        else:
            out.append(r)
    num = len(results) - len(out)
    if num > 0:
        print(f"\nRejected {num} outliers based on {m}×MAD for '{data_key}'.")
    return out


def gaussian_3d(coords, A, x0, y0, z0, sigma_x, sigma_y, sigma_z, C):
    """3D Gaussian function for curve fitting."""
    z, y, x = coords
    return A * np.exp(
        -0.5 * (((x - x0) / sigma_x) ** 2 +
                ((y - y0) / sigma_y) ** 2 +
                ((z - z0) / sigma_z) ** 2)
    ) + C


def fit_gaussian_3d(volume, scale_xy, scale_z, max_iter=2000, robust=False):
    """
    Fit a 3D Gaussian to a bead volume.

    Parameters
    ----------
    volume : ndarray
        3D array (Z, Y, X) containing the bead
    scale_xy : float
        XY pixel size in µm
    scale_z : float
        Z pixel size in µm
    max_iter : int
        Maximum iterations for curve_fit
    robust : bool
        If True, use soft-L1 loss (Huber-like) to downweight outliers
        from clipped edges or asymmetric contamination.

    Returns
    -------
    dict or None
        Dictionary with fwhm_x_um, fwhm_y_um, fwhm_z_um, fit_params, residual_norm
        or None if fitting failed.
    """
    if volume is None or volume.size < 27:
        return None
    try:
        nz, ny, nx = volume.shape
        if min(nx, ny, nz) < 3:
            return None
        z_coords, y_coords, x_coords = np.meshgrid(
            np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij'
        )
        coords = np.vstack([z_coords.ravel(), y_coords.ravel(), x_coords.ravel()])
        data = volume.ravel().astype(np.float64)
        
        peak_idx = np.unravel_index(np.argmax(volume), volume.shape)
        z0_init, y0_init, x0_init = peak_idx
        d_min = float(np.min(data))
        d_max = float(np.max(data))
        A_init = float(d_max - d_min)
        if A_init <= 0 or not np.isfinite(A_init):
            return None
        C_init = d_min

        # Use data moments for less brittle initial widths.
        w = np.clip(volume.astype(np.float64) - d_min, 0, None)
        w_sum = float(np.sum(w))
        if w_sum > 0:
            z_m = float(np.sum(w * z_coords) / w_sum)
            y_m = float(np.sum(w * y_coords) / w_sum)
            x_m = float(np.sum(w * x_coords) / w_sum)
            sz_init = float(np.sqrt(np.sum(w * (z_coords - z_m) ** 2) / w_sum))
            sy_init = float(np.sqrt(np.sum(w * (y_coords - y_m) ** 2) / w_sum))
            sx_init = float(np.sqrt(np.sum(w * (x_coords - x_m) ** 2) / w_sum))
        else:
            sx_init = sy_init = sz_init = min(nx, ny, nz) / 4.0
        sx_init = max(0.8, min(sx_init, nx / 2.0))
        sy_init = max(0.8, min(sy_init, ny / 2.0))
        sz_init = max(0.8, min(sz_init, nz / 2.0))
        
        p0 = [A_init, x0_init, y0_init, z0_init, sx_init, sy_init, sz_init, C_init]
        
        bounds_lower = [0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, d_min - 0.5 * A_init]
        bounds_upper = [max(1.0, A_init * 5.0), float(nx - 1), float(ny - 1), float(nz - 1),
                        float(nx), float(ny), float(nz), d_max]
        
        fit_kwargs = dict(
            bounds=(bounds_lower, bounds_upper),
            maxfev=max_iter,
            method='trf',
        )
        if robust:
            fit_kwargs['loss'] = 'soft_l1'
        popt, _ = curve_fit(
            gaussian_3d, coords, data, p0=p0, **fit_kwargs
        )
        
        A, x0, y0, z0, sigma_x, sigma_y, sigma_z, C = popt
        
        fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
        fwhm_x_um = abs(sigma_x) * fwhm_factor * scale_xy
        fwhm_y_um = abs(sigma_y) * fwhm_factor * scale_xy
        fwhm_z_um = abs(sigma_z) * fwhm_factor * scale_z
        
        fitted = gaussian_3d(coords, *popt)
        rmse = float(np.sqrt(np.mean((data - fitted) ** 2)))
        residual = rmse / A_init if A_init > 0 else 1.0
        # Guard against unstable fits that can pass numerical optimization.
        if (not np.isfinite(residual)) or residual > 0.75:
            return None
        if any((s <= 0.25 or not np.isfinite(s)) for s in [sigma_x, sigma_y, sigma_z]):
            return None
        
        return {
            'fwhm_x_um': fwhm_x_um,
            'fwhm_y_um': fwhm_y_um,
            'fwhm_z_um': fwhm_z_um,
            'fit_params': {
                'A': A, 'x0': x0, 'y0': y0, 'z0': z0,
                'sigma_x': sigma_x, 'sigma_y': sigma_y, 'sigma_z': sigma_z, 'C': C
            },
            'residual_norm': residual,
        }
    except Exception:
        return None


def filter_by_qa(results, min_snr=3.0, min_symmetry=0.6, snr_key='qa_z_snr', sym_key='qa_z_symmetry'):
    """
    Filter results by QA metrics (SNR and symmetry).
    
    Parameters
    ----------
    results : list of dict
        Bead measurement results
    min_snr : float
        Minimum acceptable SNR
    min_symmetry : float
        Minimum acceptable symmetry (0-1)
    snr_key : str
        Key for SNR in result dict
    sym_key : str
        Key for symmetry in result dict
        
    Returns
    -------
    tuple
        (accepted_results, rejected_results)
    """
    if not results:
        return [], []
    
    accepted = []
    rejected = []
    
    for r in results:
        snr = r.get(snr_key)
        sym = r.get(sym_key)
        
        reject = False
        reason = []
        
        if snr is not None and snr < min_snr:
            reject = True
            reason.append(f"SNR={snr:.1f}<{min_snr}")
        if sym is not None and sym < min_symmetry:
            reject = True
            reason.append(f"sym={sym:.2f}<{min_symmetry}")
        
        if reject:
            r_copy = r.copy()
            r_copy['qa_reject_reason'] = "; ".join(reason)
            rejected.append(r_copy)
        else:
            accepted.append(r)
    
    if rejected:
        print(f"\nQA filter: rejected {len(rejected)} beads, kept {len(accepted)}.")
    
    return accepted, rejected
