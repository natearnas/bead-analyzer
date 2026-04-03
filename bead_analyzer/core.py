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

import textwrap

import numpy as np
from matplotlib.widgets import RectangleSelector
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

INTERACTIVE_PREVIEW_NOTE = (
    "The lower resolution image used here is to facilitate speed; "
    "full resolution is used for analyses."
)


def add_interaction_key(fig, key_lines):
    """Render a compact controls key on the right side of an interactive figure."""
    fig.subplots_adjust(right=0.70)
    key_ax = fig.add_axes([0.72, 0.14, 0.26, 0.72])
    key_ax.axis("off")
    key_ax.set_title("Controls", fontsize=10, pad=6)
    wrapped = []
    for line in key_lines:
        parts = textwrap.wrap(str(line), width=34, break_long_words=False)
        wrapped.extend(parts if parts else [""])
    key_ax.text(
        0.0, 1.0, "\n".join(wrapped),
        transform=key_ax.transAxes,
        va="top", ha="left", fontsize=9, wrap=True,
    )


def get_preview_downsample_factor(image_2d, target_max_dim=1400):
    """Pick an integer downsample factor for interactive display."""
    if image_2d is None or getattr(image_2d, "ndim", 0) != 2:
        return 1
    max_dim = int(max(image_2d.shape))
    if max_dim <= target_max_dim:
        return 1
    return max(1, int(np.ceil(max_dim / float(target_max_dim))))


def make_preview_image(image_2d, downsample_factor=None):
    """Return (preview_image, downsample_factor) using stride sampling."""
    if image_2d is None or getattr(image_2d, "ndim", 0) != 2:
        return image_2d, 1
    f = int(downsample_factor or get_preview_downsample_factor(image_2d))
    if f <= 1:
        return image_2d, 1
    return image_2d[::f, ::f], f


def preview_to_full_point(x, y, factor):
    """Map preview-space point to full-resolution coordinates."""
    f = max(1, int(factor))
    return float(x) * f, float(y) * f


def full_to_preview_point(x, y, factor):
    """Map full-resolution point to preview-space coordinates."""
    f = max(1, int(factor))
    return float(x) / f, float(y) / f


def preview_rect_to_full(rect_coords, factor, full_shape):
    """Map preview rectangle (x1,y1,x2,y2) to full-resolution bounds."""
    if not rect_coords:
        return None
    h, w = int(full_shape[0]), int(full_shape[1])
    f = max(1, int(factor))
    x1, y1, x2, y2 = rect_coords
    x1f = int(np.clip(round(min(x1, x2) * f), 0, w - 1))
    x2f = int(np.clip(round(max(x1, x2) * f), 0, w - 1))
    y1f = int(np.clip(round(min(y1, y2) * f), 0, h - 1))
    y2f = int(np.clip(round(max(y1, y2) * f), 0, h - 1))
    if x2f <= x1f:
        x2f = min(w - 1, x1f + 1)
    if y2f <= y1f:
        y2f = min(h - 1, y1f + 1)
    return x1f, y1f, x2f, y2f


class RectangleDrawer:
    """Helper class to draw a rectangle on a matplotlib axes (right-click and drag)."""

    def __init__(self, ax):
        self.ax = ax
        self.rect_coords = None
        self.rs = RectangleSelector(
            ax, self.onselect,
            button=[3], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True, useblit=True
        )

    def onselect(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.rect_coords = (x1, y1, x2, y2)


def add_mousewheel_zoom(ax, extent=None):
    """Connect scroll event to zoom in/out centered on the mouse cursor. Clamps to image extent."""
    if extent is None and ax.images:
        extent = ax.images[0].get_extent()
    if extent is None:
        return

    left, right, bottom, top = extent
    x_min, x_max = min(left, right), max(left, right)
    y_min, y_max = min(bottom, top), max(bottom, top)

    x_total = max(1e-9, x_max - x_min)
    y_total = max(1e-9, y_max - y_min)
    min_span_ratio = 0.02
    min_x_span = max(4.0, x_total * min_span_ratio)
    min_y_span = max(4.0, y_total * min_span_ratio)
    zoom_factor = 1.2

    def on_scroll(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        # event.step: positive = scroll up = zoom in; event.button: 'up' = zoom in
        zoom_in = getattr(event, 'step', 0) > 0 or getattr(event, 'button', None) == 'up'
        x, y = event.xdata, event.ydata
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_desc = xlim[0] > xlim[1]
        y_desc = ylim[0] > ylim[1]

        x0, x1 = min(xlim), max(xlim)
        y0, y1 = min(ylim), max(ylim)
        x_span = max(1e-9, x1 - x0)
        y_span = max(1e-9, y1 - y0)

        if zoom_in:
            new_x_span = max(min_x_span, x_span / zoom_factor)
            new_y_span = max(min_y_span, y_span / zoom_factor)
        else:
            new_x_span = min(x_total, x_span * zoom_factor)
            new_y_span = min(y_total, y_span * zoom_factor)

        frac_x = 0.5 if x_span <= 0 else np.clip((x - x0) / x_span, 0.0, 1.0)
        frac_y = 0.5 if y_span <= 0 else np.clip((y - y0) / y_span, 0.0, 1.0)
        new_x0 = x - frac_x * new_x_span
        new_y0 = y - frac_y * new_y_span

        new_x0 = np.clip(new_x0, x_min, x_max - new_x_span)
        new_y0 = np.clip(new_y0, y_min, y_max - new_y_span)
        new_x1 = new_x0 + new_x_span
        new_y1 = new_y0 + new_y_span

        if x_desc:
            ax.set_xlim(new_x1, new_x0)
        else:
            ax.set_xlim(new_x0, new_x1)
        if y_desc:
            ax.set_ylim(new_y1, new_y0)
        else:
            ax.set_ylim(new_y0, new_y1)
        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect('scroll_event', on_scroll)


def add_left_drag_pan(ax, extent=None):
    """Pan image with left-click drag while clamping to image bounds."""
    if extent is None and ax.images:
        extent = ax.images[0].get_extent()
    if extent is None:
        return

    left, right, bottom, top = extent
    x_min, x_max = min(left, right), max(left, right)
    y_min, y_max = min(bottom, top), max(bottom, top)
    state = {'dragging': False, 'last_x': None, 'last_y': None}

    def _set_limits_with_orientation(x0, x1, y0, y1, x_desc, y_desc):
        if x_desc:
            ax.set_xlim(x1, x0)
        else:
            ax.set_xlim(x0, x1)
        if y_desc:
            ax.set_ylim(y1, y0)
        else:
            ax.set_ylim(y0, y1)

    def on_press(event):
        if event.inaxes != ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        state['dragging'] = True
        state['last_x'] = event.xdata
        state['last_y'] = event.ydata

    def on_release(event):
        if event.button == 1:
            state['dragging'] = False
            state['last_x'] = None
            state['last_y'] = None

    def on_move(event):
        if not state['dragging'] or event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        dx = event.xdata - state['last_x']
        dy = event.ydata - state['last_y']
        state['last_x'] = event.xdata
        state['last_y'] = event.ydata

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_desc = xlim[0] > xlim[1]
        y_desc = ylim[0] > ylim[1]
        x0, x1 = min(xlim), max(xlim)
        y0, y1 = min(ylim), max(ylim)
        x_span = x1 - x0
        y_span = y1 - y0

        if x_span >= (x_max - x_min):
            new_x0, new_x1 = x_min, x_max
        else:
            new_x0 = np.clip(x0 - dx, x_min, x_max - x_span)
            new_x1 = new_x0 + x_span
        if y_span >= (y_max - y_min):
            new_y0, new_y1 = y_min, y_max
        else:
            new_y0 = np.clip(y0 - dy, y_min, y_max - y_span)
            new_y1 = new_y0 + y_span

        _set_limits_with_orientation(new_x0, new_x1, new_y0, new_y1, x_desc, y_desc)
        ax.figure.canvas.draw_idle()

    canvas = ax.figure.canvas
    canvas.mpl_connect('button_press_event', on_press)
    canvas.mpl_connect('button_release_event', on_release)
    canvas.mpl_connect('motion_notify_event', on_move)


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
