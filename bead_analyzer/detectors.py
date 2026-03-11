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
Bead detection: manual (interactive), blob, trackpy, StarDist, and Cellpose.
"""

import numpy as np
from pathlib import Path
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter

from .core import (
    INTERACTIVE_PREVIEW_NOTE,
    MultiPointClicker,
    add_interaction_key,
    add_left_drag_pan,
    add_mousewheel_zoom,
    full_to_preview_point,
    make_preview_image,
)

# Optional: StarDist
try:
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    STARDIST_AVAILABLE = True
except ImportError:
    STARDIST_AVAILABLE = False

# Optional: Cellpose
try:
    from cellpose import models
    from cellpose import plot as cellpose_plot
    from skimage.measure import label, regionprops
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

# Optional: trackpy
try:
    import trackpy as tp
    tp.quiet()
    TRACKPY_AVAILABLE = True
except ImportError:
    TRACKPY_AVAILABLE = False

# When set by the GUI, review windows run on the main thread to avoid Matplotlib warnings.
_main_thread_runner = None
REVIEW_DISPLAY_GAMMA = 0.75


def _raise_figure_window(fig):
    """Bring the figure window to the front so the user sees it (TkAgg backend)."""
    try:
        win = fig.canvas.manager.window
        win.attributes('-topmost', True)
        win.lift()
        win.focus_force()
        win.after(300, lambda: win.attributes('-topmost', False))
    except Exception:
        pass


def _gamma_brighten_for_display(img, gamma=REVIEW_DISPLAY_GAMMA):
    """Return a display-only gamma-adjusted image (analysis data is unchanged)."""
    arr = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    vals = arr[finite]
    lo = float(np.percentile(vals, 1.0))
    hi = float(np.percentile(vals, 99.8))
    if hi <= lo:
        hi = float(np.max(vals))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.power(norm, float(gamma)).astype(np.float32, copy=False)


def get_points_manual(mip_image, ax, fig):
    """
    Interactive manual selection: right-click on beads, Escape when done.
    Returns list of (x, y) tuples.
    """
    clicker = MultiPointClicker(ax)
    fig.canvas.draw()
    import matplotlib.pyplot as plt
    plt.show()
    return clicker.points


def load_points_from_file(points_file):
    """Load bead coordinates from CSV or TXT. Returns list of (x, y) or empty list."""
    try:
        path = Path(points_file)
        separator = r'\s+' if path.suffix.lower() != '.csv' else ','
        df = pd.read_csv(path, sep=separator, engine='python')
        if 'x_coord' in df.columns and 'y_coord' in df.columns:
            return list(zip(df['x_coord'].astype(float), df['y_coord'].astype(float)))
        if 'x' in df.columns and 'y' in df.columns:
            return list(zip(df['x'].astype(float), df['y'].astype(float)))
        print("ERROR: Points file must contain 'x_coord'/'y_coord' or 'x'/'y' columns.")
        return []
    except Exception as e:
        print(f"ERROR: Could not read points file: {e}")
        return []


def _detect_points_blob_localmax(mip_image, sigma=1.2, threshold_rel=0.2, min_distance_px=5):
    """
    Simple blob detector fallback: smooth + local maxima above relative threshold.
    Returns list of (x, y) points.
    """
    img = mip_image.astype(np.float32)
    sm = gaussian_filter(img, sigma=sigma) if sigma and sigma > 0 else img
    dyn = float(sm.max() - sm.min())
    if dyn <= 0:
        return []
    thr = float(sm.min() + threshold_rel * dyn)
    neighborhood = max(3, int(min_distance_px) * 2 + 1)
    mx = maximum_filter(sm, size=neighborhood, mode='nearest')
    peaks = (sm == mx) & (sm >= thr)
    ys, xs = np.where(peaks)
    pts = [(float(x), float(y)) for y, x in zip(ys, xs)]
    print(f"Found {len(pts)} beads with blob detector.")
    return pts


def _detect_points_trackpy(mip_image, diameter=5, minmass=5000, separation=None):
    """
    Detect beads using trackpy's bandpass + local-max algorithm.

    trackpy applies a Difference-of-Gaussians bandpass filter before peak
    finding, which naturally handles spatially varying background and intensity
    gradients — no CLAHE or global threshold needed.

    Parameters
    ----------
    mip_image : 2D array
        Maximum intensity projection.
    diameter : int (odd)
        Expected feature diameter in pixels.  Must be odd.
    minmass : float
        Minimum integrated brightness for a feature.  Lower values accept
        dimmer beads; set to 0 to accept all.
    separation : int or None
        Minimum separation between features (default: diameter + 1).

    Returns list of (x, y) sub-pixel coordinates.
    """
    if not TRACKPY_AVAILABLE:
        print("\nERROR: trackpy not found. Install with: pip install trackpy")
        return []
    print("\n--- Detecting beads with trackpy ---")
    img = mip_image.astype(np.float64)
    kwargs = dict(diameter=diameter, minmass=minmass)
    if separation is not None:
        kwargs['separation'] = separation
    features = tp.locate(img, **kwargs)
    pts = list(zip(features['x'].astype(float), features['y'].astype(float)))
    print(f"Found {len(pts)} beads with trackpy.")
    return pts


def get_points_blob(mip_image, points_file=None, sigma=1.2, threshold_rel=0.2, min_distance_px=5):
    """Classical blob detection: Gaussian smooth + local maxima."""
    if points_file:
        return load_points_from_file(points_file)
    print("\n--- Detecting beads with Blob detector ---")
    return _detect_points_blob_localmax(
        mip_image, sigma=sigma, threshold_rel=threshold_rel, min_distance_px=min_distance_px
    )


def get_points_trackpy(mip_image, points_file=None, diameter=5, minmass=5000, separation=None):
    """trackpy bandpass + centroid detector."""
    if points_file:
        return load_points_from_file(points_file)
    return _detect_points_trackpy(
        mip_image, diameter=diameter, minmass=minmass, separation=separation
    )


def get_points_stardist(
    mip_image,
    points_file=None,
    model_name='2D_versatile_fluo',
    prob_thresh=0.6,
    nms_thresh=0.3,
    use_blob_fallback=False,
    blob_sigma=1.2,
    blob_threshold_rel=0.2,
    blob_min_distance=5,
    n_tiles=None,
    use_trackpy=False,
    trackpy_diameter=5,
    trackpy_minmass=5000,
    trackpy_separation=None,
):
    """
    StarDist automatic detection using 2D_versatile_fluo.
    If points_file is provided, load from file instead.
    Falls back to trackpy or blob if StarDist is unavailable/fails.
    Returns list of (x, y) tuples.
    """
    if points_file:
        return load_points_from_file(points_file)

    def _fallback():
        if use_trackpy:
            return _detect_points_trackpy(
                mip_image, diameter=trackpy_diameter,
                minmass=trackpy_minmass, separation=trackpy_separation,
            )
        if use_blob_fallback:
            return _detect_points_blob_localmax(
                mip_image, sigma=blob_sigma,
                threshold_rel=blob_threshold_rel,
                min_distance_px=blob_min_distance,
            )
        return []

    if not STARDIST_AVAILABLE:
        print("\nERROR: StarDist not found. Install with: pip install stardist")
        return _fallback()
    print("\n--- Detecting beads with StarDist ---")
    try:
        model = StarDist2D.from_pretrained(model_name)
        img_norm = normalize(mip_image, 1, 99.8, axis=None)
        predict_kwargs = dict(
            prob_thresh=float(prob_thresh), nms_thresh=float(nms_thresh),
        )
        if n_tiles is not None:
            predict_kwargs['n_tiles'] = n_tiles
        _, details = model.predict_instances(img_norm, **predict_kwargs)
        coords = details.get('points', [])
        pts = [(float(c[1]), float(c[0])) for c in coords]
        print(f"Found {len(pts)} beads with StarDist.")
        if len(pts) == 0:
            return _fallback()
        return pts
    except Exception as e:
        print(f"StarDist failed: {e}")
        return _fallback()


def get_points_cellpose(mip_image, model_path, diameter=None, gauss_sigma=None,
                        cellprob_threshold=0.0, min_size=3, flow_threshold=0.4):
    """
    Cellpose detection with custom model.
    Returns (pts, masks) where pts is list of (x, y), masks is label image.
    """
    if not CELLPOSE_AVAILABLE:
        print("\nERROR: Cellpose not found. Install with: pip install cellpose")
        return [], None
    print("\n--- Detecting beads with Cellpose ---")
    if gauss_sigma and gauss_sigma > 0:
        mip_image = gaussian_filter(mip_image, sigma=gauss_sigma)
    if diameter is None:
        print("  Note: --cellpose_diameter not set; using model's trained default.")
    try:
        import torch
        gpu_flag = torch.cuda.is_available()
        print(f"  Cellpose using GPU: {gpu_flag}")
    except ImportError:
        gpu_flag = False
    model = models.CellposeModel(gpu=gpu_flag, pretrained_model=model_path)
    masks, _, _ = model.eval(
        mip_image, diameter=diameter, cellprob_threshold=cellprob_threshold,
        channels=[0, 0], min_size=min_size, flow_threshold=flow_threshold,
    )
    bead_ids = np.unique(masks)[1:]
    if len(bead_ids) == 0:
        print("Found 0 beads.")
        return [], masks
    pts = []
    for bead_id in bead_ids:
        coords = np.argwhere(masks == bead_id)
        intensities = mip_image[coords[:, 0], coords[:, 1]]
        peak_idx = np.argmax(intensities)
        peak_yx = coords[peak_idx]
        pts.append((float(peak_yx[1]), float(peak_yx[0])))
    print(f"Found {len(pts)} beads.")
    return pts, masks


def get_points_cellpose_3d(
    stack,
    model_path,
    diameter=None,
    cellprob_threshold=0.0,
    anisotropy=None,
    min_size=3,
    flow_threshold=0.4,
):
    """
    Native Cellpose 3D detection.
    Expects stack shape (Z, Y, X), returns pts as (x, y, z) and 3D masks.
    """
    if not CELLPOSE_AVAILABLE:
        print("\nERROR: Cellpose not found. Install with: pip install cellpose")
        return [], None
    print("\n--- Detecting beads with Cellpose 3D ---")
    if diameter is None:
        print("  Note: --cellpose_diameter not set; using model's trained default.")
    try:
        import torch
        gpu_flag = torch.cuda.is_available()
        print(f"  Cellpose using GPU: {gpu_flag}")
    except ImportError:
        gpu_flag = False
    model = models.CellposeModel(gpu=gpu_flag, pretrained_model=model_path)
    masks, _, _ = model.eval(
        stack.astype(np.float32),
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        channels=[0, 0],
        do_3D=True,
        anisotropy=anisotropy,
        min_size=min_size,
        flow_threshold=flow_threshold,
    )
    if masks is None:
        return [], None
    bead_ids = np.unique(masks)[1:]
    if len(bead_ids) == 0:
        print("Found 0 beads.")
        return [], masks
    pts = []
    for prop in regionprops(masks):
        zc, yc, xc = prop.centroid
        pts.append((float(xc), float(yc), float(zc)))
    print(f"Found {len(pts)} beads.")
    return pts, masks


def review_detection_cellpose(mip_image, masks):
    """Show Cellpose detection overlay; user presses 'y' to accept or 'n' to abort."""
    if not CELLPOSE_AVAILABLE:
        return False
    if _main_thread_runner is not None:
        return _main_thread_runner(_review_detection_cellpose_impl, mip_image, masks)
    return _review_detection_cellpose_impl(mip_image, masks)


def _review_detection_cellpose_impl(mip_image, masks):
    """Actual Cellpose review UI (run on main thread when called from GUI)."""
    import matplotlib.pyplot as plt
    preview_mip, ds_factor = make_preview_image(mip_image)
    preview_masks = masks[::ds_factor, ::ds_factor] if ds_factor > 1 else masks
    fig, ax = plt.subplots(figsize=(12, 10))
    display_mip = _gamma_brighten_for_display(preview_mip, REVIEW_DISPLAY_GAMMA)
    outlines = cellpose_plot.mask_overlay(
        display_mip, preview_masks,
        colors=np.array([[255, 0, 0]] * int(preview_masks.max()))
    )
    ax.imshow(outlines)
    ax.set_title("Review Detection. Press 'y' to accept, 'n' to abort.")
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
        "Right-click: No action",
        "Y: Accept, N: Abort",
    ])
    proceed = False

    def on_key(event):
        nonlocal proceed
        if event.key.lower() == 'y':
            print("Detection accepted.")
            proceed = True
            plt.close(fig)
        elif event.key.lower() == 'n':
            print("Detection aborted.")
            proceed = False
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    _raise_figure_window(fig)
    plt.show()
    return proceed


def review_detection_points(mip_image, points, title="Review Detection"):
    """Show point overlay; user presses 'y' to accept or 'n' to abort."""
    if not points:
        return False
    if _main_thread_runner is not None:
        return _main_thread_runner(_review_detection_points_impl, mip_image, points, title)
    return _review_detection_points_impl(mip_image, points, title)


def _review_detection_points_impl(mip_image, points, title="Review Detection"):
    """Actual points review UI (run on main thread when called from GUI)."""
    import matplotlib.pyplot as plt
    preview_mip, ds_factor = make_preview_image(mip_image)
    fig, ax = plt.subplots(figsize=(12, 10))
    display_mip = _gamma_brighten_for_display(preview_mip, REVIEW_DISPLAY_GAMMA)
    ax.imshow(display_mip, cmap='gray', vmin=0.0, vmax=1.0)
    scaled_pts = [full_to_preview_point(p[0], p[1], ds_factor) for p in points]
    xs = [p[0] for p in scaled_pts]
    ys = [p[1] for p in scaled_pts]
    ax.scatter(xs, ys, s=40, facecolors='none', edgecolors='yellow')
    ax.set_title(f"{title}. Press 'y' to accept, 'n' to abort.")
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
        "Right-click: No action",
        "Y: Accept, N: Abort",
    ])
    proceed = False

    def on_key(event):
        nonlocal proceed
        if event.key.lower() == 'y':
            print("Detection accepted.")
            proceed = True
            plt.close(fig)
        elif event.key.lower() == 'n':
            print("Detection aborted.")
            proceed = False
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    _raise_figure_window(fig)
    plt.show()
    return proceed
