"""
Bead detection: manual (interactive), StarDist, Cellpose, and trackpy.
"""

import numpy as np
from pathlib import Path
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter

from .core import MultiPointClicker

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
    print(f"Blob fallback found {len(pts)} candidate beads.")
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
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 10))
    outlines = cellpose_plot.mask_overlay(
        mip_image, masks,
        colors=np.array([[255, 0, 0]] * int(masks.max()))
    )
    ax.imshow(outlines)
    ax.set_title("Review Detection. Press 'y' to accept, 'n' to abort.")
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
    plt.show()
    return proceed


def review_detection_points(mip_image, points, title="Review Detection"):
    """Show point overlay; user presses 'y' to accept or 'n' to abort."""
    if not points:
        return False
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(mip_image, cmap='gray')
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, s=40, facecolors='none', edgecolors='yellow')
    ax.set_title(f"{title}. Press 'y' to accept, 'n' to abort.")
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
    plt.show()
    return proceed
