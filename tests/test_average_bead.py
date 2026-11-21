"""Average-bead construction (`_average_bead_volumes`).

Synthetic Gaussian beads at random sub-voxel positions, with a camera offset, noise,
different brightness and some crops cut short at the image edge.
"""

import numpy as np

from bead_analyzer.analysis import _average_bead_volumes

SIGMA = (2.0, 1.2, 1.2)  # z, y, x in voxels
SHAPE = (21, 21, 21)
OFFSET = 100.0


def _bead(rng, amplitude, center, shape=SHAPE, noise=2.0):
    z, y, x = np.indices(shape, dtype=float)
    g = np.exp(-((z - center[0]) ** 2) / (2 * SIGMA[0] ** 2)
               - ((y - center[1]) ** 2) / (2 * SIGMA[1] ** 2)
               - ((x - center[2]) ** 2) / (2 * SIGMA[2] ** 2))
    return amplitude * g + OFFSET + rng.normal(0.0, noise, shape)


def _beads(rng, n=12):
    vols = []
    for i in range(n):
        c = np.array([10.0, 10.0, 10.0]) + rng.uniform(-0.5, 0.5, 3)
        vols.append(_bead(rng, amplitude=rng.uniform(500, 3000), center=c))
    return vols


def _fwhm_px(profile):
    p = np.asarray(profile, float)
    half = p.max() / 2
    idx = np.where(p >= half)[0]
    lo, hi = idx[0], idx[-1]
    left = lo - (p[lo] - half) / max(p[lo] - p[lo - 1], 1e-12)
    right = hi + (p[hi] - half) / max(p[hi] - p[hi + 1], 1e-12)
    return right - left


def test_pedestal_is_removed_and_width_is_kept():
    rng = np.random.default_rng(0)
    up = 4
    avg, info = _average_bead_volumes(_beads(rng), upsample_factor=up)
    assert info["used"] >= 10
    assert abs(info["median_floor"] - OFFSET) < 3.0
    peak = avg.max()
    edge = np.median(np.r_[avg[:4].ravel(), avg[-4:].ravel()])
    assert abs(edge) < 0.01 * peak, "average still carries a pedestal"
    zc, yc, xc = np.unravel_index(int(np.argmax(avg)), avg.shape)
    true_fwhm_x = 2.3548 * SIGMA[2] * up
    assert abs(_fwhm_px(avg[zc, yc, :]) - true_fwhm_x) < 0.12 * true_fwhm_x


def test_peak_sits_at_the_canvas_centre_even_with_edge_cut_crops():
    rng = np.random.default_rng(1)
    vols = _beads(rng, n=8)
    # Two crops cut short at the image edge: four voxels missing on one side.
    vols[0] = vols[0][:, 4:, :]
    vols[1] = vols[1][4:, :, :]
    avg, _ = _average_bead_volumes(vols, upsample_factor=4)
    peak = np.array(np.unravel_index(int(np.argmax(avg)), avg.shape), float)
    center = (np.array(avg.shape) - 1) / 2.0
    assert np.all(np.abs(peak - center) <= 2.0)
    zc = int(round(center[0]))
    yc = int(round(center[1]))
    true_fwhm_x = 2.3548 * SIGMA[2] * 4
    assert abs(_fwhm_px(avg[zc, yc, :]) - true_fwhm_x) < 0.15 * true_fwhm_x


def test_an_odd_bead_is_set_aside():
    rng = np.random.default_rng(2)
    vols = _beads(rng, n=10)
    # A clump: two beads side by side in one crop.
    clump = _bead(rng, 2000, (10.0, 10.0, 7.0)) + _bead(rng, 2000, (10.0, 10.0, 13.0)) - OFFSET
    avg, info = _average_bead_volumes(vols + [clump], upsample_factor=2)
    assert info["rejected"] >= 1
    assert info["used"] >= 3


def test_bright_beads_do_not_dominate():
    rng = np.random.default_rng(3)
    narrow = [_bead(rng, 500.0, (10, 10, 10)) for _ in range(6)]
    wide = []
    for _ in range(1):
        z, y, x = np.indices(SHAPE, dtype=float)
        g = np.exp(-((z - 10) ** 2) / (2 * 2.0 ** 2) - ((y - 10) ** 2 + (x - 10) ** 2) / (2 * 2.4 ** 2))
        wide.append(20000.0 * g + OFFSET + rng.normal(0, 2.0, SHAPE))
    avg, _ = _average_bead_volumes(narrow + wide, upsample_factor=1, reject_mad=1e9)
    zc, yc, xc = np.unravel_index(int(np.argmax(avg)), avg.shape)
    # One very bright wide bead among six dim narrow ones moves the width only a little.
    assert _fwhm_px(avg[zc, yc, :]) < 1.25 * 2.3548 * SIGMA[2]


def test_unusable_volumes_are_skipped():
    rng = np.random.default_rng(4)
    vols = _beads(rng, n=3) + [np.zeros((0, 0, 0)), np.zeros((2, 2))]
    avg, info = _average_bead_volumes(vols, upsample_factor=2)
    assert info["skipped"] == 2
    assert avg.ndim == 3
