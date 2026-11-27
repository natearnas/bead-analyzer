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
Export a measured PSF in the form deconvolution software expects.

The PSF is the average of isolated beads at the native voxel size: each bead is
background-subtracted, shifted so its fitted center sits on the central voxel,
and scaled to unit sum before averaging. The result has odd dimensions, its
peak on the central voxel, no negative values and unit sum.
"""

import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import shift

from .analysis import _crop_floor
from .core import fit_gaussian_fwhm
from .lightsheet import MIN_SAMPLES_SINGLE, measure_beads


def build_psf(stack, results, scale_xy, scale_z, box_size=15, max_beads=200, bead_table=None):
    """Average clean beads into a PSF. Returns (psf, info), or (None, info) if no bead qualifies."""
    stack = np.asarray(stack)
    nz, ny, nx = stack.shape
    if bead_table is None:
        bead_table = measure_beads(stack, results, scale_xy, scale_z, box_size)[0]
    info = {'n_beads': 0, 'voxel_size_um': {'z': scale_z, 'y': scale_xy, 'x': scale_xy}, 'notes': []}
    if not len(bead_table):
        info['notes'].append("No beads to average.")
        return None, info
    clean = ((bead_table['sampling'] != 'fit_failed') & ~bead_table['saturated']
             & ~bead_table['truncated'] & ~bead_table['crowded']).to_numpy()
    if not clean.any():
        info['notes'].append("No isolated, unsaturated bead with a complete Z profile.")
        return None, info

    samples = float(np.median(bead_table.loc[clean, 'samples_per_fwhm']))
    z_fwhm = bead_table['fwhm_z_box_um'].to_numpy()
    order = np.argsort(np.abs(z_fwhm - np.median(z_fwhm[clean])))
    picked = [i for i in order if clean[i]][:int(max_beads)]

    half_xy = int(box_size) // 2
    half_z = int(min(max(4, np.ceil(2.0 * samples)), (nz - 3) // 2))
    # Sub-slice shifts interpolate along Z, which is only sound when Z is sampled well.
    register_z = samples >= MIN_SAMPLES_SINGLE
    if not register_z:
        info['notes'].append(f"Z sampling is {samples:.1f} slices per FWHM: beads were aligned to the nearest "
                             "slice, so the PSF is broadened along Z by up to one Z step.")

    acc = np.zeros((2 * half_z + 1, 2 * half_xy + 1, 2 * half_xy + 1))
    n_used = 0
    for i in picked:
        x_c, y_c = float(results[i]['x_coord']), float(results[i]['y_coord'])
        z_c = float(bead_table['z_um'].iloc[i]) / scale_z
        xi, yi, zi = int(round(x_c)), int(round(y_c)), int(round(z_c))
        # One extra voxel on each side absorbs the sub-voxel shift.
        z0, z1 = zi - half_z - 1, zi + half_z + 2
        y0, y1 = yi - half_xy - 1, yi + half_xy + 2
        x0, x1 = xi - half_xy - 1, xi + half_xy + 2
        if z0 < 0 or y0 < 0 or x0 < 0 or z1 > nz or y1 > ny or x1 > nx:
            continue
        vol = stack[z0:z1, y0:y1, x0:x1].astype(np.float64)
        # Same floor rule as the average bead: a low percentile of the smoothed outer
        # shell. The median of the lateral border counts the defocused light of a
        # widefield bead as background (about 8 % of its light at a 15-pixel box).
        vol -= _crop_floor(vol)
        offsets = (zi - z_c if register_z else 0.0, yi - y_c, xi - x_c)
        vol = shift(vol, offsets, order=3, mode='nearest')[1:-1, 1:-1, 1:-1]
        total = vol.sum()
        if total <= 0:
            continue
        acc += vol / total
        n_used += 1
    if n_used == 0:
        info['notes'].append("No bead far enough from the stack edges to crop.")
        return None, info

    psf = np.clip(acc / n_used, 0.0, None)
    psf = (psf / psf.sum()).astype(np.float32)
    center = (half_z, half_xy, half_xy)
    peak = np.unravel_index(int(np.argmax(psf)), psf.shape)
    fwhm = {}
    for name, profile, scale in (('z', psf[:, half_xy, half_xy], scale_z),
                                 ('y', psf[half_z, :, half_xy], scale_xy),
                                 ('x', psf[half_z, half_xy, :], scale_xy)):
        fit = fit_gaussian_fwhm(profile.astype(np.float64), scale, peak_hint=int(np.argmax(profile)))
        fwhm[name] = fit['fwhm_um'] if fit else None
    info.update({
        'n_beads': n_used, 'shape_zyx': list(psf.shape), 'center_voxel_zyx': list(center),
        'peak_voxel_zyx': [int(v) for v in peak], 'z_registration': 'sub-slice' if register_z else 'nearest slice',
        'samples_per_fwhm_z': samples, 'fwhm_um': fwhm, 'sum': 1.0,
        'background': '10th percentile of the smoothed outer shell of each bead crop',
    })
    return psf, info


def export_psf(stack, results, scale_xy, scale_z, box_size, file_path, max_beads=200, bead_table=None):
    """Write ``<stem>_psf.tif`` (float32, ZYX) and ``<stem>_psf.json``. Returns their paths or None."""
    file_path = Path(file_path)
    psf, info = build_psf(stack, results, scale_xy, scale_z, box_size, max_beads, bead_table)
    info['source'] = file_path.name
    json_path = file_path.with_name(f"{file_path.stem}_psf.json")
    if psf is None:
        print("WARNING: PSF not exported: " + " ".join(info['notes']))
        return None
    tif_path = file_path.with_name(f"{file_path.stem}_psf.tif")
    tifffile.imwrite(tif_path, psf, imagej=True, resolution=(1 / scale_xy, 1 / scale_xy),
                     metadata={'unit': 'um', 'spacing': scale_z, 'axes': 'ZYX'})
    json_path.write_text(json.dumps(info, indent=2), encoding='utf-8')
    return {'psf': tif_path, 'json': json_path}
