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
Synthetic bead stacks with a known light sheet, for tests and examples.

The image of each bead is the product of a scalar pupil-model detection PSF and
a Gaussian light sheet whose thickness can vary across the field like a focused
beam. The sheet sits in the detection focal plane and the sample is stepped
through it, so a bead away from its brightest slice is both dimmer (it leaves
the sheet) and defocused.

Run as a module to write an example stack:

    python -m bead_analyzer.synthetic beads.tif --sheet_fwhm 5 --rayleigh 60
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .optics import rayleigh_range_um, sheet_fwhm_at


def _odd(n):
    n = max(1, int(n))
    return n if n % 2 else n + 1


def detection_psf_planes(na, wavelength_um, ri, fine_um, half_fine, defocus_um):
    """Scalar pupil-model detection PSF, one plane per defocus value.

    Returns a float32 array (len(defocus_um), 2*half_fine+1, 2*half_fine+1)
    sampled at ``fine_um`` and cropped around the optical axis. Every plane
    carries unit energy before cropping: defocus spreads light, it does not
    remove it.
    """
    defocus_um = np.asarray(defocus_um, dtype=float)
    blur_um = float(np.max(np.abs(defocus_um))) * np.tan(np.arcsin(na / ri))
    need = 2.0 * (half_fine + blur_um / fine_um) + 16
    # Keep the pupil at least ~12 frequency samples in radius so its edge is resolved.
    need = max(need, 12.0 * wavelength_um / (na * fine_um), 64)
    n = min(2048, int(2 ** np.ceil(np.log2(need))))
    f = np.fft.fftfreq(n, fine_um)
    fx, fy = np.meshgrid(f, f)
    fr2 = fx ** 2 + fy ** 2
    pupil = fr2 <= (na / wavelength_um) ** 2
    kz = np.sqrt(np.maximum((ri / wavelength_um) ** 2 - fr2, 0.0)) - ri / wavelength_um
    c = n // 2
    lo, hi = c - half_fine, c + half_fine + 1
    if lo < 0:
        raise ValueError("PSF patch is larger than the 2048-sample simulation grid; "
                         "use a smaller patch_px or a coarser oversample_xy")
    planes = np.empty((defocus_um.size, 2 * half_fine + 1, 2 * half_fine + 1), dtype=np.float32)
    norm = None
    for i, z in enumerate(defocus_um):
        field = np.fft.ifft2(pupil * np.exp(2j * np.pi * z * kz))
        inten = np.fft.fftshift(field.real ** 2 + field.imag ** 2)
        if norm is None:
            norm = inten.sum()
        planes[i] = inten[lo:hi, lo:hi] / norm
    return planes


def make_bead_field(shape=(41, 200, 200), scale_xy=0.5, scale_z=1.0, n_beads=50,
                    na=0.14, emission_nm=525.0, ri=1.0,
                    sheet_fwhm_um=5.0, sheet_axis='x', waist_pos_um=None, rayleigh_um=None,
                    excitation_nm=488.0, bead_diameter_um=0.0, scan_mode='step',
                    photons=20000.0, brightness_cv=0.2, background=5.0, read_noise=2.0,
                    gain=1.0, offset=100.0, patch_px=None, min_separation_px=None,
                    oversample_xy=None, oversample_z=5, seed=0):
    """Simulate a Z stack of beads imaged through a light sheet.

    Parameters
    ----------
    shape : (nz, ny, nx)
    scale_xy, scale_z : float
        Pixel size and Z step in µm.
    na, emission_nm, ri : float
        Detection numerical aperture, emission wavelength and refractive index.
    sheet_fwhm_um : float or None
        Sheet FWHM at its waist. None simulates widefield illumination.
    sheet_axis : 'x' or 'y'
        Image axis along which the sheet propagates.
    waist_pos_um : float or None
        Waist position along ``sheet_axis`` (default: the field center).
    rayleigh_um : float, 'ideal' or None
        Rayleigh range of the sheet. None keeps the thickness uniform; 'ideal'
        uses the Gaussian-beam value for ``excitation_nm``.
    bead_diameter_um : float
        Diameter of the uniformly labeled beads (0 for point sources).
    scan_mode : 'step' or 'continuous'
        'continuous' integrates each slice over one Z step.
    photons : float
        Mean photoelectrons collected from a bead in its brightest slice at the waist.
    background, read_noise : float
        Background (e-/pixel) and read noise (e- rms).
    gain, offset : float
        Camera gain (ADU/e-) and offset (ADU).

    Returns
    -------
    stack : uint16 array (nz, ny, nx)
    truth : dict
        Per-bead arrays ``x``, ``y`` (pixels), ``z`` (slices), ``sheet_fwhm_um``
        and ``photons``, plus the simulation parameters.
    """
    if scan_mode not in ('step', 'continuous'):
        raise ValueError("scan_mode must be 'step' or 'continuous'")
    if sheet_axis not in ('x', 'y'):
        raise ValueError("sheet_axis must be 'x' or 'y'")
    rng = np.random.default_rng(seed)
    nz, ny, nx = (int(v) for v in shape)
    lam = emission_nm / 1000.0
    tan_t = np.tan(np.arcsin(na / ri))
    widefield = sheet_fwhm_um is None

    field_um = (nx if sheet_axis == 'x' else ny) * scale_xy
    if waist_pos_um is None:
        waist_pos_um = field_um / 2.0
    if rayleigh_um == 'ideal':
        rayleigh_um = rayleigh_range_um(sheet_fwhm_um, excitation_nm / 1000.0, ri)

    def local_sheet(u_um):
        if rayleigh_um is None:
            return np.full(np.shape(u_um), float(sheet_fwhm_um))
        return sheet_fwhm_at(u_um, sheet_fwhm_um, waist_pos_um, rayleigh_um)

    if widefield:
        hz = nz - 1
        z_margin = nz // 4
        hp = 12 if patch_px is None else int(patch_px)
    else:
        thickest = float(np.max(local_sheet(np.array([0.0, field_um]))))
        hz = int(np.ceil(2.5 * thickest / scale_z)) + 1
        z_margin = min(int(np.ceil(1.5 * thickest / scale_z)) + 1, (nz - 1) // 2)
        if patch_px is None:
            hp = int(np.ceil((1.5 * thickest * tan_t + 1.22 * lam / na) / scale_xy)) + 2
            hp = int(np.clip(hp, 7, 40))
        else:
            hp = int(patch_px)

    # Intensity is band-limited to 2 NA / lambda, so lambda / (4 NA) samples it fully.
    if oversample_xy is None:
        oversample_xy = max(3, int(np.ceil(scale_xy / (lam / (4.0 * na)))))
    os_xy, os_z = _odd(oversample_xy), _odd(oversample_z)
    fine, fz = scale_xy / os_xy, scale_z / os_z

    # Bead positions on the fine grid, kept apart so their images do not overlap.
    if min_separation_px is None:
        min_separation_px = 1.5 * hp
    lo_xy, hi_x, hi_y = hp + 1, nx - hp - 2, ny - hp - 2
    if hi_x < lo_xy or hi_y < lo_xy or nz - 1 - z_margin < z_margin:
        raise ValueError("stack is too small for the bead image patch; enlarge shape or reduce patch_px")
    placed = []
    for _ in range(200 * int(n_beads)):
        if len(placed) >= n_beads:
            break
        cand = (rng.integers(lo_xy, hi_x + 1), rng.integers(lo_xy, hi_y + 1))
        if all((cand[0] - p[0]) ** 2 + (cand[1] - p[1]) ** 2 >= min_separation_px ** 2 for p in placed):
            placed.append(cand)
    n = len(placed)
    ix = np.array([p[0] for p in placed], dtype=int)
    iy = np.array([p[1] for p in placed], dtype=int)
    iz = rng.integers(z_margin, nz - z_margin, size=n)
    jx, jy = rng.integers(0, os_xy, size=n), rng.integers(0, os_xy, size=n)
    jz = rng.integers(0, os_z, size=n)
    x = ix - 0.5 + (jx + 0.5) / os_xy
    y = iy - 0.5 + (jy + 0.5) / os_xy
    z = iz - 0.5 + (jz + 0.5) / os_z
    u_um = (x if sheet_axis == 'x' else y) * scale_xy
    sheet_local = np.full(n, np.nan) if widefield else local_sheet(u_um)
    sigma_b = np.sqrt(np.log(1.0 + brightness_cv ** 2)) if brightness_cv > 0 else 0.0
    bead_photons = photons * np.exp(rng.normal(-0.5 * sigma_b ** 2, sigma_b, size=n)) if sigma_b else np.full(n, float(photons))

    # A sphere is built from thin layers along Z, weighted by their cross-section.
    n_layer = int((bead_diameter_um / 2.0) // fz) if bead_diameter_um else 0
    layer = np.arange(-n_layer, n_layer + 1)
    w_layer = np.clip((bead_diameter_um / 2.0) ** 2 - (layer * fz) ** 2, 0.0, None) if n_layer else np.ones(1)
    w_layer = w_layer / w_layer.sum()
    half_q = (os_z - 1) // 2 if scan_mode == 'continuous' else 0
    slab = np.arange(-half_q, half_q + 1)
    w_slab = np.full(slab.size, 1.0 / slab.size)

    m_max = hz * os_z + (os_z - 1) // 2 + half_q + n_layer
    defocus = np.arange(-m_max, m_max + 1) * fz
    nd = defocus.size
    half_fine = (hp + 1) * os_xy + os_xy // 2
    table = detection_psf_planes(na, lam, ri, fine, half_fine, defocus)
    if bead_diameter_um:
        s = bead_diameter_um / np.sqrt(20.0) / fine
        table = gaussian_filter(table, sigma=(0.0, s, s), mode='constant')

    patch = 2 * hp + 1
    binned = {}

    def pixel_table(off_y, off_x):
        key = (int(off_y), int(off_x))
        if key not in binned:
            a_y = half_fine - hp * os_xy - key[0]
            a_x = half_fine - hp * os_xy - key[1]
            sub = table[:, a_y:a_y + patch * os_xy, a_x:a_x + patch * os_xy]
            binned[key] = sub.reshape(nd, patch, os_xy, patch, os_xy).sum(axis=(2, 4), dtype=np.float64)
        return binned[key]

    img = np.zeros((nz, ny, nx), dtype=np.float64)
    wts = (w_slab[:, None] * w_layer[None, :]).ravel()
    for b in range(n):
        k = np.arange(max(0, iz[b] - hz), min(nz - 1, iz[b] + hz) + 1)
        m_k = (k - iz[b]) * os_z + (os_z - 1) // 2 - jz[b]
        cols = (m_k[:, None, None] + slab[None, :, None] - layer[None, None, :] + m_max).reshape(k.size, -1)
        coef = np.zeros((k.size, nd))
        np.add.at(coef, (np.repeat(np.arange(k.size), cols.shape[1]), cols.ravel()), np.tile(wts, k.size))
        if not widefield:
            f_b = sheet_local[b]
            coef *= (sheet_fwhm_um / f_b) * np.exp(-4.0 * np.log(2.0) * (defocus / f_b) ** 2)[None, :]
        bead_img = np.tensordot(coef, pixel_table(jy[b], jx[b]), axes=(1, 0))
        img[k[0]:k[-1] + 1, iy[b] - hp:iy[b] + hp + 1, ix[b] - hp:ix[b] + hp + 1] += bead_photons[b] * bead_img

    img += background
    counts = rng.poisson(img).astype(np.float64)
    if read_noise > 0:
        counts += rng.normal(0.0, read_noise, size=counts.shape)
    stack = np.clip(np.rint(counts * gain + offset), 0, 65535).astype(np.uint16)

    truth = {
        'x': x, 'y': y, 'z': z, 'sheet_fwhm_um': sheet_local, 'photons': bead_photons,
        'scale_xy': scale_xy, 'scale_z': scale_z, 'na': na, 'emission_nm': emission_nm, 'ri': ri,
        'sheet_axis': sheet_axis, 'waist_fwhm_um': sheet_fwhm_um, 'waist_pos_um': waist_pos_um,
        'rayleigh_um': rayleigh_um, 'excitation_nm': excitation_nm,
        'bead_diameter_um': bead_diameter_um, 'scan_mode': scan_mode, 'patch_px': hp,
    }
    return stack, truth


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Write a synthetic light-sheet bead stack with known ground truth.")
    parser.add_argument("output", type=str, help="Output TIFF path")
    parser.add_argument("--shape", type=int, nargs=3, default=[61, 400, 400], metavar=("NZ", "NY", "NX"))
    parser.add_argument("--scale_xy", type=float, default=0.5, help="Pixel size (µm)")
    parser.add_argument("--scale_z", type=float, default=1.0, help="Z step (µm)")
    parser.add_argument("--n_beads", type=int, default=150)
    parser.add_argument("--na", type=float, default=0.14, help="Detection NA")
    parser.add_argument("--emission_nm", type=float, default=525.0)
    parser.add_argument("--excitation_nm", type=float, default=488.0)
    parser.add_argument("--ri", type=float, default=1.0, help="Refractive index")
    parser.add_argument("--sheet_fwhm", type=float, default=5.0, help="Sheet FWHM at the waist (µm)")
    parser.add_argument("--sheet_axis", choices=["x", "y"], default="x")
    parser.add_argument("--waist_pos", type=float, default=None, help="Waist position along the sheet axis (µm)")
    parser.add_argument("--rayleigh", type=str, default=None,
                        help="Rayleigh range in µm, or 'ideal' for the Gaussian-beam value (default: uniform sheet)")
    parser.add_argument("--bead_diameter", type=float, default=0.0, help="Bead diameter (µm)")
    parser.add_argument("--scan_mode", choices=["step", "continuous"], default="step")
    parser.add_argument("--photons", type=float, default=20000.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import pandas as pd
    import tifffile

    rayleigh = args.rayleigh
    if rayleigh is not None and rayleigh != 'ideal':
        rayleigh = float(rayleigh)
    stack, truth = make_bead_field(
        shape=tuple(args.shape), scale_xy=args.scale_xy, scale_z=args.scale_z, n_beads=args.n_beads,
        na=args.na, emission_nm=args.emission_nm, excitation_nm=args.excitation_nm, ri=args.ri,
        sheet_fwhm_um=args.sheet_fwhm, sheet_axis=args.sheet_axis, waist_pos_um=args.waist_pos,
        rayleigh_um=rayleigh, bead_diameter_um=args.bead_diameter, scan_mode=args.scan_mode,
        photons=args.photons, seed=args.seed,
    )
    out = Path(args.output)
    tifffile.imwrite(str(out), stack, imagej=True, resolution=(1 / args.scale_xy, 1 / args.scale_xy),
                     metadata={'unit': 'um', 'spacing': args.scale_z, 'axes': 'ZYX'})
    truth_path = out.with_name(f"{out.stem}_truth.csv")
    pd.DataFrame({k: truth[k] for k in ('x', 'y', 'z', 'sheet_fwhm_um', 'photons')}).to_csv(truth_path, index=False)
    print(f"Wrote {out} ({len(truth['x'])} beads) and {truth_path}")
    extra = f" --bead_diameter {args.bead_diameter}" if args.bead_diameter else ""
    print("Analyze it with:")
    print(f"  bead-analyzer {out} --scale_xy {args.scale_xy} --scale_z {args.scale_z} --fit_gaussian "
          f"--lightsheet --na {args.na} --emission_nm {args.emission_nm} --ri {args.ri} "
          f"--sheet_axis {args.sheet_axis} --scan_mode {args.scan_mode}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
