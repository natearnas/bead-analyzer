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
Closed-form optics used to interpret bead measurements.

All lengths are in micrometers and all widths are full widths at half maximum
(FWHM) unless a name says otherwise.
"""

import numpy as np

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

# A uniformly labeled sphere of diameter d projected onto one axis has variance
# d^2 / 20, so it broadens a Gaussian profile like a Gaussian of FWHM 0.527 d.
SPHERE_FWHM_PER_DIAMETER = FWHM_PER_SIGMA / np.sqrt(20.0)

# Averaging over one Z step (continuous stage motion during the exposure) adds a
# uniform distribution of variance dz^2 / 12, i.e. 0.462 dz^2 to the FWHM squared.
SLAB_FWHM2_PER_STEP2 = FWHM_PER_SIGMA ** 2 / 12.0

# A Gaussian beam of intensity exp(-2 z^2 / w^2) has FWHM = w sqrt(2 ln 2), so its
# 1/e^2 full width 2w is 1.70 times the FWHM.
E2_WIDTH_PER_FWHM = 2.0 / np.sqrt(2.0 * np.log(2.0))


def lateral_fwhm_um(na, wavelength_um):
    """Diffraction-limited lateral FWHM of a widefield detection PSF: 0.51 lambda / NA."""
    return 0.51 * wavelength_um / na


def axial_fwhm_um(na, wavelength_um, ri=1.0):
    """On-axis axial FWHM of a widefield detection PSF: 0.88 lambda / (n - sqrt(n^2 - NA^2))."""
    if na >= ri:
        raise ValueError(f"NA ({na}) must be smaller than the refractive index ({ri})")
    return 0.88 * wavelength_um / (ri - np.sqrt(ri ** 2 - na ** 2))


def _quadrature_remove(fwhm_um, extra_um):
    fwhm = np.asarray(fwhm_um, dtype=float)
    rest = fwhm ** 2 - float(extra_um) ** 2
    out = np.sqrt(np.where(rest > 0, rest, np.nan))
    return float(out) if out.ndim == 0 else out


def remove_bead_size(fwhm_um, bead_diameter_um):
    """Remove the broadening of a uniformly labeled bead from a measured FWHM.

    Accurate to about 1% while the bead diameter does not exceed the corrected
    FWHM. Returns NaN where the bead term is larger than the measurement.
    """
    if not bead_diameter_um:
        return fwhm_um
    return _quadrature_remove(fwhm_um, SPHERE_FWHM_PER_DIAMETER * bead_diameter_um)


def remove_scan_slab(fwhm_um, z_step_um):
    """Remove the broadening from continuous Z motion over one step per exposure."""
    return _quadrature_remove(fwhm_um, np.sqrt(SLAB_FWHM2_PER_STEP2) * z_step_um)


def system_axial_fwhm_um(sheet_fwhm_um, detection_fwhm_um):
    """On-axis axial FWHM of a light-sheet system: the product of sheet and detection profiles."""
    return (sheet_fwhm_um ** -2.0 + detection_fwhm_um ** -2.0) ** -0.5


def rayleigh_range_um(waist_fwhm_um, wavelength_um, ri=1.0):
    """Rayleigh range of an ideal Gaussian sheet: 2.27 n FWHM^2 / lambda (vacuum wavelength)."""
    return np.pi * ri * waist_fwhm_um ** 2 / (2.0 * np.log(2.0) * wavelength_um)


def waist_fwhm_from_divergence(slope, wavelength_um, ri=1.0):
    """Waist FWHM of the ideal Gaussian sheet whose far-field FWHM grows by ``slope`` per unit length."""
    return 2.0 * np.log(2.0) * wavelength_um / (np.pi * ri * abs(slope))


def sheet_fwhm_at(u_um, waist_fwhm_um, waist_pos_um, rayleigh_um):
    """Sheet FWHM at position u along the propagation axis of a Gaussian beam."""
    u = np.asarray(u_um, dtype=float)
    return waist_fwhm_um * np.sqrt(1.0 + ((u - waist_pos_um) / rayleigh_um) ** 2)


def defocus_halfwidth_um(sheet_fwhm_um, na, wavelength_um, ri=1.0):
    """Half-width of the XY box needed for a box-averaged Z profile to follow the sheet.

    A bead inside the sheet but away from the detection focal plane is imaged as
    a blur of radius |z| tan(asin(NA / n)). The box must hold that blur over the
    part of the sheet that carries signal, plus the in-focus spot.
    """
    return 0.75 * sheet_fwhm_um * np.tan(np.arcsin(na / ri)) + 0.61 * wavelength_um / na
