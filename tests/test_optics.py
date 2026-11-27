"""Tests for the closed-form optics helpers."""

import numpy as np
import pytest

from bead_analyzer import optics


def test_constants_match_their_derivations():
    assert np.isclose(optics.FWHM_PER_SIGMA, 2.3548, atol=1e-4)
    assert np.isclose(optics.SPHERE_FWHM_PER_DIAMETER, 0.5266, atol=1e-4)
    assert np.isclose(optics.SLAB_FWHM2_PER_STEP2, 0.4621, atol=1e-4)
    assert np.isclose(optics.E2_WIDTH_PER_FWHM, 1.6986, atol=1e-4)


def test_detection_psf_formulas():
    assert np.isclose(optics.lateral_fwhm_um(0.14, 0.5), 1.821, atol=1e-3)
    # At low NA the axial form reduces to 1.77 n lambda / NA^2.
    assert np.isclose(optics.axial_fwhm_um(0.1, 0.525, 1.0), 1.77 * 0.525 / 0.01, rtol=0.01)
    assert np.isclose(optics.axial_fwhm_um(0.28, 0.525, 1.52), 17.76, atol=0.05)
    with pytest.raises(ValueError):
        optics.axial_fwhm_um(1.4, 0.525, 1.33)


def test_bead_and_scan_corrections_undo_their_broadening():
    sheet, bead, step = 7.0, 3.0, 3.0
    broadened = np.sqrt(sheet ** 2 + (optics.SPHERE_FWHM_PER_DIAMETER * bead) ** 2)
    assert np.isclose(optics.remove_bead_size(broadened, bead), sheet)
    assert optics.remove_bead_size(sheet, None) == sheet
    assert np.isnan(optics.remove_bead_size(1.0, 5.0))
    slab = np.sqrt(sheet ** 2 + optics.SLAB_FWHM2_PER_STEP2 * step ** 2)
    assert np.isclose(optics.remove_scan_slab(slab, step), sheet)
    assert np.allclose(optics.remove_bead_size(np.array([broadened, 1.0]), bead), [sheet, np.nan], equal_nan=True)


def test_sheet_and_beam_relations():
    # A sheet much thinner than the detection depth of field sets the axial resolution.
    assert np.isclose(optics.system_axial_fwhm_um(5.0, 50.0), 5.0, rtol=0.01)
    assert np.isclose(optics.system_axial_fwhm_um(10.0, 10.0), 10.0 / np.sqrt(2.0))
    # 2.27 n FWHM^2 / lambda; the thickness grows by sqrt(2) one Rayleigh range from the waist.
    zr = optics.rayleigh_range_um(7.0, 0.488, 1.52)
    assert np.isclose(zr, 2.266 * 1.52 * 49.0 / 0.488, rtol=1e-3)
    assert np.isclose(optics.sheet_fwhm_at(100.0 + zr, 7.0, 100.0, zr), 7.0 * np.sqrt(2.0))
    # Far from the waist the FWHM grows by waist / Rayleigh range per unit length.
    assert np.isclose(optics.waist_fwhm_from_divergence(7.0 / zr, 0.488, 1.52), 7.0)
    # Box half-width: the defocus blur over the sheet plus the in-focus spot radius.
    expected = 0.75 * 4.0 * np.tan(np.arcsin(0.8 / 1.33)) + 0.61 * 0.525 / 0.8
    assert np.isclose(optics.defocus_halfwidth_um(4.0, 0.8, 0.525, 1.33), expected)
    assert optics.defocus_halfwidth_um(8.0, 0.8, 0.525, 1.33) > expected
