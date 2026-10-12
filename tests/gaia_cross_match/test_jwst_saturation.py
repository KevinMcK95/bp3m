"""Tests for gaia_cross_match.jwst_saturation (JWST saturated-star cross-match guards, 2026-10-12)."""

import numpy as np
import pytest
from astropy.io import fits

from gaia_cross_match import jwst_saturation as js
from bp3m.pipeline.download_hst import _nonscience_reason_jwst, _failed_reason_jwst


def test_filter_name_pupil_and_filter():
    assert js.jwst_filter_name({'FILTER': 'F150W', 'PUPIL': 'CLEAR'}) == 'F150W'
    assert js.jwst_filter_name({'FILTER': 'F150W2', 'PUPIL': 'F162M'}) == 'F162M'
    assert js.jwst_filter_name({'FILTER': 'CLEAR', 'PUPIL': 'F200W'}) == 'F200W'      # NIRISS imaging
    assert js.jwst_filter_name({'FILTER': 'F770W'}) == 'F770W'                         # MIRI


def test_abg_window():
    lo, hi = js.jwst_abg_window('NIRCAM', 'NRCA1', 'F115W')
    assert lo < 0.86 < hi and hi == pytest.approx(0.86 + js.JWST_ABG_HI)
    lo, hi = js.jwst_abg_window('NIRCAM', 'NRCALONG', 'F444W')       # unmeasured LW filter: LW default
    assert hi == pytest.approx(js.JWST_ABG_DEFAULT['NIRCAM_LW'] + js.JWST_ABG_HI)
    assert js.jwst_abg_window('MIRI', 'MIRIMAGE', 'F770W') is None
    # artefact locks (AB - G 5-8) never pass; the confirmed-ok 99th percentiles do
    for f in ('F070W', 'F090W', 'F115W', 'F150W', 'F200W'):
        lo, hi = js.jwst_abg_window('NIRCAM', 'NRCA1', f)
        assert hi < 4.4 and lo < -1.9


def _dq_with_stars(stars, shape=(200, 200)):
    dq = np.zeros(shape, np.uint32)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    for x, y, r in stars:
        dq[(xx - x) ** 2 + (yy - y) ** 2 <= r * r] |= 2
    dq[50, 150] |= 2                                   # single saturated pixel: not a cluster
    return dq


def test_saturated_clusters_and_near_saturation(tmp_path):
    dq = _dq_with_stars([(40.3, 60.7, 3), (120, 140, 6)])
    p = tmp_path / 'f_cal.fits'
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros(dq.shape, np.float32), name='SCI'),
                  fits.ImageHDU(dq, name='DQ')]).writeto(p)
    cx, cy, req = js.saturated_clusters(str(p))
    assert len(cx) == 2
    o = np.argsort(cx)
    assert cx[o][0] == pytest.approx(40.0, abs=0.5) and cy[o][0] == pytest.approx(61.0, abs=0.5)
    # core (kept), core-masked star 3.5 px off (kept), spike at 10 px (dropped), field star 60 px away (kept);
    # big star: 3 r_eq exclusion
    x = np.array([40.1, 43.5, 50.0, 100.0, 120.0 + 19.0, 120.0 + 25.0])
    y = np.array([60.9, 61.0, 61.0, 61.0, 140.0, 140.0])
    drop, core = js.near_saturation(x, y, cx, cy, req)
    assert list(core) == [True, True, False, False, False, False]
    assert list(drop) == [False, False, True, False, True, False]


def test_raw_to_gdc_recovers_polynomial():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 2048, 500), rng.uniform(0, 2048, 500)
    xg = x + 1e-4 * (x - 1024) ** 2 * 1e-2 + 0.3; yg = y - 2e-7 * (x - 1024) * (y - 1024) - 0.2
    f = js.raw_to_gdc(x, y, xg, yg)
    px, py = f(np.array([10.0, 2000.0]), np.array([1500.0, 30.0]))
    tx = np.array([10.0, 2000.0]) + 1e-6 * (np.array([10.0, 2000.0]) - 1024) ** 2 + 0.3
    assert np.allclose(px, tx, atol=1e-3)


def test_anchor_offset_and_agreement():
    rng = np.random.default_rng(1)
    n = 40
    gx, gy = rng.uniform(50, 1950, n), rng.uniform(50, 1950, n)
    gmag = np.r_[rng.uniform(12, 17.5, 30), rng.uniform(19, 21, 10)]
    true = np.array([12.3, -7.6])
    cx = np.r_[gx[:30] + true[0] + rng.normal(0, 0.3, 30), rng.uniform(0, 2048, 60)]   # cores + unrelated clusters
    cy = np.r_[gy[:30] + true[1] + rng.normal(0, 0.3, 30), rng.uniform(0, 2048, 60)]
    a = js.anchor_offset(cx, cy, gx, gy, gmag, 97.0, (0, 2048, 0, 2048))
    assert a is not None
    assert a['dx'] == pytest.approx(true[0], abs=0.5) and a['dy'] == pytest.approx(true[1], abs=0.5)
    assert len(a['anchors'][0]) >= 25
    # identity scale, Gaia frame = guess frame: the right translation agrees, a 15 px artefact lock does not
    med, frac = js.anchor_agreement(1, 0, 0, 1, 0, 0, -true[0], -true[1], gx, gy, a['anchors'])
    assert med < 1.0 and frac > 0.9
    med, frac = js.anchor_agreement(1, 0, 0, 1, 0, 0, -true[0] + 15, -true[1], gx, gy, a['anchors'])
    assert med > js.JWST_ANCHOR_TOL_PX and frac < js.JWST_ANCHOR_MIN_FRAC
    # no bright stars -> no anchors
    assert js.anchor_offset(cx, cy, gx, gy, np.full(n, 20.0), 97.0, (0, 2048, 0, 2048)) is None


@pytest.mark.parametrize('h0,bad', [
    ({'EXP_TYPE': 'NRC_IMAGE', 'PUPIL': 'CLEAR'}, False),
    ({'EXP_TYPE': 'NIS_IMAGE', 'PUPIL': 'F150W'}, False),
    ({'EXP_TYPE': 'MIR_IMAGE'}, False),
    ({'EXP_TYPE': 'NRC_TACONFIRM', 'PUPIL': 'MASKRND'}, True),
    ({'EXP_TYPE': 'MIR_TACQ'}, True),
    ({'EXP_TYPE': 'NRC_IMAGE', 'PUPIL': 'MASKRND'}, True),
    ({'EXP_TYPE': 'NRC_IMAGE', 'PUPIL': 'WLP8'}, True),
    ({'EXP_TYPE': 'NRC_TSIMAGE', 'PUPIL': 'WLP8'}, True),
    ({'EXP_TYPE': 'NRC_GRISM', 'PUPIL': 'GRISMR'}, True),
])
def test_nonscience_reason(h0, bad):
    assert (_nonscience_reason_jwst(h0) is not None) == bad
    assert (_failed_reason_jwst(dict(h0, EFFEXPTM=100.0, ENG_QUAL='OK', VISITSTA='SUCCESSFUL')) is not None) == bad
