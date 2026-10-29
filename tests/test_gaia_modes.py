"""bp3m.gaia_modes (2026-10-29): field-mode file on a synthetic field (tiny MILLIQUAS, QSO sample, Gaia table)."""
import json
import numpy as np
import pandas as pd
import pytest

from bp3m import gaia_modes as GMo
from bp3m.gaia_sys import gaia_v3 as GV3


def _setup(tmp_path, n=120, nq=160, seed=5):
    from astropy.io import fits
    r = np.random.default_rng(seed); c = (150.0, 12.0)
    G = pd.DataFrame({'source_id': (np.arange(n, dtype=np.int64) + 10) * 128, 'ra': c[0] + r.uniform(-0.3, 0.3, n), 'dec': c[1] + r.uniform(-0.3, 0.3, n),
                      'gmag': r.uniform(17, 21, n), 'pmra': r.normal(0, 1, n), 'astrometric_params_solved': r.choice([31, 95], n),
                      'pseudocolour': np.nan})
    G.loc[G.astrometric_params_solved == 95, 'pseudocolour'] = 1.5
    fp = tmp_path / 'T_G0.0_gaia.csv'; G.to_csv(fp, index=False)
    qra, qde = c[0] + r.uniform(-3, 3, nq) / np.cos(np.radians(c[1])), c[1] + r.uniform(-3, 3, nq)
    q = pd.DataFrame({'source_id': (np.arange(nq, dtype=np.int64) + 5000) * 128, 'ra': qra, 'dec': qde, 'pmra': r.normal(0, 0.4, nq), 'pmdec': r.normal(0, 0.4, nq),
                      'pmra_error': 0.4, 'pmdec_error': 0.4, 'pmra_pmdec_corr': 0.0, 'parallax': r.normal(0, 0.3, nq), 'parallax_error': 0.3,
                      'parallax_pmra_corr': 0.0, 'parallax_pmdec_corr': 0.0, 'phot_g_mean_mag': r.uniform(18.5, 20.8, nq), 'phot_bp_mean_mag': 20.0,
                      'phot_rp_mean_mag': 19.5, 'ruwe': 1.0, 'nu_eff_used_in_astrometry': 1.55, 'pseudocolour': np.nan, 'ecl_lat': 0.5,
                      'astrometric_params_solved': 31})
    q.loc[0, ['ra', 'dec']] = G.loc[0, ['ra', 'dec']].values; q.loc[0, 'source_id'] = G.source_id[0]   # a QSO that is a field source
    qp = tmp_path / 'q.csv'; q.to_csv(qp, index=False)
    mq = tmp_path / 'mq.fits'
    fits.BinTableHDU.from_columns([fits.Column('RA', 'D', array=q.ra.values[:140]), fits.Column('DEC', 'D', array=q.dec.values[:140])]).writeto(mq)
    an = tmp_path / 'T_qso_anchors_v3real2p.csv'; pd.DataFrame({'source_id': q.source_id[1:4]}).to_csv(an, index=False)
    return fp, an, qp, mq, c, G


def test_build_modes_synthetic(tmp_path):
    fp, an, qp, mq, c, G = _setup(tmp_path)
    out = GMo.build_modes([fp], [an], qp, mq, c, 0.42, tmp_path / 'm.npz', chunk=50, verbose=False)
    z = np.load(out); meta = json.loads(str(z['meta']))
    N = len(G); K = z['W'].shape[2]
    assert z['M'].shape == (N, 3) and z['R'].shape == (N, 3, 3) and z['W'].shape[:2] == (N, 3) and K == meta['K']
    assert set(pd.read_csv(an).source_id) <= set(z['qso_excluded'])          # anchors excluded from the conditioning
    assert int(G.source_id[0]) in set(z['qso_excluded'])                      # a QSO that is a field source too
    assert meta['qso']['n_excluded_in_sample'] >= 1 and meta['trace_fraction'] >= 0.999
    assert (np.linalg.eigvalsh(z['R']).min(axis=1) > -1e-12).all()            # R is a covariance
    # W W^T + R reproduces the full projected field covariance W3 C_Z W3^T + FITC at each source
    cal = GV3.load(str(GMo.KERNEL))
    q = pd.read_csv(qp); ra, de, res, S, Gq, sol, exc, st = GMo.prepare_qsos(qp, mq, set(pd.read_csv(an).source_id) | set(G.source_id), c)
    g = GV3.build_gp_v3(cal, c, max(0.3, 0.42), G.ra.values, G.dec.values, G.gmag.values, G.astrometric_params_solved.values, ra, de, res, S, Gq, sol,
                        n_rings=meta['n_rings'], max_qso=GMo.MAX_QSO)
    full = np.einsum('nia,ab,njb->nij', g['W3'], g['L_Z'] @ g['L_Z'].T, g['W3']) + g['fitc']
    recon = np.einsum('nik,njk->nij', z['W'], z['W']) + z['R']
    assert np.allclose(recon, full, rtol=1e-6, atol=1e-12)
    assert np.allclose(z['M'], g['W3'] @ g['m_Z'], rtol=1e-10, atol=1e-14)


def test_in_footprints_padding():
    """Inside a square footprint, within the padding outside it, and beyond the padding."""
    P = np.array([[10.0, 0.0], [10.01, 0.0], [10.01, 0.01], [10.0, 0.01]])
    ra = np.array([10.005, 10.0105, 10.0130, 9.9990, 10.005])
    de = np.array([0.005, 0.005, 0.005, 0.005, 0.0118])          # last: 6.5" above the top edge
    keep = GMo.in_footprints(ra, de, [P], pad_arcsec=5.0)          # 5" = 0.00139 deg
    assert keep.tolist() == [True, True, False, True, False]
