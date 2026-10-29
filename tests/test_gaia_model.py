"""bp3m.gaia_model (2026-10-28): the shared Gaia error model must reproduce every pre-refactor implementation BIT FOR BIT
with the default model, honour fab_mag / as_given / floor=none, and reach forkserver workers through the environment."""
import multiprocessing as mp
import numpy as np
import pandas as pd
import pytest

from bp3m.astro_utils import GAIA_SYS_DICT as GS, gaia_cov_scale_vector
from bp3m import gaia_model as GM


def _gaia(n=400, seed=1):
    r = np.random.default_rng(seed)
    t = r.choice([2, 5, 6], n, p=[0.2, 0.4, 0.4])
    d = pd.DataFrame({'ra_error': r.uniform(0.05, 2, n), 'dec_error': r.uniform(0.05, 2, n),
                      'pmra_error': r.uniform(0.05, 3, n), 'pmdec_error': r.uniform(0.05, 3, n),
                      'parallax_error': r.uniform(0.05, 3, n), 'gmag': r.uniform(15, 21.5, n),
                      'pmra': r.normal(0, 1, n), 'pmdec': r.normal(0, 1, n), 'parallax': r.normal(0, 1, n),
                      'pseudocolour': r.uniform(1.3, 1.8, n)})
    for c in ('ra_dec_corr', 'ra_pmra_corr', 'ra_pmdec_corr', 'ra_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
              'dec_parallax_corr', 'pmra_pmdec_corr', 'parallax_pmra_corr', 'parallax_pmdec_corr'):
        d[c] = r.uniform(-0.6, 0.6, n)
    d.loc[t != 6, 'pseudocolour'] = np.nan
    for c in ('pmra', 'pmdec', 'parallax', 'pmra_error', 'pmdec_error', 'parallax_error'):
        d.loc[t == 2, c] = np.nan
    d['astrometric_params_solved'] = np.where(t == 6, 95, np.where(t == 5, 31, 3))
    return d


def _cov_canonical(d, fill_pos=1e6, fill_pm=1e3):
    n = len(d); g = lambda c, f=0.0: d[c].fillna(f).to_numpy(float)
    s = np.stack([g('ra_error', fill_pos), g('dec_error', fill_pos), g('pmra_error', fill_pm), g('pmdec_error', fill_pm),
                  g('parallax_error', fill_pm)], 1)
    R = np.zeros((n, 5, 5)); R[:, range(5), range(5)] = 1
    for i, j, c in ((0, 1, 'ra_dec_corr'), (0, 2, 'ra_pmra_corr'), (0, 3, 'ra_pmdec_corr'), (0, 4, 'ra_parallax_corr'),
                    (1, 2, 'dec_pmra_corr'), (1, 3, 'dec_pmdec_corr'), (1, 4, 'dec_parallax_corr'), (2, 3, 'pmra_pmdec_corr'),
                    (2, 4, 'parallax_pmra_corr'), (3, 4, 'parallax_pmdec_corr')):
        R[:, i, j] = R[:, j, i] = g(c)
    return s[:, :, None] * R * s[:, None, :]


@pytest.fixture(autouse=True)
def _default_model(monkeypatch):
    monkeypatch.delenv(GM.ENV, raising=False)


def test_parse_and_spec():
    assert GM.GaiaModel.parse('').is_default and GM.GaiaModel.parse('default').is_default
    m = GM.GaiaModel.parse('floor=none; cov=fab_mag')
    assert (m.cov, m.floor) == ('fab_mag', 'none') and GM.GaiaModel.parse(m.spec) == m
    for bad in ('cov=nope', 'colour=1', 'fab_mag'):
        with pytest.raises(ValueError):
            GM.GaiaModel.parse(bad)


def test_solver_style_default_bitwise():
    d = _gaia(); C0 = _cov_canonical(d); is6 = np.isfinite(d.pseudocolour).to_numpy(); is5 = np.isfinite(d.pmra).to_numpy() & ~is6
    is2 = ~is5 & ~is6
    old = C0.copy()                                       # pre-2026-10-28 solver._cache_gaia
    old[is6] *= GS['mult_6p'] ** 2; old[is5] *= GS['mult_5p'] ** 2; old[is2] *= GS['mult_2p']
    old += np.diag(np.array([0, 0, GS['pm_sys_err'], GS['pm_sys_err'], GS['parallax_sys_err']]) ** 2)
    new = GM.inflate(C0.copy(), is6, is5, gmag=d.gmag.to_numpy())
    assert np.array_equal(old, new)


def test_solver_style_fab_mag_bitwise():
    d = _gaia(); C0 = _cov_canonical(d); is6 = np.isfinite(d.pseudocolour).to_numpy(); is5 = np.isfinite(d.pmra).to_numpy() & ~is6
    old = C0.copy(); f = gaia_cov_scale_vector(d.gmag.to_numpy(), is6, is5); old *= f[:, :, None] * f[:, None, :]
    old += np.diag(np.array([0, 0, GS['pm_sys_err'], GS['pm_sys_err'], GS['parallax_sys_err']]) ** 2)
    new = GM.inflate(C0.copy(), is6, is5, gmag=d.gmag.to_numpy(), model='cov=fab_mag')
    assert np.array_equal(old, new)


def test_as_given_and_no_floor():
    d = _gaia(); C0 = _cov_canonical(d); is6 = np.isfinite(d.pseudocolour).to_numpy(); is5 = np.isfinite(d.pmra).to_numpy() & ~is6
    assert np.array_equal(GM.inflate(C0.copy(), is6, is5, model='cov=as_given;floor=none'), C0)
    a = GM.inflate(C0.copy(), is6, is5, model='cov=as_given')
    assert np.allclose(np.diagonal(a - C0, axis1=1, axis2=2), [0, 0, GS['pm_sys_err'] ** 2, GS['pm_sys_err'] ** 2, GS['parallax_sys_err'] ** 2])


def test_header_frame_order_bitwise():
    d = _gaia(); n = len(d); perm = [0, 1, 4, 2, 3]                   # header_frame order: ra, dec, plx, pmra, pmdec
    C0 = _cov_canonical(d)[:, perm][:, :, perm]
    has_pm = np.isfinite(d.pmra_error.to_numpy()) & (d.pmra_error.to_numpy() > 0)
    is6 = np.isfinite(d.pseudocolour.to_numpy()) & has_pm; is5 = has_pm & ~is6
    old = C0.copy(); mult = np.where(is6, GS['mult_6p'], np.where(is5, GS['mult_5p'], GS['mult_2p']))
    old *= (mult ** 2)[:, None, None]
    old[:, 2, 2] += GS['parallax_sys_err'] ** 2; old[:, 3, 3] += GS['pm_sys_err'] ** 2; old[:, 4, 4] += GS['pm_sys_err'] ** 2
    new = GM.inflate(C0.copy(), is6, is5, gmag=d.gmag.to_numpy(), idx=(0, 1, 3, 4, 2))
    assert np.array_equal(old, new)
    # and the magnitude-dependent model puts the parallax curve on the parallax axis
    fm = GM.inflate(C0.copy(), is6, is5, gmag=d.gmag.to_numpy(), idx=(0, 1, 3, 4, 2), model='cov=fab_mag')
    ref = GM.inflate(_cov_canonical(d), is6, is5, gmag=d.gmag.to_numpy(), model='cov=fab_mag')[:, perm][:, :, perm]
    assert np.allclose(fm, ref, rtol=0, atol=0)


def test_crossmatch_row_and_pm_sigma_bitwise():
    from bp3m.pipeline.hst_catalog_crossmatch import _build_gaia_cov5
    d = _gaia(80)
    for _, row in d.iterrows():
        C, full = _build_gaia_cov5(row)
        # pre-refactor arithmetic: raw matrix (no parallax correlations, 1e6 fallbacks), scalar multiplier by type, floors
        is6 = np.isfinite(row.pseudocolour); is5 = np.isfinite(row.pmra) and not is6
        s = [row.ra_error, row.dec_error, row.pmra_error, row.pmdec_error, row.parallax_error]
        c = {k: (row[k] if np.isfinite(row[k]) else 0.0) for k in ('ra_dec_corr', 'ra_pmra_corr', 'ra_pmdec_corr', 'dec_pmra_corr',
                                                                    'dec_pmdec_corr', 'pmra_pmdec_corr')}
        if not np.isfinite(s[2]):
            s[2] = s[3] = 1e6; c.update({k: 0.0 for k in c if 'pm' in k})
        if not np.isfinite(s[4]):
            s[4] = 1e6
        R = np.eye(5); R[0, 1] = R[1, 0] = c['ra_dec_corr']; R[0, 2] = R[2, 0] = c['ra_pmra_corr']; R[0, 3] = R[3, 0] = c['ra_pmdec_corr']
        R[1, 2] = R[2, 1] = c['dec_pmra_corr']; R[1, 3] = R[3, 1] = c['dec_pmdec_corr']; R[2, 3] = R[3, 2] = c['pmra_pmdec_corr']
        old = np.outer(s, s) * R
        if is6: old *= GS['mult_6p'] ** 2
        elif is5: old *= GS['mult_5p'] ** 2
        old[2, 2] += GS['pm_sys_err'] ** 2; old[3, 3] += GS['pm_sys_err'] ** 2; old[4, 4] += GS['parallax_sys_err'] ** 2
        assert np.array_equal(C, old) and full == (is5 or is6)
    sig = d.pmra_error.to_numpy(); has_pm = np.isfinite(sig) & (sig > 0); is6 = np.isfinite(d.pseudocolour.to_numpy())
    mult = np.where(is6, GS['mult_6p'], np.where(has_pm, GS['mult_5p'], GS['mult_2p']))
    old = np.where(has_pm, np.sqrt(mult ** 2 * sig ** 2 + GS['pm_sys_err'] ** 2), sig)
    new, _ = GM.inflate_pm_sigma(sig, sig, is6, has_pm)
    assert np.array_equal(old, new, equal_nan=True)


def test_qso_block_bitwise():
    d = _gaia(); d = d[np.isfinite(d.pmra)].reset_index(drop=True); n = len(d)
    is6 = np.isfinite(d.pseudocolour.to_numpy())
    C0 = _cov_canonical(d)[:, 2:5][:, :, 2:5]
    old = C0 * (np.where(is6, GS['mult_6p'], GS['mult_5p']) ** 2)[:, None, None]
    old[:, 0, 0] += GS['pm_sys_err'] ** 2; old[:, 1, 1] += GS['pm_sys_err'] ** 2; old[:, 2, 2] += GS['parallax_sys_err'] ** 2
    new = GM.inflate(C0.copy(), is6, ~is6, idx=(None, None, 0, 1, 2))
    assert np.array_equal(old, new)


def test_catalog_utils_build_gaia_cov_bitwise():
    from bp3m.pipeline.catalog_utils import build_gaia_cov
    d = _gaia(); C0 = _cov_canonical(d); is6 = np.isfinite(d.pseudocolour).to_numpy(); is5 = np.isfinite(d.pmra).to_numpy() & ~is6
    old = C0.copy(); old[is6] *= GS['mult_6p'] ** 2; old[is5] *= GS['mult_5p'] ** 2; old[~is5 & ~is6] *= GS['mult_2p'] ** 2
    old += np.diag(np.array([0, 0, GS['pm_sys_err'], GS['pm_sys_err'], GS['parallax_sys_err']]) ** 2)
    assert np.array_equal(old, build_gaia_cov(d))


def _worker_spec(_):
    from bp3m.gaia_model import current
    return current().spec


@pytest.mark.parametrize('method', ['forkserver', 'spawn', 'fork'])
def test_model_reaches_workers(method):
    GM.set_current('cov=fab_mag;floor=none')
    try:
        with mp.get_context(method).Pool(2) as p:
            got = p.map(_worker_spec, range(2))
        assert [GM.GaiaModel.parse(x) for x in got] == [GM.GaiaModel(cov='fab_mag', floor='none')] * 2
    finally:
        GM.set_current('')


def test_set_gaia_uwu_mag_alias():
    from bp3m.astro_utils import set_gaia_uwu_mag
    set_gaia_uwu_mag(True)
    try:
        assert GM.current().cov == 'fab_mag'
    finally:
        set_gaia_uwu_mag(False)
    assert GM.current().is_default


def test_zero_point_spec_and_legacy():
    assert GM.GaiaModel().zp == 'l21' and 'zp=l21' in GM.GaiaModel().spec
    old = GM.parse_recorded('cov=fab_scalar;floor=diag')           # recorded on 2026-10-28 before the zp key existed
    assert old.zp == 'none' and old == GM.LEGACY and GM.parse_recorded(None) == GM.LEGACY
    assert GM.GaiaModel.parse('cov=fab_mag').zp == 'l21'           # a user-given spec keeps the default zero point
    assert GM.GaiaModel.parse('cov=fab_mag;floor=none;zp=none').zp == 'none'
    with pytest.raises(ValueError):
        GM.GaiaModel.parse('zp=l22')


def test_zero_point_consistency(tmp_path):
    raw = tmp_path / 'f_gaia_raw.csv'; cor = tmp_path / 'f_gaiacorr_l21.csv'
    pd.DataFrame({'source_id': [1], 'parallax': [0.1]}).to_csv(raw, index=False)
    pd.DataFrame({'source_id': [1], 'parallax': [0.12], 'parallax_orig': [0.1], 'zp_l21': [-0.02]}).to_csv(cor, index=False)
    try:
        GM.set_current('cov=fab_scalar;floor=diag;zp=l21'); GM.set_override_files([str(cor)]); GM.check_zp_consistency()
        GM.set_override_files([str(raw)].copy() if False else None)
        with pytest.raises(SystemExit):
            GM.check_zp_consistency()                                # zp=l21 without a corrected table
        GM.set_current('cov=fab_scalar;floor=diag;zp=none'); GM.set_override_files([str(cor)])
        with pytest.raises(SystemExit):
            GM.check_zp_consistency()                                # zp=none but a corrected table
        GM.set_override_files(None); GM.check_zp_consistency()       # legacy: canonical files, no zero point
    finally:
        GM.set_override_files(None); GM.set_current('')
