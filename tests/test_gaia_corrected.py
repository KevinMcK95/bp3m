"""bp3m.gaia_corrected (2026-10-29, handoff 3.4): the BP3M-native corrected Gaia tables on a synthetic field."""
import json
import numpy as np
import pandas as pd
import pytest

from bp3m import gaia_corrected as GC
from bp3m.gaia_sys import gaia_v3 as GV3, prep as P
from bp3m.gaia_sys.qso_sys import correction as qs_correction

P5 = ['ra', 'dec', 'pmra', 'pmdec', 'parallax']
CORR = [f'{P5[i]}_{P5[j]}_corr' for i in range(5) for j in range(i + 1, 5)]


def _corr_name(a, b):
    return {'pmra_parallax_corr': 'parallax_pmra_corr', 'pmdec_parallax_corr': 'parallax_pmdec_corr'}.get(f'{a}_{b}_corr', f'{a}_{b}_corr')


def _sample(n=900, seed=3):
    r = np.random.default_rng(seed)
    sol = r.choice([3, 31, 95], n, p=[0.1, 0.45, 0.45])
    d = pd.DataFrame({'source_id': (np.arange(n, dtype=np.int64) + 1000) * 128 * 7, 'ra': 150 + r.uniform(-0.3, 0.3, n),
                      'dec': 12 + r.uniform(-0.3, 0.3, n), 'astrometric_params_solved': sol, 'gmag': r.uniform(16, 20.9, n),
                      'pmra': r.normal(0, 2, n), 'pmdec': r.normal(0, 2, n), 'parallax': r.normal(0.2, 0.5, n), 'ruwe': r.uniform(0.9, 1.3, n),
                      'bp_rp': r.uniform(0.4, 2.0, n), 'bp_rp_error': r.uniform(0.01, 0.1, n), 'ecl_lat': 0.5,
                      'nu_eff_used_in_astrometry': r.uniform(1.35, 1.65, n), 'pseudocolour': r.uniform(1.4, 1.6, n),
                      'pseudocolour_error': r.uniform(0.02, 0.1, n), 'phot_bp_rp_excess_factor': 1.2,
                      'astrometric_excess_noise_sig': 0.5, 'ipd_frac_multi_peak': 0, 'classprob_dsc_combmod_galaxy': 0.0})
    for c in P5:
        d[f'{c}_error'] = r.uniform(0.1, 0.8, n)
    for c in CORR:
        d[_corr_name(*c.split('_')[:2]) if c.count('_') == 2 else c] = r.uniform(-0.2, 0.2, n)
    for c in P5:
        d[f'{c}_pseudocolour_corr'] = r.uniform(-0.3, 0.3, n)
    d.loc[sol != 31, 'nu_eff_used_in_astrometry'] = np.nan
    d.loc[sol != 95, ['pseudocolour', 'pseudocolour_error'] + [f'{c}_pseudocolour_corr' for c in P5]] = np.nan
    twop = sol == 3
    d.loc[twop, ['pmra', 'pmdec', 'parallax', 'pmra_error', 'pmdec_error', 'parallax_error']] = np.nan
    return d


def _field(tmp_path, drop_col=None):
    S = _sample(); g = tmp_path / 'Gaia'; g.mkdir()
    F = S.drop(columns=['ipd_frac_multi_peak'] + [f'{c}_pseudocolour_corr' for c in P5] + ['pseudocolour_error'])
    F = F.sample(frac=0.6, random_state=1).reset_index(drop=True)            # field = a subset, in its own order
    fp = g / 'T_ra150_G0.0_gaia.csv'; F.to_csv(fp, index=False)
    s = S if drop_col is None else S.assign(**{drop_col: np.where(S.astrometric_params_solved == 95, np.nan, S[drop_col])})
    sp = g / 'T_condsample_r0.5.csv'; s.to_csv(sp, index=False)
    q = F[F.astrometric_params_solved.isin([31, 95])].head(6)
    A = pd.DataFrame({'source_id': q.source_id, 'catalog_match': [True] * 4 + [False] * 2, 'is_qso_anchor': [True] * 4 + [False] * 2})
    ap = g / 'T_ra150_qso_anchors.csv'; A.to_csv(ap, index=False)
    return fp, ap, sp, F, S, set(A.source_id[A.catalog_match])


def test_build_synthetic(tmp_path):
    fp, ap, sp, F, S, qso = _field(tmp_path)
    w = GC.build(fp, ap, sp, verbose=False)
    cal = GV3.load(str(GC.KERNEL))
    for model, (cp, apath) in w.items():
        C = pd.read_csv(cp); meta = json.loads(open(str(cp) + '.json').read())
        assert (C.source_id.values == F.source_id.values).all()
        for c in ('pmra', 'pseudocolour', 'ra', 'dec', 'parallax'):
            assert (C[c].isna() == F[c].isna()).all(), (model, c)
        two = F.astrometric_params_solved == 3
        assert np.allclose(C.loc[two, 'pmra_error'].fillna(-1), F.loc[two, 'pmra_error'].fillna(-1))       # 2p untouched
        assert (C.value_source[two] == 'none').all()
        # 5p star rows: published - L21 - qso_sys (- star zp for the parallax); no conditioning
        five = (F.astrometric_params_solved == 31) & ~F.source_id.isin(qso)
        Ff = F[five]; qs = qs_correction(str(GC.QSOSYS), Ff.ra.values, Ff.dec.values, Ff.gmag.values, Ff.astrometric_params_solved.values, Ff.pseudocolour.values)
        zp = P.l21(Ff.reset_index(drop=True)).parallax_zp.values
        assert np.allclose(C.pmra[five], Ff.pmra - qs[:, 0], atol=1e-12)
        assert np.allclose(C.parallax[five], Ff.parallax - zp - qs[:, 2] - GV3.zp_star(cal, Ff.gmag.values), atol=1e-12)
        has = C.pmra.notna()
        Cv = P.cov(C[has], P5)
        assert (np.linalg.eigvalsh(Cv).min(axis=1) > 0).all(), model
        assert meta['corrections_qso_rows'] == ['l21_zp', 'qso_sys'] and 'l21_zp' in meta['corrections']
        if meta['position_inflation'] == 'gm':
            # f = geometric mean of the PM / parallax inflation relative to the conditioned published block
            k = np.cbrt((C.pmra_error / F.pmra_error) * (C.pmdec_error / F.pmdec_error) * (C.parallax_error / F.parallax_error))
            f = C.ra_error / F.ra_error
            assert np.allclose(f[five], k[five], rtol=1e-9)                  # 5p rows are unconditioned: exact
        if meta['position_inflation'] == 'hst':
            # 5p rows: ra_error^2 = (k_pos(G) sigma_published)^2 + c_pos^2 with the table recorded in the sidecar
            tb = meta['position_table']; b = np.digitize(F.gmag.values, tb['g_edges'])
            k = np.asarray(tb['k_pos']['31'])[b]; c = tb['c_pos_mas']['31']
            assert np.allclose(C.ra_error[five] ** 2, (k[five] * F.ra_error[five]) ** 2 + c ** 2, rtol=1e-9)
        if meta['position_inflation'] == 'fab':
            assert np.allclose((C.ra_error / F.ra_error)[five], 1.05, rtol=1e-9)
        if model == 'corrvals':
            assert np.allclose(C.pmra_error[five], F.pmra_error[five])
        A = pd.read_csv(apath)
        assert set(A.source_id) <= set(C.source_id) and A.is_qso_anchor.sum() <= 4


def test_missing_conditioning_input_is_an_error(tmp_path):
    fp, ap, sp, F, S, qso = _field(tmp_path, drop_col='ra_pseudocolour_corr')
    with pytest.raises(SystemExit):
        GC.build(fp, ap, sp, verbose=False)


def test_field_rows_must_be_in_the_sample(tmp_path):
    fp, ap, sp, F, S, qso = _field(tmp_path)
    s = pd.read_csv(sp); s = s[~s.source_id.isin(F.source_id[F.astrometric_params_solved == 31].head(3))]; s.to_csv(sp, index=False)
    with pytest.raises(SystemExit):
        GC.build(fp, ap, sp, verbose=False)
