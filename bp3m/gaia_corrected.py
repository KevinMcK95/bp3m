"""BP3M-native corrected Gaia tables + QSO anchor files (handoff hst_dist_corr/GAIA_SYSTEMATICS_BP3M_HANDOFF.md 3.1 / 3.2a /
3.2b / 3.4; user 2026-10-29: "bp3m itself [must be] able to entirely run everything it needs ... especially for a new target
downloaded from scratch").  Port of the pop_fit reference builder hst_dist_corr/gaia_sys/build_corrected_copy.py (v2, 10-29),
whose 10-29 outputs in GaiaHub_results/{Leo_I,Draco_dSph,Sculptor_dSph,Fornax_dSph}/Gaia/corrected/ are frozen regression
fixtures.  Model inputs (kernel-v3 real2p JSON, qso_sys npz) and code are vendored in bp3m/gaia_sys + bp3m/data/gaia_sys with
their sha256 checked; pop_fit owns them and announces changes in the handoff.

Inputs per field: the active field Gaia CSV (Gaia/<F>_gaia_files.json), its vetted anchors file (<anchor stem>_qso_anchors.csv;
catalog_match marks the QSO rows), and a CONDITIONING SAMPLE: BP3M's own Gaia cone download (download_conditioning_sample: 5p/6p,
all G, every column the preparation needs) covering the field with a margin.  The 6p conditioning is sample dependent (5p-
calibrated colour reference, per-bin centring / mean preservation) and n10 (Gaia 5p/6p sources within 10", the calibration's
convention: pop_fit's downloads hold 5p/6p only) is counted on it.
Values (rows, order, NaN pattern of every column kept): star rows = pop_fit production preparation (RUWE < 1.4 / 5p-6p / G cuts,
L21, 6p conditioning of ALL FIVE parameters, quality cuts with C* = missing colour, qso_sys) + the kernel's stars' parallax offset;
rows of the sample passing the cuts take that preparation (value_source calib_sample), other star rows the same steps without the
cuts (extended); catalog-matched QSO rows L21 + qso_sys only (qso); 2p rows untouched (none).
Covariance (v3real2p*): PM / parallax block = kernel-v3 noise_cov of the conditioned block (n10 = 0 for QSO rows); positions
scaled by f: 'gm' (default, user 10-29 interim) = per-star geometric mean of the v3 inflation of (pmra, pmdec, parallax),
f = (k_pmra k_pmdec k_plx)^(1/3), k = sigma_v3 / sigma_conditioned; 'fab' = Fabricius 1.05 (5p) / 1.22 (6p) as the reference
build.  Full 5x5 = B C5c B^T, B = blockdiag(f I2, L_new L_c^-1): PD by construction, PM / parallax block exactly the v3 model.
corrvals: the same values with the published errors / correlations (arm A', inflated by bp3m as usual).
Outputs go to Gaia/corrected_bp3m/ (never over the reference fixtures in Gaia/corrected/):
  <stem>_gaiacorr_<model>.csv + .json, <anchor stem>_qso_anchors_<model>.csv + .json, models v3real2p (positions 'gm'),
  v3real2p_fab, v3real2p_mc (median-centred conditioning, positions 'gm'), corrvals.
Missing inputs are hard errors (no fallback)."""
from __future__ import annotations
import datetime, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd

from .gaia_sys import prep as P
from .gaia_sys import gaia_v3 as GV3

DATA = Path(__file__).resolve().parent / 'data' / 'gaia_sys'
KERNEL = DATA / 'gaia_dr3_kernel_v3_real2p.json'
QSOSYS = DATA / 'qso_sys_model_dr3.npz'
SHA256 = {KERNEL.name: '30a5aa238a52c8f8e9e30716bc4229dd9c21c1013042b85b13fe233c0dd8b302',
          QSOSYS.name: '5a6477c1e45c63e180feb1a896b0b3dcb6c09fab1e206475a437ed34f644c5ca'}
MULT5, MULT6, FLOOR_PM, FLOOR_PLX, SIGMA_CUT = 1.05, 1.22, 0.026, 0.011, 3.0      # bp3m GAIA_SYS_DICT, qso_vetting
QSO_CORR = ['l21_zp', 'qso_sys']
STAR_CORR = ['l21_zp', '6p_pseudocolour', 'qso_sys', 'v3_star_zp']
MODELS = {'v3real2p': ('mp', 'gm', 'v3_noise_cov'), 'v3real2p_fab': ('mp', 'fab', 'v3_noise_cov'),
          'v3real2p_mc': ('mc', 'gm', 'v3_noise_cov'), 'corrvals': ('mp', None, 'published')}
OUT_DIR = 'corrected_bp3m'
BUILD_VERSION = 2       # 2: C*-masked 6p rows restore the position block with the PM block
P5, ASTRO = P.P5, P.ASTRO
POS_COLS = ['ra_error', 'dec_error', 'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr', 'dec_parallax_corr',
            'dec_pmra_corr', 'dec_pmdec_corr']
# every column the preparation reads (pop_fit's download + positions and their pseudocolour correlations)
SAMPLE_QUERY_COLS = (
    'source_id, ra, dec, ra_error, dec_error, ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr, '
    'dec_pmra_corr, dec_pmdec_corr, pmra, pmra_error, pmdec, pmdec_error, pmra_pmdec_corr, parallax, parallax_error, '
    'parallax_pmra_corr, parallax_pmdec_corr, phot_g_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error, phot_bp_mean_mag, '
    'phot_bp_mean_flux, phot_bp_mean_flux_error, phot_rp_mean_mag, phot_rp_mean_flux, phot_rp_mean_flux_error, bp_rp, ruwe, '
    'phot_bp_rp_excess_factor, astrometric_excess_noise_sig, ipd_frac_multi_peak, classprob_dsc_combmod_galaxy, '
    'ipd_gof_harmonic_amplitude, visibility_periods_used, phot_bp_n_blended_transits, phot_rp_n_blended_transits, '
    'phot_bp_n_obs, phot_rp_n_obs, nu_eff_used_in_astrometry, pseudocolour, ecl_lat, astrometric_params_solved, '
    'pseudocolour_error, ra_pseudocolour_corr, dec_pseudocolour_corr, parallax_pseudocolour_corr, pmra_pseudocolour_corr, '
    'pmdec_pseudocolour_corr')
G_SLICES = [(None, 17.0), (17.0, 18.5), (18.5, 19.5), (19.5, 20.0), (20.0, 20.5), (20.5, 21.0), (21.0, None), ('null', None)]


def _sha(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def check_inputs():
    """The vendored model inputs must be the announced versions (hard error otherwise)."""
    for p in (KERNEL, QSOSYS):
        if _sha(p) != SHA256[p.name]:
            raise SystemExit(f'{p}: sha256 differs from the announced {SHA256[p.name]} (re-vendor from pop_fit)')


# -- conditioning sample -------------------------------------------------------------------------------------------------
def sample_radius_deg(width_deg: float, height_deg: float) -> float:
    """Cone covering the field box with a 10 % margin, at least 30' (the r3b minimum)."""
    return max(0.5, 1.1 * 0.5 * math.hypot(width_deg, height_deg))


def standardise_sample(df: pd.DataFrame) -> pd.DataFrame:
    """pop_fit's column layout (_standardise_gaia_df): lower-case names, magnitude errors from fluxes with 2.5/ln 10,
    bp_rp_error = quadrature sum, gmag / bpmag / rpmag names, booleans untouched (none queried)."""
    df = df.copy(); df.columns = [c.lower() for c in df.columns]
    c = 2.5 / math.log(10)
    for f, fe, e in (('phot_g_mean_flux', 'phot_g_mean_flux_error', 'gmag_error'),
                     ('phot_bp_mean_flux', 'phot_bp_mean_flux_error', 'bpmag_error'),
                     ('phot_rp_mean_flux', 'phot_rp_mean_flux_error', 'rpmag_error')):
        fl = df[f].to_numpy(float); fe_ = df[fe].to_numpy(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            df[e] = np.where(fl > 0, c * fe_ / fl, np.nan)
    df['bp_rp_error'] = np.sqrt(df['bpmag_error']**2 + df['rpmag_error']**2)
    df = df.rename(columns={'phot_g_mean_mag': 'gmag', 'phot_bp_mean_mag': 'bpmag', 'phot_rp_mean_mag': 'rpmag'})
    df['source_id'] = df['source_id'].astype(np.int64)
    return df


def sample_path(gaia_dir: Path, stem: str, radius_deg: float) -> Path:
    return Path(gaia_dir) / f'{stem}_condsample_r{radius_deg:.4f}.csv'


def download_conditioning_sample(gaia_dir, stem: str, ra: float, dec: float, radius_deg: float,
                                 gaia_tap_server: str | None = None) -> Path:
    """Gaia DR3 5p/6p sources (all G) in a cone, every column the preparation needs, in G slices (each checked against the TAP
    output limits) with the total verified by COUNT(*).  Cached with a sidecar (query, counts, sha256); a cache whose sidecar
    does not verify is re-queried, never used."""
    from .pipeline.download_gaia import _submit_gaia_async
    out = sample_path(gaia_dir, stem, radius_deg); side = Path(str(out) + '.json')
    if out.exists() and side.exists():
        meta = json.loads(side.read_text())
        if meta.get('sha256') == _sha(out) and meta.get('row_count_verified') and meta.get('columns') == SAMPLE_QUERY_COLS:
            return out
    where = (f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_deg:.6f})) = 1 "
             f"AND astrometric_params_solved >= 31")
    parts, counts = [], []
    for lo, hi in G_SLICES:
        if lo == 'null':
            gw = 'phot_g_mean_mag IS NULL'
        else:
            gw = ' AND '.join(([f'phot_g_mean_mag >= {lo}'] if lo is not None else []) + ([f'phot_g_mean_mag < {hi}'] if hi is not None else []))
        d = _submit_gaia_async(f'SELECT {SAMPLE_QUERY_COLS} FROM gaiadr3.gaia_source WHERE {where} AND {gw}', gaia_tap_server)
        parts.append(d); counts.append(len(d))
    n_tot = int(_submit_gaia_async(f'SELECT COUNT(*) AS n FROM gaiadr3.gaia_source WHERE {where}', gaia_tap_server).iloc[0, 0])
    df = pd.concat(parts, ignore_index=True)
    if len(df) != n_tot or df.source_id.duplicated().any():
        raise SystemExit(f'conditioning sample: {len(df)} rows from the G slices vs COUNT(*) {n_tot} (or duplicates)')
    df = standardise_sample(df)
    tmp = Path(str(out) + '.tmp'); df.to_csv(tmp, index=False); tmp.replace(out)
    side.write_text(json.dumps({'ra': ra, 'dec': dec, 'radius_deg': radius_deg, 'where': where, 'columns': SAMPLE_QUERY_COLS,
                                'g_slices': [list(map(str, s)) for s in G_SLICES], 'slice_counts': counts, 'n_rows': len(df),
                                'row_count_verified': True, 'sha256': _sha(out),
                                'created': datetime.datetime.now().isoformat(timespec='seconds')}, indent=1))
    return out


# -- builder -------------------------------------------------------------------------------------------------------------
def _position_factor(kind: str, C5c, S_new, sol):
    if kind == 'fab':
        return np.where(sol == 95, MULT6, MULT5)
    if kind == 'gm':
        k = np.sqrt(np.einsum('nii->ni', S_new) / np.einsum('nii->ni', C5c[:, 2:, 2:]))
        return np.cbrt(np.prod(k, axis=1))
    raise ValueError(kind)


def build(field_csv, anchors_csv, sample_csv, out_dir=None, models=None, sample_meta: dict | None = None,
          verbose: bool = True) -> dict:
    """Build the corrected tables for one field Gaia CSV.  Returns {model: (copy path, anchors path)}."""
    check_inputs()
    field_csv, anchors_csv, sample_csv = Path(field_csv), Path(anchors_csv), Path(sample_csv)
    out_dir = Path(out_dir) if out_dir else field_csv.parent / OUT_DIR
    models = models or list(MODELS)
    cal = GV3.load(str(KERNEL))
    G0 = pd.read_csv(field_csv, dtype={'source_id': 'int64'}, low_memory=False)
    if not (G0.source_id.is_unique and (G0.source_id % 128 == 0).all()):
        raise SystemExit(f'{field_csv.name}: source_id integrity')
    S0 = pd.read_csv(sample_csv, dtype={'source_id': 'int64'}, low_memory=False)
    A = pd.read_csv(anchors_csv, dtype={'source_id': 'int64'}, low_memory=False)
    out_csv = ~A.source_id.isin(G0.source_id)
    if (out_csv & (A.catalog_match.astype(bool) | A.is_qso_anchor.astype(bool))).any():
        raise SystemExit(f'{anchors_csv.name}: catalog-matched QSOs / anchors not in the Gaia CSV')
    dropped_ids = A.source_id[out_csv].astype(np.int64).tolist()
    A = A[~out_csv].reset_index(drop=True)
    qso_ids = set(A.loc[A.catalog_match.astype(bool), 'source_id'])
    # every 5p/6p field row must be in the conditioning sample (its preparation columns come from there); published values
    # must agree (same DR3 rows)
    f56 = G0.astrometric_params_solved.isin([31, 95])
    miss = ~G0.source_id[f56].isin(S0.source_id)
    if miss.any():
        raise SystemExit(f'{field_csv.name}: {int(miss.sum())} 5p/6p field rows are not in the conditioning sample (enlarge it)')
    chk = G0[f56][['source_id', 'pmra', 'parallax', 'ra']].merge(S0[['source_id', 'pmra', 'parallax', 'ra']], on='source_id', suffixes=('', '_s'))
    for c in ('pmra', 'parallax', 'ra'):
        if not np.allclose(chk[c], chk[c + '_s'], rtol=0, atol=1e-6 if c != 'ra' else 1e-9, equal_nan=True):
            raise SystemExit(f'{field_csv.name}: {c} differs between the field CSV and the conditioning sample (different release?)')
    U = S0.copy()
    six_u = U.astrometric_params_solved == 95
    for c in ['pseudocolour_error'] + [f'{p}_pseudocolour_corr' for p in P5] + POS_COLS:
        bad = six_u & U[c].isna() & U.pseudocolour.notna()
        if bad.any():
            raise SystemExit(f'conditioning sample: {int(bad.sum())} 6p rows lack {c}')
    base = U[U.astrometric_params_solved.isin([31, 95])].reset_index(drop=True)
    cut0 = P._drop_no_g(base[base.ruwe < P.RUWE_MAX].reset_index(drop=True))
    dom_ids = set(P.apply_quality_cuts(cut0.copy()).source_id)          # calibration cuts -> v3_domain
    vals, stats = {}, {}
    modes = sorted({MODELS[m][0] for m in models})
    for mode in modes:
        a, sa = P.cond5(P.l21(cut0, 'sample'), mode, 'sample (cuts)')
        a = P.qsosys(P.restore_positions(P.apply_quality_cuts(a)), str(QSOSYS)).set_index('source_id')
        b, sb = P.cond5(P.l21(base, 'sample'), mode, 'sample (no cuts)')
        b = P.qsosys(b, str(QSOSYS)).set_index('source_id')
        vals[mode], stats[mode] = (a, b), dict(calib_sample=sa, extended=sb)
    Q = base[base.source_id.isin(qso_ids)].reset_index(drop=True)
    Q = P.qsosys(P.l21(Q, 'QSOs'), str(QSOSYS)).set_index('source_id')
    # -- per field row
    sid = G0.source_id.to_numpy(np.int64); sol = G0.astrometric_params_solved.to_numpy()
    has = np.isin(sol, [31, 95]) & G0.pmra.notna().to_numpy()
    is_q = np.isin(sid, list(qso_ids)) & has
    in_a = np.isin(sid, vals[modes[0]][0].index.to_numpy()) & has & ~is_q
    in_b = has & ~is_q & ~in_a
    if not np.isin(sid[in_b], vals[modes[0]][1].index.to_numpy()).all():
        raise SystemExit(f'{field_csv.name}: rows missing from the uncut pass')
    src = np.where(is_q, 'qso', np.where(in_a, 'calib_sample', np.where(in_b, 'extended', 'none')))
    Gm = G0.gmag.to_numpy(float)
    zp = np.zeros(len(G0)); star = has & ~is_q; zp[star] = GV3.zp_star(cal, Gm[star])
    n10 = np.zeros(len(G0))
    n10[has] = GV3.count_n10(G0.ra.to_numpy(float)[has], G0.dec.to_numpy(float)[has], base.ra.to_numpy(float),
                             base.dec.to_numpy(float), cal['n10_radius_arcsec'])
    n10[is_q] = 0.0
    X, C5c = {}, None
    for mode in modes:
        Xm = np.full((len(G0), 5), np.nan); Cm = np.full((len(G0), 5, 5), np.nan)
        for m, T in ((in_a, vals[mode][0]), (in_b, vals[mode][1]), (is_q, Q)):
            t = T.loc[sid[m]]
            Xm[m] = t[P5].to_numpy(float); Cm[m] = P.cov(t, P5)
        Xm[:, 4] -= zp
        X[mode] = Xm
        if C5c is None:
            C5c = Cm
        elif not np.allclose(Cm[has], C5c[has], rtol=1e-10, atol=1e-15):
            raise SystemExit(f'{field_csv.name}: conditioned covariance depends on the centring mode')
    S_new = np.full((len(G0), 3, 3), np.nan)
    S_new[has] = GV3.noise_cov(cal, C5c[has, 2:, 2:], Gm[has], n10[has], sol[has])
    L = np.linalg.cholesky(S_new[has]) @ np.linalg.inv(np.linalg.cholesky(C5c[has, 2:, 2:]))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = field_csv.name[:-4]
    code = {p.name: _sha(p) for p in [Path(__file__), Path(P.__file__), Path(GV3.__file__), Path(P.__file__).parent / 'qso_sys.py']}
    created = datetime.datetime.now().isoformat(timespec='seconds')
    written = {}
    for model in models:
        mode, pos, covname = MODELS[model]
        out = G0.copy()
        for j, c in enumerate(P5):
            out[f'{c}_orig'] = G0[c]
            out[c] = out[c].astype('float64'); out.loc[has, c] = X[mode][has, j]
        out['v3_domain'] = np.isin(sid, list(dom_ids)) & has
        out['value_source'] = src; out['n10'] = n10; out['gaia_model'] = model
        fpos = None
        if covname == 'v3_noise_cov':
            fpos = _position_factor(pos, C5c[has], S_new[has], sol[has])
            B = np.zeros((has.sum(), 5, 5)); B[:, 0, 0] = B[:, 1, 1] = fpos; B[:, 2:, 2:] = L
            C5v3 = B @ C5c[has] @ np.transpose(B, (0, 2, 1))
            if not np.allclose(C5v3[:, 2:, 2:], S_new[has], rtol=1e-8, atol=1e-14):
                raise SystemExit('v3 block not reproduced')
            if not (np.linalg.eigvalsh(C5v3).min(axis=1) > 0).all():
                raise SystemExit(f'{field_csv.name}: non-PD covariance')
            for c in [f'{p}_error' for p in P5] + [P.corr_col(out, P5[i], P5[j]) for i in range(5) for j in range(i + 1, 5)]:
                out[c] = out[c].astype('float64')
            P.put_cov(out, C5v3, P5, out.index[has])
        p = out_dir / f'{stem}_gaiacorr_{model}.csv'
        out.to_csv(p, index=False)
        meta = dict(model=model, builder='bp3m.gaia_corrected', build_version=BUILD_VERSION, corrections=(
                        STAR_CORR if mode == 'mp' else ['l21_zp', '6p_pseudocolour_mc', 'qso_sys', 'v3_star_zp']),
                    corrections_qso_rows=QSO_CORR, covariance=covname, position_inflation=pos,
                    position_factor_median_5p_6p=(None if fpos is None else [float(np.median(fpos[sol[has] == 31])),
                                                                              float(np.median(fpos[sol[has] == 95]))]),
                    conditioned_parameters=P5, six_p_mode={'mp': 'exact per-bin weighted-mean preserving (r3b, 10-21)',
                                                           'mc': 'median-centred (the 10-16 kernel-calibration inputs)'}[mode],
                    six_p_stats=stats[mode], n10_convention='5p/6p sources of the conditioning sample within 10 arcsec',
                    source_csv=field_csv.name, sha256_source_csv=_sha(field_csv),
                    conditioning_sample=sample_csv.name, sha256_conditioning_sample=_sha(sample_csv), conditioning_sample_meta=sample_meta,
                    calibration=KERNEL.name, sha256_calibration=SHA256[KERNEL.name], qso_sys=QSOSYS.name,
                    sha256_qso_sys=SHA256[QSOSYS.name], code=code, qso_rows='catalog_match in ' + anchors_csv.name,
                    counts={s_: int((src == s_).sum()) for s_ in ('calib_sample', 'extended', 'qso', 'none')},
                    n_v3_domain=int(out.v3_domain.sum()), created=created)
        Path(str(p) + '.json').write_text(json.dumps(meta, indent=1))
        # anchors: astrometric test on this model's values and covariance
        a = A.copy(); m = a.source_id.isin(sid[has]).to_numpy()
        r = pd.Series(np.arange(len(G0)), index=sid).loc[a.source_id[m]].to_numpy()
        Cq = (S_new[r] if covname == 'v3_noise_cov' else
              P.cov(G0.iloc[r], ASTRO) * np.where(sol[r] == 95, MULT6, MULT5)[:, None, None]**2
              + np.diag([FLOOR_PM**2, FLOOR_PM**2, FLOOR_PLX**2]))
        from .pipeline.secular_aberration import secular_aberration_pm
        ab = np.column_stack(secular_aberration_pm(G0.ra.to_numpy(float)[r], G0.dec.to_numpy(float)[r])) / 1e3
        d = X[mode][r, 2:] - np.column_stack([ab, np.zeros(len(r))])
        mah = np.sqrt(np.einsum('ni,nij,nj->n', d, np.linalg.inv(Cq), d))
        a['has_5p_solution'] = m
        a['chi2_astrometric'] = np.nan; a.loc[m, 'chi2_astrometric'] = mah
        a['astrometric_pass'] = False; a.loc[m, 'astrometric_pass'] = mah < SIGMA_CUT
        a['is_qso_anchor'] = a.catalog_match.astype(bool) & a.astrometric_pass
        pa = out_dir / f'{anchors_csv.name[:-4]}_{model}.csv'
        a.to_csv(pa, index=False)
        Path(str(pa) + '.json').write_text(json.dumps(dict(
            model=model, builder='bp3m.gaia_corrected', corrections=QSO_CORR,
            covariance=covname + (' (n10 = 0)' if covname != 'published' else ' x Fabricius + VB21 floor'),
            source_anchors=anchors_csv.name, sha256_source_anchors=_sha(anchors_csv), dropped_not_in_gaia_csv=dropped_ids,
            sigma_cut=SIGMA_CUT, source_csv=field_csv.name, sha256_source_csv=_sha(field_csv), calibration=KERNEL.name,
            sha256_calibration=SHA256[KERNEL.name], code=code, created=created), indent=1))
        written[model] = (p, pa)
        if verbose:
            print(f'  {model}: anchors {int(A.is_qso_anchor.sum())} -> {int(a.is_qso_anchor.sum())}; rows {meta["counts"]}; '
                  f'v3 domain {meta["n_v3_domain"]:,}; position factor median 5p / 6p {meta["position_factor_median_5p_6p"]}')
    return written


# -- bp3m_run integration (--gaia_corrected MODEL) ----------------------------------------------------------------------
def default_spec(model: str) -> str:
    """Gaia-model spec that consumes a table: v3 covariances AS GIVEN without the VB21 floor (field modes add floor=modes);
    corrvals keeps bp3m's Fabricius + VB21 inflation (arm A').  All carry L21 (the table has it; never applied twice)."""
    if model not in MODELS:
        raise SystemExit(f'--gaia_corrected {model!r}: one of {sorted(MODELS)}')
    return 'cov=fab_scalar;floor=diag;zp=l21' if MODELS[model][2] == 'published' else 'cov=as_given;floor=none;zp=l21'


def legacy_anchors(output_dir, field_name, ra, dec, width, height, lib_dir=None) -> Path:
    """The field's QSO vetting on the PUBLISHED values (canonical *_qso_anchors.csv, zp=none): its catalog_match flags define the
    QSO rows; the builder redoes the astrometric test on each model's values.  Runs the vetting if the file is missing."""
    import os
    from .pipeline import qso_vetting as QV
    old = os.environ.get('BP3M_GAIA_MODEL'), os.environ.get('BP3M_GAIA_FILES')
    os.environ['BP3M_GAIA_MODEL'] = 'cov=fab_scalar;floor=diag;zp=none'; os.environ.pop('BP3M_GAIA_FILES', None)
    try:
        p = QV.qso_anchors_path(Path(output_dir) / field_name / 'Gaia', field_name, ra, dec, width, height)
        if not p.exists():
            QV.vet_qso_candidates(field_name=field_name, output_dir=Path(output_dir), ra=ra, dec=dec,
                                  search_width=width, search_height=height, lib_dir=lib_dir)
    finally:
        for k, v in zip(('BP3M_GAIA_MODEL', 'BP3M_GAIA_FILES'), old):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    if not p.exists():
        raise SystemExit(f'QSO vetting did not produce {p}')
    return p


def ensure_field(field_csv, anchors_csv, ra: float, dec: float, width: float, height: float, model: str,
                 gaia_tap_server: str | None = None) -> Path:
    """The model table for one field Gaia CSV: conditioning sample downloaded (or reused when its sidecar verifies), tables
    built (or reused when every input sha256, the code and BUILD_VERSION match).  Returns the table's path."""
    field_csv, anchors_csv = Path(field_csv), Path(anchors_csv)
    gdir = field_csv.parent; stem = field_csv.name[:-len('_G0.0_gaia.csv')] if field_csv.name.endswith('_G0.0_gaia.csv') else field_csv.stem
    r = sample_radius_deg(width, height)
    sp = download_conditioning_sample(gdir, stem, ra, dec, r, gaia_tap_server)
    out = gdir / OUT_DIR / f'{field_csv.name[:-4]}_gaiacorr_{model}.csv'
    side = Path(str(out) + '.json')
    code = {p.name: _sha(p) for p in [Path(__file__), Path(P.__file__), Path(GV3.__file__), Path(P.__file__).parent / 'qso_sys.py']}
    if out.exists() and side.exists():
        m = json.loads(side.read_text())
        if (m.get('build_version') == BUILD_VERSION and m.get('sha256_source_csv') == _sha(field_csv)
                and m.get('sha256_conditioning_sample') == _sha(sp) and m.get('code') == code
                and json.loads(Path(str(gdir / OUT_DIR / f'{anchors_csv.name[:-4]}_{model}.csv') + '.json').read_text()).get(
                    'sha256_source_anchors') == _sha(anchors_csv)):
            return out
    meta = json.loads(Path(str(sp) + '.json').read_text())
    w = build(field_csv, anchors_csv, sp, sample_meta={k: meta[k] for k in ('ra', 'dec', 'radius_deg', 'n_rows')})
    return w[model][0]
