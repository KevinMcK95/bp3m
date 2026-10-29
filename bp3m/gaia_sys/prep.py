"""Gaia preparation steps of the pop_fit production (r3b) that the corrected-copy builder needs (handoff 3.4, user 10-29:
BP3M builds the corrected tables itself).  The functions below marked VERBATIM are copied unchanged from
pop_fit_dev/data.py (sha256 3e60df3a41125ac67ad3f2c80c209c7934ca2a7b882a91f6566ef00eaf678808, 2026-10-29); constants from pop_fit_dev/config.py and data.py
module level with the production settings of the reference build (QUALITY_CSTAR_AS_MISSING_COLOUR = True, strict cuts off,
RUWE < 1.4).  cond5 / l21 / qsosys follow hst_dist_corr/gaia_sys/build_corrected_copy.py (reference implementation).
Re-vendor (never edit) when pop_fit announces a change in the handoff."""
import numpy as np
import pandas as pd

# -- constants (pop_fit_dev/config.py, data.py) --------------------------------------------------------------------------
QUALITY_CSTAR_NSIGMA = 3.0      # |C*| < N sigma_C*(G), Riello+2021 eq. 6 and 18
QUALITY_AEN_SIG_MAX = 2.0       # astrometric_excess_noise_sig
QUALITY_IPD_MULTI_MAX = 5       # ipd_frac_multi_peak (per cent)
QUALITY_GALAXY_PROB_MAX = 0.5   # classprob_dsc_combmod_galaxy (NaN passes)
_GAIA_QUALITY_COLS = ('phot_bp_rp_excess_factor', 'astrometric_excess_noise_sig',
                      'ipd_frac_multi_peak', 'classprob_dsc_combmod_galaxy')
RUWE_MAX = 1.4
QUALITY_IPD_GOF_MAX = None
QUALITY_VIS_MIN = None
QUALITY_BLEND_MAX = None
QUALITY_CSTAR_AS_MISSING_COLOUR = True          # production (--cstar-as-missing-colour), as the reference build
_COLOUR_COLS = ('bpmag', 'bpmag_error', 'rpmag', 'rpmag_error', 'bp_rp', 'bp_rp_error',
                'phot_bp_mean_mag', 'phot_rp_mean_mag', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error')
_COND_6P_COLS = ('pmra', 'pmdec', 'parallax', 'pmra_error', 'pmdec_error', 'parallax_error',
                 'pmra_pmdec_corr', 'parallax_pmra_corr', 'parallax_pmdec_corr')
_GAIA_STRICT_COLS = ('ipd_gof_harmonic_amplitude', 'visibility_periods_used',
                     'phot_bp_n_blended_transits', 'phot_rp_n_blended_transits', 'phot_bp_n_obs', 'phot_rp_n_obs')
NU_FORMULA_SCATTER = 0.02       # um^-1: scatter of nu_eff(BP-RP) about the true effective wavenumber


# VERBATIM pop_fit_dev/data.py
def nu_eff_from_bp_rp(bp_rp):
    """Effective wavenumber (µm⁻¹) from BP−RP, Lindegren et al. (2021, A&A 649, A2):
    ν = 1.76 − (1.61/π) atan(0.531 (BP−RP))."""
    return 1.76 - (1.61 / np.pi) * np.arctan(0.531 * np.asarray(bp_rp, float))


# VERBATIM pop_fit_dev/data.py
def _drop_no_g(df):
    """Drop sources without G photometry: a few DR3 sources have 5/6-parameter astrometry
    but no (or rejected) G-band photometry; they carry no CMD information."""
    bad = ~np.isfinite(df['gmag'].to_numpy(dtype=float))
    if bad.any():
        print(f'  Dropped {bad.sum():,} sources without G photometry')
    return df[~bad].reset_index(drop=True)


# VERBATIM pop_fit_dev/data.py
def corrected_bp_rp_excess(excess, bp_rp):
    """C* of Riello et al. (2021), eq. 6 / Table 2."""
    x = np.asarray(bp_rp, dtype=float)
    c = np.full_like(x, np.nan)
    lo  = x < 0.5
    mid = (x >= 0.5) & (x < 4.0)
    hi  = x >= 4.0
    c[lo]  = 1.154360 + 0.033772 * x[lo] + 0.032277 * x[lo]**2
    c[mid] = (1.162004 + 0.011464 * x[mid] + 0.049255 * x[mid]**2
              - 0.005879 * x[mid]**3)
    c[hi]  = 1.057572 + 0.140537 * x[hi]
    return np.asarray(excess, dtype=float) - c


# VERBATIM pop_fit_dev/data.py
def apply_quality_cuts(df):
    """
    Remove sources whose photometry or astrometry is unreliable for the CMD
    prior and the kinematic mixture: BP/RP flux excess (|C*| beyond
    QUALITY_CSTAR_NSIGMA sigma_C*(G)), astrometric excess noise, multi-peaked
    image parameters, and a high DSC galaxy probability.  Each cut is skipped
    if its column is absent; NaN values pass (e.g. no BP/RP photometry, since
    such stars carry no CMD information anyway).
    """
    missing = [c for c in _GAIA_QUALITY_COLS if c not in df.columns]
    if missing:
        print(f'  WARNING: quality columns {missing} not in the Gaia cache; '
              f'those cuts are skipped (use --redownload to refresh).')
    keep = np.ones(len(df), dtype=bool)
    phot_bad = np.zeros(len(df), dtype=bool)       # photometric flags (C*, blend): colour contaminated
    report = []
    if 'phot_bp_rp_excess_factor' in df.columns:
        cstar = corrected_bp_rp_excess(df['phot_bp_rp_excess_factor'], df['bp_rp'])
        g     = df['gmag'].to_numpy(dtype=float)
        sig_c = 0.0059898 + 8.817481e-12 * g**7.618399
        bad   = np.isfinite(cstar) & (np.abs(cstar) > QUALITY_CSTAR_NSIGMA * sig_c)
        phot_bad |= bad; report.append(f'C* {bad.sum():,}')
    for col, thr, lab in [('astrometric_excess_noise_sig', QUALITY_AEN_SIG_MAX, 'AEN_sig'),
                          ('ipd_frac_multi_peak', QUALITY_IPD_MULTI_MAX, 'ipd_multi'),
                          ('classprob_dsc_combmod_galaxy', QUALITY_GALAXY_PROB_MAX, 'P_gal')]:
        if col in df.columns:
            v   = df[col].to_numpy(dtype=float)
            bad = np.isfinite(v) & (v > thr)
            keep &= ~bad; report.append(f'{lab} {bad.sum():,}')
    if _strict_on():
        miss = [c for c in _GAIA_STRICT_COLS if c not in df.columns]
        if miss:
            raise RuntimeError(f'strict quality cuts need {miss} (use --redownload)')
        if QUALITY_IPD_GOF_MAX is not None:
            v = df['ipd_gof_harmonic_amplitude'].to_numpy(dtype=float)
            bad = np.isfinite(v) & (v > QUALITY_IPD_GOF_MAX)
            keep &= ~bad; report.append(f'ipd_gof {bad.sum():,}')
        if QUALITY_VIS_MIN is not None:
            v = df['visibility_periods_used'].to_numpy(dtype=float)
            bad = ~(v >= QUALITY_VIS_MIN)
            keep &= ~bad; report.append(f'vis_periods {bad.sum():,}')
        if QUALITY_BLEND_MAX is not None:
            nb = df['phot_bp_n_blended_transits'].to_numpy(float) + df['phot_rp_n_blended_transits'].to_numpy(float)
            no = df['phot_bp_n_obs'].to_numpy(float) + df['phot_rp_n_obs'].to_numpy(float)
            with np.errstate(invalid='ignore', divide='ignore'):
                beta = np.where(no > 0, nb / no, np.nan)
            bad = np.isfinite(beta) & (beta > QUALITY_BLEND_MAX)      # no BP/RP: passes (no CMD information)
            phot_bad |= bad; report.append(f'blend {bad.sum():,}')
    df = df.copy()
    df['colour_masked'] = False
    rescue = keep & phot_bad
    if QUALITY_CSTAR_AS_MISSING_COLOUR and rescue.any():
        for c in _COLOUR_COLS:
            if c in df.columns:
                df[c] = df[c].astype('float64'); df.loc[rescue, c] = np.nan
        six = rescue & (df['astrometric_params_solved'].to_numpy() == 95) if 'astrometric_params_solved' in df else rescue & False
        n6 = 0
        if six.any() and all(c + '_raw' in df.columns for c in _COND_6P_COLS):
            for c in _COND_6P_COLS:
                df.loc[six, c] = df.loc[six, c + '_raw']
            n6 = int(six.sum())
        df.loc[rescue, 'colour_masked'] = True
        report.append(f'kept with colour masked {rescue.sum():,} ({n6:,} 6p unconditioned)')
    else:
        keep &= ~phot_bad
    if report:
        print(f'  Quality cuts removed {(~keep).sum():,} / {len(df):,} '
              f'(C*/AEN/ipd/galaxy flags overlap: {", ".join(report)})')
    return df[keep].reset_index(drop=True)


# VERBATIM pop_fit_dev/data.py
def _strict_on():
    return any(v is not None for v in (QUALITY_IPD_GOF_MAX, QUALITY_VIS_MIN, QUALITY_BLEND_MAX))


# -- corrected-copy steps (hst_dist_corr/gaia_sys/build_corrected_copy.py, reference implementation 10-29 v2) --------------
P5 = ['ra', 'dec', 'pmra', 'pmdec', 'parallax']       # 5x5 order; (pmra, pmdec, parallax) = pop order
ASTRO = P5[2:]
CB, GB = np.array([-1, 0.5, 0.9, 1.2, 1.6, 2.2, 10]), np.array([0, 18.5, 19.5, 20.2, 30])


def need(df, cols, what):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise SystemExit(f'{what}: missing columns {miss}')


def corr_col(df, a, b):
    return f'{a}_{b}_corr' if f'{a}_{b}_corr' in df.columns else f'{b}_{a}_corr'


def cov(df, names):
    e = np.column_stack([df[f'{c}_error'].to_numpy(float) for c in names])
    C = np.einsum('ni,ij->nij', e**2, np.eye(len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            C[:, i, j] = C[:, j, i] = np.nan_to_num(df[corr_col(df, names[i], names[j])].to_numpy(float)) * e[:, i] * e[:, j]
    return C


def put_cov(df, C, names, rows):
    e = np.sqrt(np.einsum('nii->ni', C))
    for i, a in enumerate(names):
        df.loc[rows, f'{a}_error'] = e[:, i]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            df.loc[rows, corr_col(df, names[i], names[j])] = C[:, i, j] / (e[:, i] * e[:, j])


def l21(df, what=''):
    """Lindegren+21 zero point subtracted (bp3m.gaia_zp.compute_zp = pop_fit's apply_parallax_zeropoint, verified identical
    10-29); keeps parallax_raw / parallax_zp as pop_fit does.  Rows without a zero point (no parallax / not 5p-6p) get 0."""
    from ..gaia_zp import compute_zp
    need(df, ['nu_eff_used_in_astrometry', 'pseudocolour', 'ecl_lat', 'astrometric_params_solved', 'gmag', 'parallax'],
         f'L21 ({what})')
    if 'parallax_raw' in df.columns:
        return df
    zp, _ = compute_zp(df)
    zp = np.nan_to_num(zp, nan=0.0)
    df = df.copy(); df['parallax_raw'] = df['parallax']; df['parallax_zp'] = zp; df['parallax'] = df['parallax'] - zp
    return df


def cond5(df, mode, what=''):
    """pop_fit's condition_6p_on_colour extended to (ra*, dec, pmra, pmdec, parallax); mode 'mp' (exact per-bin weighted-mean
    preserving) or 'mc' (median-centred).  Returns (df, stats).  ra*/dec shifts in mas (ra, dec updated in degrees)."""
    need(df, ['pseudocolour', 'pseudocolour_error', 'astrometric_params_solved', 'bp_rp', 'bp_rp_error', 'gmag',
              'nu_eff_used_in_astrometry'] + [f'{p}_pseudocolour_corr' for p in P5] + [f'{p}_error' for p in P5]
         + [corr_col(df, P5[i], P5[j]) for i in range(5) for j in range(i + 1, 5)], f'6p conditioning ({what})')
    df = df.copy().reset_index(drop=True)
    for c in _COND_6P_COLS + _COND_POS_COLS:        # raw values kept, as pop_fit (apply_quality_cuts restores the PM block for
        df[c] = df[c].astype('float64'); df[c + '_raw'] = df[c]          # C*; restore_positions does the position block)
    sol = df.astrometric_params_solved.to_numpy(); br = df.bp_rp.to_numpy(float)
    six = sol == 95
    ok = six & np.isfinite(br) & np.isfinite(df.pseudocolour.to_numpy(float)) & np.isfinite(df.pseudocolour_error.to_numpy(float))
    stats = dict(n_6p=int(six.sum()), n_6p_conditioned=int(ok.sum()),
                 n_6p_no_colour=int((six & ~np.isfinite(br)).sum()),
                 n_6p_no_pseudocolour_error=int((six & np.isfinite(br) & ~np.isfinite(df.pseudocolour_error.to_numpy(float))).sum()))
    if not ok.any():
        return df, stats
    idx = np.where(ok)[0]
    g = df.gmag.to_numpy(float)
    br_err = np.nan_to_num(df.bp_rp_error.to_numpy(float)[idx], nan=0.05)
    dnudc = -(1.61 / np.pi) * 0.531 / (1.0 + (0.531 * br[idx])**2)
    var_p = (dnudc * br_err)**2 + NU_FORMULA_SCATTER**2
    nu_ps = df.pseudocolour.to_numpy(float)[idx]; s_nu = df.pseudocolour_error.to_numpy(float)[idx]
    nu_ref = nu_eff_from_bp_rp(br[idx])
    five = (sol == 31) & np.isfinite(br) & np.isfinite(df.nu_eff_used_in_astrometry.to_numpy(float))
    c6, g6 = br[idx], g[idx]
    if five.sum() >= 50:
        off5 = df.nu_eff_used_in_astrometry.to_numpy(float)[five] - nu_eff_from_bp_rp(br[five])
        glob_off = float(np.median(off5)); off = np.full(len(idx), glob_off)
        c5, g5 = br[five], g[five]
        for i in range(len(CB) - 1):
            for j in range(len(GB) - 1):
                in5 = (c5 >= CB[i]) & (c5 < CB[i + 1]) & (g5 >= GB[j]) & (g5 < GB[j + 1])
                in6 = (c6 >= CB[i]) & (c6 < CB[i + 1]) & (g6 >= GB[j]) & (g6 < GB[j + 1])
                if in5.sum() >= 20 and in6.any():
                    off[in6] = np.median(off5[in5])
        nu_ref = nu_ref + off
        var_p = var_p + (1.4826 * np.median(np.abs(off5 - glob_off)))**2 / max(five.sum() / 50.0, 1.0)
    dnu = nu_ps - nu_ref
    centre = np.full(len(idx), float(np.median(dnu)))
    grp = np.full(len(idx), -1); k_ = 0
    for i in range(len(CB) - 1):
        for j in range(len(GB) - 1):
            m = (c6 >= CB[i]) & (c6 < CB[i + 1]) & (g6 >= GB[j]) & (g6 < GB[j + 1])
            if m.sum() >= 20:
                centre[m] = np.median(dnu[m]); grp[m] = k_; k_ += 1
    dnu = dnu - centre
    e = np.column_stack([df[f'{p}_error'].to_numpy(float)[idx] for p in P5])
    r_nu = np.column_stack([np.nan_to_num(df[f'{p}_pseudocolour_corr'].to_numpy(float)[idx]) for p in P5])
    C_xnu = r_nu * e * s_nu[:, None]
    denom = s_nu**2 + var_p
    sh = -C_xnu * (dnu / denom)[:, None]                                # (n, 5): ra*, dec in mas
    if mode == 'mp':
        wv = 1.0 / e**2
        for gv in np.unique(grp):
            m = grp == gv
            sh[m] -= np.sum(wv[m] * sh[m], 0) / np.sum(wv[m], 0)
    elif mode != 'mc':
        raise ValueError(mode)
    C5 = cov(df.iloc[idx], P5)
    C5c = C5 - np.einsum('ni,nj->nij', C_xnu, C_xnu) / denom[:, None, None]
    for c in ('ra', 'dec') + tuple(ASTRO):
        df[c] = df[c].astype('float64')
    dec_r = np.radians(df.dec.to_numpy(float)[idx])
    df.loc[idx, 'ra'] = df.ra.to_numpy(float)[idx] + sh[:, 0] / 3.6e6 / np.cos(dec_r)
    df.loc[idx, 'dec'] = df.dec.to_numpy(float)[idx] + sh[:, 1] / 3.6e6
    for k, p in enumerate(ASTRO):
        df.loc[idx, p] = df[p].to_numpy(float)[idx] + sh[:, 2 + k]
    for c in [f'{p}_error' for p in P5] + [corr_col(df, P5[i], P5[j]) for i in range(5) for j in range(i + 1, 5)]:
        df[c] = df[c].astype('float64')
    put_cov(df, C5c, P5, idx)
    df['pos_shift_ra_mas'] = 0.0; df['pos_shift_dec_mas'] = 0.0
    df.loc[idx, 'pos_shift_ra_mas'] = sh[:, 0]; df.loc[idx, 'pos_shift_dec_mas'] = sh[:, 1]
    stats['pos_shift_mas_median'] = float(np.median(np.hypot(sh[:, 0], sh[:, 1])))
    stats['pos_shift_mas_p90'] = float(np.percentile(np.hypot(sh[:, 0], sh[:, 1]), 90))
    return df, stats


_COND_POS_COLS = ('ra', 'dec', 'ra_error', 'dec_error', 'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
                  'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr')


def restore_positions(df):
    """After apply_quality_cuts: pop_fit restores the UNCONDITIONED PM / parallax block of 6p rows kept with their colour masked
    (C*: contaminated colour).  With five-parameter conditioning the position block (values, errors, every position
    correlation) must be restored with it, or the row's 5x5 mixes conditioned positions with unconditioned PMs."""
    if 'colour_masked' not in df.columns or not all(c + '_raw' in df.columns for c in _COND_POS_COLS):
        return df
    six = df['colour_masked'].to_numpy(bool) & (df['astrometric_params_solved'].to_numpy() == 95)
    if six.any():
        df = df.copy()
        for c in _COND_POS_COLS:
            df.loc[six, c] = df.loc[six, c + '_raw']
        if 'pos_shift_ra_mas' in df.columns:
            df.loc[six, ['pos_shift_ra_mas', 'pos_shift_dec_mas']] = 0.0
    return df


def qsosys(df, npz_path):
    """Subtract the QSO-calibrated magnitude-dependent PM / parallax term (relative to G = 20)."""
    from .qso_sys import correction
    qs = correction(npz_path, df.ra.to_numpy(float), df.dec.to_numpy(float), df.gmag.to_numpy(float),
                    df.astrometric_params_solved.to_numpy(), df.pseudocolour.to_numpy(float))
    for j, c in enumerate(ASTRO):
        df[c] = df[c].to_numpy(float) - qs[:, j]
    return df
