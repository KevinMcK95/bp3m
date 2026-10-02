"""Header-frame GDC label inputs (moved from hst_dist_corr/ml/extract_header_frame.py, 2026-10-02).

Model-independent per-detection quantities for the GDC-correction work: every cross-matched Gaia
5-parameter star propagated to the exposure epoch and projected into Anderson's GDC frame with the
HEADER geometry only (tangent point / reference pixel = mean over chips of the catalogue's
CHIPn_CRVAL1/2 and CHIPn_CRPIX{1,2}_GDC, nominal GDC scale x initial_scale x VAFACTOR, rotation by
-ORIENTAT; both chips in one frame, so the inter-chip geometry is a prediction, not a parameter).

No frame map, no per-image solve and no error floor are applied here: those are analysis choices the
label build (extract_header_frame / iterate_alignment) makes later, in table space.  The indv fit
writes these rows next to its results (residual_export.export_gdc_labels -> gdc_hdr.csv.gz) so the
GDC work never has to re-open FLC headers, catalogues or matched_gaia.csv over NFS; the label build's
slow path calls header_frame_inputs() too, so both paths compute identical numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GAIA_EPOCH = 2016.0
STAR_GROSS_PX = 0.5    # a use_for_fit star whose indv residual exceeds this is not trusted (ml/quality.py)
CAT_COLS = ['x', 'y', 'chip_ext', 'x_gdc', 'y_gdc', 'cov_xx_gdc', 'cov_yy_gdc', 'cov_xy_gdc', 'flux', 'sky',
            'qfit', 'chi2', 'psf_frac', 'n_sat', 'n_neighbors', 'dist_nearest', 'dist_nearest_brighter', 'mag',
            'concentration', 'cr_recovered', 'dq_3x3']


def gaia_pos_cov_mas2(sa, dt_yr, f_ra, f_dec):
    """(n,2,2) position covariance at the HST epoch from the Gaia 5x5 covariance.

    Weighted EXACTLY as BP3M weights Gaia (bp3m.astro_utils.GAIA_SYS_DICT), because the labels are
    residuals of a fit BP3M will redo: the published errors are underestimated, so the covariance is
    scaled by the SQUARE of the per-solution-type sigma multiplier (1.22 for 6-parameter solutions,
    identified by a finite pseudocolour; 1.05 for 5-parameter; 1.00 for 2-parameter), and the
    Vasiliev & Baumgardt (2021) systematic floors are added in quadrature to the proper-motion and
    parallax diagonal. Before 2026-09-21 this used the raw catalogue covariance, which made every
    reported chi2 too large -- the Gaia term is ~95% of the ACS/WFC error budget."""
    n = len(sa)
    e = {c: sa[c].to_numpy(float) for c in ('ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error')}
    def corr(name):
        return sa[name].to_numpy(float) if name in sa.columns else np.zeros(n)
    names = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
    sig = np.stack([e['ra_error'], e['dec_error'], e['parallax_error'], e['pmra_error'], e['pmdec_error']], axis=1)
    C = np.zeros((n, 5, 5))
    for i in range(5):
        C[:, i, i] = sig[:, i] ** 2
    pairs = {(0, 1): 'ra_dec_corr', (0, 2): 'ra_parallax_corr', (0, 3): 'ra_pmra_corr', (0, 4): 'ra_pmdec_corr',
             (1, 2): 'dec_parallax_corr', (1, 3): 'dec_pmra_corr', (1, 4): 'dec_pmdec_corr',
             (2, 3): 'parallax_pmra_corr', (2, 4): 'parallax_pmdec_corr', (3, 4): 'pmra_pmdec_corr'}
    for (i, j), nm in pairs.items():
        c = np.nan_to_num(corr(nm)) * sig[:, i] * sig[:, j]
        C[:, i, j] = c; C[:, j, i] = c
    # --- BP3M Gaia systematics: covariance x mult^2, then the systematic floors -------------
    # Catalogues written after 2026-09-21 already carry the inflated + floored errors and mark
    # themselves with gaia_err_inflated; inflating those again would double-count.
    if 'gaia_err_inflated' in sa.columns and bool(np.asarray(sa['gaia_err_inflated']).any()):
        U = np.zeros((n, 2, 5))
        U[:, 0, 0] = 1; U[:, 0, 2] = f_ra; U[:, 0, 3] = dt_yr
        U[:, 1, 1] = 1; U[:, 1, 2] = f_dec; U[:, 1, 4] = dt_yr
        return np.einsum('nij,njk,nlk->nil', U, C, U)
    from bp3m.astro_utils import GAIA_SYS_DICT as _GS
    pc = sa['pseudocolour'].to_numpy(float) if 'pseudocolour' in sa.columns else np.full(n, np.nan)
    has_pm = np.isfinite(e['pmra_error']) & (e['pmra_error'] > 0)
    is_6p = np.isfinite(pc) & has_pm
    is_5p = has_pm & ~is_6p
    mult = np.where(is_6p, _GS['mult_6p'], np.where(is_5p, _GS['mult_5p'], _GS['mult_2p']))
    C *= (mult ** 2)[:, None, None]
    # column order here is [ra, dec, parallax, pmra, pmdec]
    C[:, 2, 2] += _GS['parallax_sys_err'] ** 2
    C[:, 3, 3] += _GS['pm_sys_err'] ** 2
    C[:, 4, 4] += _GS['pm_sys_err'] ** 2

    U = np.zeros((n, 2, 5))
    U[:, 0, 0] = 1; U[:, 0, 2] = f_ra; U[:, 0, 3] = dt_yr
    U[:, 1, 1] = 1; U[:, 1, 2] = f_dec; U[:, 1, 4] = dt_yr
    return np.einsum('nij,njk,nlk->nil', U, C, U)


def filter_name(ph) -> str:
    return ph.get('FILTER') or (ph.get('FILTER1') if str(ph.get('FILTER1', '')).startswith('F') else ph.get('FILTER2'))


def refpix(cat_hdr1, h1):
    """(crpix (2,), ra_cen, dec_cen): mean over chips of the catalogue's CHIPn_CRPIX{1,2}_GDC / CHIPn_CRVAL1/2,
    exactly as gaia_cross_match.process_single_image; FLC ext-1 CRPIX/CRVAL when the catalogue lacks them."""
    pfx = sorted({k.split('_CRPIX1_GDC')[0] for k in cat_hdr1.keys() if k.endswith('_CRPIX1_GDC') and k.startswith('CHIP')})
    if pfx:
        return (np.array([np.mean([float(cat_hdr1[f'{q}_CRPIX1_GDC']) for q in pfx]),
                          np.mean([float(cat_hdr1[f'{q}_CRPIX2_GDC']) for q in pfx])]),
                float(np.mean([float(cat_hdr1[f'{q}_CRVAL1']) for q in pfx])),
                float(np.mean([float(cat_hdr1[f'{q}_CRVAL2']) for q in pfx])))
    return np.array([float(h1['CRPIX1']), float(h1['CRPIX2'])]), float(h1['CRVAL1']), float(h1['CRVAL2'])


def frame_geometry(ph, h1, cat_hdr1) -> dict:
    """Per-image header-frame constants: pred = crpix + M @ (xi, eta)[deg]."""
    from bp3m.instrument_config import get_instrument_config
    crpix, ra_cen, dec_cen = refpix(cat_hdr1, h1)
    icfg = get_instrument_config(str(ph.get('INSTRUME')), str(ph.get('DETECTOR')))
    vaf = float(h1.get('VAFACTOR', ph.get('VAFACTOR', 1.0)) or 1.0)
    pscale_deg = icfg['pixel_scale'] * icfg['initial_scale'] * vaf / 3600.0   # deg/px in Anderson's GDC frame
    orientat = float(h1.get('ORIENTAT', 0.0))
    th = np.radians(-orientat); R0 = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    M = np.linalg.inv(R0) @ (np.array([[-1.0, 0.0], [0.0, 1.0]]) / pscale_deg)   # px/deg; +X = -RA
    mjd = float(ph['EXPSTART']); t_yr = 2000.0 + (mjd - 51544.5) / 365.25
    return dict(crpix_x=float(crpix[0]), crpix_y=float(crpix[1]), ra_cen=ra_cen, dec_cen=dec_cen, orientat=orientat,
                vafactor=vaf, pscale_deg=pscale_deg, pscale_mas=pscale_deg * 3.6e6, M=M.tolist(),
                mjd=mjd, year=t_yr, dt_gaia_yr=t_yr - GAIA_EPOCH, filter=filter_name(ph),
                instrument=ph.get('INSTRUME'), detector=ph.get('DETECTOR'), wcsname=str(h1.get('WCSNAME', '')))


def trusted_ids(indv_dirs) -> set:
    """Gaia ids the indv fit aligned on (use_for_fit) and left within STAR_GROSS_PX."""
    from pathlib import Path
    out = set()
    for d in indv_dirs:
        d = Path(d); p, det = d / 'stellar_astrometry.csv', d / 'detections.npz'
        if not (p.exists() and det.exists()):
            continue
        ids = pd.read_csv(p, usecols=['Gaia_id'], dtype={'Gaia_id': np.int64}).Gaia_id.to_numpy(np.int64)
        z = np.load(det, allow_pickle=True)
        for k in z.files:
            if k.endswith('_sidx'):
                sub = k[:-5]; uff = z[f'{sub}_use_for_fit'].astype(bool)
                if f'{sub}_dx_gdc' in z.files:
                    uff &= np.hypot(z[f'{sub}_dx_gdc'], z[f'{sub}_dy_gdc']) < STAR_GROSS_PX
                out |= set(ids[z[k].astype(int)[uff]])
    return out


def header_frame_inputs(matched: pd.DataFrame, sa: pd.DataFrame, cat, geom: dict, trusted: set):
    """Rows (DataFrame) of the pre-solve header-frame prediction for the 5p matched stars, or None (<5 stars).

    matched : matched_gaia.csv (gaia_source_id, hst_index, has_gaia_pms)
    sa      : Gaia astrometry incl. errors/correlations (indv stellar_astrometry.csv), one row per Gaia_id
    cat     : pypass catalogue table (ext 1)
    geom    : frame_geometry()
    """
    from astropy.time import Time
    from astropy.coordinates import solar_system_ephemeris
    from bp3m.astro_utils import propagate_gaia_positions, get_parallax_factors, get_tele_position
    from gaia_cross_match.miracle_match import rd2x, rd2y
    m = matched[matched.has_gaia_pms.astype(str).str.lower().isin(['true', '1'])]
    if len(m) < 5:
        return None
    sa = sa.drop_duplicates('Gaia_id').set_index('Gaia_id')
    m = m[m.gaia_source_id.isin(sa.index)]
    s = sa.loc[m.gaia_source_id.to_numpy()]
    p5 = (np.isfinite(s.pmra.to_numpy(float)) & np.isfinite(s.pmdec.to_numpy(float))
          & np.isfinite(s.parallax.to_numpy(float)))
    m, s = m[p5], s[p5]
    if len(m) < 5:
        return None
    mjd, dt = geom['mjd'], geom['dt_gaia_yr']
    with solar_system_ephemeris.set('builtin'):          # as bp3m.solver: Earth barycentric position
        tele = get_tele_position(Time(mjd, format='mjd'), curr_id='earth')
    ra_p, dec_p = propagate_gaia_positions(s.ra.to_numpy(float), s.dec.to_numpy(float), s.pmra.to_numpy(float),
                                           s.pmdec.to_numpy(float), s.parallax.to_numpy(float), dt, tele)
    f_ra, f_dec = get_parallax_factors(s.ra.to_numpy(float), s.dec.to_numpy(float), tele)
    Cg = gaia_pos_cov_mas2(s, dt, np.asarray(f_ra), np.asarray(f_dec))
    crpix = np.array([geom['crpix_x'], geom['crpix_y']]); M = np.asarray(geom['M'])
    xi = rd2x(ra_p, dec_p, geom['ra_cen'], geom['dec_cen']); eta = rd2y(ra_p, dec_p, geom['ra_cen'], geom['dec_cen'])
    pred = crpix + (M @ np.vstack([xi, eta])).T
    idx = m.hst_index.to_numpy(int)
    ok = (idx >= 0) & (idx < len(cat))
    m, s, pred, Cg, idx = m[ok], s[ok], pred[ok], Cg[ok], idx[ok]
    c = cat[idx]; names = c.dtype.names
    meas = np.column_stack([c['x_gdc'].astype(float), c['y_gdc'].astype(float)])
    fin = np.isfinite(meas).all(axis=1)
    m, s, pred, Cg, c = m[fin], s[fin], pred[fin], Cg[fin], c[fin]
    if len(m) < 5:
        return None
    df = pd.DataFrame({'gaia_id': m.gaia_source_id.to_numpy(np.int64), 'pred_x': pred[:, 0], 'pred_y': pred[:, 1],
                       'cg_mas_xx': Cg[:, 0, 0], 'cg_mas_yy': Cg[:, 1, 1], 'cg_mas_xy': Cg[:, 0, 1],
                       'trusted': m.gaia_source_id.isin(trusted).to_numpy()})
    for col in CAT_COLS:
        if col in names:
            df[col] = c[col].astype(int if col in ('chip_ext', 'n_sat') else float)
    if 'chip_ext' not in df:
        df['chip_ext'] = np.where(c['y'] >= 2048, 4, 1)
    for col in ('gmag', 'bp_rp', 'ruwe', 'parallax', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'parallax_error'):
        df[col] = s[col].to_numpy(float) if col in s.columns else np.nan
    return df.rename(columns={'mag': 'mag_inst'})
