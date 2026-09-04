"""
Step 4d: Cross-match HST PSF catalogs against CFHT/UNIONS detector catalogs.

Unlike the DELVE step, both sides are already Gaia-aligned (HST via
transformation.csv from the Gaia cross-match; CFHT via the per-detector
UNIONS alignment), so no affine discovery is needed. The match runs in the
shared Gaia tangent frame in three tiers per (HST image, CFHT detector)
pair:

  1. ANCHOR — Gaia stars matched independently on both sides (5p and 2p)
     are trusted as correct. They calibrate a per-pair magnitude zeropoint
     (with a linear colour term when enough stars) and a residual relative
     offset (dxi, deta) after propagating each star from the CFHT epoch to
     the HST epoch with its own Gaia PM/parallax (field-typical fill for
     2p stars).
  2. FAINT — HST detections without a Gaia match vs CFHT sources without a
     Gaia match (G >~ 21.5). CFHT positions are propagated to the HST epoch
     with the field-typical PM/parallax (parallax factors evaluated at BOTH
     epochs), the anchor offset applied, and pairs accepted within a k-sigma
     positional gate plus a magnitude gate through the anchor zeropoint.
  3. Output per HST image: matched_cfht.csv (anchor + faint rows),
     diagnostic_plots_cfht.png, processing_log_cfht.txt — mirroring the
     DELVE outputs. A xmatch_cfht_status.json sidecar caches success.

The HST side uses the GDC-corrected pixel frame; whether the flc_catalog
x/y are already GDC-corrected is VERIFIED at runtime against the known
matched_gaia pairs rather than assumed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ground_to_gaia_xmatch.geometry import gnomonic, inverse_gnomonic
from ground_to_gaia_xmatch.match_table import read_transformation
from bp3m.astro_utils import (get_tele_position, get_parallax_factors,
                              field_typical_astrometry)

MAS_PER_DEG = 3.6e6
CFHT_DET_HALF_DEG = 0.06     # ~200" square MegaCam detector half-diagonal
HST_HALF_DEG = 0.035         # generous HST FOV half-diagonal
SIGMA_CFHT_MAS = 30.0        # per-source CFHT centroid floor (seeing-limited)
SIGMA_HST_MAS = 5.0          # HST faint-detection floor in the Gaia frame
SIGMA_FILL_PM = 5.0          # mas/yr — unknown-PM scatter for faint sources
K_SIGMA_GATE = 4.0
MAG_GATE = 1.5               # |dmag - zp| acceptance after zeropoint


# ── CFHT store access ────────────────────────────────────────────────────────

MEGACAM_HALF_DEG = 0.75      # MegaCam mosaic half-size (exposure-level gate)


def cfht_exposure_inventory(cfht_dir: Path, ra: float, dec: float,
                            radius_deg: float) -> pd.DataFrame:
    """UNIONS EXPOSURES whose mosaic could overlap (ra, dec)+radius.

    NOTE: ra0/dec0 in the roll-up are EXPOSURE-level (one tangent point for
    all 40 detectors), so this is only the coarse gate — per-detector
    coverage comes from the catalog footprints (detector_footprints).
    """
    comb = pd.read_csv(Path(cfht_dir) / 'all_transformations_combined.csv',
                       usecols=['expnum', 'ra0', 'dec0', 'mjd'])
    comb = comb.drop_duplicates('expnum')
    cosd = np.cos(np.radians(dec))
    sep = np.hypot((comb.ra0 - ra) * cosd, comb.dec0 - dec)
    out = comb[sep < radius_deg + MEGACAM_HALF_DEG].copy()
    out['sep_deg'] = sep[sep < radius_deg + MEGACAM_HALF_DEG]
    return out.sort_values('sep_deg').reset_index(drop=True)


def detector_footprints(cfht_dir: Path, expnum: int,
                        cat: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-detector sky bounding boxes for one exposure, from its catalog."""
    if cat is None:
        cat = _read_cat(cfht_dir, expnum)
    g = cat.groupby('ext').agg(ra_min=('ra', 'min'), ra_max=('ra', 'max'),
                               dec_min=('dec', 'min'), dec_max=('dec', 'max'),
                               n_src=('ra', 'size')).reset_index()
    g['expnum'] = expnum
    return g


def _is_num(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _read_cat(cfht_dir: Path, expnum: int) -> pd.DataFrame:
    cols = ['ra', 'dec', 'x', 'y', 'ext', 'mag', 'magerr',
            'flux_radius', 'flags_ext', 'flags_phot']
    df = pd.read_csv(Path(cfht_dir) / 'cat_files' / f'{expnum}p.photcal.cat',
                     comment='#', sep=r'\s+', names=cols, usecols=range(10))
    return df


def load_cfht_detector(cfht_dir: Path, expnum: int, ext: int):
    """(sources_df, matched_gaia_df, transform_dict) for one detector.

    sources_df gains ra_al/dec_al — positions through the fitted alignment
    (i.e. in the Gaia frame at the CFHT epoch).
    """
    cfht_dir = Path(cfht_dir)
    det_dir = cfht_dir / 'xmatch' / f'cfht_{expnum}' / f'det_{ext:02d}'
    t = {k: (float(v) if _is_num(v) else v) for k, v in
         read_transformation(det_dir / 'transformation.csv').items()}
    src = _read_cat(cfht_dir, expnum)
    src = src[src.ext == ext].reset_index(drop=True)
    # QUALITY FILTER — must replicate CFHTInstrument.load_catalog exactly:
    # the xmatch's src_index is the POSITIONAL index into this filtered
    # frame, and the match tables are joined on it.
    pf = src['flags_phot'].astype(int)
    src = src[(src['flags_ext'] == 0) & ((pf & 2) == 0)
              & ((pf & 4) == 0)].reset_index(drop=True)
    src['src_index'] = src.index

    # Conventions read from the g2g source (align/solver.py geometry build,
    # match_table.build_match_table, instruments/cfht.load_catalog):
    #   src_xi_mas = gnomonic(RAW cat ra/dec, exposure tangent ra0/dec0)
    #   ALIGN model: gaia_at_epoch(xi at ra0_current) = M_align @ src_xi
    #   -> source sky in the Gaia frame at the CFHT epoch is
    #      inverse_gnomonic(M_align @ src_xi, ra0_final, dec0_final).
    # The xmatch transformation (pivoted affine) is only the initial fit and
    # is used solely as the fallback when align outputs are absent.
    xi, eta = gnomonic(src.ra.to_numpy(), src.dec.to_numpy(),
                       t['ra0'], t['dec0'])
    al_csv = (cfht_dir / 'align' / f'cfht_{expnum}' / f'det_{ext:02d}'
              / 'image_transformations.csv')
    if al_csv.exists():
        ar = pd.read_csv(al_csv).iloc[0]
        Ma = np.array([[ar.A, ar.B], [ar.C, ar.D]])
        v2 = Ma @ np.vstack([xi, eta])
        xi_al, eta_al = v2[0], v2[1]
        ra_t, de_t = float(ar.ra0_final), float(ar.dec0_final)
        t['mjd'] = float(ar.mjd)
        t['posterior_align'] = True
    else:
        M = np.array([[t['A'], t['B']], [t['C'], t['D']]])
        al = M @ np.vstack([xi - t['xs_o'], eta - t['ys_o']])
        xi_al, eta_al = al[0] + t['xt_o'], al[1] + t['yt_o']
        ra_t, de_t = t['ra0'], t['dec0']
        t['posterior_align'] = False
    src['xi_al'], src['eta_al'] = xi_al, eta_al
    src['ra_al'], src['dec_al'] = inverse_gnomonic(xi_al, eta_al, ra_t, de_t)
    mg_path = det_dir / 'matched_gaia.csv'
    mg = (pd.read_csv(mg_path, dtype={'gaia_source_id': str})
          if mg_path.exists() else pd.DataFrame())
    return src, mg, t


# ── HST side ─────────────────────────────────────────────────────────────────

def load_hst_image(img_dir: Path):
    """(catalog_df, matched_gaia_df, transform_dict, gdc_mode) for one image.

    catalog_df gains ra_al/dec_al through the HST transformation. gdc_mode
    records whether flc_catalog x/y needed the matched_gaia GDC offsets
    ('shifted') or matched directly ('direct') — verified empirically on the
    known Gaia pairs, never assumed.
    """
    from astropy.io import fits
    img_dir = Path(img_dir)
    name = img_dir.name
    t = read_transformation(img_dir / 'transformation.csv')
    mg = pd.read_csv(img_dir / 'matched_gaia.csv',
                     dtype={'gaia_source_id': str})
    with fits.open(img_dir / f'{name}_flc_catalog.fits') as h:
        # native-endian copies (numpy 2.x removed ndarray.newbyteorder)
        cat = pd.DataFrame({
            c: np.ascontiguousarray(h[1].data[c]).astype(
                h[1].data[c].dtype.newbyteorder('='), copy=False)
            for c in ('x', 'y', 'mag', 'mag_err', 'qfit', 'chi2', 'flux')})
    cat['hst_index'] = cat.index

    # Empirical GDC check: compare catalog (x,y) at matched hst_index rows
    # against matched_gaia (hst_x_gdc, hst_y_gdc).
    j = mg.merge(cat, left_on='hst_index', right_on='hst_index', how='inner')
    ddx = j['hst_x_gdc'] - j['x']
    gdc_mode = 'direct' if np.nanmedian(np.abs(ddx)) < 0.05 else 'shifted'
    if gdc_mode == 'shifted':
        # Interpolate the (smooth) GDC offset from the matched stars onto
        # every catalog detection — adequate for matching (GDC gradients are
        # smooth on 100-px scales); the joint solve later re-derives exact
        # positions through the standard loader.
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        pts = j[['x', 'y']].to_numpy()
        for axis, gcol, rcol in (('x', 'hst_x_gdc', '_dx'),
                                 ('y', 'hst_y_gdc', '_dy')):
            dv = (j[gcol] - j[axis]).to_numpy()
            try:
                f = LinearNDInterpolator(pts, dv)
                v = f(cat[['x', 'y']].to_numpy())
                fn = NearestNDInterpolator(pts, dv)
                v = np.where(np.isfinite(v), v,
                             fn(cat[['x', 'y']].to_numpy()))
            except Exception:
                v = np.full(len(cat), np.nanmedian(dv))
            cat[rcol] = v
        cat['x_gdc'] = cat['x'] + cat['_dx']
        cat['y_gdc'] = cat['y'] + cat['_dy']
    else:
        cat['x_gdc'], cat['y_gdc'] = cat['x'], cat['y']

    # GaiaHub pseudo-image convention (verified empirically on Draco,
    # 1.6 mas median vs gaia_ra_prop): the transformation maps GDC px into
    # Gaia pseudo-image px; sky offsets are
    #   (xi, eta) = R(-orientat) @ diag(-1, +1) @ (XY_t - [x_cen, y_cen]) * ps
    # with pixel_scale stored in ARCSEC.
    ps = float(t.get('pixel_scale', 0.05)) * 1000.0
    M = np.array([[t['A'], t['B']], [t['C'], t['D']]])
    v = M @ np.vstack([(cat.x_gdc - t['xs_o']).to_numpy(),
                       (cat.y_gdc - t['ys_o']).to_numpy()])
    Xt, Yt = v[0] + t['xt_o'], v[1] + t['yt_o']
    a = -(Xt - t['x_cen']) * ps
    b = (Yt - t['y_cen']) * ps
    rot = np.radians(-float(t.get('orientat', 0.0)))
    xi_al = np.cos(rot) * a - np.sin(rot) * b
    eta_al = np.sin(rot) * a + np.cos(rot) * b
    cat['xi_al'], cat['eta_al'] = xi_al, eta_al
    cat['ra_al'], cat['dec_al'] = inverse_gnomonic(
        xi_al, eta_al, t['ra_cen'], t['dec_cen'])
    return cat, mg, t, gdc_mode


def _hst_frame_check(cat, mg, t):
    """Median |predicted - gaia| [mas] over the matched pairs — sanity that
    the transformation convention reproduces the Gaia positions."""
    j = mg.merge(cat[['hst_index', 'ra_al', 'dec_al']], on='hst_index')
    cosd = np.cos(np.radians(t['dec_cen']))
    d = np.hypot((j.ra_al - j.gaia_ra_prop) * cosd,
                 j.dec_al - j.gaia_dec_prop) * MAS_PER_DEG
    return float(np.nanmedian(d)), len(j)


# ── Matching core ────────────────────────────────────────────────────────────

def _parallax_shift(ra, dec, mjd_from, mjd_to, plx_mas):
    """Apparent shift (dra*cosd, ddec) [mas] between two epochs for a source
    of the given parallax."""
    from astropy.time import Time
    fr_r, fr_d = get_parallax_factors(
        ra, dec, get_tele_position(Time(mjd_from, format='mjd'), curr_id='earth'))
    to_r, to_d = get_parallax_factors(
        ra, dec, get_tele_position(Time(mjd_to, format='mjd'), curr_id='earth'))
    return (to_r - fr_r) * plx_mas, (to_d - fr_d) * plx_mas


def match_one_pair(hst, det, fill, log):
    """Match one (HST image, CFHT detector) pair. Returns rows (list of dict).

    hst: dict(cat, mg, t, mjd, name); det: dict(src, mg, t, expnum, ext).
    """
    cat, hmg, ht = hst['cat'], hst['mg'], hst['t']
    src, cmg, ct = det['src'], det['mg'], det['t']
    dt_yr = (hst['mjd'] - ct['mjd']) / 365.25
    cosd = np.cos(np.radians(ht['ra_cen'] * 0 + ht['dec_cen']))

    # ── tier 1: common-Gaia anchor ──────────────────────────────────────────
    if not len(cmg) or not len(hmg):
        return [], None
    common = hmg.merge(cmg, on='gaia_source_id', suffixes=('_h', '_c'))
    if len(common) < 5:
        log(f"    det {det['expnum']}/{det['ext']:02d}: only {len(common)} "
            f"common Gaia stars — skipped")
        return [], None

    # CFHT-side aligned positions of the common stars: from src via src_index
    c_pos = src.set_index('src_index').loc[
        common.src_index.to_numpy(), ['ra_al', 'dec_al', 'mag']]
    h_pos = cat.set_index('hst_index').loc[
        common.hst_index.to_numpy(), ['ra_al', 'dec_al']]

    # Per-star propagation CFHT epoch -> HST epoch with its own Gaia PM/plx
    # (from the field Gaia catalog — the matched tables don't carry PMs);
    # 2p stars fall back to the field-typical fill.
    gl = hst['gaia_lookup']
    common = common.merge(gl, on='gaia_source_id', how='left')
    pm_ra = common['pmra'].fillna(fill['pmra']).to_numpy()
    pm_de = common['pmdec'].fillna(fill['pmdec']).to_numpy()
    plx = common['parallax'].fillna(fill['plx']).to_numpy()
    ra_c, de_c = c_pos.ra_al.to_numpy(), c_pos.dec_al.to_numpy()
    p_ra, p_de = _parallax_shift(ra_c, de_c, ct['mjd'], hst['mjd'], plx)
    ra_prop = ra_c + (pm_ra * dt_yr + p_ra) / MAS_PER_DEG / cosd
    de_prop = de_c + (pm_de * dt_yr + p_de) / MAS_PER_DEG

    dxi = (h_pos.ra_al.to_numpy() - ra_prop) * cosd * MAS_PER_DEG
    deta = (h_pos.dec_al.to_numpy() - de_prop) * MAS_PER_DEG
    off_xi, off_eta = float(np.median(dxi)), float(np.median(deta))
    mad = float(np.median(np.hypot(dxi - off_xi, deta - off_eta))) * 1.4826

    # colour zeropoint: HST instrumental (st) mag - CFHT mag
    hmag = hmg.set_index('hst_index').loc[
        common.hst_index.to_numpy(), 'hst_mag_st_gdc'].to_numpy()
    cmag = c_pos.mag.to_numpy()
    ok = np.isfinite(hmag) & np.isfinite(cmag)
    zp = float(np.median((hmag - cmag)[ok])) if ok.sum() else np.nan
    zp_scatter = (float(np.median(np.abs((hmag - cmag)[ok]
                                         - zp))) * 1.4826 if ok.sum() else np.nan)
    zp_slope = 0.0
    if 'bp_rp' in common.columns and ok.sum() >= 8:
        cc = common['bp_rp'].to_numpy()[ok]
        good = np.isfinite(cc)
        if good.sum() >= 8:
            zp_slope = float(np.polyfit(cc[good],
                                        (hmag - cmag)[ok][good], 1)[0])

    anchor = dict(n_common=len(common), off_xi=off_xi, off_eta=off_eta,
                  mad_mas=mad, zp=zp, zp_scatter=zp_scatter,
                  zp_slope=zp_slope, dt_yr=dt_yr)
    log(f"    det {det['expnum']}/{det['ext']:02d}: {len(common)} common "
        f"Gaia, offset ({off_xi:+.1f},{off_eta:+.1f}) mas, mad {mad:.1f}, "
        f"zp {zp:+.2f}±{zp_scatter:.2f} (slope {zp_slope:+.3f}), "
        f"dt {dt_yr:+.1f} yr")

    rows = []
    for i in range(len(common)):
        rows.append(dict(
            hst_index=int(common.hst_index.iloc[i]),
            cfht_expnum=det['expnum'], cfht_ext=det['ext'],
            cfht_src_index=int(common.src_index.iloc[i]),
            cfht_mjd=ct['mjd'], dt_yr=dt_yr,
            gaia_source_id=common.gaia_source_id.iloc[i],
            tier='anchor',
            sep_mas=float(np.hypot(dxi[i] - off_xi, deta[i] - off_eta)),
            cfht_mag=float(cmag[i]),
            cfht_ra_hst_epoch=float(ra_prop[i]),
            cfht_dec_hst_epoch=float(de_prop[i]),
            sigma_tot_mas=mad))

    # ── tier 2: faint / non-Gaia matching ───────────────────────────────────
    h_faint = cat[~cat.hst_index.isin(hmg.hst_index)]
    c_faint = src[~src.src_index.isin(cmg.src_index if len(cmg) else [])]
    
    if not len(h_faint) or not len(c_faint):
        return rows, anchor

    p_ra_f, p_de_f = _parallax_shift(
        c_faint.ra_al.to_numpy(), c_faint.dec_al.to_numpy(),
        ct['mjd'], hst['mjd'], fill['plx'])
    ra_f = (c_faint.ra_al.to_numpy()
            + (fill['pmra'] * dt_yr + p_ra_f + off_xi) / MAS_PER_DEG / cosd)
    de_f = (c_faint.dec_al.to_numpy()
            + (fill['pmdec'] * dt_yr + p_de_f + off_eta) / MAS_PER_DEG)

    sig = np.sqrt(SIGMA_HST_MAS**2 + SIGMA_CFHT_MAS**2 + mad**2
                  + (SIGMA_FILL_PM * abs(dt_yr))**2)
    gate_deg = K_SIGMA_GATE * sig / MAS_PER_DEG

    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[ra_f * cosd, de_f])
    q = np.c_[h_faint.ra_al.to_numpy() * cosd, h_faint.dec_al.to_numpy()]
    dist, idx = tree.query(q, distance_upper_bound=gate_deg)
    okm = np.isfinite(dist) & (dist < gate_deg)
    # magnitude gate through the anchor zeropoint
    hm = h_faint['mag'].to_numpy() + (ht.get('zp', 0.0) * 0.0)
    hst_st = h_faint['mag'].to_numpy()  # instrumental; zp maps st->cfht
    n_pos = int(okm.sum())
    taken = {}
    for qi in np.where(okm)[0]:
        ci = idx[qi]
        dmag = np.nan
        if np.isfinite(zp):
            # hst_mag_st_gdc = mag + (st offset); catalog 'mag' is
            # instrumental — the st offset is absorbed into zp via anchors,
            # so compare (mag_st - cfht_mag) to zp with a wide gate.
            dmag = (hst_st[qi] + _st_offset(hmg)) - \
                   c_faint['mag'].to_numpy()[ci]
            if np.isfinite(dmag) and abs(dmag - zp) > MAG_GATE:
                continue
        key = int(c_faint.src_index.to_numpy()[ci])
        cand = (float(dist[qi] * MAS_PER_DEG), qi, ci, dmag)
        if key not in taken or cand < taken[key]:
            taken[key] = cand
    for key, (sep, qi, ci, dmag) in taken.items():
        rows.append(dict(
            hst_index=int(h_faint.hst_index.to_numpy()[qi]),
            cfht_expnum=det['expnum'], cfht_ext=det['ext'],
            cfht_src_index=key, cfht_mjd=ct['mjd'], dt_yr=dt_yr,
            gaia_source_id='', tier='faint',
            sep_mas=sep, cfht_mag=float(c_faint['mag'].to_numpy()[ci]),
            cfht_ra_hst_epoch=float(ra_f[ci]),
            cfht_dec_hst_epoch=float(de_f[ci]),
            sigma_tot_mas=float(sig)))
    log(f"      faint: {len(taken)} matches of {len(h_faint)} HST x "
        f"{len(c_faint)} CFHT candidates (gate {K_SIGMA_GATE:.0f}x"
        f"{sig:.0f} mas, {n_pos} positional)")
    return rows, anchor


def _st_offset(hmg):
    """Median (hst_mag_st_gdc - hst_mag_gdc): converts catalog instrumental
    mags to the st system the anchors' zeropoint was fit in."""
    if 'hst_mag_st_gdc' in hmg and 'hst_mag_gdc' in hmg:
        return float(np.nanmedian(hmg.hst_mag_st_gdc - hmg.hst_mag_gdc))
    return 0.0


# ── Per-image worker + orchestrator ─────────────────────────────────────────

def _plot_diagnostics(rows_df, anchors, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    a = rows_df[rows_df.tier == 'anchor']
    f = rows_df[rows_df.tier == 'faint']
    ax[0, 0].hist([a.sep_mas, f.sep_mas], bins=40,
                  label=[f'anchor ({len(a)})', f'faint ({len(f)})'],
                  stacked=False, histtype='step')
    ax[0, 0].set_xlabel('separation [mas]'); ax[0, 0].legend()
    ax[0, 1].scatter(f.cfht_mag, f.sep_mas, s=4, alpha=0.4)
    ax[0, 1].set_xlabel('CFHT mag'); ax[0, 1].set_ylabel('sep [mas]')
    if anchors:
        ad = pd.DataFrame(anchors)
        ax[1, 0].errorbar(ad.off_xi, ad.off_eta, xerr=ad.mad_mas,
                          yerr=ad.mad_mas, fmt='o', ms=4, alpha=0.7)
        ax[1, 0].set_xlabel('offset xi [mas]'); ax[1, 0].set_ylabel('offset eta [mas]')
        ax[1, 0].set_title('per-detector anchor offsets')
        ax[1, 1].scatter(ad.dt_yr, ad.zp, s=16)
        ax[1, 1].set_xlabel('dt [yr]'); ax[1, 1].set_ylabel('mag zeropoint')
    fig.suptitle(out_png.parent.name)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def match_one_image(img_dir: Path, dets: pd.DataFrame, cfht_dir: Path,
                    gaia_lookup: pd.DataFrame, fill: dict,
                    det_cache: dict, make_plots=True):
    """Cross-match one HST image against all overlapping CFHT detectors."""
    from astropy.io import fits
    img_dir = Path(img_dir)
    name = img_dir.name
    logf = open(img_dir / 'processing_log_cfht.txt', 'w')

    def log(msg):
        print(msg, file=logf, flush=True)

    try:
        cat, hmg, ht, gdc_mode = load_hst_image(img_dir)
        chk_mas, n_chk = _hst_frame_check(cat, hmg, ht)
        mjd = float(fits.getheader(img_dir / f'{name}_flc.fits', 0)['EXPSTART'])
        log(f'{name}: {len(cat)} detections, {len(hmg)} Gaia-matched, '
            f'gdc_mode={gdc_mode}, frame check {chk_mas:.1f} mas '
            f'(n={n_chk}), mjd {mjd:.2f}')
        if chk_mas > 60.0:
            log('  FRAME CHECK FAILED (>60 mas) — aborting this image')
            return name, 0, 'frame check failed'
        # HST footprint from the aligned detections; overlap against the
        # per-detector CATALOG footprints (the roll-up ra0/dec0 are
        # exposure-level, useless for detector selection).
        h_ra_lo, h_ra_hi = cat.ra_al.min(), cat.ra_al.max()
        h_de_lo, h_de_hi = cat.dec_al.min(), cat.dec_al.max()
        pad = 30.0 / 3600.0
        near = []
        for expnum in dets.expnum.astype(int):
            fp_key = ('fp', expnum)
            if fp_key not in det_cache:
                try:
                    det_cache[fp_key] = detector_footprints(cfht_dir, expnum)
                except Exception as e:
                    log(f'  exp {expnum}: footprint load failed ({e})')
                    det_cache[fp_key] = None
            fp = det_cache[fp_key]
            if fp is None:
                continue
            hit = fp[(fp.ra_min - pad < h_ra_hi) & (fp.ra_max + pad > h_ra_lo)
                     & (fp.dec_min - pad < h_de_hi)
                     & (fp.dec_max + pad > h_de_lo)]
            near.extend((expnum, int(e)) for e in hit.ext)
        log(f'  {len(near)} overlapping CFHT detectors '
            f'(of {len(dets)} candidate exposures)')
        hst = dict(cat=cat, mg=hmg, t=ht, mjd=mjd, name=name,
                   gaia_lookup=gaia_lookup)
        all_rows, anchors = [], []
        for key in near:
            if key not in det_cache:
                try:
                    det_cache[key] = load_cfht_detector(
                        cfht_dir, key[0], key[1])
                except Exception as e:
                    log(f'    det {key}: load failed ({e})')
                    det_cache[key] = None
            if det_cache[key] is None:
                continue
            src, cmg, ct = det_cache[key]
            rows, anchor = match_one_pair(
                hst, dict(src=src, mg=cmg, t=ct,
                          expnum=key[0], ext=key[1]), fill, log)
            all_rows.extend(rows)
            if anchor:
                anchor.update(expnum=key[0], ext=key[1])
                anchors.append(anchor)
        if not all_rows:
            log('  no matches')
            return name, 0, None
        df = pd.DataFrame(all_rows)
        df.to_csv(img_dir / 'matched_cfht.csv', index=False)
        n_faint = int((df.tier == 'faint').sum())
        log(f'  TOTAL: {len(df)} matches ({n_faint} faint/non-Gaia) across '
            f'{df.cfht_expnum.nunique()} exposures')
        if make_plots:
            try:
                _plot_diagnostics(df, anchors,
                                  img_dir / 'diagnostic_plots_cfht.png')
            except Exception as e:
                log(f'  plot failed: {e}')
        return name, len(df), None
    except Exception as exc:
        import traceback
        log(traceback.format_exc())
        return name, 0, str(exc)
    finally:
        logf.close()


def run_cross_match_cfht(output_dir, field_name, cfht_dir,
                         ra, dec, radius_deg, gaia_csv=None,
                         make_plots=True, force=False):
    """Step 4d driver: HST x CFHT/UNIONS cross-match for every PSF-fit image.

    Serial over images with a shared detector cache (each CFHT detector is
    loaded once per field, not once per image).
    """
    t0 = time.time()
    field_dir = Path(output_dir) / field_name
    hst_root = field_dir / 'HST' / 'mastDownload' / 'HST'
    cfht_dir = Path(cfht_dir)
    print(f'\nStep 4d: HST x CFHT/UNIONS cross-match  (store: {cfht_dir})')
    dets = cfht_exposure_inventory(cfht_dir, ra, dec, radius_deg)
    if not len(dets):
        print('  no CFHT/UNIONS coverage for this field — skipping')
        return []
    print(f'  {len(dets)} candidate CFHT exposures within '
          f'{radius_deg + MEGACAM_HALF_DEG:.2f} deg')

    from bp3m.data_loader_flc import resolve_gaia_csvs
    if gaia_csv is not None:
        paths = ([Path(p) for p in gaia_csv]
                 if isinstance(gaia_csv, (list, tuple)) else [Path(gaia_csv)])
    else:
        paths, _ = resolve_gaia_csvs(field_dir)
    gl = pd.concat([
        pd.read_csv(p, usecols=lambda c: c in
                    ('SOURCE_ID', 'source_id', 'pmra', 'pmdec',
                     'parallax', 'bp_rp'))
        .rename(columns={'SOURCE_ID': 'source_id'}) for p in paths]) \
        .drop_duplicates('source_id')
    gl['gaia_source_id'] = gl['source_id'].astype(str)
    fill = field_typical_astrometry(gl.pmra.dropna(), gl.pmdec.dropna(),
                                    plx=gl.parallax.dropna())
    print(f"  2p/faint propagation fill: pm=({fill['pmra']:+.2f},"
          f"{fill['pmdec']:+.2f}) mas/yr  plx={fill['plx']:+.3f} mas")

    img_dirs = [d for d in sorted(hst_root.iterdir())
                if (d / 'matched_gaia.csv').exists()
                and (d / 'transformation.csv').exists()]
    todo = [d for d in img_dirs
            if force or not (d / 'matched_cfht.csv').exists()]
    print(f'  {len(img_dirs)} images with Gaia xmatch, {len(todo)} to do')
    det_cache: dict = {}
    results = []
    for i, d in enumerate(todo, 1):
        nm, n, err = match_one_image(d, dets, cfht_dir, gl, fill,
                                     det_cache, make_plots=make_plots)
        status = f'{n} matches' if err is None else f'FAILED: {err}'
        print(f'  [{i:3d}/{len(todo)}] {nm}: {status}')
        results.append((nm, n, err))
    n_ok = sum(1 for _, n, e in results if e is None and n > 0)
    print(f'  Step 4d done: {n_ok}/{len(todo)} images matched '
          f'({time.time()-t0:.0f}s)')
    return results
