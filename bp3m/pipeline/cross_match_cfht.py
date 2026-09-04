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


# ── Matching via the shared gaia_cross_match machinery ──────────────────────
#
# Per (HST image, CFHT exposure): build an external-catalog DataFrame in the
# DELVE schema and call process_single_image_delve(label='cfht',
# out_suffix=f'_{expnum}') — the SAME iterative 4p->6p relative-alignment
# fits, sigma-clipping, magnitude/colour diagnostics, 10-panel figure and
# processing log as the Gaia/DELVE cross-matches. Gaia-matched CFHT sources
# carry their own Gaia PM/plx (+errors); non-Gaia sources carry the
# field-typical fill with SIGMA_FILL_PM errors, so the match covariance
# widens with dt in the standard way. CFHT positions are apparent at the
# CFHT epoch: the per-source parallax displacement at t_cfht is subtracted
# (barycentric-ising them) so the machinery's plx*p(t_hst) term then models
# the parallax difference between BOTH epochs.

SIGMA_CFHT_POS_MAS = 20.0   # per-source CFHT centroid floor (ra/dec_error)


def build_cfht_ext_df(cfht_dir, expnum, exts, gaia_lookup, fill,
                      det_cache) -> "pd.DataFrame | None":
    """External-catalog rows (DELVE schema) for one exposure's detectors."""
    from astropy.time import Time
    frames = []
    for ext in exts:
        key = (int(expnum), int(ext))
        if key not in det_cache:
            try:
                det_cache[key] = load_cfht_detector(cfht_dir, expnum, ext)
            except Exception:
                det_cache[key] = None
        if det_cache[key] is None:
            continue
        src, mg, t = det_cache[key]
        d = src[['ra_al', 'dec_al', 'mag', 'magerr', 'src_index']].copy()
        d = d.rename(columns={'ra_al': 'ra', 'dec_al': 'dec'})
        d['source_id'] = (int(expnum) * 10**8 + int(ext) * 10**6
                          + d.src_index.astype(np.int64))
        d['cfht_mjd'] = float(t['mjd'])
        # Gaia astrometry for the Gaia-matched sources
        if len(mg):
            d = d.merge(mg[['src_index', 'gaia_source_id']],
                        on='src_index', how='left')
            gm = mg[['src_index', 'gaia_source_id']].merge(
                gaia_lookup.rename(columns={'ra': 'gaia_ra',
                                            'dec': 'gaia_dec'}),
                on='gaia_source_id', how='left')
            d = d.merge(gm.drop(columns=['gaia_source_id']),
                        on='src_index', how='left')
        else:
            d['gaia_source_id'] = ''
            for c in ('pmra', 'pmdec', 'parallax', 'pmra_error',
                      'pmdec_error', 'parallax_error', 'bp_rp', 'gmag',
                      'rpmag', 'gaia_ra', 'gaia_dec'):
                d[c] = np.nan
        frames.append(d)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    has_gaia = df['pmra'].notna()
    df['has_gaia'] = has_gaia
    df['pmra'] = df.pmra.fillna(fill['pmra'])
    df['pmdec'] = df.pmdec.fillna(fill['pmdec'])
    df['parallax'] = df.parallax.fillna(fill['plx'])
    df['pmra_error'] = df.get('pmra_error', pd.Series(np.nan, df.index)) \
        .fillna(SIGMA_FILL_PM)
    df['pmdec_error'] = df.get('pmdec_error', pd.Series(np.nan, df.index)) \
        .fillna(SIGMA_FILL_PM)
    df['parallax_error'] = df.get('parallax_error',
                                  pd.Series(np.nan, df.index)).fillna(1.0)
    df['ra_error'] = SIGMA_CFHT_POS_MAS
    df['dec_error'] = SIGMA_CFHT_POS_MAS
    df['ref_epoch'] = Time(df.cfht_mjd.iloc[0], format='mjd').jyear
    # Barycentric-ise the apparent CFHT positions: subtract the parallax
    # displacement AT THE CFHT EPOCH (per-source plx).
    p_ra, p_de = get_parallax_factors(
        df.ra.to_numpy(), df.dec.to_numpy(),
        get_tele_position(Time(df.cfht_mjd.iloc[0], format='mjd'),
                          curr_id='earth'))
    cosd = np.cos(np.radians(df.dec.to_numpy()))
    df['ra'] = df.ra - df.parallax.to_numpy() * p_ra / MAS_PER_DEG / cosd
    df['dec'] = df.dec - df.parallax.to_numpy() * p_de / MAS_PER_DEG
    return df


def match_one_image(img_dir, dets, cfht_dir, gaia_lookup, fill, det_cache,
                    make_plots=True,
                    sigma_rot_deg=0.01, sigma_scale=1e-3, sigma_skew=1e-3,
                    discovery_max_offset=10, max_mag_diff=5.0,
                    init_resid_max=5.0):
    """Cross-match one HST image against each overlapping CFHT exposure via
    the shared single-image machinery; union the per-exposure tables into
    matched_cfht.csv."""
    from gaia_cross_match.cross_match_delve import process_single_image_delve
    img_dir = Path(img_dir)
    name = img_dir.name
    hst = dict(root=str(img_dir), flc=str(img_dir / f'{name}_flc.fits'),
               catalog=str(img_dir / f'{name}_flc_catalog.fits'))
    # HST filter wavelength -> blue-minus-red colour ordering vs CFHT r(~640nm)
    from astropy.io import fits as _fits
    _hdr = _fits.getheader(hst['flc'], 0)
    _filt = next((str(_hdr.get(k, '')) for k in ('FILTER', 'FILTER1', 'FILTER2')
                  if str(_hdr.get(k, '')).startswith('F')), 'F606W')
    import re as _re
    _lam_hst = float((_re.findall(r'F(\d+)', _filt) or ['606'])[0])
    _lam_cfht = 640.0
    mjd_hst = float(_hdr.get('EXPSTART', 0.0))
    if _lam_hst < _lam_cfht:
        _c_sign, _c_lbl = -1.0, f'HST {_filt} − r_CFHT'
    else:
        _c_sign, _c_lbl = 1.0, f'r_CFHT − HST {_filt}'
    # footprint selection (per-detector catalog bounding boxes)
    try:
        cat, hmg, ht, gdc_mode = load_hst_image(img_dir)
        chk_mas, n_chk = _hst_frame_check(cat, hmg, ht)
    except Exception as exc:
        return name, 0, f'HST load failed: {exc}'
    if chk_mas > 60.0:
        return name, 0, f'HST frame check failed ({chk_mas:.0f} mas)'
    h_ra_lo, h_ra_hi = cat.ra_al.min(), cat.ra_al.max()
    h_de_lo, h_de_hi = cat.dec_al.min(), cat.dec_al.max()
    pad = 30.0 / 3600.0
    n_total = 0
    parts = []
    diag_all, field_all = [], []
    for expnum in dets.expnum.astype(int):
        fp_key = ('fp', expnum)
        if fp_key not in det_cache:
            try:
                det_cache[fp_key] = detector_footprints(cfht_dir, expnum)
            except Exception:
                det_cache[fp_key] = None
        fp = det_cache[fp_key]
        if fp is None:
            continue
        hit = fp[(fp.ra_min - pad < h_ra_hi) & (fp.ra_max + pad > h_ra_lo)
                 & (fp.dec_min - pad < h_de_hi) & (fp.dec_max + pad > h_de_lo)]
        if not len(hit):
            continue
        ext_df = build_cfht_ext_df(cfht_dir, expnum, list(hit.ext),
                                   gaia_lookup, fill, det_cache)
        if ext_df is None or not len(ext_df):
            continue
        # forced anchors: Gaia stars matched on BOTH sides
        _fp = None
        if len(hmg):
            _j = hmg[['hst_index', 'gaia_source_id']].merge(
                ext_df[['source_id', 'gaia_source_id']].dropna(),
                on='gaia_source_id')
            if len(_j):
                _fp = _j[['hst_index', 'source_id']].to_numpy()
        try:
            diag_ret = process_single_image_delve(
                hst, ext_df, label='cfht', mag_col='mag',
                out_suffix=f'_{expnum}',
                sigma_rot_deg=sigma_rot_deg, sigma_scale=sigma_scale,
                sigma_skew=sigma_skew,
                discovery_max_offset=discovery_max_offset,
                max_mag_diff=max_mag_diff, init_resid_max=init_resid_max,
                gaia_cmd=True, color_hst_label=_c_lbl,
                color_hst_sign=_c_sign,
                make_plots=False, make_offset_plots=make_plots,
                return_diag=True,
                forced_pairs=_fp)
        except Exception as exc:
            parts.append((expnum, None, str(exc)))
            continue
        if diag_ret is not None:
            dd, fdf = diag_ret
            dd = dd.assign(cfht_expnum=expnum,
                           cfht_mjd=float(ext_df.cfht_mjd.iloc[0]))
            diag_all.append(dd)
            if fdf is not None:
                field_all.append(fdf.assign(cfht_expnum=expnum))
        mcsv = img_dir / f'matched_cfht_{expnum}.csv'
        if mcsv.exists():
            m = pd.read_csv(mcsv)
            m['cfht_expnum'] = expnum
            sid = m['cfht_source_id'].astype(np.int64)
            m['cfht_ext'] = (sid // 10**6) % 100
            m['cfht_src_index'] = sid % 10**6
            m['cfht_mjd'] = float(ext_df.cfht_mjd.iloc[0])
            gaia_ids = {}
            for key in {( int(expnum), int(e)) for e in m['cfht_ext']}:
                dc = det_cache.get(key)
                if dc is not None and len(dc[1]):
                    gaia_ids.update(dict(zip(dc[1].src_index,
                                             dc[1].gaia_source_id)))
            m['gaia_source_id'] = [gaia_ids.get(int(i), '')
                                   for i in m['cfht_src_index']]
            parts.append((expnum, m, None))
            n_total += len(m)
        else:
            parts.append((expnum, None, 'no matches'))
    good = [m for _, m, e in parts if m is not None]
    union = None
    if good:
        union = pd.concat(good, ignore_index=True)
        union.to_csv(img_dir / 'matched_cfht.csv', index=False)
    if make_plots and diag_all:
        try:
            _plot_union_diagnostics(
                img_dir, name, pd.concat(diag_all, ignore_index=True),
                pd.concat(field_all, ignore_index=True) if field_all else None,
                union, ht, hst_mjd=mjd_hst, gaia_lookup=gaia_lookup,
                det_cache=det_cache, color_label=_c_lbl, color_sign=_c_sign)
        except Exception as exc:
            import traceback; traceback.print_exc()
    errs = '; '.join(f'{e}:{err}' for e, m, err in parts if err)
    return name, n_total, (errs or None)


def run_cross_match_cfht(output_dir, field_name, cfht_dir,
                         ra, dec, radius_deg, gaia_csv=None,
                         make_plots=True, force=False):
    """Step 4d driver: HST x CFHT/UNIONS cross-match for every PSF-fit image,
    via the shared gaia_cross_match single-image machinery (DELVE-parity
    outputs: per-exposure matched_cfht_<exp>.csv + 10-panel diagnostics +
    processing logs, plus a union matched_cfht.csv per image)."""
    t0 = time.time()
    field_dir = Path(output_dir) / field_name
    hst_root = field_dir / 'HST' / 'mastDownload' / 'HST'
    cfht_dir = Path(cfht_dir)
    print(f'\nStep 4d: HST x CFHT/UNIONS cross-match  (store: {cfht_dir})')
    dets = cfht_exposure_inventory(cfht_dir, ra, dec, radius_deg)
    if not len(dets):
        print('  no CFHT/UNIONS coverage for this field — skipping')
        return []
    print(f'  {len(dets)} candidate CFHT exposures')

    from bp3m.data_loader_flc import resolve_gaia_csvs
    if gaia_csv is not None:
        paths = ([Path(p) for p in gaia_csv]
                 if isinstance(gaia_csv, (list, tuple)) else [Path(gaia_csv)])
    else:
        paths, _ = resolve_gaia_csvs(field_dir)
    use = ('SOURCE_ID', 'source_id', 'ra', 'dec', 'pmra', 'pmdec',
           'parallax', 'bp_rp', 'gmag', 'rpmag',
           'pmra_error', 'pmdec_error', 'parallax_error')
    gl = pd.concat([pd.read_csv(p, usecols=lambda c: c in use)
                    .rename(columns={'SOURCE_ID': 'source_id'})
                    for p in paths]).drop_duplicates('source_id')
    gl['gaia_source_id'] = gl['source_id'].astype(str)
    gl = gl.drop(columns=['source_id'])
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
        nm, n, err = match_one_image(d, dets, cfht_dir, gl, fill, det_cache,
                                     make_plots=make_plots)
        status = f'{n} matches' + (f'  [{err}]' if err else '')
        print(f'  [{i:3d}/{len(todo)}] {nm}: {status}')
        results.append((nm, n, err))
    n_ok = sum(1 for _, n, e in results if n > 0)
    print(f'  Step 4d done: {n_ok}/{len(todo)} images matched '
          f'({time.time()-t0:.0f}s)')
    return results


def _plot_union_diagnostics(img_dir, name, diag, field, union, ht,
                            hst_mjd, gaia_lookup, det_cache,
                            color_label, color_sign):
    """One figure per HST image concatenating ALL exposures — the SAME
    10-panel figure as the DELVE/Gaia cross-matches (shared plot function),
    with the CFHT panel substitutions: sky-offset field map, 5p Gaia +
    2p Gaia+CFHT PM VPD, Gaia CMD, HST-depth HSTxCFHT CMD, and the
    CFHT-HST vs Gaia G-RP colour-colour."""
    from gaia_cross_match.cross_match_delve import _save_diagnostic_plots_delve
    m = diag[diag.is_matched]
    r = diag[~diag.is_matched]
    # 2p Gaia+CFHT PM estimates from the union table
    pm2p = None
    if union is not None and len(union):
        u = union[union.gaia_source_id.notna()].copy()
        u['gaia_source_id'] = u.gaia_source_id.astype(str)
        gl = gaia_lookup.rename(columns={'ra': 'gaia_ra', 'dec': 'gaia_dec'})
        j = u.merge(gl, on='gaia_source_id', how='left', suffixes=('', '_gl'))
        j2 = j[j.pmra.isna() & j.gaia_ra.notna()]
        if len(j2):
            from astropy.time import Time
            cosd = np.cos(np.radians(ht['dec_cen']))
            dt = Time(j2.cfht_mjd.to_numpy(), format='mjd').jyear - 2016.0
            ok = np.abs(dt) > 0.5
            pm2p = pd.DataFrame({
                'pmra': ((j2.cfht_ra_prop - j2.gaia_ra) * cosd * 3.6e6
                         / np.where(ok, dt, np.nan)),
                'pmdec': ((j2.cfht_dec_prop - j2.gaia_dec) * 3.6e6
                          / np.where(ok, dt, np.nan))}).dropna()
    _save_diagnostic_plots_delve(
        str(img_dir), f'{name} (all exposures)', m, r,
        delve_field_df=field, label='cfht', out_suffix='',
        gaia_cmd=True, color_hst_label=color_label,
        color_hst_sign=color_sign,
        sky_center=(ht['ra_cen'], ht['dec_cen']),
        pm2p_df=pm2p, cmd2_hst_axis=True)
