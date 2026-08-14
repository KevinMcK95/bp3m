"""
cross_match_validator.py - Cross-image validation of Gaia-HST cross-matches.

For each target, groups processed images by filter/camera, then:

  1. Writes per-image source_quality.csv annotating each matched source with:
       mag_normalized     = hst_mag_st_gdc + cross_image_zp  (comparable across images)
       n_same_filter      = number of same-filter/camera images also matching this Gaia source
       mag_norm_mad       = MAD of mag_normalized across those images
       mag_residual       = deviation from cross-image median
       is_mag_consistent  = mag_norm_mad < threshold
       expected/observed inter-image magnitude delta vs reference image
       wcs_offset_px      = pointing change from WCS headers (image-level constant)
       is_trustworthy     = combined flag

  2. Writes a per-target cross_match_catalog.csv with one row per
     (gaia_source_id, filter_camera) pair:
       gaia_source_id, filter_camera, n_images, image_list, hst_index_list,
       mag_norm_mean, mag_norm_std, mag_norm_mad, is_consistent

Usage:
    conda activate pymc_new
    python cross_match_validator.py --target Fornax_dSph --data-dir ./data
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from collections import defaultdict


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _science_filter(h0):
    # WFC3 (and other single-wheel instruments) use a single FILTER keyword.
    single = h0.get('FILTER', '').strip()
    if single:
        return single
    # ACS uses two filter wheels (FILTER1/FILTER2); return the non-CLEAR one.
    f1 = h0.get('FILTER1', 'CLEAR').strip()
    f2 = h0.get('FILTER2', 'CLEAR').strip()
    return f2 if 'CLEAR' in f1 else f1


def load_image_data(image_dir, image_name):
    matched_path   = os.path.join(image_dir, 'matched_gaia.csv')
    transform_path = os.path.join(image_dir, 'transformation.csv')
    flc_paths      = glob.glob(os.path.join(image_dir, '*_flc.fits'))
    if not (os.path.exists(matched_path) and
            os.path.exists(transform_path) and flc_paths):
        return None

    matched   = pd.read_csv(matched_path)
    transform = pd.read_csv(transform_path, index_col='parameter')['value']

    with fits.open(flc_paths[0]) as h:
        h0 = h[0].header
        h1 = h[1].header
        exptime  = float(h0.get('EXPTIME', 1.0))
        filt     = _science_filter(h0)
        instrume = h0.get('INSTRUME', '').strip()
        detector = h0.get('DETECTOR', '').strip()
        crval1   = float(h1.get('CRVAL1', 0.0))
        crval2   = float(h1.get('CRVAL2', 0.0))

    has_stmag = ('hst_mag_st_gdc' in matched.columns and
                 matched['hst_mag_st_gdc'].notna().any())

    return {
        'image_name':    image_name,
        'image_dir':     image_dir,
        'matched':       matched,
        'transform':     transform,
        'exptime':       exptime,
        'filter':        filt,
        'instrume':      instrume,
        'detector':      detector,
        'camera':        f'{instrume}/{detector}',
        'filter_camera': f'{filt}/{instrume}/{detector}',
        'has_stmag':     has_stmag,
        'zp':            float(transform['zp']),
        'crval1':        crval1,
        'crval2':        crval2,
        'pixel_scale':   float(transform.get('pixel_scale', 0.05)),
        'dec_cen':       float(transform['dec_cen']),
    }


def find_processed_images(target, data_dir):
    hst_root = os.path.join(data_dir, target, 'HST')
    images = {}
    for root, dirs, files in os.walk(hst_root):
        name = os.path.basename(root)
        if f'{name}_flc_catalog.fits' in files and 'matched_gaia.csv' in files:
            data = load_image_data(root, name)
            if data is not None:
                images[name] = data
    return images


# ---------------------------------------------------------------------------
# Magnitude helpers
# ---------------------------------------------------------------------------

def has_valid_stmag(matched_df):
    return ('hst_mag_st_gdc' in matched_df.columns and
            matched_df['hst_mag_st_gdc'].notna().any())


def compute_pairwise_zps(group):
    """
    For every pair (a, b) of images in `group` that share ≥ 3 sources, compute:

        ZP(a→b) = median( mag_st_gdc_a_j − mag_st_gdc_b_j )
                  for Gaia sources j detected in both images.

    The ZP is always computed from direct per-star differences — never from
    population medians and never via Gaia magnitudes as an intermediary.
    Both directions are stored so BFS traversal is straightforward.

    Returns dict {(name_a, name_b): (zp, n_shared)} for pairs with n_shared ≥ 3,
    with both (a,b) and (b,a) entries (ZPs negated for the reverse direction).
    """
    names = list(group.keys())
    result = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            df_a = (group[a]['matched']
                    [['gaia_source_id', 'hst_mag_st_gdc']]
                    .rename(columns={'hst_mag_st_gdc': 'mag_a'}))
            df_b = (group[b]['matched']
                    [['gaia_source_id', 'hst_mag_st_gdc']]
                    .rename(columns={'hst_mag_st_gdc': 'mag_b'}))
            shared = df_a.merge(df_b, on='gaia_source_id')
            ok = (np.isfinite(shared['mag_a'].values) &
                  np.isfinite(shared['mag_b'].values))
            n_ok = int(ok.sum())
            if n_ok >= 3:
                zp = float(np.median(
                    shared['mag_a'].values[ok] - shared['mag_b'].values[ok]))
                result[(a, b)] = ( zp, n_ok)
                result[(b, a)] = (-zp, n_ok)
    return result


def find_overlap_components(names, pairwise_zps):
    """
    Find connected components of the image overlap graph.
    Two images are connected if they share ≥ 3 sources (i.e. have an entry in
    pairwise_zps).  Images with no overlap with anyone are their own component
    (solo).

    Returns a list of sets of image names.
    """
    adj = defaultdict(set)
    for (a, b) in pairwise_zps:
        adj[a].add(b)

    visited = set()
    components = []
    for name in names:
        if name not in visited:
            component = set()
            queue = [name]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                queue.extend(adj[node] - visited)
            components.append(component)
    return components


def propagate_zps_bfs(component, pairwise_zps, ref_name):
    """
    BFS from ref_name through the overlap graph, accumulating ZPs:

        ZP_neighbor = ZP_current + pairwise_zps[(current, neighbor)][0]

    so that mag_norm_i = mag_st_gdc_i + ZP_i ≈ mag_st_gdc_ref for all i.
    ref_name always has ZP = 0.0.

    Returns dict {name: zp} for every image reachable from ref_name.
    """
    zps = {ref_name: 0.0}
    queue = [ref_name]
    while queue:
        current = queue.pop(0)
        for neighbor in component:
            if neighbor not in zps and (current, neighbor) in pairwise_zps:
                zp_edge, _ = pairwise_zps[(current, neighbor)]
                zps[neighbor] = zps[current] + zp_edge
                queue.append(neighbor)
    return zps


def mad(x):
    return float(np.median(np.abs(x - np.median(x))))


def weighted_mean_and_err(mags, errs):
    """Inverse-variance weighted mean and its uncertainty."""
    w = 1.0 / errs**2
    mu  = np.sum(w * mags) / np.sum(w)
    sig = 1.0 / np.sqrt(np.sum(w))
    return mu, sig


# ---------------------------------------------------------------------------
# Pointing offset from WCS headers
# ---------------------------------------------------------------------------

def wcs_offset_px(d_i, d_ref):
    """
    First-order pixel offset of image i's pointing relative to reference,
    from CRVAL differences.  Sign convention: +RA → −X (East-Left).
    Returns scalar distance in pixels.
    """
    cos_dec = np.cos(np.radians(d_ref['dec_cen']))
    pix     = d_ref['pixel_scale']
    dx = -(d_i['crval1'] - d_ref['crval1']) * 3600.0 * cos_dec / pix
    dy =  (d_i['crval2'] - d_ref['crval2']) * 3600.0 / pix
    return float(np.hypot(dx, dy))


# ---------------------------------------------------------------------------
# Per-group validation
# ---------------------------------------------------------------------------

def validate_filter_group(group, ref_name, zp_dict, mag_scatter_thr, offset_tol_mag,
                          offset_tol_px, cross_camera_extra_tol=0.05, z_outlier=3.0):
    """
    Validate a set of images forming one overlap-connected component.

    zp_dict: {name: zp} pre-computed via propagate_zps_bfs so that
             mag_norm_i = hst_mag_st_gdc_i + zp_i ≈ hst_mag_st_gdc_ref.
             ref_name has zp = 0.0 by construction.

    All images in `group` must have hst_mag_st_gdc (caller guarantees this).
    ZPs are never assumed to be 0; they are always derived from direct per-star
    differences along a spanning tree of the overlap graph.
    """
    ref = group[ref_name]

    for name, d in group.items():
        zp = zp_dict[name]
        d['mag_norm']   = d['matched']['hst_mag_st_gdc'].values + zp
        d['cross_zp']   = zp
        d['used_stmag'] = True

    # --- Cross-image source magnitude table ---
    # Error for mag_st_gdc equals hst_mag_err_gdc (STMAG is a linear flux
    # scaling, so fractional errors are identical).  Add a 0.01 mag floor so
    # that near-perfect formal errors don't make the pull statistic too
    # aggressive.
    has_err = all('hst_mag_err_gdc' in d['matched'].columns for d in group.values())
    rows = []
    for name, d in group.items():
        tmp = d['matched'][['gaia_source_id']].copy()
        tmp['mag_norm'] = d['mag_norm']
        tmp['image']    = name
        if has_err:
            tmp['mag_err'] = d['matched']['hst_mag_err_gdc'].values
        rows.append(tmp)
    combined = pd.concat(rows, ignore_index=True)

    def per_source_stats(g):
        mags = g['mag_norm'].values
        imgs = g['image'].values
        if has_err:
            errs = np.clip(g['mag_err'].values, 1e-4, None) + 0.01
            mu_w, sig_w = weighted_mean_and_err(mags, errs)
            pulls = (mags - mu_w) / errs
        else:
            mu_w  = np.median(mags)
            sig_w = mad(mags)
            pulls = (mags - mu_w) / (sig_w if sig_w > 0 else 1.0)

        outlier_mask = np.abs(pulls) > z_outlier
        n_consistent = int((~outlier_mask).sum())
        outlier_imgs = ','.join(sorted(imgs[outlier_mask]))
        return pd.Series({
            'n_same_filter':   len(mags),
            'mag_norm_median': float(np.median(mags)),
            'mag_norm_mad':    mad(mags),
            'mag_norm_wmean':  float(mu_w),
            'mag_norm_werr':   float(sig_w),
            'n_consistent':    n_consistent,
            'outlier_images':  outlier_imgs,
        })

    source_stats = (combined
        .groupby('gaia_source_id')
        .apply(per_source_stats)
        .reset_index())

    # --- Per-image stats vs reference ---
    image_stats = {}
    for name, d in group.items():
        zp = zp_dict[name]
        same_camera = (d['camera'] == ref['camera'])
        tol = offset_tol_mag if same_camera else offset_tol_mag + cross_camera_extra_tol

        ref_df  = ref['matched'][['gaia_source_id']].assign(mag_ref=ref['mag_norm'])
        this_df = d['matched'][['gaia_source_id']].assign(mag_this=d['mag_norm'])
        shared  = ref_df.merge(this_df, on='gaia_source_id')
        ok      = (np.isfinite(shared['mag_ref'].values) &
                   np.isfinite(shared['mag_this'].values))
        n_shared = int(ok.sum())
        if n_shared >= 3:
            residual_after_zp = float(np.median(
                shared['mag_ref'].values[ok] - shared['mag_this'].values[ok]))
            offset_mag_ok = abs(residual_after_zp) < tol
        else:
            residual_after_zp = np.nan
            offset_mag_ok     = True

        image_stats[name] = {
            'ref_image':          ref_name,
            'mag_scale':          'mag_st_gdc+CrossZP',
            'same_camera_as_ref': same_camera,
            'used_stmag':         True,
            'n_shared_with_ref':  n_shared,
            'cross_image_zp':     zp,
            'residual_after_zp':  residual_after_zp,
            'offset_mag_ok':      offset_mag_ok,
            'wcs_offset_px':      wcs_offset_px(d, ref),
        }

    return source_stats, image_stats


# ---------------------------------------------------------------------------
# Per-image output
# ---------------------------------------------------------------------------

def write_source_quality(data, source_stats, image_stats, mag_scatter_thr):
    df = data['matched'].copy()
    df['mag_normalized'] = data['mag_norm']

    df = df.merge(source_stats, on='gaia_source_id', how='left')
    df['mag_residual_from_wmean'] = df['mag_normalized'] - df['mag_norm_wmean']

    # This image is consistent if it is not in the outlier list for this source
    name = data['image_name']
    df['is_mag_consistent'] = ~df['outlier_images'].fillna('').str.contains(
        name, regex=False)
    df.loc[df['n_same_filter'] == 1, 'is_mag_consistent'] = True  # solo: can't assess

    s = image_stats[data['image_name']]
    for col, val in s.items():
        df[col] = val

    # pointing_ok is informational only; WCS offset alone doesn't make a
    # cross-match untrustworthy — dithered images can be 50+ px apart.
    df['is_trustworthy'] = df['is_mag_consistent'] & df['offset_mag_ok']

    out = os.path.join(data['image_dir'], 'source_quality.csv')
    df.to_csv(out, index=False)
    return out


def write_solo_quality(data):
    df = data['matched'].copy()
    if not has_valid_stmag(data['matched']):
        print(f"  WARNING: {data['image_name']} missing hst_mag_st_gdc "
              f"(stale py1pass output?) — solo image skipped")
        return None
    mag_norm = data['matched']['hst_mag_st_gdc'].values.copy()
    df['mag_normalized']    = mag_norm
    df['n_same_filter']     = 1
    df['mag_norm_median']   = df['mag_normalized']
    df['mag_norm_mad']      = np.nan
    df['is_mag_consistent'] = True
    df['ref_image']         = data['image_name']
    df['mag_scale']         = 'STMAG'
    df['n_shared_with_ref'] = len(df)
    df['cross_image_zp']    = 0.0
    df['residual_after_zp'] = 0.0
    df['wcs_offset_px']     = 0.0
    df['same_camera_as_ref'] = True
    df['used_stmag']        = True
    df['offset_mag_ok']     = True
    df['is_trustworthy']    = True

    out = os.path.join(data['image_dir'], 'source_quality.csv')
    df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Gaia photometry enrichment
# ---------------------------------------------------------------------------

# Columns to pull from the Gaia catalog and how to rename them in the output.
_GAIA_PHOT_COLS = {
    'gmag_error':                   'gaia_gmag_error',
    'bpmag':                        'gaia_bpmag',
    'bpmag_error':                  'gaia_bpmag_error',
    'rpmag':                        'gaia_rpmag',
    'rpmag_error':                  'gaia_rpmag_error',
    'phot_bp_rp_excess_factor':     'gaia_bp_rp_excess_factor',
    'corrected_flux_excess_factor': 'gaia_corrected_excess_factor',
    'ruwe':                         'gaia_ruwe',
}


def _load_gaia_phot(data_dir, target):
    """
    Locate Gaia catalog CSV(s) under {data_dir}/{target}/Gaia/ and return a
    DataFrame indexed by int64 source_id containing photometric columns from
    _GAIA_PHOT_COLS (only those present in the file(s) are included; missing
    columns are silently skipped).  Returns None if no suitable file is found.
    """
    import glob as _glob
    gaia_dir = os.path.join(data_dir, target, 'Gaia')
    gaia_files = sorted(_glob.glob(os.path.join(gaia_dir, '*_gaia.csv')))
    if not gaia_files:
        return None

    want = set(_GAIA_PHOT_COLS.keys()) | {'source_id'}
    frames = []
    for path in gaia_files:
        try:
            df = pd.read_csv(path, usecols=lambda c: c in want)
            if 'source_id' in df.columns:
                frames.append(df)
        except Exception:
            pass

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset='source_id')
    combined['source_id'] = combined['source_id'].astype(np.int64)
    return combined.set_index('source_id')


# ---------------------------------------------------------------------------
# DELVE photometry enrichment
# ---------------------------------------------------------------------------

_DELVE_PHOT_COLS = ['delve_rmag', 'delve_gmag', 'delve_imag', 'delve_zmag',
                    'delve_pmra', 'delve_pmdec',
                    'delve_pmra_error', 'delve_pmdec_error', 'delve_pmra_pmdec_corr',
                    'delve_parallax', 'delve_parallax_error',
                    'delve_ra_error', 'delve_dec_error']
_DELVE_SENTINEL_LO, _DELVE_SENTINEL_HI = -90.0, 50.0


def _mask_delve_sentinels(df):
    """Replace DELVE photometric sentinel values (−99 / +99) with NaN in-place."""
    for col in _DELVE_PHOT_COLS:
        if col.endswith('mag') and col in df.columns:
            bad = (df[col] < _DELVE_SENTINEL_LO) | (df[col] > _DELVE_SENTINEL_HI)
            df.loc[bad, col] = np.nan
    return df


def _collect_delve_info(images):
    """
    Read matched_delve.csv + source_quality.csv per image and partition into:

    - gaia_linked : rows where hst_index appears in source_quality.csv (has Gaia ID)
    - delve_only  : rows where hst_index has no Gaia match

    Returns (gaia_linked_df, delve_only_df).  Either may be empty.
    """
    gaia_linked, delve_only = [], []

    for name, d in images.items():
        delve_path = os.path.join(d['image_dir'], 'matched_delve.csv')
        if not os.path.exists(delve_path):
            continue

        delve = _mask_delve_sentinels(pd.read_csv(delve_path))
        delve['image_name']    = name
        delve['filter_camera'] = d['filter_camera']

        sq_path = os.path.join(d['image_dir'], 'source_quality.csv')
        if os.path.exists(sq_path):
            sq = pd.read_csv(sq_path, usecols=['hst_index', 'gaia_source_id'])
            sq['gaia_source_id'] = sq['gaia_source_id'].astype(np.int64)
            linked = delve.merge(sq, on='hst_index', how='inner')
            gaia_linked.append(linked)
            only = delve[~delve['hst_index'].isin(sq['hst_index'])].copy()
        else:
            only = delve.copy()

        if len(only):
            delve_only.append(only)

    gl = pd.concat(gaia_linked, ignore_index=True) if gaia_linked else pd.DataFrame()
    do = pd.concat(delve_only,  ignore_index=True) if delve_only  else pd.DataFrame()
    return gl, do


def _agg_delve_for_gaia_source(g):
    """Aggregate DELVE columns across multiple images for one gaia_source_id."""
    row = {'n_delve_images': len(g)}
    for col in ['delve_source_id']:
        if col in g.columns:
            row[col] = g[col].iloc[0]
    for col in _DELVE_PHOT_COLS:
        if col in g.columns:
            vals = g[col].dropna()
            row[col] = float(vals.median()) if len(vals) else np.nan
    if 'delve_mtype' in g.columns:
        modes = g['delve_mtype'].dropna().mode()
        row['delve_mtype'] = modes.iloc[0] if len(modes) else np.nan
    return pd.Series(row)


# ---------------------------------------------------------------------------
# Global target-level catalog
# ---------------------------------------------------------------------------

def build_global_catalog(images, target, data_dir):
    """
    Aggregate all source_quality.csv files into a single cross_match_catalog.csv
    at the target level.  One row per (gaia_source_id, filter_camera).
    """
    rows = []
    for name, d in images.items():
        sq_path = os.path.join(d['image_dir'], 'source_quality.csv')
        if not os.path.exists(sq_path):
            continue
        sq = pd.read_csv(sq_path)
        sq['image_name']    = name
        sq['filter_camera'] = d['filter_camera']
        cols = ['gaia_source_id', 'hst_index', 'image_name', 'filter_camera',
                'mag_normalized', 'mag_norm_mad', 'mag_norm_wmean', 'mag_norm_werr',
                'n_consistent', 'outlier_images', 'is_trustworthy',
                'has_gaia_pms', 'gaia_gmag', 'residual_sigma', 'hst_is_star']
        rows.append(sq[[c for c in cols if c in sq.columns]])

    if not rows:
        return

    all_data = pd.concat(rows, ignore_index=True)

    # Per (gaia_source_id, filter_camera) aggregation
    def agg_group(g):
        mags   = g['mag_normalized'].values
        trusts = g['is_trustworthy'].values
        # Weighted mean/err from the per-image source_quality files (same value
        # repeated per image, so just take the first non-null entry)
        wmean = g['mag_norm_wmean'].dropna().iloc[0] if 'mag_norm_wmean' in g and g['mag_norm_wmean'].notna().any() else float(np.mean(mags))
        werr  = g['mag_norm_werr'].dropna().iloc[0]  if 'mag_norm_werr'  in g and g['mag_norm_werr'].notna().any()  else np.nan
        n_con = int(g['n_consistent'].dropna().iloc[0]) if 'n_consistent' in g and g['n_consistent'].notna().any() else len(g)
        out_imgs = g['outlier_images'].dropna().iloc[0] if 'outlier_images' in g and g['outlier_images'].notna().any() else ''

        # is_star aggregation across images
        if 'hst_is_star' in g.columns:
            star_vals = g['hst_is_star'].astype(bool)
            is_star_all = bool(star_vals.all())
            is_star_any = bool(star_vals.any())
            non_star_imgs = ','.join(sorted(g.loc[~star_vals, 'image_name'].tolist()))
        else:
            is_star_all = np.nan
            is_star_any = np.nan
            non_star_imgs = ''

        return pd.Series({
            'n_images':              len(g),
            'image_list':            ','.join(g['image_name'].tolist()),
            'hst_index_list':        ','.join(g['hst_index'].astype(str).tolist()),
            'mag_norm_wmean':        float(wmean),
            'mag_norm_werr':         float(werr) if not np.isnan(werr) else np.nan,
            'mag_norm_mad':          float(g['mag_norm_mad'].median()),
            'n_consistent':          n_con,
            'outlier_images':        out_imgs,
            'n_trustworthy':         int(trusts.sum()),
            'all_trustworthy':       bool(trusts.all()),
            'any_trustworthy':       bool(trusts.any()),
            'has_gaia_pms':          bool(g['has_gaia_pms'].any()),
            'gaia_gmag':             float(g['gaia_gmag'].iloc[0]),
            'median_residual_sigma': float(g['residual_sigma'].median()),
            'is_star_all_images':    is_star_all,
            'is_star_any_image':     is_star_any,
            'non_star_images':       non_star_imgs,
        })

    catalog = (all_data
               .groupby(['gaia_source_id', 'filter_camera'])
               .apply(agg_group, include_groups=False)
               .reset_index())

    # Enrich with Gaia photometry (BP/RP mags + errors, excess factor, RUWE).
    # These are source-level constants; the join is on gaia_source_id.
    gaia_phot = _load_gaia_phot(data_dir, target)
    if gaia_phot is not None:
        present = {src: dst for src, dst in _GAIA_PHOT_COLS.items()
                   if src in gaia_phot.columns}
        if present:
            phot_df = (gaia_phot[list(present.keys())]
                       .rename(columns=present)
                       .reset_index()
                       .rename(columns={'source_id': 'gaia_source_id'}))
            phot_df['gaia_source_id'] = phot_df['gaia_source_id'].astype(np.int64)
            catalog['gaia_source_id'] = catalog['gaia_source_id'].astype(np.int64)
            catalog = catalog.merge(phot_df, on='gaia_source_id', how='left')

    # ── DELVE enrichment ──────────────────────────────────────────────────────
    gaia_linked, delve_only = _collect_delve_info(images)

    # 1. For Gaia-matched sources: add median DELVE photometry and IDs.
    if len(gaia_linked):
        delve_agg = (gaia_linked
                     .groupby('gaia_source_id', sort=False)
                     .apply(_agg_delve_for_gaia_source, include_groups=False)
                     .reset_index())
        delve_agg['gaia_source_id'] = delve_agg['gaia_source_id'].astype(np.int64)
        catalog = catalog.merge(delve_agg, on='gaia_source_id', how='left')
        catalog['has_delve_match'] = catalog['n_delve_images'].notna()
        n_enriched = int(catalog['has_delve_match'].sum())
        print(f'  DELVE: enriched {n_enriched} Gaia-matched sources with DELVE photometry')
    else:
        catalog['has_delve_match'] = False

    # 2. DELVE-only sources: one row per (delve_source_id, filter_camera).
    if len(delve_only) and 'delve_source_id' in delve_only.columns:
        def _agg_delve_only(g):
            # NOTE: filter_camera and delve_source_id are groupby keys —
            # pandas 3.x excludes them from g; restored via reset_index() below.
            row = {
                'gaia_source_id':        np.nan,
                'n_images':              len(g),
                'image_list':            ','.join(g['image_name'].tolist()),
                'hst_index_list':        ','.join(g['hst_index'].astype(str).tolist()),
                'mag_norm_wmean':        float(g['hst_mag_st_gdc'].median())
                                         if 'hst_mag_st_gdc' in g.columns else np.nan,
                'mag_norm_werr':         np.nan,
                'mag_norm_mad':          np.nan,
                'n_consistent':          len(g),
                'n_trustworthy':         len(g),
                'all_trustworthy':       True,
                'any_trustworthy':       True,
                'has_gaia_pms':          False,
                'gaia_gmag':             np.nan,
                'median_residual_sigma': float(g['residual_sigma'].median())
                                         if 'residual_sigma' in g.columns else np.nan,
                'has_delve_match':       True,
                'n_delve_images':        len(g),
                'delve_mtype':           g['delve_mtype'].mode().iloc[0]
                                         if 'delve_mtype' in g.columns and len(g['delve_mtype'].dropna()) else np.nan,
                'delve_ra':              float(g['delve_ra_prop'].median())
                                         if 'delve_ra_prop' in g.columns else np.nan,
                'delve_dec':             float(g['delve_dec_prop'].median())
                                         if 'delve_dec_prop' in g.columns else np.nan,
            }
            for col in _DELVE_PHOT_COLS:
                if col in g.columns:
                    vals = g[col].dropna()
                    row[col] = float(vals.median()) if len(vals) else np.nan
            return pd.Series(row)

        # reset_index() restores delve_source_id and filter_camera from the groupby keys
        delve_only_cat = (delve_only
                          .groupby(['delve_source_id', 'filter_camera'], sort=False)
                          .apply(_agg_delve_only, include_groups=False)
                          .reset_index())
        catalog = pd.concat([catalog, delve_only_cat], ignore_index=True)
        print(f'  DELVE: added {len(delve_only_cat)} DELVE-only source rows')

    out = os.path.join(data_dir, target, 'cross_match_catalog.csv')
    catalog.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Photometry CMD / colour-colour plots
# ---------------------------------------------------------------------------

def _plot_delve_photometry(cat, wide, data_dir, target, n_filters):
    """
    Produce DELVE-based CMD and colour-colour plots saved alongside the Gaia ones.

    plots_validate_delve_cmds.png  — HST mag vs DELVE g−r colour per filter
    plots_validate_delve_cc.png    — DELVE g−r vs r−i colour-colour
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Work from the per-(source, filter_camera) catalog, keeping best row per
    # (source, filter) so the wide pivot works cleanly.
    def _src_key(row):
        if pd.notna(row.get('gaia_source_id')):
            return f"g_{int(row['gaia_source_id'])}"
        return f"d_{row.get('delve_source_id', 'NA')}"

    cat = cat.copy()
    cat['_src_key'] = cat.apply(_src_key, axis=1)
    cat['_filter']  = cat['filter_camera'].str.split('/').str[0]

    # Per-source DELVE photometry (constant across filters — take first non-null).
    delve_src = (cat.drop_duplicates(subset='_src_key')
                 [['_src_key'] + [c for c in ['delve_rmag', 'delve_gmag', 'delve_imag', 'delve_zmag']
                                  if c in cat.columns]])

    # Wide HST magnitudes (one col per filter).
    cat_best = (cat.sort_values('n_trustworthy', ascending=False)
                   .drop_duplicates(subset=['_src_key', '_filter']))
    wide_d = (cat_best
              .pivot_table(index='_src_key', columns='_filter',
                           values='mag_norm_wmean', aggfunc='first')
              .reset_index())
    wide_d.columns.name = None
    hst_fcols = [c for c in wide_d.columns if c != '_src_key']
    wide_d = wide_d.rename(columns={c: f'hst_{c}' for c in hst_fcols})
    wide_d = wide_d.merge(delve_src, on='_src_key', how='left')

    # Compute DELVE colours; mask sentinels.
    def _safe(col):
        if col not in wide_d.columns:
            return None
        v = wide_d[col].copy()
        v[(v < _DELVE_SENTINEL_LO) | (v > _DELVE_SENTINEL_HI)] = np.nan
        return v

    r = _safe('delve_rmag'); g = _safe('delve_gmag')
    ri = _safe('delve_imag'); z = _safe('delve_zmag')

    has_gr = g is not None and r is not None
    has_ri = ri is not None and r is not None
    gr = g - r if has_gr else None
    rmi = r - ri if has_ri else None

    # ── CMDs: one panel per HST filter ──────────────────────────────────────
    hst_filters = sorted(hst_fcols)
    n = len(hst_filters)
    if n and has_gr:
        ncols = min(4, n); nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                                 squeeze=False)
        fig.suptitle(f'{target}  —  DELVE CMDs ({n_filters} HST filters)', fontsize=14)
        for idx, filt in enumerate(hst_filters):
            ax = axes[idx // ncols][idx % ncols]
            hcol = f'hst_{filt}'
            if hcol not in wide_d.columns:
                ax.set_visible(False); continue
            ok = np.isfinite(wide_d[hcol].values) & np.isfinite(gr.values)
            ax.scatter(gr[ok], wide_d[hcol][ok], s=2, alpha=0.4, color='steelblue')
            ax.invert_yaxis()
            ax.set_xlabel('g − r  (DELVE DES)', fontsize=9)
            ax.set_ylabel(f'HST {filt}', fontsize=9)
            ax.set_title(filt, fontsize=10)
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(data_dir, target, 'plots_validate_delve_cmds.png')
        plt.savefig(out, dpi=150); plt.close()
        print(f'  DELVE CMDs: {out}')

    # ── Colour-colour: g−r vs r−i (no HST axis needed) ───────────────────────
    if has_gr and has_ri:
        ok = np.isfinite(gr.values) & np.isfinite(rmi.values)
        if ok.sum() > 10:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(gr[ok], rmi[ok], s=3, alpha=0.4, color='steelblue')
            ax.set_xlabel('g − r  (DELVE DES)', fontsize=11)
            ax.set_ylabel('r − i  (DELVE DES)', fontsize=11)
            ax.set_title(f'{target}  —  DELVE colour-colour  (n={ok.sum():,})', fontsize=12)
            plt.tight_layout()
            out = os.path.join(data_dir, target, 'plots_validate_delve_cc.png')
            plt.savefig(out, dpi=150); plt.close()
            print(f'  DELVE colour-colour: {out}')


def plot_photometry_catalog(data_dir, target):
    """
    Read cross_match_catalog.csv, pivot to wide format, and produce CMDs
    and colour-colour diagrams using the same _plot_cmds layout as the v2
    catalogue code.  Saved as {data_dir}/{target}/plots_validate_cmds.png.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print('  plot_photometry_catalog: matplotlib not available, skipping')
        return

    cat_path = os.path.join(data_dir, target, 'cross_match_catalog.csv')
    if not os.path.exists(cat_path):
        print(f'  plot_photometry_catalog: {cat_path} not found, skipping')
        return

    cat = pd.read_csv(cat_path)

    # Extract filter name from filter_camera (e.g. "F814W/WFC3/UVIS" → "F814W")
    cat['_filter'] = cat['filter_camera'].str.split('/').str[0]

    # When the same filter appears from multiple cameras, keep the row
    # with the most trustworthy observations per (source, filter).
    cat_best = (cat
                .sort_values('n_trustworthy', ascending=False)
                .drop_duplicates(subset=['gaia_source_id', '_filter'])
                .copy())

    # Pivot to wide: one row per source, one mag_wmean_{FILTER} column per filter.
    wide = (cat_best
            .pivot_table(index='gaia_source_id',
                         columns='_filter',
                         values='mag_norm_wmean',
                         aggfunc='first')
            .reset_index())
    wide.columns.name = None
    filter_cols = [c for c in wide.columns if c != 'gaia_source_id']
    wide = wide.rename(columns={c: f'mag_wmean_{c}' for c in filter_cols})

    # Merge source-level Gaia photometry back in.
    gaia_meta_cols = ['gaia_source_id', 'gaia_gmag', 'gaia_bpmag', 'gaia_rpmag']
    gaia_meta_cols = [c for c in gaia_meta_cols if c in cat.columns]
    gaia_meta = (cat_best[gaia_meta_cols]
                 .drop_duplicates(subset=['gaia_source_id']))
    wide = wide.merge(gaia_meta, on='gaia_source_id', how='left')

    # gaia_df for _plot_cmds: source_id, gmag, bp_rp
    # gaia_cc_df for _plot_color_color: source_id, gmag, bpmag, rpmag (separate)
    gaia_df    = None
    gaia_cc_df = None
    if {'gaia_gmag', 'gaia_bpmag', 'gaia_rpmag'}.issubset(wide.columns):
        _gaia_base = (wide[['gaia_source_id', 'gaia_gmag', 'gaia_bpmag', 'gaia_rpmag']]
                      .rename(columns={'gaia_source_id': 'source_id',
                                       'gaia_gmag':      'gmag',
                                       'gaia_bpmag':     'bpmag',
                                       'gaia_rpmag':     'rpmag'})
                      .copy())
        gaia_df = _gaia_base.copy()
        gaia_df['bp_rp'] = gaia_df['bpmag'] - gaia_df['rpmag']
        gaia_df = gaia_df.drop(columns=['bpmag', 'rpmag'])
        gaia_cc_df = _gaia_base  # keeps bpmag/rpmag for colour-colour

    try:
        from bp3m.pipeline.hst_catalog_crossmatch import _plot_cmds, _plot_color_color
    except ImportError:
        print('  plot_photometry_catalog: bp3m not importable, skipping plots')
        return

    n_filters = len([c for c in wide.columns if c.startswith('mag_wmean_')])
    out_cmds = os.path.join(data_dir, target, 'plots_validate_cmds.png')
    try:
        _plot_cmds(wide, gaia_df, out_cmds,
                   title=f'{target}  —  validate photometry ({n_filters} HST filters)')
        print(f'  CMDs: {out_cmds}')
    except Exception as e:
        print(f'  Warning: plots_validate_cmds.png failed: {e}')

    out_cc = os.path.join(data_dir, target, 'plots_validate_cc.png')
    try:
        _plot_color_color(wide, gaia_cc_df, out_cc,
                          title=f'{target}  —  colour-colour ({n_filters} HST filters)')
        print(f'  Colour-colour: {out_cc}')
    except Exception as e:
        print(f'  Warning: plots_validate_cc.png failed: {e}')

    # ── DELVE CMD + colour-colour ─────────────────────────────────────────────
    delve_cols = ['delve_rmag', 'delve_gmag', 'delve_imag', 'delve_zmag']
    has_delve = any(c in cat.columns for c in delve_cols)
    if has_delve:
        _plot_delve_photometry(cat, wide, data_dir, target, n_filters)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_target(target, data_dir, mag_scatter_thr=0.1,
                    offset_tol_mag=0.05, offset_tol_px=10.0):
    images = find_processed_images(target, data_dir)
    if not images:
        print(f'No processed images found for {target}'); return

    print(f'{target}: {len(images)} processed images')

    # Group by filter only — overlap-connectivity within each filter is
    # determined from the pairwise ZP graph, not assumed from the camera/detector.
    # Two images with the same filter but different sky positions will end up in
    # separate connected components and be validated independently.
    by_filter = defaultdict(list)
    for name, d in images.items():
        by_filter[d['filter']].append(name)

    all_image_stats = {}

    for filt, names in sorted(by_filter.items()):
        # Drop images without hst_mag_st_gdc before building the graph.
        missing = [n for n in names if not has_valid_stmag(images[n]['matched'])]
        if missing:
            print(f'\n  [{filt}] WARNING: {len(missing)} image(s) missing '
                  f'hst_mag_st_gdc (stale py1pass?) — skipped: {missing}')
        valid = [n for n in names if n not in missing]
        if not valid:
            continue

        # Pairwise ZPs between all images in this filter that share ≥ 3 sources.
        group_all = {n: images[n] for n in valid}
        pairwise  = compute_pairwise_zps(group_all)

        # Connected components: each is an independently calibratable set of images.
        components = find_overlap_components(valid, pairwise)
        cameras_all = sorted({images[n]['camera'] for n in valid})
        print(f'\n  [{filt}]  {len(valid)} image(s)  cameras: {", ".join(cameras_all)}'
              f'  →  {len(components)} overlap component(s)')

        for comp in components:
            comp_names = sorted(comp)

            if len(comp) == 1:
                name = comp_names[0]
                out  = write_solo_quality(images[name])
                if out is None:
                    continue
                n_total = len(images[name]['matched'])
                all_image_stats[name] = {
                    'filter':            filt,
                    'ref_image':         name,
                    'mag_scale':         'STMAG',
                    'cross_image_zp':    0.0,
                    'n_shared_with_ref': n_total,
                    'residual_after_zp': 0.0,
                    'wcs_offset_px':     0.0,
                    'n_trustworthy':     n_total,
                    'n_total':           n_total,
                }
                print(f'    {name}: solo → {out}')
                continue

            ref_name = max(comp, key=lambda n: len(images[n]['matched']))
            zp_dict  = propagate_zps_bfs(comp, pairwise, ref_name)
            group    = {n: images[n] for n in comp}

            source_stats, image_stats = validate_filter_group(
                group, ref_name, zp_dict, mag_scatter_thr, offset_tol_mag, offset_tol_px)

            for name in comp_names:
                out = write_source_quality(images[name], source_stats, image_stats,
                                           mag_scatter_thr)
                sq = pd.read_csv(out)
                n_trust = int(sq['is_trustworthy'].sum())
                n_total = len(sq)
                s = image_stats[name]
                cam_tag = '' if s['same_camera_as_ref'] else ' [x-cam]'
                zp_str  = f'{s["cross_image_zp"]:+.3f}'
                res_str = (f'{s["residual_after_zp"]:+.3f}'
                           if not np.isnan(s['residual_after_zp']) else 'N/A')
                print(f'    {name}{cam_tag} [{s["mag_scale"]}]: '
                      f'{n_trust}/{n_total} trustworthy | '
                      f'ZP={zp_str} residual={res_str} | '
                      f'WCS={s["wcs_offset_px"]:.1f}px')
                all_image_stats[name] = {
                    'filter': filt,
                    **s,
                    'n_trustworthy': n_trust,
                    'n_total':       n_total,
                }

    # Write ZP offset table: one row per image, showing which reference was used
    # and what ZP offset was applied.  Rows where cross_image_zp == 0.0 and
    # ref_image == image_name are the per-filter photometric anchors.
    if all_image_stats:
        zp_rows = []
        for name, s in all_image_stats.items():
            zp_rows.append({
                'image':               name,
                'filter':              s.get('filter', ''),
                'ref_image':           s.get('ref_image', ''),
                'mag_scale':           s.get('mag_scale', ''),
                'cross_image_zp': s.get('cross_image_zp', np.nan),
                'n_shared_with_ref':   s.get('n_shared_with_ref', 0),
                'residual_after_zp':   s.get('residual_after_zp', np.nan),
                'wcs_offset_px':       s.get('wcs_offset_px', np.nan),
                'n_trustworthy':       s.get('n_trustworthy', 0),
                'n_total':             s.get('n_total', 0),
            })
        zp_df = pd.DataFrame(zp_rows).sort_values(['filter', 'ref_image', 'image'])
        zp_path = os.path.join(data_dir, target, 'magnitude_zp_offsets.csv')
        zp_df.to_csv(zp_path, index=False)
        print(f'\n  ZP offsets: {len(zp_df)} images → {zp_path}')

    # Global catalog
    out = build_global_catalog(images, target, data_dir)
    if out:
        cat = pd.read_csv(out)
        n_sources = len(cat)
        n_multi   = int((cat['n_images'] > 1).sum())
        print(f'  Global catalog: {n_sources} source/filter entries '
              f'({n_multi} seen in >1 image) → {out}')

    # CMD / colour-colour plots from the enriched catalog
    plot_photometry_catalog(data_dir, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cross-image validation of Gaia-HST cross-matches.')
    parser.add_argument('--target', required=True)
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--mag-scatter-threshold', type=float, default=0.1,
                        help='MAD threshold for magnitude consistency flag. Default: 0.1 mag')
    parser.add_argument('--offset-tolerance-mag', type=float, default=0.05,
                        help='Tolerance for expected vs observed inter-image ZP offset. Default: 0.05 mag')
    parser.add_argument('--offset-tolerance-px', type=float, default=10.0,
                        help='WCS pointing offset tolerance (small dithers only). Default: 10 px')
    args = parser.parse_args()

    validate_target(args.target, args.data_dir,
                    mag_scatter_thr=args.mag_scatter_threshold,
                    offset_tol_mag=args.offset_tolerance_mag,
                    offset_tol_px=args.offset_tolerance_px)
