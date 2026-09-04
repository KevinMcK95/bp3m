"""
CFHT/UNIONS ingestion for the bp3m joint alignment (--use_cfht).

CFHT detectors become IMAGES in the joint solve — their alignments are
solved simultaneously with the HST images and the stellar astrometry.
The bulk CFHT+Gaia posteriors provide ONLY the initialization and a
deliberately weak prior (widths = cfht_prior_inflate x the posterior
sigmas; default 10x, i.e. covariance x100 — the posterior widths would
double-count the Gaia stars that are in this solve again). The Gaia+CFHT
astrometric solutions are NOT used as priors on the stars: only the
measured positions enter, exactly as for HST images.

Star identities:
  * Gaia rows: the Gaia source id (joins the existing star list).
  * Faint (non-Gaia) rows: connected components of the bipartite match
    graph {(HST image, hst_index)} <-> {(expnum, ext, src_index)} from the
    matched_cfht tables — a CFHT source matched to HST detections in
    several HST images IS one star. Ids are deterministic negatives:
    -(component_rank + 1), stable for fixed inputs.

Frames:
  CFHT "detector coordinates" are pseudo-pixels: raw gnomonic offsets at
  the exposure tangent divided by CFHT_PSCALE_MAS. The bulk alignment
  model (gaia_at_epoch = M @ raw_xi, tangent ra0_final) then maps onto
  bp3m's convention directly: init (A,B,C,D) = bulk M, dRA0 = dDec0 = 0,
  tangent point = ra0_final/dec0_final.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CFHT_PSCALE_MAS = 187.0     # MegaCam plate scale [mas/px] for pseudo-pixels
SIGMA_CFHT_POS_MAS = 20.0   # per-detection centroid floor (matches Step 4d)


def _union_find(edges):
    """Connected components over hashable nodes. Returns {node: root}."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra_, rb_ = find(a), find(b)
        if ra_ != rb_:
            parent[ra_] = rb_
    return {n: find(n) for n in parent}


def collect_cfht_matches(field_dir: Path) -> pd.DataFrame:
    """All matched_cfht rows across the field's HST images, one table."""
    field_dir = Path(field_dir)
    hst_root = field_dir / 'HST' / 'mastDownload' / 'HST'
    frames = []
    for d in sorted(hst_root.iterdir()):
        f = d / 'matched_cfht.csv'
        if not f.exists():
            continue
        m = pd.read_csv(f, dtype={'gaia_source_id': str})
        m['hst_image'] = d.name
        frames.append(m)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df['is_gaia'] = (df.gaia_source_id.notna()
                     & (df.gaia_source_id.astype(str).str.len() > 3))
    return df


def assign_faint_star_ids(matches: pd.DataFrame) -> pd.DataFrame:
    """Deterministic negative star ids for the faint (non-Gaia) matches via
    connected components of the HST<->CFHT match graph."""
    f = matches[~matches.is_gaia]
    edges = [((r.hst_image, int(r.hst_index)),
              (int(r.cfht_expnum), int(r.cfht_ext), int(r.cfht_src_index)))
             for r in f.itertuples()]
    roots = _union_find(edges)
    # deterministic ordering: sort components by their smallest node repr
    comp_nodes: dict = {}
    for n, r in roots.items():
        comp_nodes.setdefault(r, []).append(n)
    comp_key = {r: min(map(repr, ns)) for r, ns in comp_nodes.items()}
    ordered = sorted(comp_key, key=comp_key.get)
    comp_id = {r: -(i + 1) for i, r in enumerate(ordered)}
    node_id = {n: comp_id[r] for n, r in roots.items()}
    out = matches.copy()
    ids = np.zeros(len(out), dtype=np.int64)
    for i, r in enumerate(out.itertuples()):
        if r.is_gaia:
            ids[i] = np.int64(r.gaia_source_id)
        else:
            ids[i] = node_id[(r.hst_image, int(r.hst_index))]
    out['star_id'] = ids
    return out


def load_cfht_bulk_posterior(cfht_dir: Path, expnum: int, ext: int):
    """(r_init dict, sigma dict, tangent, mjd) from the bulk align outputs."""
    al_csv = (Path(cfht_dir) / 'align' / f'cfht_{expnum}'
              / f'det_{ext:02d}' / 'image_transformations.csv')
    if not al_csv.exists():
        return None
    ar = pd.read_csv(al_csv).iloc[0]
    return dict(
        abcd=(float(ar.A), float(ar.B), float(ar.C), float(ar.D)),
        sigma_abcd=(float(ar.sigma_A), float(ar.sigma_B),
                    float(ar.sigma_C), float(ar.sigma_D)),
        sigma_pointing_mas=(float(ar.get('sigma_dx', 10.0)),
                            float(ar.get('sigma_dy', 10.0))),
        ra0=float(ar.ra0_final), dec0=float(ar.dec0_final),
        mjd=float(ar.mjd),
        n_stars=int(ar.get('n_stars_alignment', 0)),
    )


def build_cfht_images(field_dir, cfht_dir, matches: pd.DataFrame,
                      cfht_prior_inflate: float = 10.0,
                      min_stars: int = 5):
    """CFHT detector entries for the joint solve.

    Returns (images meta dict, stars_per_image dict, faint_catalog df).
    Meta mirrors the HST loader's fields; stars_df carries the pseudo-pixel
    detector coordinates, per-detection covariance, star ids and provenance
    tier for plotting (gaia_cfht_hst / gaia_cfht / cfht_hst).
    """
    from ground_to_gaia_xmatch.geometry import gnomonic
    from bp3m.pipeline.cross_match_cfht import (load_cfht_detector,
                                                _read_cat)
    cfht_dir = Path(cfht_dir)
    images, stars_per_image = {}, {}
    if 'star_id' not in matches.columns:
        matches = assign_faint_star_ids(matches)
    for (expnum, ext), grp in matches.groupby(['cfht_expnum', 'cfht_ext']):
        post = load_cfht_bulk_posterior(cfht_dir, int(expnum), int(ext))
        if post is None:
            continue
        src, mg, t = load_cfht_detector(cfht_dir, int(expnum), int(ext))
        name = f'cfht_{int(expnum)}_{int(ext):02d}'
        # detections used in the joint solve: every matched row of this det
        rows = grp.drop_duplicates('cfht_src_index')
        sel = src.set_index('src_index').loc[
            rows.cfht_src_index.to_numpy()]
        xi, eta = gnomonic(sel.ra.to_numpy(), sel.dec.to_numpy(),
                           post['ra0'], post['dec0'])
        # ALSO include the detector's other Gaia-matched sources (they are
        # measurements of stars already in the solve, even without an HST
        # counterpart in this field -> tier gaia_cfht)
        extra = mg[~mg.src_index.isin(rows.cfht_src_index)]
        tier = np.where(rows.is_gaia.to_numpy(), 'gaia_cfht_hst', 'cfht_hst')
        star_ids = rows.star_id.to_numpy()
        if len(extra):
            e_sel = src.set_index('src_index').loc[extra.src_index.to_numpy()]
            xi_e, eta_e = gnomonic(e_sel.ra.to_numpy(), e_sel.dec.to_numpy(),
                                   post['ra0'], post['dec0'])
            xi = np.concatenate([xi, xi_e])
            eta = np.concatenate([eta, eta_e])
            sel = pd.concat([sel, e_sel])
            star_ids = np.concatenate([
                star_ids, extra.gaia_source_id.astype(np.int64).to_numpy()])
            tier = np.concatenate([tier, ['gaia_cfht'] * len(extra)])
        if len(sel) < min_stars:
            continue
        stars_df = pd.DataFrame({
            'Gaia_id': star_ids,
            'X': xi / CFHT_PSCALE_MAS, 'Y': eta / CFHT_PSCALE_MAS,
            'x_hst_err': SIGMA_CFHT_POS_MAS / CFHT_PSCALE_MAS,
            'y_hst_err': SIGMA_CFHT_POS_MAS / CFHT_PSCALE_MAS,
            'xy_hst_corr': 0.0,
            'mag': sel.mag.to_numpy(), 'mag_err': sel.magerr.to_numpy(),
            'q_hst': 0.05, 'use_for_alignment': True,
            'use_for_align_init_flag': True,
            'provenance': tier,
        })
        # duplicate star in one detector (should not happen) — keep first
        stars_df = stars_df.drop_duplicates('Gaia_id', keep='first')
        A, B, C, D = post['abcd']
        sA, sB, sC, sD = [max(sg, 1e-7) * cfht_prior_inflate
                          for sg in post['sigma_abcd']]
        sdx, sdy = [max(sg, 1.0) * cfht_prior_inflate
                    for sg in post['sigma_pointing_mas']]
        # prior mean in bp3m's (rot, scale) parameterisation from the bulk M;
        # prior widths mapped from the inflated per-parameter sigmas
        rot_deg = float(np.degrees(np.arctan2(B - C, A + D)))
        scale = float(np.sqrt(max(A * D - B * C, 1e-12)))
        images[name] = dict(
            instrument='CFHT', detector='MEGACAM', filter='r',
            expnum=int(expnum), ext=int(ext),
            hst_time_mjd=post['mjd'],
            ra0=post['ra0'], dec0=post['dec0'],
            pixel_scale=CFHT_PSCALE_MAS,
            orig_pixel_scale=CFHT_PSCALE_MAS,
            orig_rot_deg=rot_deg,
            initial_scale_ratio=scale,
            pixel_scale_ratio=scale,
            rotation_deg=rot_deg,
            on_skew=0.5 * (A - D), off_skew=0.5 * (B + C),
            fcm_abcd=np.array([A, B, C, D, 0.0, 0.0]),
            # solver prior hooks (_make_image_prior meta overrides)
            sigma_rot_deg=float(np.degrees(np.hypot(sB, sC))),
            sigma_scale=float(np.hypot(sA, sD)),
            sigma_skew=float(0.5 * np.hypot(sA + sD, sB + sC)
                             * 0 + max(sA, sB, sC, sD)),
            sigma_pointing=float(max(sdx, sdy)),
            n_bulk_stars=post['n_stars'],
        )
        stars_per_image[name] = stars_df
    # faint-star catalog entries (2p-style: flat PM/plx priors, position at
    # the mean matched HST epoch)
    faint = matches[~matches.is_gaia]
    cat_rows = []
    for sid, g in faint.groupby('star_id'):
        cat_rows.append(dict(
            Gaia_id=np.int64(sid),
            ra=float(g.cfht_ra_hst_epoch.mean()
                     if 'cfht_ra_hst_epoch' in g else g.cfht_ra_prop.mean()),
            dec=float(g.cfht_dec_hst_epoch.mean()
                      if 'cfht_dec_hst_epoch' in g
                      else g.cfht_dec_prop.mean()),
            n_hst_images=g.hst_image.nunique(),
            n_cfht_dets=g.groupby(['cfht_expnum', 'cfht_ext']).ngroups,
        ))
    faint_catalog = pd.DataFrame(cat_rows)
    return images, stars_per_image, faint_catalog


def build_fallback_hst_images(field_dir, matches: pd.DataFrame,
                              pos_err_floor: float = 0.05,
                              pos_corr_table=None):
    """HST images with CFHT matches but NO Gaia cross-match: initialize from
    transformation_cfht_<exp>.csv (same schema and Gaia-frame convention as
    transformation.csv — the CFHT side was pre-aligned by the bulk posterior,
    so the relative fit already sits on top of it) and build their stars_df
    from the matched_cfht pairs (star ids: Gaia where known, faint negatives
    otherwise). Returns (images meta dict, stars_per_image dict)."""
    from bp3m.data_loader_flc import _read_image_meta, _build_stars_df
    pos_corr = None
    if pos_corr_table is not None:
        from bp3m.pos_corr import PseudoGDCSet
        pos_corr = PseudoGDCSet(pos_corr_table)
    field_dir = Path(field_dir)
    hst_root = field_dir / 'HST' / 'mastDownload' / 'HST'
    matches = (matches if 'star_id' in matches.columns
               else assign_faint_star_ids(matches))
    images, stars_per_image = {}, {}
    for img_name, grp in matches.groupby('hst_image'):
        d = hst_root / img_name
        if (d / 'transformation.csv').exists():
            continue        # normal Gaia-initialized image
        # best exposure = most matches with an existing transformation file
        best = None
        for exp, g in grp.groupby('cfht_expnum'):
            tf = d / f'transformation_cfht_{int(exp)}.csv'
            if tf.exists() and (best is None or len(g) > best[1]):
                best = (tf.name, len(g))
        if best is None:
            continue
        meta = _read_image_meta(d, img_name, transformation_file=best[0])
        if meta is None:
            continue
        meta['cfht_initialized'] = True
        override = (grp.drop_duplicates('hst_index')
                    [['hst_index', 'star_id']]
                    .rename(columns={'star_id': 'gaia_source_id'}))
        stars_df = _build_stars_df(d, img_name, None, pos_err_floor,
                                   pos_corr=pos_corr, meta=meta,
                                   match_override=override)
        if stars_df is None or len(stars_df) < 5:
            continue
        images[img_name] = meta
        stars_per_image[img_name] = stars_df
        print(f'    {img_name}: CFHT-initialized ({best[0]}, '
              f'{len(stars_df)} matched detections)')
    return images, stars_per_image
