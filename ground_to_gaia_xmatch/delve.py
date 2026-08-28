"""
DELVE proper-motion priors for ground-based fields.

Mirrors bp3m's ``--use_delve`` support, using the same DELVE extraction and the
same cross-matching machinery, so a ground field can carry the same combined
Gaia+DELVE priors an HST field does.

How it works
------------
1. ``fetch_delve`` calls ``bp3m.pipeline.download_delve.download_delve`` — no
   reimplementation — which pulls the nside=32 HEALPix tiles for the field
   footprint and caches a **Gaia-format** CSV.  That format is the whole trick:
   DELVE's columns are named exactly as Gaia's (``ra``, ``pmra``,
   ``pmra_error``, all ten correlations, ``ref_epoch``), so the existing
   ``cross_match_image`` consumes it unchanged and DELVE gets the identical
   discovery / refinement / one-to-one treatment Gaia gets.

2. ``xmatch.run(..., delve_df=...)`` runs that second pass per image and writes
   ``matched_delve.csv`` beside ``matched_gaia.csv``.

3. ``merge_delve`` joins the two per image on ``src_index`` — the index of the
   ground detection in that image's source catalogue.  A detection present in
   both files is one star seen by Gaia and DELVE, so its Gaia row gains
   ``delve_*`` prior columns.  A detection present only in the DELVE file is a
   DELVE-only star, and gets a synthetic catalogue row.

   bp3m associates the two catalogues the same way — through the shared image
   detection rather than a sky match between catalogues — which is what makes
   the association as reliable as the cross-match itself.

DELVE-only stars matter most exactly where the priors are needed: Gaia-sparse
ultra-faint dwarf fields.  A catalogue-level Gaia<->DELVE merge would gain the
improved priors but forfeit those stars entirely.

Coverage is southern: of the fields in play, DELVE fully covers Eridanus II,
Horologium I, Reticulum II, Fornax and Sculptor, and does not reach Sagittarius
dSph, Virgo III, M49 or COSMOS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import layout

DEFAULT_DELVE_DIR = '/bootes_raid6/users/kmckinnon/bp3m/DELVE_ProperMotion/PMCatalog'

DELVE_DIR = 'DELVE'
MATCHED_DELVE_CSV = 'matched_delve.csv'

# The prior columns attached to a survey row, named exactly as bp3m names them
# so the solver code ports across unchanged.
DELVE_PRIOR_COLS = [
    'delve_pmra', 'delve_pmdec',
    'delve_pmra_error', 'delve_pmdec_error',
    'delve_parallax', 'delve_parallax_error',
    'delve_ra_error', 'delve_dec_error',
    'delve_ra_cat', 'delve_dec_cat',
    'delve_corr_ra_dec', 'delve_corr_ra_plx', 'delve_corr_ra_pmra',
    'delve_corr_ra_pmdec', 'delve_corr_dec_plx', 'delve_corr_dec_pmra',
    'delve_corr_dec_pmdec', 'delve_corr_plx_pmra', 'delve_corr_plx_pmdec',
    'delve_corr_pmra_pmdec',
    'delve_gmag', 'delve_rmag', 'delve_imag', 'delve_zmag',
]

# DELVE catalogue column -> the delve_* prior column it becomes.
_SRC_TO_PRIOR = {
    'pmra': 'delve_pmra', 'pmdec': 'delve_pmdec',
    'pmra_error': 'delve_pmra_error', 'pmdec_error': 'delve_pmdec_error',
    'parallax': 'delve_parallax', 'parallax_error': 'delve_parallax_error',
    'ra_error': 'delve_ra_error', 'dec_error': 'delve_dec_error',
    'ra': 'delve_ra_cat', 'dec': 'delve_dec_cat',
    'ra_dec_corr': 'delve_corr_ra_dec',
    'ra_parallax_corr': 'delve_corr_ra_plx',
    'ra_pmra_corr': 'delve_corr_ra_pmra',
    'ra_pmdec_corr': 'delve_corr_ra_pmdec',
    'dec_parallax_corr': 'delve_corr_dec_plx',
    'dec_pmra_corr': 'delve_corr_dec_pmra',
    'dec_pmdec_corr': 'delve_corr_dec_pmdec',
    'parallax_pmra_corr': 'delve_corr_plx_pmra',
    'parallax_pmdec_corr': 'delve_corr_plx_pmdec',
    'pmra_pmdec_corr': 'delve_corr_pmra_pmdec',
    'g_mag': 'delve_gmag', 'r_mag': 'delve_rmag',
    'i_mag': 'delve_imag', 'z_mag': 'delve_zmag',
}

# The five sigmas that must all be finite and positive for a DELVE row to supply
# a usable 5x5 prior.  bp3m nullifies the whole astrometric block otherwise.
_SIGMA_COLS = ['delve_ra_error', 'delve_dec_error', 'delve_pmra_error',
               'delve_pmdec_error', 'delve_parallax_error']


def delve_root(field_root) -> Path:
    return Path(field_root) / DELVE_DIR


def fetch_delve(field_root, name: str, delve_dir=DEFAULT_DELVE_DIR,
                force: bool = False, quiet: bool = False):
    """
    Extract the DELVE catalogue over the field footprint.

    The footprint comes from the visit-detector table (the sky the images
    actually cover), not the requested cone, for the same reason the Gaia box
    does: it must contain the detectors that came back.
    """
    field_root = Path(field_root)
    gdir = delve_root(field_root)
    gdir.mkdir(parents=True, exist_ok=True)

    existing = sorted(gdir.glob('*_delve.csv'))
    if existing and not force:
        if not quiet:
            print(f'  DELVE catalogue already present: {existing[0].name}')
        return existing[0]

    from bp3m.pipeline.download_delve import download_delve
    from .scripts.new_field import footprint
    ra0, dec0, w, h = footprint(field_root, name)
    if not quiet:
        print(f'  DELVE: centre ({ra0:.4f}, {dec0:.4f}) box {w:.4f} x {h:.4f} deg')
    download_delve(ra=ra0, dec=dec0, search_width=w, search_height=h,
                   output_dir=str(field_root), field_name=name,
                   delve_dir=delve_dir, force_redownload=force)

    # download_delve nests under <output_dir>/<field_name>/DELVE/; flatten so the
    # loader's glob is a single level, matching how the Gaia fetch behaves.
    import shutil
    for f in sorted(field_root.rglob('*_delve.csv')):
        if f.parent != gdir:
            dest = gdir / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
    nested = field_root / name
    if nested.is_dir():
        for d in sorted((p for p in nested.rglob('*') if p.is_dir()),
                        key=lambda x: -len(x.parts)):
            try:
                d.rmdir()
            except OSError:
                pass
        try:
            nested.rmdir()
        except OSError:
            pass

    hits = sorted(gdir.glob('*_delve.csv'))
    if not hits:
        if not quiet:
            print('  DELVE: no tiles overlap this field — continuing without it')
        return None
    return hits[0]


# The DELVE catalogue is ~350k rows and takes seconds to parse.  Cached on the
# resolved path so repeated calls across images are free, the same reason the
# CFHT adapter caches its 848k-row detector table.
_DELVE_CACHE: dict = {}


def load_delve(field_root, quiet: bool = False) -> pd.DataFrame | None:
    """
    The DELVE catalogue for this field.

    Delegates the quality cuts to
    ``gaia_cross_match.cross_match_delve.load_delve_data`` — the same loader
    bp3m uses — rather than reimplementing them.  That drops rows with
    non-finite astrometry and applies DELVE's +/-99 photometry sentinel cut as a
    ROW filter (not a mask), and keeps every ``mtype`` category, leaving the
    cross-match's own magnitude and sigma cuts to reject poorly-constrained
    sources.

    ``gmag`` is aliased to DES ``r_mag`` because that is the magnitude bp3m
    matches DELVE on (``mag_d_in = delve_df['r_mag']``), and it is what drives
    our discovery tier-walk.
    """
    hits = sorted(delve_root(field_root).glob('*_delve.csv'))
    if not hits:
        return None
    key = str(hits[0].resolve())
    if key in _DELVE_CACHE:
        return _DELVE_CACHE[key]
    from gaia_cross_match.cross_match_delve import load_delve_data
    df = load_delve_data(str(hits[0]))
    if df is None or not len(df):
        return None
    df['source_id'] = df['source_id'].astype('int64')
    if 'gmag' not in df.columns:
        df['gmag'] = pd.to_numeric(df['r_mag'], errors='coerce')
    if 'bp_rp' not in df.columns and {'g_mag', 'i_mag'} <= set(df.columns):
        df['bp_rp'] = (pd.to_numeric(df['g_mag'], errors='coerce')
                       - pd.to_numeric(df['i_mag'], errors='coerce'))
    _DELVE_CACHE[key] = df
    return df


def build_delve_field(delve_df: pd.DataFrame, meta, xi_src=None, eta_src=None,
                      margin: float = 3500.0):
    """
    Per-image DELVE reference field — the DELVE analogue of build_gaia_field.

    Uses ``gaia_cross_match.cross_match_delve._propagate_delve``, NOT the Gaia
    propagation.  DELVE needs its own path for two reasons that matter:

      * its covariance is built by ``_construct_delve_cov``, which adds DELVE's
        systematic floors in quadrature — 10 mas on position (DECam astrometric
        calibration residuals) and 1 mas/yr on PM.  Propagating with the Gaia
        routine would silently drop those and treat DELVE positions as ~10x
        more precise than they are;
      * the Gaia routine indexes Gaia-only columns (``pseudocolour`` to flag 6p
        solutions) that DELVE has no analogue for.

    Note ``_construct_delve_cov`` orders parameters (ra, dec, plx, pmra, pmdec)
    — parallax at index 2 — which is the catalogue convention, not the solver's
    (dRA*, dDec, pmra, pmdec, plx).  Only the propagated 2x2 position block is
    used here, so the ordering does not leak out of that function.
    """
    from gaia_cross_match.cross_match_delve import _propagate_delve
    from .discovery import GaiaField
    from .geometry import gnomonic

    ra_prop, dec_prop, Ct = _propagate_delve(delve_df, meta.mjd)
    C = np.asarray(Ct)[:, 0:2, 0:2].copy()
    err = np.power(np.maximum(np.linalg.det(C), 1e-30), 0.25)
    xi, eta = gnomonic(ra_prop, dec_prop, meta.ra0, meta.dec0)

    keep = np.ones(len(delve_df), dtype=bool)
    if xi_src is not None and len(xi_src):
        keep = ((xi >= xi_src.min() - margin) & (xi <= xi_src.max() + margin)
                & (eta >= eta_src.min() - margin) & (eta <= eta_src.max() + margin))

    def col(name, default=np.nan):
        if name in delve_df.columns:
            return pd.to_numeric(delve_df[name],
                                 errors='coerce').to_numpy(float)[keep]
        return np.full(int(keep.sum()), default)

    n_keep = int(keep.sum())
    return GaiaField(
        xi=xi[keep], eta=eta[keep], C=C[keep], err=err[keep],
        mag=col('r_mag', 20.0),
        has_pms=np.ones(n_keep, dtype=bool),      # every DELVE row has a PM
        # DELVE rows have already passed download_delve's star-galaxy cut and
        # load_delve_data's finiteness cuts, so all of them are legitimate
        # discovery seeds.  There is no RUWE analogue to be conservative about.
        clean=np.ones(n_keep, dtype=bool),
        extra={
            'source_id': delve_df['source_id'].to_numpy(dtype=np.int64)[keep],
            'ra_prop': ra_prop[keep], 'dec_prop': dec_prop[keep],
            'pmra': col('pmra'), 'pmdec': col('pmdec'), 'bp_rp': col('bp_rp'),
            'index': np.where(keep)[0],
        },
    )


def to_bp3m_schema(matched: pd.DataFrame, delve_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the columns bp3m's matched_delve.csv carries, so the file is consumable
    by gaia_cross_match.validator without translation.

    The validator's ``_collect_delve_info`` reads matched_delve.csv straight off
    disk and joins ``source_quality.csv`` on ``hst_index`` to split DELVE
    detections into Gaia-linked and DELVE-only.  It also reads
    ``delve_source_id`` and the ``_DELVE_PHOT_COLS`` block.  None of those exist
    in our MATCH_COLUMNS schema, so they are added here rather than the file
    being left in a private format only this package understands.

    Our own columns are kept alongside, so the file serves both consumers.
    """
    out = matched.copy()
    out['hst_index'] = out['src_index']
    out['delve_source_id'] = out['gaia_source_id'].astype('int64')
    # bp3m names the ground magnitude columns after HST; the validator computes
    # cross-image zero-points from hst_mag_st_gdc, which for us is the image's
    # own calibrated magnitude.
    if 'src_mag' in out.columns:
        out['hst_mag_st_gdc'] = out['src_mag']
        out['hst_mag_gdc'] = out['src_mag']
    if 'src_magerr' in out.columns:
        out['hst_mag_err_gdc'] = out['src_magerr']
    if 'src_is_star' in out.columns:
        out['hst_is_star'] = out['src_is_star']

    src = delve_df.set_index('source_id')
    idx = out['delve_source_id'].to_numpy()
    keep = np.isin(idx, src.index.to_numpy())
    for scol, pcol in _SRC_TO_PRIOR.items():
        if scol not in src.columns:
            continue
        vals = np.full(len(out), np.nan, dtype=float)
        if keep.any():
            vals[keep] = pd.to_numeric(
                src[scol].reindex(idx[keep]).to_numpy(), errors='coerce')
        out[pcol] = vals
    if 'mtype' in src.columns:
        m = np.full(len(out), None, dtype=object)
        if keep.any():
            m[keep] = src['mtype'].reindex(idx[keep]).to_numpy()
        out['delve_mtype'] = m
    return out


def _valid_prior(df: pd.DataFrame) -> np.ndarray:
    """Rows whose full 5x5 DELVE covariance is usable."""
    ok = np.ones(len(df), dtype=bool)
    for c in _SIGMA_COLS:
        if c not in df.columns:
            return np.zeros(len(df), dtype=bool)
        s = pd.to_numeric(df[c], errors='coerce')
        ok &= s.notna().to_numpy() & (s.to_numpy(float) > 0)
    return ok


def merge_delve(gaia_df: pd.DataFrame, field_root, metas,
                delve_df: pd.DataFrame | None = None,
                delve_use_for_align: bool = False,
                quiet: bool = False) -> pd.DataFrame:
    """
    Attach DELVE priors to the survey catalogue and add DELVE-only stars.

    Returns a copy of `gaia_df` with the `delve_*` columns filled where a
    matched DELVE source exists, plus one synthetic row per DELVE-only star.
    Synthetic rows carry `delve_only=True` and, unless `delve_use_for_align`,
    `use_for_alignment=False`: DELVE astrometry is not currently trusted to
    calibrate an image transform, but a DELVE-only star's ground positions still
    constrain its own PM through the astrometry-only path.
    """
    field_root = Path(field_root)
    if delve_df is None:
        delve_df = load_delve(field_root, quiet=True)
    out = gaia_df.copy()
    for c in DELVE_PRIOR_COLS:
        if c not in out.columns:
            out[c] = np.nan
    if 'delve_only' not in out.columns:
        out['delve_only'] = False
    if delve_df is None:
        return out

    dsrc = delve_df.set_index('source_id')
    prior_by_gaia: dict[int, dict] = {}
    delve_only: dict[int, dict] = {}

    for meta in metas:
        d = layout.xmatch_root(field_root) / meta.rel_dir()
        pg, pdv = d / layout.MATCHED_CSV, d / MATCHED_DELVE_CSV
        if not pdv.exists():
            continue
        mdv = pd.read_csv(pdv)
        if 'src_index' not in mdv.columns or not len(mdv):
            continue
        gaia_by_src: dict[int, int] = {}
        if pg.exists():
            mg = pd.read_csv(pg, dtype={'gaia_source_id': 'int64'})
            if 'src_index' in mg.columns:
                gaia_by_src = dict(zip(mg['src_index'].astype(int),
                                       mg['gaia_source_id'].astype('int64')))
        for src_i, dv_id in zip(mdv['src_index'].astype(int),
                                mdv['gaia_source_id'].astype('int64')):
            if dv_id not in dsrc.index:
                continue
            row = dsrc.loc[dv_id]
            if isinstance(row, pd.DataFrame):      # duplicate ids: take the first
                row = row.iloc[0]
            payload = {pc: row[sc] for sc, pc in _SRC_TO_PRIOR.items()
                       if sc in dsrc.columns}
            gid = gaia_by_src.get(src_i)
            if gid is not None:
                prior_by_gaia.setdefault(int(gid), payload)
            else:
                delve_only.setdefault(int(dv_id), payload)

    # ── priors onto existing Gaia rows ───────────────────────────────────────
    n_prior = 0
    if prior_by_gaia:
        pri = pd.DataFrame.from_dict(prior_by_gaia, orient='index')
        pri.index.name = 'source_id'
        pri = pri.reset_index()
        idcol = 'source_id' if 'source_id' in out.columns else 'gaia_source_id'
        out['_sid'] = pd.to_numeric(out[idcol], errors='coerce').astype('Int64')
        pri['_sid'] = pri['source_id'].astype('Int64')
        pri = pri.drop(columns=['source_id'])
        for c in [c for c in pri.columns if c != '_sid']:
            mapping = dict(zip(pri['_sid'], pri[c]))
            filled = out['_sid'].map(mapping)
            out[c] = filled.where(filled.notna(), out[c])
        out = out.drop(columns=['_sid'])
        bad = ~_valid_prior(out)
        out.loc[bad, [c for c in DELVE_PRIOR_COLS if c in out.columns]] = np.nan
        n_prior = int(_valid_prior(out).sum())

    # ── DELVE-only synthetic rows ────────────────────────────────────────────
    n_only = 0
    if delve_only:
        rows = []
        for dv_id, payload in delve_only.items():
            src = dsrc.loc[dv_id]
            if isinstance(src, pd.DataFrame):
                src = src.iloc[0]
            r = {c: np.nan for c in out.columns}
            r.update(payload)
            # Position/epoch come from DELVE; Gaia PM columns stay NaN so the
            # solver treats the star as needing a diffuse Gaia prior, exactly as
            # a Gaia 2p source is treated, while the delve_* block supplies the
            # real information.
            # NEGATIVE synthetic id, exactly as bp3m does
            # (delve_id_to_gaia_id[did] = -did): unique, int64-safe, and
            # instantly distinguishable from a real Gaia source_id.  It MUST
            # match the id driver._append_delve_only stamps on the detections,
            # or the catalogue row and its observations never join and the star
            # is dropped.
            r['source_id'] = -int(dv_id)
            if 'gaia_source_id' in out.columns:
                r['gaia_source_id'] = -int(dv_id)
            for c, v in (('ra', src.get('ra')), ('dec', src.get('dec')),
                         ('ref_epoch', src.get('ref_epoch', 2016.0)),
                         ('gmag', src.get('gmag', src.get('r_mag'))),
                         ('bp_rp', src.get('bp_rp'))):
                if c in out.columns:
                    r[c] = v
            r['delve_only'] = True
            if 'use_for_alignment' in out.columns:
                r['use_for_alignment'] = bool(delve_use_for_align)
            if 'astrometric_params_solved' in out.columns:
                r['astrometric_params_solved'] = 0     # neither 5p nor 6p
            rows.append(r)
        add = pd.DataFrame(rows)
        keep = _valid_prior(add)
        add = add[keep]
        n_only = len(add)
        if n_only:
            out = pd.concat([out, add], ignore_index=True)

    if not quiet:
        print(f'  DELVE priors merged: {n_prior} survey sources with DELVE '
              f'covariance, {n_only} DELVE-only star(s) added'
              + ('' if delve_use_for_align else ' (astrometry only)'))
    return out


__all__ = ['DEFAULT_DELVE_DIR', 'DELVE_DIR', 'MATCHED_DELVE_CSV',
           'DELVE_PRIOR_COLS', 'delve_root', 'fetch_delve', 'load_delve',
           'merge_delve']
