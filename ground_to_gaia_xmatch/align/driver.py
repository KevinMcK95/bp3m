"""
Alignment driver — instrument-independent.

Reads the cross-match outputs for a set of images, builds the star catalogue,
runs AlignmentSolver, and writes the results.

Two grouping modes, both using the same solver:

  per-image   one solve per image.  Each image is aligned independently.
  joint       ONE solve across many images.  Stars observed in more than one
              image are shared, so the alignment and the stellar parameters are
              constrained simultaneously.

The solver has always accepted a list of image records — `_precompute_geometry`
reports "across N image(s)".  Only the old drivers were per-image; joint mode is
a grouping choice here, not new solver code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from .. import layout
from ..instruments.base import ImageMeta
from ..match_table import read_transformation
from .solver import N_R, AlignmentSolver

MIN_STARS = 6

# Parameter rows that carry image identity rather than a number.  They collide
# with identity columns on the wide frame and must be dropped explicitly — CFHT
# used to drop 'ext' by hand and LSST relied on assignment order to clobber
# 'detector'/'band', which silently mis-keyed a detector if the order changed.
IDENTITY_PARAMS = {'ext', 'detector', 'band', 'visit', 'expnum',
                   'instrument', 'exposure', 'image_id'}


def _as_float(params: dict, key: str, default: float) -> float:
    """
    Read a numeric transform parameter.

    Values arrive as strings, because the long-format `value` column also holds
    text (tier labels, band names) and so is object dtype.  Never do arithmetic
    on these without going through here.
    """
    v = params.get(key, default)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float(default)
    return f if np.isfinite(f) else float(default)


def load_image_record(field_root: Path, meta: ImageMeta) -> dict | None:
    """
    Build one solver image record from this image's cross-match outputs.

    Returns None (with a reason printed) if the image cannot be used.
    """
    d = layout.xmatch_root(field_root) / meta.rel_dir()
    mfile, tfile = d / layout.MATCHED_CSV, d / layout.TRANSFORM_CSV
    if not mfile.exists() or not tfile.exists():
        print(f'  [{meta.image_id}] missing cross-match output — skip')
        return None

    matched = pd.read_csv(mfile)
    # Read the id column straight to int64: a float64 round-trip silently
    # corrupts Gaia source_ids above 2^53.
    matched['gaia_source_id'] = matched['gaia_source_id'].astype('int64')
    if len(matched) < MIN_STARS:
        print(f'  [{meta.image_id}] only {len(matched)} matches — skip')
        return None

    params = {k: v for k, v in read_transformation(tfile).items()
              if k not in IDENTITY_PARAMS}

    A0 = _as_float(params, 'A', 1.0)
    B0 = _as_float(params, 'B', 0.0)
    C0 = _as_float(params, 'C', 0.0)
    D0 = _as_float(params, 'D', 1.0)
    xs_o = _as_float(params, 'xs_o', 0.0)
    ys_o = _as_float(params, 'ys_o', 0.0)
    xt_o = _as_float(params, 'xt_o', 0.0)
    yt_o = _as_float(params, 'yt_o', 0.0)

    return {
        'name': meta.image_id,
        'mjd': _as_float(params, 'mjd', meta.mjd),
        'ra0': _as_float(params, 'ra0', meta.ra0),
        'dec0': _as_float(params, 'dec0', meta.dec0),
        'matched': matched,
        'A0': A0, 'B0': B0, 'C0': C0, 'D0': D0,
        'dx0': xt_o - A0 * xs_o - B0 * ys_o,
        'dy0': yt_o - C0 * xs_o - D0 * ys_o,
        # Physical plate scale, so save_results can report pixel_scale_mas
        # alongside the dimensionless scale ratio (bp3m writes the former).
        'pixel_scale': _as_float(params, 'pixel_scale', meta.pixel_scale),
        # Instrument-native identity, written verbatim to the output tables.
        **meta.key,
        'meta': meta,
    }


def _star_catalog(records: Sequence[dict], gaia_df: pd.DataFrame):
    """Gaia rows for every star matched in any of `records`, plus the id->row map."""
    ids = set()
    for rec in records:
        ids.update(rec['matched']['gaia_source_id'].astype('int64').tolist())
    sub = gaia_df[gaia_df['gaia_source_id'].isin(ids)].copy().reset_index(drop=True)
    return sub, {int(g): i for i, g in enumerate(sub['gaia_source_id'])}


def solve(records: Sequence[dict], gaia_df: pd.DataFrame, out_dir: Path,
          label: str, n_iter=20, clip_sigma=4.5,
          sigma_rot_deg=None, sigma_scale=None, sigma_skew=None,
          sigma_pointing=None, alpha_scale_chi2=False,
          exclude_2p_from_alignment=False,
          make_plots=True, verbose=True):
    """
    Run the solver over `records` (one image or many) and write the outputs.
    """
    gaia_sub, star_id_to_idx = _star_catalog(records, gaia_df)
    if len(gaia_sub) == 0:
        print(f'  [{label}] no Gaia stars — skip')
        return None

    # Drop the non-solver key before handing records over.
    img_records = [{k: v for k, v in r.items() if k != 'meta'} for r in records]

    solver = AlignmentSolver(
        img_records, gaia_sub, star_id_to_idx,
        sigma_rot_deg=sigma_rot_deg, sigma_scale=sigma_scale,
        sigma_skew=sigma_skew, sigma_pointing=sigma_pointing,
        exclude_2p_from_alignment=exclude_2p_from_alignment,
    )

    if verbose:
        print(f'  [{label}] {len(gaia_sub)} stars across {len(records)} '
              f'image(s)  fitting...')

    r_hat, C_r, v_hat, C_vT, a_arr, K_img = solver.fit(
        n_iter=n_iter, clip_sigma=clip_sigma,
        inflate_errors=True, inflate_from_iter=3, inflate_alpha_max=3.0,
        alpha_scale_chi2=alpha_scale_chi2, verbose=verbose)
    v_mean, v_cov = solver.compute_analytic_posteriors(a_arr, K_img, C_vT, C_r)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    solver.save_results(r_hat, C_r, v_hat, C_vT, v_mean, v_cov, out_dir)
    if make_plots:
        solver.make_plots(r_hat, v_hat, v_mean, v_cov, C_vT, C_r, out_dir)

    resid = solver.compute_residuals(r_hat, v_hat)
    total_used = total_n = 0
    for rec in records:
        name = rec['name']
        rd = resid[name]
        use = rd['use']
        n_used = int(solver._img_data[name]['use_for_fit'].sum())
        n_tot = int(solver._img_data[name]['n'])
        total_used += n_used
        total_n += n_tot
        rms_xi = float(np.sqrt(np.mean(rd['resid_xi'][use]**2))) if use.any() else np.nan
        rms_eta = float(np.sqrt(np.mean(rd['resid_eta'][use]**2))) if use.any() else np.nan
        print(f'  [{name}] {n_used}/{n_tot} stars  '
              f'rms xi={rms_xi:.2f} eta={rms_eta:.2f} mas')
    if len(records) > 1:
        print(f'  [{label}] TOTAL {total_used}/{total_n} star-image pairs, '
              f'{len(gaia_sub)} distinct stars')
    return out_dir


def run_per_image(inst, field_root: Path, **kw):
    """One independent solve per image."""
    field_root = Path(field_root)
    out_dirs = []
    images = list(inst.iter_images())
    for i, meta in enumerate(images, 1):
        print(f'[{i:3d}/{len(images)}] {meta.image_id}', flush=True)
        rec = load_image_record(field_root, meta)
        if rec is None:
            continue
        out = solve([rec], _gaia(inst, meta),
                    layout.align_root(field_root) / meta.rel_dir(),
                    meta.image_id, **kw)
        if out:
            out_dirs.append(out)
    print(f'\nDone: {len(out_dirs)}/{len(images)} images solved')
    return out_dirs


def run_joint(inst, field_root: Path, label='all', **kw):
    """One solve across every image, sharing stars between them."""
    field_root = Path(field_root)
    records, gaia_parts = [], []
    for meta in inst.iter_images():
        rec = load_image_record(field_root, meta)
        if rec is not None:
            records.append(rec)
            gaia_parts.append(_gaia(inst, meta))
    if not records:
        print('No usable images — nothing to solve.')
        return None
    # Exposures may carry different Gaia catalogues; pool and de-duplicate.
    gaia_df = (pd.concat(gaia_parts, ignore_index=True)
                 .drop_duplicates('gaia_source_id')
                 .reset_index(drop=True))

    n_pairs = sum(len(r['matched']) for r in records)
    ids = set()
    for r in records:
        ids.update(r['matched']['gaia_source_id'].astype('int64').tolist())
    print(f'Joint solve: {len(records)} images, {n_pairs} star-image pairs, '
          f'{len(ids)} distinct stars '
          f'({n_pairs / max(len(ids), 1):.2f} observations per star)')

    return solve(records, gaia_df,
                 layout.joint_align_dir(field_root, label), f'joint_{label}', **kw)


def _gaia(inst, meta=None) -> pd.DataFrame:
    """Gaia catalogue with the id column named and typed as the solver expects."""
    g = inst.gaia_catalog(meta)
    if 'gaia_source_id' not in g.columns:
        g = g.rename(columns={'source_id': 'gaia_source_id'})
    g['gaia_source_id'] = g['gaia_source_id'].astype('int64')
    return g


__all__ = ['load_image_record', 'solve', 'run_per_image', 'run_joint', 'MIN_STARS']
