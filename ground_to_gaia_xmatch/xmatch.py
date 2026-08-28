"""
Cross-match driver: adapter -> discovery -> refinement -> final pass -> tables.

Instrument-independent.  Everything telescope-specific arrived through the
Instrument adapter before this module ran.
"""

from __future__ import annotations

import io
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import layout
from .discovery import discover, magnitude_tiers
from .gaia_field import build_gaia_field
from .instruments.base import ImageMeta, Instrument
from .match_table import (
    build_match_table, check_one_to_one, transformation_dict,
    write_all_transformations, write_transformation,
)
from .refinement import final_pass, run_affine_refinement

INIT_RESID_MAX_MAS = 500.0     # sanity gate on the 4P seed


def _ts() -> str:
    return datetime.now().strftime('%H:%M:%S')


def _spread(v):
    return 0.5 * float(np.diff(np.nanpercentile(v, [16, 84]))[0])


def cross_match_image(inst: Instrument, meta: ImageMeta, gaia_df: pd.DataFrame,
                      source_tiers=magnitude_tiers, verbose=True,
                      field_builder=None):
    """
    Cross-match a single image.

    Returns a result dict, or None if the image could not be solved.
    """
    cfg = inst.config
    cat = inst.load_catalog(meta)
    if len(cat) < cfg.min_matches:
        print(f'  Too few sources: {len(cat)}')
        return None
    print(f'  Catalog: {len(cat)} sources, {int(cat.is_star.sum())} stars '
          f'({100*cat.is_star.mean():.1f}%)')

    # field_builder lets a non-Gaia reference catalogue supply its own
    # propagation and covariance.  DELVE must: its errors carry systematic
    # floors added by _construct_delve_cov, and the Gaia routine indexes
    # Gaia-only columns it does not have.
    _build = field_builder or build_gaia_field
    gaia = _build(gaia_df, meta, cat.xi, cat.eta,
                  margin=cfg.disc_max + 500.0)
    if len(gaia) < cfg.min_matches:
        print(f'  Too few Gaia in footprint: {len(gaia)}')
        return None
    print(f'  Gaia in footprint: {len(gaia)} ({int(gaia.clean.sum())} clean, '
          f'{int(gaia.has_pms.sum())} 5p/6p, {int((~gaia.has_pms).sum())} 2p)')

    best, tier = discover(cat, gaia, cfg, source_tiers=source_tiers, verbose=verbose)
    if best is None:
        return None
    best['gaia_tier'] = tier
    print(f"  4P Discovery Succeeded [{tier}]: {best['tier']}  "
          f"{best['n_match']} pairs  red_cost={best['red_cost']:.2f}")

    fit = run_affine_refinement(cat, gaia, cfg, best, verbose=verbose)
    if fit is None:
        return None
    if max(fit['init_resid_xi'], fit['init_resid_eta']) > INIT_RESID_MAX_MAS:
        print(f"  Rejected: init 6P residual "
              f"[{fit['init_resid_xi']:.0f},{fit['init_resid_eta']:.0f}] mas "
              f"exceeds {INIT_RESID_MAX_MAS:.0f} mas — spurious 4P seed.")
        return None

    matched, rejected = final_pass(cat, gaia, cfg, fit, verbose=verbose)
    if matched is None:
        return None

    m_df = build_match_table(cat, gaia, matched, fit['zp'])
    r_df = build_match_table(cat, gaia, rejected, fit['zp']) if len(rejected) else None
    check_one_to_one(m_df, meta.image_id)

    resid_xi, resid_eta = _spread(matched['dx']), _spread(matched['dy'])
    params = transformation_dict(meta, fit, best, len(m_df), resid_xi, resid_eta)

    return {'meta': meta, 'matched': m_df, 'rejected': r_df,
            'params': params, 'best': best, 'fit': fit,
            'n_sources': len(cat), 'n_gaia': len(gaia)}


def write_image_result(field_root: Path, result: dict, log_text: str = '') -> Path:
    """Write one image's outputs into the canonical layout."""
    meta = result['meta']
    out = layout.xmatch_root(field_root) / meta.rel_dir()
    out.mkdir(parents=True, exist_ok=True)
    result['matched'].to_csv(out / layout.MATCHED_CSV, index=False)
    write_transformation(out / layout.TRANSFORM_CSV, result['params'])
    if log_text:
        (out / layout.LOG_TXT).write_text(log_text)
    return out


def run(inst: Instrument, field_root: Path, source_tiers=magnitude_tiers,
        make_plots=True, force=False, delve_df=None) -> pd.DataFrame:
    """
    Cross-match every image in the dataset.

    Returns the per-image summary table (also written to disk).
    """
    field_root = Path(field_root)
    images = list(inst.iter_images())
    print(f'[{_ts()}] Processing {len(images)} images\n', flush=True)

    summary, per_exposure = [], {}
    n_skipped_done = 0
    for i, meta in enumerate(images, 1):
        # Resume: a directory is only "complete" once its sentinel exists, which
        # is written after every output file.  The sentinel carries the summary
        # row, so skipping still produces a complete summary table.
        out_dir = layout.xmatch_root(field_root) / meta.rel_dir()
        if not force and layout.is_complete(out_dir):
            row = layout.read_complete(out_dir)
            summary.append(row or {'image_id': meta.image_id, 'status': 'ok',
                                   **meta.key})
            n_skipped_done += 1
            continue

        print(f'[{_ts()}] [{i:3d}/{len(images)}] {meta.image_id} ...', flush=True)
        t0 = time.time()

        # Capture the per-image log while still echoing it.
        buf, orig = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            # Per-image: CFHT has one Gaia catalogue per exposure, LSST one per
            # field.  Adapters cache, so this is cheap on repeat calls.
            gaia_df = inst.gaia_catalog(meta)
            result = cross_match_image(inst, meta, gaia_df, source_tiers)
        finally:
            sys.stdout = orig
            log_text = buf.getvalue()
            print(log_text, end='', flush=True)

        dt = time.time() - t0
        if result is None:
            out = layout.xmatch_root(field_root) / meta.rel_dir()
            out.mkdir(parents=True, exist_ok=True)
            (out / layout.LOG_TXT).write_text(log_text)
            print(f'  -> SKIPPED ({dt:.1f}s)\n', flush=True)
            row = {'image_id': meta.image_id, 'status': 'skipped', **meta.key}
            summary.append(row)
            # Mark complete: "too few sources to match" is a property of the
            # data, not a transient failure, so a rerun would reach the same
            # verdict.  --force still redoes it.
            layout.mark_complete(out, row)
            continue

        write_image_result(field_root, result, log_text)

        # Second pass against DELVE.  Same cross_match_image, same discovery /
        # refinement / one-to-one machinery -- DELVE's catalogue is written with
        # Gaia column names precisely so this works unchanged.  Written beside
        # matched_gaia.csv and joined later on src_index (see delve.merge_delve).
        if delve_df is not None:
            from .delve import MATCHED_DELVE_CSV
            try:
                from .delve import build_delve_field
                dres = cross_match_image(inst, meta, delve_df, source_tiers,
                                         verbose=False,
                                         field_builder=build_delve_field)
            except Exception as exc:
                dres = None
                print(f'  DELVE cross-match failed: {type(exc).__name__}: {exc}')
            if dres is not None:
                out_d = layout.xmatch_root(field_root) / meta.rel_dir()
                dres['matched'].to_csv(out_d / MATCHED_DELVE_CSV, index=False)
                print(f"  DELVE: {dres['params']['n_matches']} matches")
            else:
                print('  DELVE: no solution for this image')

        if make_plots:
            from .plots.xmatch_plots import make_xmatch_plots
            make_xmatch_plots(result, layout.plots_dir(
                layout.xmatch_root(field_root) / meta.rel_dir()))

        p = result['params']
        n = p['n_matches']
        n2p = int((~result['matched']['has_gaia_pms'].astype(bool)).sum())
        print(f"  -> {n} matches ({n2p} 2p), "
              f"resid=({p['resid_xi_rms_mas']:.1f},{p['resid_eta_rms_mas']:.1f}) mas, "
              f"scale={p['scale']:.6f}, rot={p['rotation_deg']:.4f}deg  ({dt:.1f}s)\n",
              flush=True)

        summary.append({
            'image_id': meta.image_id, 'status': 'ok',
            'n_sources': result['n_sources'], 'n_matched': n, 'n_matched_2p': n2p,
            'match_rate': n / max(result['n_sources'], 1),
            'resid_xi_rms_mas': p['resid_xi_rms_mas'],
            'resid_eta_rms_mas': p['resid_eta_rms_mas'],
            'scale': p['scale'], 'rotation_deg': p['rotation_deg'],
            'mjd': meta.mjd, 'ra0': meta.ra0, 'dec0': meta.dec0,
            **meta.key,
        })
        layout.mark_complete(out_dir, summary[-1])
        per_exposure.setdefault(meta.exposure_id, []).append(result)

    # Exposure-level roll-ups.
    for exp_id, results in per_exposure.items():
        exp_dir = layout.xmatch_root(field_root) / exp_id
        pd.concat([r['matched'] for r in results], ignore_index=True) \
          .to_csv(exp_dir / layout.ALL_MATCHED_CSV, index=False)
        write_all_transformations(exp_dir / layout.ALL_TRANSFORM_CSV,
                                 [r['params'] for r in results])

    df = pd.DataFrame(summary)
    layout.xmatch_root(field_root).mkdir(parents=True, exist_ok=True)
    df.to_csv(layout.xmatch_root(field_root) / layout.SUMMARY_CSV, index=False)

    ok = df[df['status'] == 'ok'] if len(df) else df
    if len(ok):
        tot, t2p = int(ok['n_matched'].sum()), int(ok['n_matched_2p'].sum())
        print(f'[{_ts()}] {"="*55}')
        print(f'[{_ts()}] {len(ok)}/{len(df)} images  |  {tot} matches  '
              f'|  {t2p} Gaia 2p ({100*t2p/max(tot,1):.1f}%)')
        print(f'[{_ts()}] {"="*55}', flush=True)
    return df


__all__ = ['cross_match_image', 'write_image_result', 'run', 'INIT_RESID_MAX_MAS']
