"""
Step 4c: Cross-match HST PSF catalogs against DELVE using the same machinery
as the Gaia cross-match (Step 4).

For each image directory that contains {obs_id}_flc_catalog.fits, calls
gaia_cross_match.cross_match_delve.process_single_image_delve and writes:
    matched_delve.csv           — HST↔DELVE matched pairs
    diagnostic_plots_delve.png  — 8-panel diagnostic figure
    offset_histogram_delve.png  — 2D offset histogram from discovery step
    processing_log_delve.txt    — per-image console log

DELVE data is loaded from the CSV produced by download_delve.py.

Cache logic mirrors the Gaia cross-match: a xmatch_delve_status.json sidecar
records outcomes and params so re-runs can skip unchanged images.
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from bp3m.instrument_config import (
    SIGMA_ROT_DEG as _DEFAULT_SIGMA_ROT_DEG,
    SIGMA_SCALE   as _DEFAULT_SIGMA_SCALE,
    SIGMA_SKEW    as _DEFAULT_SIGMA_SKEW,
)
from bp3m.pipeline.cross_match import _find_image_folders, _has_mag_calibration


def _write_delve_status(root: Path, status: str, params_meta: dict,
                         reason: str = '', n_matched: int = 0) -> None:
    (root / 'xmatch_delve_status.json').write_text(json.dumps({
        'status':    status,
        'reason':    reason,
        'n_matched': n_matched,
        'params':    params_meta,
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
    }, indent=2))


def _delve_cache_status(hst_root: Path, params_meta: dict) -> tuple[str, str]:
    """Return ('skip'|'run', reason) based on xmatch_delve_status.json."""
    status_path = hst_root / 'xmatch_delve_status.json'
    if not status_path.exists():
        return 'run', 'no previous DELVE result'
    try:
        saved = json.loads(status_path.read_text())
    except Exception:
        return 'run', 'could not read xmatch_delve_status.json'
    if saved.get('params') != params_meta:
        return 'run', 'params changed'
    st = saved.get('status', 'unknown')
    if st == 'success':
        if (hst_root / 'matched_delve.csv').exists():
            return 'skip', f"previously matched ({saved.get('n_matched', '?')} stars)"
        return 'run', 'status=success but matched_delve.csv missing'
    if st in ('failed', 'skipped'):
        return 'skip', f"previously {st}: {saved.get('reason', '')}"
    return 'run', f'unknown status: {st}'


def _match_one_delve(args):
    """Worker: DELVE cross-match one image. Returns (image_name, n_matched, error)."""
    hst_dict, delve_df, kwargs = args
    from gaia_cross_match.cross_match_delve import process_single_image_delve

    root        = Path(hst_dict['root'])
    name        = root.name
    params_meta = kwargs.get('params_meta', {})

    try:
        out = root / 'matched_delve.csv'
        pre_mtime = out.stat().st_mtime if out.exists() else None

        process_single_image_delve(
            hst_dict, delve_df,
            hst_pix_floor=kwargs.get('hst_pix_floor', 0.5),
            min_matches=kwargs.get('min_matches', 3),
            max_mag_diff=kwargs.get('max_mag_diff', 5.0),
            scale_sweep=kwargs.get('scale_sweep', False),
            discovery_max_offset=kwargs.get('discovery_max_offset', 50),
            use_resid_floor=kwargs.get('use_resid_floor', True),
            sigma_rot_deg=kwargs.get('prior_sigma_rot_deg', None),
            sigma_scale=kwargs.get('prior_sigma_scale', None),
            sigma_skew=kwargs.get('prior_sigma_skew', None),
            init_resid_max=kwargs.get('init_resid_max', 5.0),
        )
        post_mtime = out.stat().st_mtime if out.exists() else None
        updated = post_mtime is not None and post_mtime != pre_mtime
        n = len(pd.read_csv(str(out))) if updated else 0
        if updated and n > 0:
            _write_delve_status(root, 'success', params_meta, n_matched=n)
        else:
            if out.exists():
                out.unlink()
            _write_delve_status(root, 'failed', params_meta,
                                  reason='no DELVE matches found')
        return name, n, None

    except Exception as exc:
        _write_delve_status(root, 'failed', params_meta, reason=str(exc))
        return name, 0, str(exc)


def run_cross_match_delve(
    output_dir: Path,
    field_name: str,
    delve_csv_path: "Path | str",
    telescope: str = 'HST',
    im_type: str = '_flc',
    n_processes: int = 4,
    hst_pix_floor: float = 0.5,
    min_matches: int = 3,
    max_mag_diff: float = 5.0,
    scale_sweep: bool = False,
    discovery_max_offset: int = 50,
    use_resid_floor: bool = True,
    force_rematch: bool = False,
    image_id: "str | None" = None,
    restrict_to_obsids: "list[str] | None" = None,
    lib_dir: "Path | None" = None,
    prior_sigma_rot_deg: "float | None" = None,
    prior_sigma_scale: "float | None" = None,
    prior_sigma_skew: "float | None" = None,
    init_resid_max: float = 5.0,
) -> list[Path]:
    """
    Cross-match all PSF-fit HST catalogs in a field against the DELVE catalogue.

    Parameters
    ----------
    delve_csv_path      : path to the *_delve.csv produced by download_delve.py
    hst_pix_floor       : HST positional uncertainty floor (pixels). Default 0.5
                          (larger than Gaia's 0.05 to match DELVE's lower precision)
    max_mag_diff        : maximum r_DELVE − HST magnitude difference. Default 5.0
                          (larger than Gaia's 3.0 due to photometric scatter)
    All other parameters mirror run_cross_match().
    All mtype sources (modest1/2/3, fast) are included — the cross-match sigma
    and magnitude cuts handle source quality naturally. mtype is preserved in
    matched_delve.csv for downstream filtering.

    Returns
    -------
    List of matched_delve.csv paths
    """
    from gaia_cross_match.cross_match_delve import load_delve_data

    print('\n' + '─'*50)
    print('Step 4c: Cross-matching HST ↔ DELVE')
    print('─'*50)

    delve_df = load_delve_data(str(delve_csv_path))
    if delve_df is None or len(delve_df) == 0:
        print('  ERROR: could not load DELVE catalogue.')
        return []

    folders = _find_image_folders(output_dir, field_name,
                                   telescope=telescope, im_type=im_type)
    if image_id:
        folders = [f for f in folders if Path(f['root']).name == image_id]
    if restrict_to_obsids is not None:
        keep = set(restrict_to_obsids)
        folders = [f for f in folders if Path(f['root']).name in keep]
    if not folders:
        print('  No image catalogs found for DELVE cross-match.')
        return []

    params_meta = {
        'hst_pix_floor':        hst_pix_floor,
        'min_matches':          min_matches,
        'max_mag_diff':         max_mag_diff,
        'scale_sweep':          scale_sweep,
        'discovery_max_offset': discovery_max_offset,
        'use_resid_floor':      use_resid_floor,
        'sigma_rot_deg':  prior_sigma_rot_deg if prior_sigma_rot_deg is not None else _DEFAULT_SIGMA_ROT_DEG,
        'sigma_scale':    prior_sigma_scale   if prior_sigma_scale   is not None else _DEFAULT_SIGMA_SCALE,
        'sigma_skew':     prior_sigma_skew    if prior_sigma_skew    is not None else _DEFAULT_SIGMA_SKEW,
        'init_resid_max': init_resid_max,
    }

    from tqdm import tqdm
    work = []
    skipped = []
    skipped_nophot = []
    for hst in tqdm(folders, desc='  Checking DELVE xmatch cache', unit='img',
                    dynamic_ncols=True):
        root = Path(hst['root'])
        name = root.name

        if not force_rematch:
            action, reason = _delve_cache_status(root, params_meta)
            if action == 'skip':
                skipped.append(name)
                continue

        if not _has_mag_calibration(Path(hst['catalog'])):
            _write_delve_status(root, 'skipped', params_meta,
                                  reason='no photometric calibration')
            skipped_nophot.append(name)
            continue

        work.append((hst, delve_df, {**params_meta, 'params_meta': params_meta}))

    if skipped:
        print(f'  {len(skipped)} image(s) already DELVE-matched (cached).')
    if skipped_nophot:
        print(f'  {len(skipped_nophot)} image(s) skipped (no photometric calibration).')
    if not work:
        print('  All images already processed — nothing to do.')
        return sorted(Path(f['root']) / 'matched_delve.csv'
                      for f in folders
                      if (Path(f['root']) / 'matched_delve.csv').exists())

    print(f'  Running DELVE cross-match on {len(work)} image(s)...')

    n_workers = (min(len(work), n_processes)
                 if n_processes > 0 else len(work))
    n_workers = max(1, n_workers)

    results: list[tuple[str, int, "str | None"]] = []
    if n_workers == 1 or len(work) == 1:
        for args in work:
            results.append(_match_one_delve(args))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_match_one_delve, a): a[0]['root'] for a in work}
            for fut in as_completed(futs):
                results.append(fut.result())

    n_ok = n_fail = 0
    for name, n, err in results:
        if err:
            print(f'  WARNING: {name} DELVE failed — {err}')
            n_fail += 1
        elif n > 0:
            print(f'  {name}: {n} DELVE matches')
            n_ok += 1
        else:
            n_fail += 1
    print(f'  DELVE cross-match complete: {n_ok} succeeded, {n_fail} failed.')

    return sorted(Path(f['root']) / 'matched_delve.csv'
                  for f in folders
                  if (Path(f['root']) / 'matched_delve.csv').exists())
