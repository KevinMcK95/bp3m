"""Which exposures share one guide-star acquisition (a 'visit' in the physical sense: same roll and pointing
errors), from the MAST support files on disk — NOT from rootname[:6] alone (user 2026-09-30).

Group key per exposure, read from <img>_spt.fits (falls back to the FLC primary header):
    PROPOSID, OBSET_ID (visit), DGESTAR + SGESTAR (dominant / roll guide stars), PA_V3 (binned to
    PA_TOL deg), and a time chain: consecutive exposures of a group must be within GAP_DAYS of each other.
Exposures with GUIDEACT other than FINE LOCK, or with a missing guide-star id, are never grouped.
"""
from __future__ import annotations
import glob, os
import numpy as np

PA_TOL = 0.02        # deg — within-visit PA_V3 scatter is < 0.005 deg; a re-acquisition on new stars changes it more
GAP_DAYS = 2.0       # a visit rarely spans more than a day; larger gaps mean a new acquisition
_KEYS = ('PROPOSID', 'OBSET_ID', 'DGESTAR', 'SGESTAR', 'PA_V3', 'EXPSTART', 'GUIDEACT', 'ROOTNAME', 'INSTRUME', 'FILTER', 'FILTER1', 'FILTER2')


def exposure_meta(flc_path: str) -> dict:
    from astropy.io import fits
    d = os.path.dirname(flc_path); root = os.path.basename(flc_path).split('_')[0]
    spt = os.path.join(d, f'{root}_spt.fits')
    out = {}
    for p in ((spt,) if os.path.exists(spt) else ()) + (flc_path,):
        try:
            h = fits.getheader(p, 0)
        except Exception:
            continue
        for k in _KEYS:
            if k not in out and h.get(k) is not None:
                out[k] = h.get(k)
    out['root'] = root.lower()
    return out


def group_key(m: dict):
    """Hashable key or None when the exposure must stay alone."""
    dg, sg = str(m.get('DGESTAR', '') or '').strip(), str(m.get('SGESTAR', '') or '').strip()
    if not dg or str(m.get('GUIDEACT', 'FINE LOCK')).upper().find('FINE') < 0:
        return None
    pa = m.get('PA_V3')
    pa_bin = int(round(float(pa) / PA_TOL)) if pa is not None else None
    return (str(m.get('PROPOSID', '')), str(m.get('OBSET_ID', '')), dg, sg, str(m.get('INSTRUME', '')), pa_bin)


def visit_groups(flc_paths) -> dict:
    """root -> group id (int); exposures that share a key AND chain within GAP_DAYS get the same id."""
    metas = [exposure_meta(p) for p in flc_paths]
    by_key = {}
    for m in metas:
        k = group_key(m)
        if k is None:
            continue
        by_key.setdefault(k, []).append(m)
    gid, out = 0, {}
    for k, ms in by_key.items():
        ms.sort(key=lambda m: float(m.get('EXPSTART') or 0.0))
        cur, last_t = None, None
        for m in ms:
            t = float(m.get('EXPSTART') or 0.0)
            if cur is None or (last_t is not None and t - last_t > GAP_DAYS):
                gid += 1; cur = gid
            out[m['root']] = cur; last_t = t
    for m in metas:
        if m['root'] not in out:
            gid += 1; out[m['root']] = gid      # singleton
    return out


def compare_with_rootname(flc_paths):
    """Diagnostic: how the metadata grouping differs from rootname[:6]."""
    metas = {os.path.basename(p).split('_')[0].lower(): exposure_meta(p) for p in flc_paths}
    g = visit_groups(flc_paths)
    naive = {r: r[:6] for r in g}
    import collections
    # rootname visits split by metadata / metadata groups spanning several rootname visits
    per_naive = collections.defaultdict(set); per_meta = collections.defaultdict(set)
    for r in g:
        per_naive[naive[r]].add(g[r]); per_meta[g[r]].add(naive[r])
    split = {k: v for k, v in per_naive.items() if len(v) > 1}
    merged = {k: v for k, v in per_meta.items() if len(v) > 1}
    reasons = collections.Counter()
    for k, gids in split.items():
        ms = [metas[r] for r in g if naive[r] == k]
        dgs = {str(m.get('DGESTAR')) for m in ms}; pas = [float(m.get('PA_V3') or np.nan) for m in ms]; ts = [float(m.get('EXPSTART') or np.nan) for m in ms]
        if len(dgs) > 1: reasons['different guide stars'] += 1
        elif np.nanmax(pas) - np.nanmin(pas) > PA_TOL: reasons['PA_V3 differs'] += 1
        elif np.nanmax(ts) - np.nanmin(ts) > GAP_DAYS: reasons['time gap'] += 1
        else: reasons['no guide-star id / not FINE LOCK'] += 1
    return dict(n_exposures=len(g), n_rootname_visits=len(per_naive), n_meta_groups=len(per_meta), rootname_visits_split=len(split), split_reasons=dict(reasons), meta_groups_spanning_visits=len(merged))
