"""Visit-group completion of the Gaia cross-match (user 2026-10-01).

Exposures that share one guide-star acquisition (bp3m.visit_groups; split further by primary
WCSNAME, because a visit can mix a-priori and Gaia-aligned header solutions) share the error of
their header WCS.  Measured on 2026-10-01: within a visit the header pointing error agrees to
~1 mas and the rotation error to ~1", independent of the dither separation (to 60") and of the
ORIENTAT difference (to 36"), whereas it scatters by 3-9 mas (and up to ~1" on GSC-era headers)
between visits.  Each image keeps its OWN header (dither, roll); only the shared error is
transferred (gaia_cross_match.sibling_seed).

This module provides the group bookkeeping: members, trusted siblings, consensus prediction,
suspect solutions (own fit disagrees with the siblings), the group master list of matched Gaia
sources and, per member, the master sources expected in its footprint but not matched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_PASS_VERSION = 1
MIN_SIB_MATCHES = 5      # a member needs >= this many matches to define the shared header error
SUSPECT_PX = 3.0         # own solution vs sibling consensus (max over footprint) -> suspect
CONSENSUS_SPREAD_PX = 1.0  # siblings must agree among themselves this well to overrule a member
FOOTPRINT_MARGIN_PX = 10.0
ZP_TOL_MAG = 2.5         # zero point (Gaia G - calibrated STMAG) within this of the filter reference
ZP_FLOOR_MAG = -6.0      # no reference for the filter: spurious matches to faint junk sit at ~ -10
ZP_REF_MIN_MATCHES = 10
ZP_GATE_MAX_N = 20       # chance (spurious) solutions only happen at low N; >= this the zero point is not used


def filter_key(flc) -> tuple:
    from astropy.io import fits
    h = fits.getheader(flc, 0)
    f = h.get('FILTER')
    if not f:
        fs = [str(h.get(k, '')).strip() for k in ('FILTER1', 'FILTER2')]
        fs = [x for x in fs if x and not x.upper().startswith('CLEAR')]
        f = '+'.join(fs) if fs else 'CLEAR'
    return (str(h.get('INSTRUME', '')).strip(), str(h.get('DETECTOR', '')).strip(), str(f).strip())


def matched_zp(path) -> float:
    try:
        m = pd.read_csv(path, usecols=['gaia_gmag', 'hst_mag_st_gdc'])
        return float(np.median(m['gaia_gmag'] - m['hst_mag_st_gdc']))
    except Exception:
        return float('nan')


def field_zp_reference(folders: list) -> dict:
    """(instrume, detector, filter) -> median zero point over the field's matches with >= ZP_REF_MIN_MATCHES."""
    acc = {}
    for f in folders:
        r = Path(f['root']); st = _status(r)
        if st.get('status') != 'success' or int(st.get('n_matched', 0) or 0) < ZP_REF_MIN_MATCHES:
            continue
        z = matched_zp(r / 'matched_gaia.csv')
        if np.isfinite(z) and z > ZP_FLOOR_MAG:      # a deep field's majority can itself be spurious
            acc.setdefault(filter_key(f['flc']), []).append(z)
    return {k: float(np.median(v)) for k, v in acc.items()}


ZP_DECIDES = False   # 2026-10-01: zp ~ -10 is often a REAL match to saturated stars (wing-fit flux);
                     # the main-pass chance-coincidence test rejects spurious solutions instead.


def zp_sane(zp: float, ref, n: int = 0) -> bool:
    if not ZP_DECIDES or n >= ZP_GATE_MAX_N:
        return True
    if not np.isfinite(zp):
        return False
    return abs(zp - ref) < ZP_TOL_MAG if ref is not None and np.isfinite(ref) else zp > ZP_FLOOR_MAG


def _status(root: Path) -> dict:
    try:
        return json.loads((root / 'xmatch_status.json').read_text())
    except Exception:
        return {}


def file_md5(p: Path) -> str:
    try:
        return hashlib.md5(Path(p).read_bytes()).hexdigest()[:12]
    except Exception:
        return ''


class Member:
    def __init__(self, folder: dict):
        from astropy.io import fits
        from gaia_cross_match.cross_match import get_hst_params
        from gaia_cross_match import sibling_seed as ss
        self.root = Path(folder['root']); self.name = self.root.name
        self.flc, self.catalog = folder['flc'], folder['catalog']
        self.status = _status(self.root)
        self.ok = (self.status.get('status') == 'success' and (self.root / 'matched_gaia.csv').exists()
                   and (self.root / 'transformation.csv').exists())
        self.n = int(self.status.get('n_matched', 0) or 0) if self.ok else 0
        self.params = get_hst_params(self.flc, catalog_file=self.catalog)
        self.c = np.array([self.params['x_cen'], self.params['y_cen']])
        cat = fits.getdata(self.catalog)
        x = np.asarray(cat['x_gdc'], float); y = np.asarray(cat['y_gdc'], float); okp = np.isfinite(x) & np.isfinite(y)
        self.empty = int(okp.sum()) < 3          # no usable catalogue: never a sibling, never rematched
        if self.empty:
            self.xr, self.yr = (0.0, 1.0), (0.0, 1.0)
        else:
            self.xr = (float(np.percentile(x[okp], 1)), float(np.percentile(x[okp], 99)))
            self.yr = (float(np.percentile(y[okp], 1)), float(np.percentile(y[okp], 99)))
        self.wcsname = str(fits.getheader(self.flc, 1).get('WCSNAME', ''))
        self.ids, self.trans, self.affine, self.E = set(), None, None, None
        self.filt = filter_key(self.flc); self.zp = float('nan')
        if self.ok:
            self.zp = matched_zp(self.root / 'matched_gaia.csv')
            m = pd.read_csv(self.root / 'matched_gaia.csv', dtype={'gaia_source_id': np.int64})
            self.ids = set(m['gaia_source_id'].astype(np.int64))
            self.trans = ss.read_transformation(self.root / 'transformation.csv')
            self.affine = affine_from_trans(self.trans, self.c)
            if self.n >= MIN_SIB_MATCHES:
                self.E = ss.measure_header_error(self.params, self.trans, self.xr, self.yr)

    def grid(self, n=7):
        from gaia_cross_match.sibling_seed import _grid
        return _grid(self.xr, self.yr, n)


def affine_from_trans(tr: dict, c) -> tuple:
    """transformation.csv (g = M (h - s_o) + t_o) -> (M, t) with g = M (h - c) + t."""
    M = np.array([[tr['A'], tr['B']], [tr['C'], tr['D']]])
    t = M @ (np.asarray(c, float) - np.array([tr['xs_o'], tr['ys_o']])) + np.array([tr['xt_o'], tr['yt_o']])
    return M, t


def eval_affine(aff, c, hx, hy):
    M, t = aff
    return np.column_stack([hx - c[0], hy - c[1]]) @ M.T + t


def disagreement_px(member: Member, aff1, aff2) -> float:
    hx, hy = member.grid()
    return float(np.max(np.hypot(*(eval_affine(aff1, member.c, hx, hy) - eval_affine(aff2, member.c, hx, hy)).T)))


def consensus(target: Member, sibs: list):
    """Median-of-siblings predicted affine for target; returns (aff, spread_px, n_sibs) or None."""
    from gaia_cross_match import sibling_seed as ss
    sibs = [s for s in sibs if s.E is not None and s is not target]
    if not sibs:
        return None
    hx, hy = target.grid()
    G = []
    for s in sibs:
        M, t, _ = ss.predicted_affine(target.params, s.E, target.xr, target.yr)
        G.append(eval_affine((M, t), target.c, hx, hy))
    G = np.array(G)                                  # (n_sib, n_grid, 2)
    med = np.median(G, axis=0)
    spread = float(np.max(np.median(np.hypot(*(G - med).transpose(2, 0, 1)), axis=0))) if len(G) > 1 else 0.0
    X = np.column_stack([hx - target.c[0], hy - target.c[1], np.ones_like(hx)])
    px, *_ = np.linalg.lstsq(X, med[:, 0], rcond=None); py, *_ = np.linalg.lstsq(X, med[:, 1], rcond=None)
    return (np.array([[px[0], px[1]], [py[0], py[1]]]), np.array([px[2], py[2]])), spread, len(sibs)


def build_groups(folders: list) -> list:
    """[[Member, ...], ...]: visit groups (guide-star acquisition) x primary WCSNAME, >= 2 members."""
    from bp3m.visit_groups import visit_groups
    g = visit_groups([f['flc'] for f in folders])
    by = {}
    for f in folders:
        by.setdefault(g[Path(f['root']).name.lower()], []).append(f)
    out = []
    for fs in by.values():
        if len(fs) < 2:
            continue
        ms = [Member(f) for f in fs]
        for wn in sorted({m.wcsname for m in ms}):
            sub = [m for m in ms if m.wcsname == wn]
            if len(sub) >= 2:
                out.append(sub)
    return out


def classify(group: list, zp_ref: dict | None = None) -> dict:
    """Per member: 'trusted' | 'suspect' | 'weak' (success, < MIN_SIB_MATCHES) | 'failed',
    with the consensus prediction from the OTHER zero-point-sane members that define E.
    A success whose zero point (Gaia G - STMAG) is inconsistent with its filter is a suspect
    whatever its siblings say: spurious solutions match faint junk and sit ~10 mag off."""
    zp_ref = zp_ref or {}
    sane = {m.name: (m.ok and zp_sane(m.zp, zp_ref.get(m.filt), m.n)) for m in group}
    info = {}
    for m in group:
        others = [s for s in group if s is not m and s.E is not None and sane[s.name]]
        cons = consensus(m, others)
        d = disagreement_px(m, m.affine, cons[0]) if (cons and m.affine is not None) else None
        if not m.ok:
            cls = 'failed'
        elif not sane[m.name]:
            cls = 'suspect'
        elif m.E is None:
            cls = 'weak'
        else:
            cls = 'trusted'
            if d is not None and d > SUSPECT_PX and cons[1] < CONSENSUS_SPREAD_PX:
                big_other = max((s.n for s in others), default=0)
                if cons[2] >= 2 or big_other >= max(10, 2 * m.n):
                    cls = 'suspect'
        info[m.name] = dict(cls=cls, cons=cons, disagree_px=d, zp=m.zp, zp_ref=zp_ref.get(m.filt),
                            zp_bad=bool(m.ok and not sane[m.name]))
    # a 'weak' member is also suspect when it disagrees with a consistent consensus
    for m in group:
        i = info[m.name]
        if i['cls'] == 'weak' and i['disagree_px'] is not None and i['disagree_px'] > SUSPECT_PX \
                and i['cons'][1] < CONSENSUS_SPREAD_PX:
            i['cls'] = 'suspect'
    return info


def master_ids(group: list, info: dict) -> set:
    ids = set()
    for m in group:
        if info[m.name]['cls'] in ('trusted', 'weak'):
            ids |= m.ids
    return ids


def project_to_member(member: Member, aff, gaia_rows: pd.DataFrame):
    """Approximate HST pixel positions (x_gdc frame) of Gaia rows in a member (PM-propagated)."""
    from gaia_cross_match import sibling_seed as ss
    yr = 2000.0 + (member.params['obs_epoch_mjd'] - 51544.5) / 365.25
    ref = gaia_rows['ref_epoch'].fillna(2016.0).to_numpy() if 'ref_epoch' in gaia_rows else 2016.0
    dt = yr - ref
    dec = gaia_rows['dec'].to_numpy(); ra = gaia_rows['ra'].to_numpy()
    ra = ra + np.nan_to_num(gaia_rows['pmra'].to_numpy()) * dt / 3.6e6 / np.cos(np.radians(dec))
    dec = dec + np.nan_to_num(gaia_rows['pmdec'].to_numpy()) * dt / 3.6e6
    fr = {k: member.params[k] for k in ('ra_cen', 'dec_cen', 'x_cen', 'y_cen', 'pixel_scale', 'orientat')}
    gx, gy = ss.sky_to_frame(ra, dec, fr)
    M, t = aff
    h = np.einsum('ij,nj->ni', np.linalg.inv(M), np.column_stack([gx, gy]) - t) + member.c
    return h[:, 0], h[:, 1]


def expected_missing(member: Member, aff, master: set, gaia_df: pd.DataFrame) -> pd.DataFrame:
    """Master sources predicted inside member's footprint that it has not matched."""
    miss = sorted(master - member.ids)
    if not miss:
        return pd.DataFrame(columns=['source_id', 'x', 'y'])
    rows = gaia_df[gaia_df['source_id'].isin(miss)]
    x, y = project_to_member(member, aff, rows)
    mg = FOOTPRINT_MARGIN_PX
    inside = (x > member.xr[0] - mg) & (x < member.xr[1] + mg) & (y > member.yr[0] - mg) & (y < member.yr[1] + mg)
    out = rows[inside][['source_id']].copy(); out['x'] = x[inside]; out['y'] = y[inside]
    if 'gmag' in rows:
        out['gmag'] = rows['gmag'].to_numpy()[inside]
    return out


def group_signature(group: list, params_key: dict) -> str:
    items = sorted((m.name, m.status.get('status', ''), m.n, file_md5(m.root / 'matched_gaia.csv')) for m in group)
    blob = json.dumps({'v': GROUP_PASS_VERSION, 'members': items, 'params': params_key}, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Driver: Step 4c visit-group completion
# ---------------------------------------------------------------------------

RECOVER_PX = 1.5          # trusted member: rerun only if a missing master star has a detection this close
ACCEPT_PX = 2.0           # a new solution must agree with the guess it was seeded from
GUESS_MAX_OFFSET = 10.0   # px half-width of the narrowed 4P offset search
MAX_ROUNDS = 3
_SKIP_STATUS = ('skipped',)


def _ts():
    import datetime
    return datetime.datetime.now().strftime('%m-%d %H:%M:%S')


def light_groups(folders: list, cache_path: Path | None = None) -> list:
    """Visit groups x WCSNAME (>= 2 members) from headers only, cached per (image, flc mtime)."""
    from astropy.io import fits
    from bp3m.visit_groups import exposure_meta, group_key, GAP_DAYS
    cache = {}
    if cache_path is not None and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    metas, changed = {}, False
    for f in folders:
        name = Path(f['root']).name; mt = Path(f['flc']).stat().st_mtime
        c = cache.get(name)
        if c is None or c.get('mtime') != mt:
            m = exposure_meta(f['flc'])
            try:
                m['WCSNAME'] = str(fits.getheader(f['flc'], 1).get('WCSNAME', ''))
            except Exception:
                m['WCSNAME'] = ''
            c = {'mtime': mt, 'meta': {k: (v if isinstance(v, (int, float, str)) or v is None else str(v)) for k, v in m.items()}}
            cache[name] = c; changed = True
        metas[name] = c['meta']
    if cache_path is not None and changed:
        try:
            cache_path.write_text(json.dumps(cache))
        except Exception:
            pass
    by = {}
    for f in folders:
        m = metas[Path(f['root']).name]; k = group_key(m)
        if k is None:
            continue
        by.setdefault((k, m.get('WCSNAME', '')), []).append((float(m.get('EXPSTART') or 0.0), f))
    out = []
    for k, lst in by.items():
        lst.sort(key=lambda t: t[0]); cur, last = [], None
        for t, f in lst:
            if cur and last is not None and t - last > GAP_DAYS:
                if len(cur) >= 2:
                    out.append(cur)
                cur = []
            cur.append(f); last = t
        if len(cur) >= 2:
            out.append(cur)
    return out


def light_signature(fs: list, params_key: dict) -> str:
    items = []
    for f in fs:
        r = Path(f['root']); st = _status(r)
        items.append((r.name, st.get('status', ''), int(st.get('n_matched', 0) or 0), file_md5(r / 'matched_gaia.csv')))
    blob = json.dumps({'v': GROUP_PASS_VERSION, 'members': sorted(items), 'params': params_key}, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:16]


def _group_task(args):
    """Worker: rematch one image into a scratch dir with a guess transform."""
    hst, kw, guess, tmp, gaia_df = args
    import shutil
    from gaia_cross_match.cross_match import process_single_image
    if gaia_df is None:
        from bp3m.pipeline.cross_match import _get_worker_gaia
        gaia_df = _get_worker_gaia()
    import os, time as _time
    # unique per attempt: on NFS a directory whose files are still open elsewhere cannot be removed
    tmp = Path(f"{tmp}_{os.getpid()}_{int(_time.time() * 1000)}")
    tmp.mkdir(parents=True, exist_ok=True)
    err = None
    try:
        process_single_image({'root': str(tmp), 'flc': hst['flc'], 'catalog': hst['catalog']}, gaia_df,
                             hst_pix_floor=kw.get('hst_pix_floor', 0.5), min_matches=kw.get('min_matches', 3),
                             zero_pm=kw.get('zero_pm', False), max_mag_diff=kw.get('max_mag_diff', 3.0),
                             scale_sweep=kw.get('scale_sweep', False),
                             discovery_max_offset=kw.get('discovery_max_offset', 50),
                             use_resid_floor=kw.get('use_resid_floor', True),
                             sigma_rot_deg=kw.get('prior_sigma_rot_deg'), sigma_scale=kw.get('prior_sigma_scale'),
                             sigma_skew=kw.get('prior_sigma_skew'), init_resid_max=kw.get('init_resid_max', 5.0),
                             pos_corr_model=kw.get('pos_corr_model'),
                             guess_affine=guess, guess_max_offset=GUESS_MAX_OFFSET)
    except Exception as e:
        err = f'{type(e).__name__}: {e}'
    n = 0
    if (tmp / 'matched_gaia.csv').exists():
        try:
            n = len(pd.read_csv(tmp / 'matched_gaia.csv'))
        except Exception:
            n = 0
    return hst['root'], str(tmp), n, err


def _recoverable(member, aff, master, gaia_df) -> int:
    """How many missing master stars have an HST detection within RECOVER_PX of their prediction."""
    from astropy.io import fits
    ms = expected_missing(member, aff, master, gaia_df)
    if not len(ms):
        return 0
    cat = fits.getdata(member.catalog)
    from scipy.spatial import cKDTree
    xy = np.column_stack([np.asarray(cat['x_gdc'], float), np.asarray(cat['y_gdc'], float)])
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    d, _ = cKDTree(xy).query(np.column_stack([ms.x, ms.y]))
    return int((d < RECOVER_PX).sum())


def _clean(o):
    """NaN/inf -> None so the status files stay strict JSON."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _write_status(root: Path, base: dict) -> None:
    import datetime
    base = dict(base); base['timestamp'] = datetime.datetime.now().isoformat(timespec='seconds')
    (root / 'xmatch_status.json').write_text(json.dumps(_clean(base), indent=2, default=str))


def _accept(member, tmp: Path, n_new: int, params_meta: dict, note: dict, apply: bool, out_root: Path | None):
    """Install the scratch result (in place, or under out_root/<image>/ for a dry run)."""
    import shutil
    dest = member.root if apply else (out_root / member.name)
    dest.mkdir(parents=True, exist_ok=True)
    for fn in ('matched_gaia.csv', 'transformation.csv', 'diagnostic_plots.png', 'offset_histogram.png'):
        if (tmp / fn).exists():
            shutil.copy2(tmp / fn, dest / fn)
    log = tmp / 'processing_log.txt'
    if log.exists():
        with open(dest / 'processing_log.txt', 'a') as f:
            f.write(f"\n\n===== visit-group completion {_ts()} ({note.get('reason')}) =====\n")
            f.write(log.read_text())
    if apply:
        (dest / 'xmatch_params.json').write_text(json.dumps(params_meta, indent=2))
        _write_status(dest, {'status': 'success', 'reason': '', 'n_matched': int(n_new), 'params': params_meta,
                             'group_pass': note})


def run_group_pass(folders: list, gaia_df: pd.DataFrame, match_kwargs: dict, params_meta_for,
                   params_key: dict, n_processes: int = 4, force: bool = False, apply: bool = True,
                   out_root: Path | None = None, cache_path: Path | None = None, log=print) -> int:
    """Step 4c.  Returns the number of images whose cross-match changed (caller re-validates)."""
    import shutil
    groups = light_groups(folders, cache_path)
    if not groups:
        log('  visit-group completion: no visit groups with >= 2 cross-matchable images')
        return 0
    todo = []
    for fs in groups:
        sig = light_signature(fs, params_key)
        if not force and all(_status(Path(f['root'])).get('group_pass', {}).get('sig') == sig for f in fs):
            continue
        todo.append(fs)
    log(f"  [{_ts()}] visit-group completion: {len(groups)} groups ({sum(len(g) for g in groups)} images); "
        f"{len(groups) - len(todo)} unchanged since their last pass (skipped), {len(todo)} to check")
    if not todo:
        return 0
    for fs in todo:          # stale scratch dirs from an interrupted pass
        for f in fs:
            for d in Path(f['root']).glob('.xmatch_group_tmp*'):
                shutil.rmtree(d, ignore_errors=True)
    zp_ref = field_zp_reference(folders)
    log(f"  [{_ts()}] zero-point references (Gaia G - STMAG, fields matches with >= {ZP_REF_MIN_MATCHES} stars): "
        + ", ".join(f"{k[2]}={v:+.2f}" for k, v in sorted(zp_ref.items())))
    results = {}            # image -> note
    n_changed = 0
    n_demoted = 0
    active = todo
    for rnd in range(1, MAX_ROUNDS + 1):
        tasks, ctx = [], {}
        for gi, fs in enumerate(active):
            grp = [m for m in (Member(f) for f in fs) if not m.empty]
            if len(grp) < 2:
                continue
            if all(_status(m.root).get('status') in _SKIP_STATUS for m in grp):
                continue
            info = classify(grp, zp_ref); master = master_ids(grp, info)
            trusted = [s for s in grp if info[s.name]['cls'] == 'trusted']
            if not trusted:     # fall back to zero-point-sane low-N members (3-4 matches) as seeds
                from gaia_cross_match import sibling_seed as _ss
                for s in grp:
                    if info[s.name]['cls'] == 'weak' and s.n >= 3 and s.trans is not None:
                        s.E = _ss.measure_header_error(s.params, s.trans, s.xr, s.yr); trusted.append(s)
            for m in grp:
                if m.status.get('status') in _SKIP_STATUS:
                    continue
                i = info[m.name]
                # seeds come from TRUSTED siblings only (a suspect never seeds another image)
                cons = consensus(m, [s for s in trusted if s is not m]); i['cons'] = cons
                if i['cls'] in ('failed', 'suspect'):
                    if cons is None:
                        results.setdefault(m.name, dict(cls=i['cls'], result='no trusted sibling', n_before=m.n, n_after=m.n,
                                                        zp_bad=i.get('zp_bad', False), zp=i.get('zp'), zp_ref=i.get('zp_ref')))
                        continue
                    guess, reason = cons[0], f"{i['cls']}: seeded from {cons[2]} sibling(s)"
                else:
                    guess = m.affine if (i['cls'] == 'trusted' or cons is None) else cons[0]
                    k = _recoverable(m, guess, master, gaia_df)
                    if k == 0:
                        results.setdefault(m.name, dict(cls=i['cls'], result='complete (no recoverable master stars)', n_before=m.n, n_after=m.n))
                        continue
                    reason = f"{i['cls']}: {k} missing master star(s) with a detection <{RECOVER_PX} px"
                hst = {'root': str(m.root), 'flc': m.flc, 'catalog': m.catalog}
                tmp = m.root / '.xmatch_group_tmp'
                tasks.append((hst, match_kwargs, guess, str(tmp)))
                ctx[str(m.root)] = (m, i, guess, reason, gi, master)
        if not tasks:
            break
        log(f"  [{_ts()}] round {rnd}: {len(tasks)} image(s) to rematch")
        outs = []
        if n_processes > 1 and len(tasks) > 1:
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            from bp3m.pipeline.cross_match import _pool_init
            cache = Path(folders[0]['root']).parents[3] / 'Gaia' / '.xmatch_group_worker_cache.pkl'
            gaia_df.to_pickle(cache)
            try:
                with ProcessPoolExecutor(max_workers=min(n_processes, len(tasks)), mp_context=_mp.get_context('forkserver'),
                                         initializer=_pool_init, initargs=(str(cache),)) as ex:
                    futs = [ex.submit(_group_task, t + (None,)) for t in tasks]
                    for fu in as_completed(futs):
                        outs.append(fu.result())
            finally:
                try:
                    cache.unlink()
                except OSError:
                    pass
        else:
            outs = [_group_task(t + (gaia_df,)) for t in tasks]
        changed_groups = set()
        for k, (root, tmp, n_new, err) in enumerate(sorted(outs), 1):
            m, i, guess, reason, gi, master = ctx[root]; tmp = Path(tmp)
            note = dict(version=GROUP_PASS_VERSION, round=rnd, cls=i['cls'], reason=reason, n_before=m.n, n_after=m.n,
                        zp_bad=i.get('zp_bad', False), zp=i.get('zp'), zp_ref=i.get('zp_ref'))
            res = 'rejected'
            if err:
                note['result'] = f'error: {err}'
            elif n_new == 0:
                note['result'] = 'no matches'
            else:
                from gaia_cross_match import sibling_seed as ss
                new_aff = affine_from_trans(ss.read_transformation(tmp / 'transformation.csv'), m.c)
                d = disagreement_px(m, new_aff, guess); note['agree_px'] = round(d, 3)
                new_ids = set(pd.read_csv(tmp / 'matched_gaia.csv', dtype={'gaia_source_id': np.int64})['gaia_source_id'])
                note['n_from_master'] = len((new_ids - m.ids) & master)
                note['n_new_ids'] = len(new_ids - m.ids)
                zn = matched_zp(tmp / 'matched_gaia.csv'); note['zp_new'] = round(zn, 3)
                if d > ACCEPT_PX:
                    note['result'] = f'disagrees with its guess by {d:.2f} px'
                elif not zp_sane(zn, zp_ref.get(m.filt), n_new):
                    note['result'] = f'zero point {zn:+.2f} inconsistent with {m.filt[2]}'
                elif i['cls'] in ('failed', 'suspect') and n_new >= int(match_kwargs.get('min_matches', 3)):
                    res = 'accepted'
                elif i['cls'] in ('trusted', 'weak') and n_new > m.n:
                    res = 'accepted'
                else:
                    note['result'] = f'no gain ({n_new} vs {m.n})'
                if res == 'accepted':
                    note['result'] = 'accepted'; note['n_after'] = int(n_new)
                    _accept(m, tmp, n_new, params_meta_for(m), note, apply, out_root)
                    n_changed += 1; changed_groups.add(gi)
            log(f"    [{_ts()}] {k}/{len(outs)} {m.name} [{i['cls']}] {m.n} -> {note['n_after']} matches "
                f"(+{note.get('n_new_ids', 0)} new, {note.get('n_from_master', 0)} from the group master list"
                f"{', agree %.2f px' % note['agree_px'] if 'agree_px' in note else ''}): {note['result']}")
            results[m.name] = note
            shutil.rmtree(tmp, ignore_errors=True)
        if not apply:
            break           # dry run: one round (results are not installed, so the next round would repeat)
        active = [active[g] for g in sorted(changed_groups)]
        if not active:
            break
    # zero-point-inconsistent solutions that could not be re-solved are demoted (files renamed, not deleted)
    for fs in todo:
        for f in fs:
            r = Path(f['root']); nm = r.name; note = results.get(nm)
            if not note or not note.get('zp_bad') or note.get('result') == 'accepted':
                continue
            note['demoted'] = True; n_demoted += 1
            log(f"    [{_ts()}] {nm}: zero point {note.get('zp'):+.2f} vs {note.get('zp_ref')} for its filter and no "
                f"sibling-consistent re-solution -> demoted to failed")
            if apply:
                for fn in ('matched_gaia.csv', 'transformation.csv'):
                    if (r / fn).exists():
                        (r / fn).rename(r / fn.replace('.csv', '_rejected_zp.csv'))
                st = _status(r)
                _write_status(r, {'status': 'failed', 'n_matched': 0, 'params': st.get('params'),
                                  'reason': f"zero point {note.get('zp'):+.2f} inconsistent with its filter "
                                            f"(spurious solution); visit-group completion could not re-solve it",
                                  'group_pass': note})
                n_changed += 1
    # record the final per-group signature in every member, so an unchanged group is skipped next time
    if apply:
        for fs in todo:
            sig = light_signature(fs, params_key)
            for f in fs:
                r = Path(f['root']); st = _status(r)
                if not st:
                    continue
                gp = dict(st.get('group_pass', {})); gp.update(results.get(r.name, {})); gp['sig'] = sig
                gp.setdefault('version', GROUP_PASS_VERSION)
                st['group_pass'] = gp
                (r / 'xmatch_status.json').write_text(json.dumps(_clean(st), indent=2, default=str))
    acc = sum(1 for v in results.values() if v.get('result') == 'accepted')
    rec_fail = sum(1 for v in results.values() if v.get('result') == 'accepted' and v.get('cls') == 'failed')
    rec_sus = sum(1 for v in results.values() if v.get('result') == 'accepted' and v.get('cls') == 'suspect')
    gain = sum(v.get('n_after', 0) - v.get('n_before', 0) for v in results.values() if v.get('result') == 'accepted')
    log(f"  [{_ts()}] visit-group completion done: {acc} image(s) improved ({rec_fail} failed recovered, "
        f"{rec_sus} suspect re-solved), {n_demoted} spurious solution(s) demoted, {gain:+d} matched stars in total")
    return n_changed
