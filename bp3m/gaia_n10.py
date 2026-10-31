"""All-source Gaia neighbour counts n10 for kernel-v3 calibrations with "n10_sources": "all" (real3 and later; handoff
GAIA_SYSTEMATICS_BP3M_HANDOFF.md section 7, items 1-2).

n10_all = number of OTHER gaiadr3.gaia_source entries (any astrometric solution, any G) within 10" of each source, counted
server-side by a TAP upload join: the definition, query, mirrors and checks of the pop_fit reference (pop_fit_dev/n10_all.py,
pop_fit_review/n10_all_r4/count_targets.py).  The count excludes the source itself, so the field table's own completeness
does not matter.  Mirrors in order: Heidelberg gaiadr3.gaia_source (the server of the calibration counts), GAVO gaia.dr3lite
(all DR3 sources, reduced columns), ESA gaiadr3.gaia_source.
Checks (hard errors, never a fallback): every source counted exactly once and no unknown source_id returned; CROSS-MIRROR: a
random subsample (500, or all) recounted on a mirror that did not produce its counts agrees exactly, except where the
difference is explained by neighbours within 1 mas of the 10" edge (the mirrors' CONTAINS differ at the sub-mas level).
Reported in the sidecar (not fatal): sources whose count is below their 5p/6p neighbour count in a local reference table
(great-circle separations).
Cache: <out_dir>/<stem>_n10all.parquet (source_id, n10_all, mirror) + .json sidecar (sha256 of the source set, mirrors,
checks); chunk parts in <out_dir>/<stem>_n10all_parts_<set key>/ (resumable).  The .parquet suffix and the separate directory
keep the counts out of every Gaia-CSV pattern (*_gaia.csv, *_gaiacorr_*.csv, *.csv*: the pop_fit glob bug of 10-31)."""
from __future__ import annotations
import datetime, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd

MIRRORS = {'heidelberg': ('https://gaia.ari.uni-heidelberg.de/tap', 'gaiadr3.gaia_source'),
           'gavo': ('https://dc.g-vo.org/tap', 'gaia.dr3lite'),
           'esa': ('https://gea.esac.esa.int/tap-server/tap', 'gaiadr3.gaia_source')}
RADIUS_ARCSEC = 10.0
CHUNK = 2_000          # synchronous queries only: Heidelberg's async upload joins hung on 10-31 (sync 2k-source joins take ~1 s)
NCHECK = 500
BOUNDARY_MAS = 1.0
DEFINITION = ('all other gaiadr3.gaia_source entries (any astrometric solution, any G) within 10 arcsec, server-side TAP '
              'upload-join count (bp3m.gaia_n10); QSO rows n10 = 0')
QUERY = """SELECT u.source_id, COUNT(*) AS n FROM TAP_UPLOAD.t AS u JOIN {tab} AS g
ON 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', u.ra, u.dec, {r:.10f}))
WHERE g.source_id <> u.source_id GROUP BY u.source_id"""


def _run(sid, ra, dec, mirrors, retries=2):
    """Counts for one chunk from the first mirror in `mirrors` that answers -> (counts, mirror).  Hard error if all fail."""
    import pyvo
    from astropy.table import Table
    tbl = Table({'source_id': sid, 'ra': ra, 'dec': dec})
    errors = []
    for m in mirrors:
        url, tab = MIRRORS[m]
        for k in range(retries):
            try:
                svc = pyvo.dal.TAPService(url)
                q = QUERY.format(tab=tab, r=RADIUS_ARCSEC / 3600.0)
                res = svc.run_sync(q, uploads={'t': tbl}, maxrec=10_000_000, timeout=600)
                if res.query_status != 'OK':          # truncated: absent sources would silently read as 0 neighbours
                    raise RuntimeError(f'result status {res.query_status}')
                r = res.to_table().to_pandas()
                r_id = r.source_id.to_numpy(np.int64)
                if not np.isin(r_id, sid).all() or len(np.unique(r_id)) != len(r_id):
                    raise RuntimeError('unknown or repeated source_ids returned')
                n = pd.Series(r.n.to_numpy(np.int64), index=r_id).reindex(sid).fillna(0).to_numpy(np.int64)
                return n, m                               # sources absent from the result have 0 neighbours
            except Exception as exc:                     # noqa: BLE001 - retry / next mirror, then hard error
                errors.append(f'{m} attempt {k + 1}: {str(exc)[:160]}')
                time.sleep(10 * (k + 1))
    raise RuntimeError('gaia_n10: all TAP mirrors failed: ' + ' | '.join(errors))


def _boundary_neighbours(sid, ra, dec):
    """Neighbours of one source with |separation - 10"| < BOUNDARY_MAS (Heidelberg, else GAVO)."""
    import pyvo
    from astropy.table import Table
    errors = []
    for m in ('heidelberg', 'gavo'):
        url, tab = MIRRORS[m]
        q = f"""SELECT DISTANCE(POINT('ICRS', g.ra, g.dec), POINT('ICRS', u.ra, u.dec)) * 3600 AS sep FROM TAP_UPLOAD.t AS u
JOIN {tab} AS g ON 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', u.ra, u.dec, {11 / 3600:.10f}))
WHERE g.source_id <> u.source_id"""
        try:
            r = pyvo.dal.TAPService(url).run_sync(q, uploads={'t': Table({'source_id': [sid], 'ra': [ra], 'dec': [dec]})},
                                                  timeout=300).to_table().to_pandas()
            return int((np.abs(r.sep.to_numpy(float) - RADIUS_ARCSEC) < BOUNDARY_MAS / 1e3).sum())
        except Exception as exc:                         # noqa: BLE001 - next mirror, then hard error
            errors.append(f'{m}: {str(exc)[:160]}')
    raise RuntimeError('gaia_n10 boundary check: all mirrors failed: ' + ' | '.join(errors))


def local_count(ra, dec, ref_ra, ref_dec, radius_arcsec=RADIUS_ARCSEC):
    """Neighbours within radius in a local reference table (great-circle, RA wrap safe); a source in the table counts itself,
    so callers pass sources that are in it and subtract 1, or sources that are not."""
    from scipy.spatial import cKDTree
    u = lambda a, d: np.column_stack([np.cos(np.radians(d)) * np.cos(np.radians(a)), np.cos(np.radians(d)) * np.sin(np.radians(a)),
                                      np.sin(np.radians(d))])
    t = cKDTree(u(np.asarray(ref_ra, float), np.asarray(ref_dec, float)))
    return np.array([len(x) for x in t.query_ball_point(u(np.asarray(ra, float), np.asarray(dec, float)),
                                                        2 * np.sin(np.radians(radius_arcsec / 3600) / 2))])


def set_key(sid) -> str:
    return hashlib.sha256(np.sort(np.asarray(sid, np.int64)).tobytes()).hexdigest()


def _sha(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def counts_path(out_dir, stem) -> Path:
    return Path(out_dir) / f'{stem}_n10all.parquet'


def ensure(out_dir, stem, sid, ra, dec, ref=None, verbose=True) -> tuple[pd.Series, Path]:
    """All-source n10 for the sources (sid, ra, dec), counted (or reused when the cache's source set and sidecar verify).
    ref = optional (ra, dec, is_self) local 5p/6p table for the reported lower-bound check.  Returns (Series by source_id, path)."""
    sid = np.asarray(sid, np.int64); ra = np.asarray(ra, float); dec = np.asarray(dec, float)
    if len(np.unique(sid)) != len(sid):
        raise SystemExit('gaia_n10: duplicate source_ids')
    if not (sid % 128 == 0).all():
        raise SystemExit('gaia_n10: source_id integrity (not int64-exact)')
    key = set_key(sid)
    out = counts_path(out_dir, stem); side = Path(str(out) + '.json')
    if out.exists() and side.exists():
        meta = json.loads(side.read_text())
        if meta.get('source_set_sha256') == key and meta.get('sha256') == _sha(out) and meta.get('radius_arcsec') == RADIUS_ARCSEC:
            C = pd.read_parquet(out)
            return C.set_index('source_id').n10_all.loc[sid], out
        if verbose:
            print(f'  n10 counts {out.name}: source set / file changed, recounting', flush=True)
    parts = Path(out_dir) / f'{stem}_n10all_parts_{key[:12]}'
    parts.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(0, len(sid), CHUNK):
        p = parts / f'part_{i // CHUNK:04d}.parquet'
        if not p.exists():
            sl = slice(i, i + CHUNK)
            n, m = _run(sid[sl], ra[sl], dec[sl], list(MIRRORS))
            pd.DataFrame({'source_id': sid[sl], 'n10_all': n, 'mirror': m}).to_parquet(p)
            if verbose and (i // CHUNK) % 25 == 0:
                print(f'    n10_all: {min(i + CHUNK, len(sid)):,}/{len(sid):,} sources ({m}, {time.time() - t0:.0f} s)', flush=True)
    C = pd.concat([pd.read_parquet(parts / f'part_{i // CHUNK:04d}.parquet') for i in range(0, len(sid), CHUNK)], ignore_index=True)
    if len(C) != len(sid) or not (C.source_id.to_numpy(np.int64) == sid).all():
        raise SystemExit(f'gaia_n10 {stem}: counts incomplete or out of order ({len(C)} vs {len(sid)}); delete {parts}')
    # cross-mirror check on a subsample, on mirrors that did not produce those counts
    rng = np.random.default_rng(int(sid[0] % 2**32)); k = np.sort(rng.choice(len(sid), min(NCHECK, len(sid)), replace=False))
    others = [m for m in MIRRORS if m not in set(C.mirror.iloc[k])]
    if not others:
        raise SystemExit(f'gaia_n10 {stem}: no independent mirror left for the cross-check')
    n_chk, m_chk = _run(sid[k], ra[k], dec[k], others)
    n0 = C.n10_all.to_numpy(np.int64)
    diff = np.where(n_chk != n0[k])[0]
    for j in diff:
        nb = _boundary_neighbours(int(sid[k][j]), float(ra[k][j]), float(dec[k][j]))
        if abs(int(n_chk[j]) - int(n0[k[j]])) > nb:
            raise SystemExit(f'gaia_n10 {stem}: cross-mirror check FAILED for {sid[k][j]} ({m_chk} {n_chk[j]} vs {n0[k[j]]}; '
                             f'{nb} neighbours within {BOUNDARY_MAS} mas of 10")')
    below = None
    if ref is not None:
        rra, rde, self_in = ref
        lo = local_count(ra, dec, rra, rde) - np.asarray(self_in, int)
        below = int((n0 < lo).sum())
    C[['source_id', 'n10_all', 'mirror']].to_parquet(out)
    side.write_text(json.dumps(dict(
        definition=DEFINITION, radius_arcsec=RADIUS_ARCSEC, n_sources=len(sid), source_set_sha256=key, sha256=_sha(out),
        mirrors=sorted(set(C.mirror)), cross_check=dict(n=len(k), mirror=m_chk, n_boundary_explained=len(diff)),
        n_below_local_5p6p=below, mean_n10_all=float(n0.mean()), query=QUERY, chunk=CHUNK,
        created=datetime.datetime.now().isoformat(timespec='seconds')), indent=1))
    if verbose:
        print(f'  n10 counts {out.name}: {len(sid):,} sources via {sorted(set(C.mirror))}; cross-check {len(k)} on {m_chk} '
              f'identical{f" except {len(diff)} boundary case(s)" if len(diff) else ""}; mean n10 {n0.mean():.2f}'
              f'{"" if below is None else f"; below the local 5p/6p count: {below}"}', flush=True)
    return C.set_index('source_id').n10_all.loc[sid], out
