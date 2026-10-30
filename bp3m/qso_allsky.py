"""All-sky Gaia DR3 QSO-candidate table for the field-mode conditioning samples (2026-10-29).

Per-field cone queries of gaiadr3.qso_candidates (and spatial upload joins with MILLIQUAS) do not scale: the archive dropped the
LMC (31 deg cone), 47 Tuc and COSMOS jobs.  Instead, ONE all-sky table is built with the production cuts (bp3m.gaia_modes cone
query: 5p/6p with PM and parallax, RUWE < 1.4, ipd_gof_harmonic_amplitude < 0.2, G < QSO_GMAX), in source_id-range chunks (an
index range scan per job, no spatial predicate), each chunk verified by COUNT(*); MILLIQUAS counterparts within MQ_MATCH_ARCSEC are
flagged locally (mq_match; prepare_qsos keeps exactly those).  Per field the QSO sample is then a local cone selection
(gaia_modes.download_qso_sample, mode 'allsky_table'), identical to the cone query by construction.
Stored in <bp3m_lib>/qso_catalogs/gaia_dr3_qso_candidates_cuts.parquet + .json (cuts, chunk counts, MILLIQUAS sha256, created).
Build: python -m bp3m.qso_allsky <bp3m_lib dir> [n_chunks]   (resumable; N_WORKERS parallel archive jobs)   (also built on demand by ensure_table)."""
from __future__ import annotations
import datetime, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

NAME = 'gaia_dr3_qso_candidates_cuts.parquet'
N_CHUNKS = 128
N_WORKERS = 4                                             # parallel archive jobs (one chunk ~10 min on 10-29 evening)
HPX12_MAX = 12 * 4 ** 12                                   # healpix level-12 pixels; source_id = hpx12 * 2^35 + ...
TABLE_VERSION = 1


def _sha(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def table_path(lib_dir) -> Path:
    return Path(lib_dir) / 'qso_catalogs' / NAME


def build(lib_dir, n_chunks: int = N_CHUNKS, gaia_tap_server=None, verbose: bool = True, workers: int = N_WORKERS) -> Path:
    from .gaia_modes import QSO_QUERY_COLS, QSO_GMAX, MQ_MATCH_ARCSEC
    from .gaia_sys import qso_prep as QP
    from .pipeline.download_gaia import _submit_gaia_async, find_qso_catalogs
    mq = find_qso_catalogs(Path(lib_dir)).get('milliquas')
    if mq is None:
        raise SystemExit(f'MILLIQUAS not found in {lib_dir}/qso_catalogs')
    out = table_path(lib_dir); part_dir = out.parent / (out.stem + '_parts'); part_dir.mkdir(parents=True, exist_ok=True)
    cuts = (f"g.astrometric_params_solved >= 31 AND g.pmra IS NOT NULL AND g.pmdec IS NOT NULL AND g.parallax IS NOT NULL "
            f"AND g.ruwe < 1.4 AND g.ipd_gof_harmonic_amplitude < 0.2 AND g.phot_g_mean_mag < {QSO_GMAX:.2f}")
    frm = 'FROM gaiadr3.qso_candidates AS q JOIN gaiadr3.gaia_source AS g ON g.source_id = q.source_id'   # range on the driving table
    edges = np.linspace(0, HPX12_MAX, n_chunks + 1).round().astype(np.int64)
    def one(k):
        pf = part_dir / f'part_{k:04d}.parquet'
        lo, hi = int(edges[k]) << 35, (int(edges[k + 1]) << 35) - 1
        if pf.exists():                                         # resumable: a part is written only after its count check
            return k, int(json.loads(Path(str(pf) + '.json').read_text())['n']), 0.0, True
        where = f'q.source_id BETWEEN {lo} AND {hi} AND {cuts}'
        t0 = time.time()
        d = _submit_gaia_async(f'SELECT {QSO_QUERY_COLS} {frm} WHERE {where}', gaia_tap_server)
        n = int(_submit_gaia_async(f'SELECT COUNT(*) AS n {frm} WHERE {where}', gaia_tap_server).iloc[0, 0])
        d.columns = [c.lower() for c in d.columns]
        if len(d) != n or d.source_id.duplicated().any():
            raise SystemExit(f'chunk {k}: {len(d)} rows vs COUNT(*) {n} (or duplicates)')
        d['source_id'] = d.source_id.astype(np.int64)
        tmp = pf.with_suffix('.tmp'); d.to_parquet(tmp, index=False); tmp.replace(pf)
        Path(str(pf) + '.json').write_text(json.dumps(dict(k=k, lo=lo, hi=hi, n=n)))
        return k, n, time.time() - t0, False
    def one_retry(k, tries=3):
        for t in range(tries):                                  # transient archive errors (404 on a dropped async result, 5xx)
            try:
                return one(k)
            except SystemExit:
                raise                                           # a count mismatch is a real error, never retried away
            except Exception as e:
                if t == tries - 1:
                    raise
                print(f'chunk {k}: attempt {t + 1} failed ({type(e).__name__}); retrying', flush=True); time.sleep(60 * (t + 1))
    from concurrent.futures import ThreadPoolExecutor
    counts = [None] * n_chunks; done = 0
    with ThreadPoolExecutor(workers) as ex:
        for k, n, dt, cached in ex.map(one_retry, range(n_chunks)):
            counts[k] = n; done += 1
            if verbose and not cached:
                print(f'[{time.strftime("%m-%d %H:%M:%S")}] [{done}/{n_chunks}] chunk {k}: {n} rows ({dt:.0f} s)', flush=True)
    T = pd.concat([pd.read_parquet(part_dir / f'part_{k:04d}.parquet') for k in range(n_chunks)], ignore_index=True)
    if len(T) != sum(counts) or T.source_id.duplicated().any():
        raise SystemExit(f'all-sky table: {len(T)} rows vs {sum(counts)} counted (or duplicates)')
    T['mq_match'] = QP.crossmatch_milliquas(T.ra.to_numpy(float), T.dec.to_numpy(float), milliquas_path=str(mq),
                                            match_radius_arcsec=MQ_MATCH_ARCSEC)
    tmp = out.with_suffix('.tmp'); T.to_parquet(tmp, index=False); tmp.replace(out)
    Path(str(out) + '.json').write_text(json.dumps(dict(
        table_version=TABLE_VERSION, cuts=cuts, from_=frm, columns=QSO_QUERY_COLS, n_chunks=n_chunks, chunk_counts=counts,
        n_rows=len(T), n_mq_match=int(T.mq_match.sum()), mq_match_arcsec=MQ_MATCH_ARCSEC, milliquas=Path(mq).name,
        sha256_milliquas=_sha(mq), sha256=_sha(out), row_count_verified=True,
        created=datetime.datetime.now().isoformat(timespec='seconds')), indent=1))
    if verbose:
        print(f'ALL-SKY QSO TABLE DONE: {len(T)} rows, {int(T.mq_match.sum())} MILLIQUAS-matched -> {out}', flush=True)
    return out


def ensure_table(lib_dir, gaia_tap_server=None) -> Path:
    """The verified all-sky table (built on first use; a table whose sidecar does not verify is an error, never used)."""
    from .gaia_modes import QSO_QUERY_COLS
    out = table_path(lib_dir); side = Path(str(out) + '.json')
    if out.exists() and side.exists():
        m = json.loads(side.read_text())
        if m.get('row_count_verified') and m.get('columns') == QSO_QUERY_COLS and m.get('table_version') == TABLE_VERSION and m.get('sha256') == _sha(out):
            return out
        raise SystemExit(f'{out}: sidecar does not verify (columns / version / sha256); rebuild with python -m bp3m.qso_allsky')
    return build(lib_dir, gaia_tap_server=gaia_tap_server)


if __name__ == '__main__':
    build(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else N_CHUNKS)
