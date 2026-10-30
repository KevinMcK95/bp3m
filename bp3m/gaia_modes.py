"""BP3M-native Gaia field-mode files (handoff GAIA_SYSTEMATICS_BP3M_HANDOFF.md 3.2 / 3.2a / 3.4): the kernel-v3 (real2p)
smooth field term f ~ GP(0, B_mid / (1 + theta/theta_mid) + B_large exp(-theta/theta_large)) with per-source amplitudes
a(G, 5p/6p), conditioned on the target's QSOs, projected to every Gaia source of the field:
    v_meas,i = v_true,i + M_i + W_i z + e_i,   z ~ N(0, I_K),   e_i ~ N(0, C_star,i + R_i)     (pmra, pmdec, parallax)
M_i = W3_i m_Z (posterior mean), W = leading K modes of A = W3 L_Z (>= 99.9 % of trace(A A^T) over the field's sources),
R_i = FITC remainder + the per-source part of the dropped modes.  The short-range block is per-star variance (already in the
v3real2p covariance) and never a mode.  bp3m.gaia_model.FieldModes reads the file (floor=modes;modes=<npz>).

QSO conditioning sample = the pop_fit production r3b recipe: Gaia DR3 qso_candidates within 5 deg of the field centre, 5p/6p,
RUWE < 1.4, ipd_gof_harmonic_amplitude < 0.2, G < 21; L21 zero point; qso_sys subtracted; MILLIQUAS match (1.5"); two-stage
cleaning (parallax 3 sigma, iterative PM 3 sigma clip; pop_fit measure_pm_systematic); residual = (pmra, pmdec) - secular
aberration drift, parallax; raw published covariances (the v3 QSO noise model, n10 = 0, is applied inside build_gp_v3); the
1500 QSOs nearest the centre.  NO DOUBLE COUNTING: every QSO in the model's anchors file(s) and every QSO that is a source of
the field Gaia table(s) is removed from the conditioning and listed in qso_excluded (BP3M checks its injected anchors against
it).  Geometry: centre = field box centre; extent = max(0.3 deg, box half-diagonal) (the field's sources must lie inside the
inducing grid; pop_fit used clip(3 r_half, 0.3, cone)); n_rings = max(3, ceil(extent / 0.13 deg)) so the inducing spacing stays
below theta_mid / 2.  Stars are projected in chunks (memory)."""
from __future__ import annotations
import datetime, json, math
from pathlib import Path
import numpy as np
import pandas as pd

from .gaia_sys import gaia_v3 as GV3, qso_prep as QP
from .gaia_corrected import KERNEL, QSOSYS, SHA256, check_inputs, _sha, OUT_DIR

QSO_RADIUS_DEG, QSO_GMAX, MAX_QSO, TRACE_FRAC = 5.0, 21.0, 1500, 0.999
SPACING_DEG, KEEP_DEG, QSO_NEAR_DEG, MAX_QSO_TOTAL = 0.13, 0.26, 5.0, 3000   # inducing lattice / its reach; conditioning QSOs
FOOTPRINT_MARGIN_DEG = 0.5   # footprints with no vertex within the field Gaia extent + this are dropped (stray images)
MQ_MATCH_ARCSEC, MQ_CHUNK = 1.5, 2000   # match radius = qso_prep.crossmatch_milliquas default (prepare_qsos); MILLIQUAS positions per uploaded TAP job
MAX_3NZ, MAX_CHUNK_BYTES = 9000, 4e8                                         # memory guards (grid size; per-chunk arrays)
QSO_QUERY_COLS = ('g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.pmra_error, g.pmdec_error, g.pmra_pmdec_corr, g.parallax, '
                  'g.parallax_error, g.parallax_pmra_corr, g.parallax_pmdec_corr, g.phot_g_mean_mag, g.phot_g_mean_flux, '
                  'g.phot_g_mean_flux_error, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.ruwe, g.nu_eff_used_in_astrometry, '
                  'g.pseudocolour, g.ecl_lat, g.astrometric_params_solved')
MODES_VERSION = 3       # 2: sources inside the image footprints + padding; 3: pruned grid, QSOs near the sources (10-29)


def qso_sample_path(gaia_dir, stem, radius_deg=QSO_RADIUS_DEG):
    return Path(gaia_dir) / f'{stem}_qsocond_r{radius_deg:.1f}_G{QSO_GMAX:.1f}.csv'


def download_qso_sample(gaia_dir, stem, ra, dec, gaia_tap_server=None, radius_deg=QSO_RADIUS_DEG, milliquas_path=None, qso_table=None) -> Path:
    """Gaia DR3 QSO candidates (pop_fit production cuts) within radius_deg; cached with a verified sidecar.
    With milliquas_path (10-29, mosaics): MILLIQUAS-FIRST - the MILLIQUAS objects in the cone are uploaded and only Gaia QSO candidates
    within MQ_MATCH_ARCSEC of one are returned.  prepare_qsos keeps exactly those (it cross-matches to MILLIQUAS at the same radius),
    so the conditioning sample is identical while the query no longer scans every candidate of a large cone (LMC / 47 Tuc / COSMOS
    cones failed: the archive dropped the async result after > 1 h).  Without it: the original cone query.
    With qso_table (bp3m.qso_allsky, 10-29; preferred): a LOCAL cone selection from the verified all-sky table built with the same
    cuts - no archive call per field (the upload join also failed for the 31 deg LMC cone).  Identical rows to the cone query."""
    from .pipeline.download_gaia import _submit_gaia_async
    mode = 'allsky_table' if qso_table is not None else ('milliquas_upload' if milliquas_path is not None else 'cone')
    out = qso_sample_path(gaia_dir, stem, radius_deg)
    if mode != 'cone':
        out = out.with_name(out.stem + ('_tab.csv' if mode == 'allsky_table' else '_mq.csv'))
    side = Path(str(out) + '.json')
    if out.exists() and side.exists():
        m = json.loads(side.read_text())
        if m.get('sha256') == _sha(out) and m.get('row_count_verified') and m.get('columns') == QSO_QUERY_COLS and m.get('mode', 'cone') == mode:
            return out
    cuts = (f"g.astrometric_params_solved >= 31 AND g.pmra IS NOT NULL AND g.pmdec IS NOT NULL AND g.parallax IS NOT NULL "
            f"AND g.ruwe < 1.4 AND g.ipd_gof_harmonic_amplitude < 0.2 AND g.phot_g_mean_mag < {QSO_GMAX:.2f}")
    cone = f"CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_deg:.6f})) = 1"
    meta = {}
    if mode == 'allsky_table':
        tside = json.loads(Path(str(qso_table) + '.json').read_text())
        if tside.get('columns') != QSO_QUERY_COLS or not tside.get('row_count_verified'):
            raise SystemExit(f'{qso_table}: not built with the current QSO columns / unverified')
        T = pd.read_parquet(qso_table)
        c = _unit([ra], [dec])[0]
        df = T[np.degrees(np.arccos(np.clip(_unit(T.ra.to_numpy(float), T.dec.to_numpy(float)) @ c, -1, 1))) < radius_deg]
        df = df.drop(columns=['mq_match']).reset_index(drop=True); n = len(df)
        where = f'{cone} AND {cuts} (local selection from {Path(qso_table).name})'
        meta = dict(qso_table=Path(qso_table).name, sha256_qso_table=tside.get('sha256'))
    elif mode == 'cone':
        where = f'{cone} AND {cuts}'
        frm = 'FROM gaiadr3.gaia_source AS g JOIN gaiadr3.qso_candidates AS q ON q.source_id = g.source_id'
        df = _submit_gaia_async(f'SELECT {QSO_QUERY_COLS} {frm} WHERE {where}', gaia_tap_server)
        n = int(_submit_gaia_async(f'SELECT COUNT(*) AS n {frm} WHERE {where}', gaia_tap_server).iloc[0, 0])
        df.columns = [c.lower() for c in df.columns]
    else:
        from astropy.io import fits
        from astropy.table import Table
        if not Path(milliquas_path).exists():
            raise SystemExit(f'MILLIQUAS catalogue not found ({milliquas_path})')
        with fits.open(milliquas_path) as h:
            mra = h[1].data['RA'].astype(np.float64); mde = h[1].data['DEC'].astype(np.float64)
        c = _unit([ra], [dec])[0]; pad = MQ_MATCH_ARCSEC / 3600.0
        sel = np.degrees(np.arccos(np.clip(_unit(mra, mde) @ c, -1, 1))) < radius_deg + pad
        # chunks of MQ_CHUNK uploaded positions; the cone cut is applied client-side (10-29: a 31 deg CONTAINS cone inside the join made
        # the LMC job fail server-side even with only 6,264 uploaded positions); each chunk's rows verified by COUNT(DISTINCT)
        ids = np.nonzero(sel)[0]; parts, n = [], 0
        frm = (f"FROM tap_upload.mq AS u JOIN gaiadr3.gaia_source AS g ON 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), "
               f"CIRCLE('ICRS', u.ra, u.dec, {MQ_MATCH_ARCSEC / 3600.0:.8f})) JOIN gaiadr3.qso_candidates AS q ON q.source_id = g.source_id")
        where = cuts
        for k in range(0, len(ids), MQ_CHUNK):
            ch = ids[k:k + MQ_CHUNK]; up = Path(str(out) + f'.upload{k // MQ_CHUNK}.xml')
            Table(dict(mq_id=ch.astype(np.int64), ra=mra[ch], dec=mde[ch])).write(up, format='votable', overwrite=True)
            d_ = _submit_gaia_async(f'SELECT DISTINCT {QSO_QUERY_COLS} {frm} WHERE {where}', gaia_tap_server, upload_resource=str(up), upload_table_name='mq')
            n_ = int(_submit_gaia_async(f'SELECT COUNT(DISTINCT g.source_id) AS n {frm} WHERE {where}', gaia_tap_server,
                                        upload_resource=str(up), upload_table_name='mq').iloc[0, 0])
            if len(d_) != n_: raise SystemExit(f'QSO sample chunk {k // MQ_CHUNK}: {len(d_)} rows vs COUNT(DISTINCT) {n_}')
            d_.columns = [c_.lower() for c_ in d_.columns]; parts.append(d_); up.unlink()
        df = pd.concat(parts, ignore_index=True).drop_duplicates('source_id') if parts else pd.DataFrame(columns=[c_.strip().split('.')[-1] for c_ in QSO_QUERY_COLS.split(',')])
        incone = np.degrees(np.arccos(np.clip(_unit(df.ra.to_numpy(float), df.dec.to_numpy(float)) @ c, -1, 1))) < radius_deg
        df = df[incone].reset_index(drop=True); n = len(df)                     # per-chunk counts verified above; the cut is local
        where = f'{cone} AND {cuts} (cone applied client-side)'
        meta = dict(n_milliquas_uploaded=int(sel.sum()), match_arcsec=MQ_MATCH_ARCSEC, chunk=MQ_CHUNK)
    if len(df) != n or df.source_id.duplicated().any():
        raise SystemExit(f'QSO sample: {len(df)} rows vs COUNT(*) {n} (or duplicates)')
    df['source_id'] = df.source_id.astype(np.int64)
    tmp = Path(str(out) + '.tmp'); df.to_csv(tmp, index=False); tmp.replace(out)
    side.write_text(json.dumps({'ra': ra, 'dec': dec, 'radius_deg': radius_deg, 'gmax': QSO_GMAX, 'where': where, 'mode': mode, **meta,
                                'columns': QSO_QUERY_COLS, 'n_rows': len(df), 'row_count_verified': True, 'sha256': _sha(out),
                                'created': datetime.datetime.now().isoformat(timespec='seconds')}, indent=1))
    return out


def prepare_qsos(qso_csv, milliquas_path, exclude_ids, center):
    """Cleaned conditioning QSOs: (ra, dec, resid (M,3), S (M,3,3), G, sol, ids, stats)."""
    from .gaia_zp import compute_zp
    from .gaia_sys.qso_sys import correction
    from .pipeline.secular_aberration import secular_aberration_pm
    if milliquas_path is None or not Path(milliquas_path).exists():
        raise SystemExit(f'MILLIQUAS catalogue not found ({milliquas_path}): required for the QSO conditioning sample')
    q = pd.read_csv(qso_csv, dtype={'source_id': 'int64'})
    q = q.rename(columns={'phot_g_mean_mag': 'gmag'})
    zp, _ = compute_zp(q); q['parallax'] = q['parallax'] - np.nan_to_num(zp, nan=0.0)
    D = correction(str(QSOSYS), q.ra.values, q.dec.values, q.gmag.values, q.astrometric_params_solved.values, q.pseudocolour.values)
    for j, c in enumerate(('pmra', 'pmdec', 'parallax')):
        q[c] = q[c].to_numpy(float) - D[:, j]
    n_raw = len(q)
    q = q[QP.crossmatch_milliquas(q.ra.to_numpy(), q.dec.to_numpy(), milliquas_path=str(milliquas_path))].reset_index(drop=True)
    n_mq = len(q)
    res = QP.measure_pm_systematic(q)
    q = q[res['mask_kept']].reset_index(drop=True)
    n_clean = len(q)
    ex = q.source_id.isin(list(exclude_ids)).to_numpy()
    excluded_in_sample = q.source_id[ex].astype(np.int64).tolist()
    q = q[~ex].reset_index(drop=True)
    S = np.zeros((len(q), 3, 3)); e = q[['pmra_error', 'pmdec_error', 'parallax_error']].to_numpy(float)
    S[:, range(3), range(3)] = e**2
    for (i, j), c in (((0, 1), 'pmra_pmdec_corr'), ((0, 2), 'parallax_pmra_corr'), ((1, 2), 'parallax_pmdec_corr')):
        S[:, i, j] = S[:, j, i] = np.nan_to_num(q[c].to_numpy(float)) * e[:, i] * e[:, j]
    ab = np.column_stack(secular_aberration_pm(q.ra.to_numpy(float), q.dec.to_numpy(float))) / 1e3        # uas/yr -> mas/yr
    resid = np.column_stack([q.pmra - ab[:, 0], q.pmdec - ab[:, 1], q.parallax])
    sol = q.astrometric_params_solved.to_numpy()
    stats = dict(n_candidates=n_raw, n_milliquas=n_mq, n_clean=n_clean, n_excluded_in_sample=int(ex.sum()), n_conditioning=len(q))
    return q.ra.to_numpy(float), q.dec.to_numpy(float), resid, S, q.gmag.to_numpy(float), sol, excluded_in_sample, stats


def image_footprints(image_files, cache_path=None):
    """Sky polygons (deg) of every science extension of the given HST (_fl?) / JWST (_cal) images, from the header WCS.  Used
    ONLY to select which Gaia sources get a field term (the padding absorbs WCS pointing errors); never for positions."""
    from astropy.io import fits
    from astropy.wcs import WCS
    import warnings
    import os
    cache = {}
    if cache_path is not None and Path(cache_path).exists():
        try:
            cache = json.loads(Path(cache_path).read_text())
        except Exception:
            cache = {}
    polys, new = [], 0
    drop = ('CPDIS', 'DP1', 'DP2', 'D2IM', 'CPERR', 'NPOLEXT', 'D2IMEXT', 'D2IMERR', 'WCSDVARR')
    for f in image_files:
        st = os.stat(f); key = Path(f).name; tag = [st.st_size, int(st.st_mtime)]
        hit = cache.get(key)
        if hit is not None and hit.get('tag') == tag:                      # 10-29: one header read per image, ever (mosaics:
            polys += [np.asarray(q, float) for q in hit['polys']]          # 5,000-14,000 images on a loaded RAID, ~0.35 s each)
            continue
        mine = []
        with fits.open(f, memmap=True, lazy_load_hdus=True) as h:
            for e in h:                                  # headers only: the lookup-table distortions (read from the data) are
                hd = e.header                            # irrelevant for a selection footprint and cost a minute per field
                if hd.get('EXTNAME', '') == 'SCI' and hd.get('NAXIS', 0) == 2:
                    hc = hd.copy()
                    for k in [k for k in hc.keys() if k.startswith(drop)]:
                        del hc[k]
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        mine.append(WCS(hc).calc_footprint(axes=(hd['NAXIS1'], hd['NAXIS2'])))
        polys += mine; new += 1
        cache[key] = {'tag': tag, 'polys': [q.tolist() for q in mine]}
    if cache_path is not None and new:
        tmp = Path(str(cache_path) + '.tmp'); tmp.write_text(json.dumps(cache)); tmp.replace(cache_path)
    if not polys:
        raise SystemExit('field modes: no science-extension footprints in the image list')
    return polys


def in_footprints(ra, dec, polys, pad_arcsec):
    """True for sources inside any footprint polygon or within pad_arcsec of its edges (tangent plane per polygon).  Sources are
    swept in declination so each polygon only tests its own band (47 Tuc: 8,800 polygons x 3 M Gaia rows)."""
    ra = np.asarray(ra, float); dec = np.asarray(dec, float); keep = np.zeros(len(ra), bool); pad = pad_arcsec / 3600.0
    order = np.argsort(dec); ds = dec[order]
    for P in polys:
        i0, i1 = np.searchsorted(ds, P[:, 1].min() - pad - 1e-6), np.searchsorted(ds, P[:, 1].max() + pad + 1e-6, 'right')
        if i1 <= i0:
            continue
        idx = order[i0:i1]; idx = idx[~keep[idx]]
        c0, d0 = np.mean(P[:, 0]), np.mean(P[:, 1]); cd = np.cos(np.radians(d0))
        px = ((P[:, 0] - c0 + 180) % 360 - 180) * cd; py = P[:, 1] - d0
        xs = ((ra[idx] - c0 + 180) % 360 - 180) * cd; ys = dec[idx] - d0
        m = (xs > px.min() - pad - 1e-6) & (xs < px.max() + pad + 1e-6)
        idx, xs, ys = idx[m], xs[m], ys[m]
        if not len(idx):
            continue
        inside = np.zeros(len(xs), bool); dmin = np.full(len(xs), np.inf)
        for k in range(len(px)):
            x1, y1, x2, y2 = px[k], py[k], px[(k + 1) % len(px)], py[(k + 1) % len(px)]
            inside ^= ((y1 > ys) != (y2 > ys)) & (xs < (x2 - x1) * (ys - y1) / (y2 - y1 + 1e-300) + x1)
            t = np.clip(((xs - x1) * (x2 - x1) + (ys - y1) * (y2 - y1)) / max((x2 - x1) ** 2 + (y2 - y1) ** 2, 1e-300), 0, 1)
            dmin = np.minimum(dmin, np.hypot(xs - (x1 + t * (x2 - x1)), ys - (y1 + t * (y2 - y1))))
        keep[idx[inside | (dmin <= pad)]] = True
    return keep


def _unit(ra, dec):
    r, d = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


def pruned_grid(ra, dec, center, spacing=SPACING_DEG, keep=KEEP_DEG):
    """Hexagonal inducing lattice (spacing deg) kept only within `keep` deg of a selected source: the cost follows the covered
    area, not the bounding box of a mosaic.  RA spacing uses each row's own cos(dec)."""
    from scipy.spatial import cKDTree
    c0, d0 = float(center[0]), float(center[1])
    y = np.asarray(dec, float) - d0; x = ((np.asarray(ra, float) - c0 + 180) % 360 - 180) * np.cos(np.radians(np.asarray(dec, float)))
    rows = np.arange(y.min() - keep, y.max() + keep + 1e-9, spacing * np.sqrt(3) / 2)
    zr, zd = [], []
    for j, yy in enumerate(rows):
        xs = np.arange(x.min() - keep, x.max() + keep + 1e-9, spacing) + (spacing / 2 if j % 2 else 0.0)
        dd = d0 + yy; zr.append(c0 + xs / max(np.cos(np.radians(dd)), 1e-3)); zd.append(np.full(len(xs), dd))
    zr, zd = np.concatenate(zr) % 360, np.concatenate(zd)
    ok = np.abs(zd) < 89.9
    zr, zd = zr[ok], zd[ok]
    dist, _ = cKDTree(_unit(ra, dec)).query(_unit(zr, zd), distance_upper_bound=2 * np.sin(np.radians(keep) / 2))
    m = np.isfinite(dist)
    return zr[m], zd[m]


def _project(cal, zra, zde, LZZ, ra, dec, G, sol):
    """W3 (n, 3, 3nz) and the FITC remainder (n, 3, 3) of build_gp_v3's projection for any stars, given its K_ZZ Cholesky (the
    conditioning is done once; asserted equal to build_gp_v3 on the first chunk)."""
    uZ = _unit(zra, zde); us = _unit(ra, dec); N = len(us); nz = len(uZ)
    a = GV3.amplitude(cal, G, sol)
    KxZ = GV3.smooth_cov(cal, us, uZ).reshape(3 * N, 3 * nz)
    Wu = np.linalg.solve(LZZ.T, np.linalg.solve(LZZ, KxZ.T)).T
    Kxx = np.repeat((cal['B_mid_pop'] + cal['B_large_pop'])[None], N, 0)
    Q = np.einsum('nia,nja->nij', KxZ.reshape(N, 3, 3 * nz), Wu.reshape(N, 3, 3 * nz))
    return Wu.reshape(N, 3, 3 * nz) * a[:, :, None], (Kxx - Q) * a[:, :, None] * a[:, None, :]


def build_modes(field_csvs, anchors_csvs, qso_csv, milliquas_path, center, half_diag_deg, out_npz, model='v3real2p', footprints_dropped=0,
                verbose=True, footprints=None, pad_arcsec=None, chunk=None) -> Path:
    """Field-mode npz for the Gaia sources (with a PM) of the field table(s), restricted to the image footprints when given."""
    import time
    from scipy.spatial import cKDTree
    t0 = time.time()
    check_inputs()
    cal = GV3.load(str(KERNEL))
    G = pd.concat([pd.read_csv(f, dtype={'source_id': 'int64'}, usecols=['source_id', 'ra', 'dec', 'gmag', 'pmra', 'pseudocolour',
                                                                        'astrometric_params_solved']) for f in field_csvs])
    G = G.drop_duplicates('source_id')
    G = G[G.pmra.notna() & G.astrometric_params_solved.isin([31, 95])].reset_index(drop=True)
    n_table = len(G)
    if footprints is not None:
        # 10-29 (user): only Gaia sources inside the images (+ padding), not the whole Gaia box: every star the solver can see
        # (incl. saturated / unmatched ones), lightweight files; the inducing grid follows the selected sources
        G = G[in_footprints(G.ra.values, G.dec.values, footprints, pad_arcsec)].reset_index(drop=True)
        if len(G) == 0:
            raise SystemExit('field modes: no Gaia source inside the image footprints')
        center = np.array([G.ra.mean() + ((G.ra - G.ra.mean() + 180) % 360 - 180).mean(), G.dec.mean()])   # wrap-safe mean
    sol = G.astrometric_params_solved.to_numpy(); Gm = G.gmag.to_numpy(float)
    ra, dec = G.ra.to_numpy(float), G.dec.to_numpy(float); N = len(G)
    anc_ids = set()
    for a in anchors_csvs:
        anc_ids |= set(pd.read_csv(a, usecols=['source_id'], dtype={'source_id': 'int64'}).source_id.astype(int))
    q_ra, q_dec, q_res, q_S, q_G, q_sol, exc_in_sample, qst = prepare_qsos(qso_csv, milliquas_path, anc_ids | set(G.source_id.astype(int)), center)
    # conditioning QSOs: within QSO_NEAR_DEG of a selected source, the MAX_QSO_TOTAL nearest (each field patch keeps its own)
    dq, _ = cKDTree(_unit(ra, dec)).query(_unit(q_ra, q_dec))
    dq = np.degrees(2 * np.arcsin(np.clip(dq / 2, 0, 1)))
    sel = np.where(dq <= QSO_NEAR_DEG)[0]; sel = sel[np.argsort(dq[sel])][:MAX_QSO_TOTAL]
    q_ra, q_dec, q_res, q_S, q_G, q_sol = q_ra[sel], q_dec[sel], q_res[sel], q_S[sel], q_G[sel], q_sol[sel]
    qst.update(n_near=int((dq <= QSO_NEAR_DEG).sum()), n_used=len(sel))
    if len(sel) == 0:
        raise SystemExit(f'field modes: no conditioning QSO within {QSO_NEAR_DEG} deg of the field')
    sp = SPACING_DEG
    while True:                                                     # inducing grid; spacing widened only if the grid is too big
        zra, zde = pruned_grid(ra, dec, center, spacing=sp)
        if 3 * len(zra) <= MAX_3NZ:
            break
        sp *= 1.25
    if len(zra) < 7:
        zra, zde = QP.inducing_points(center[0], center[1], 0.3, n_rings=3)
    nz3 = 3 * len(zra)
    chunk = int(chunk or max(200, min(20000, MAX_CHUNK_BYTES // (nz3 * 3 * 8 * 4))))
    gp0 = GV3.build_gp_v3(cal, center, 0.3, ra[:chunk], dec[:chunk], Gm[:chunk], sol[:chunk], q_ra, q_dec, q_res, q_S, q_G, q_sol,
                          max_qso=len(q_ra) + 1, z_points=(zra, zde))
    W3c, fc = _project(cal, zra, zde, gp0['LZZ'], ra[:chunk], dec[:chunk], Gm[:chunk], sol[:chunk])
    if not (np.allclose(W3c, gp0['W3'], rtol=1e-9, atol=1e-15) and np.allclose(fc, gp0['fitc'], rtol=1e-9, atol=1e-15)):
        raise SystemExit('field modes: chunk projection differs from build_gp_v3')
    t_cond = time.time() - t0
    # pass 1: accumulate A^T A (A = W3 L_Z), M and the FITC part of R chunk by chunk (projection only; conditioning done once)
    LZ = gp0['L_Z']; AtA = np.zeros((nz3, nz3)); M = np.zeros((N, 3)); R = np.zeros((N, 3, 3))
    for s in range(0, N, chunk):
        sl = slice(s, min(N, s + chunk))
        W3, f = (gp0['W3'], gp0['fitc']) if s == 0 else _project(cal, zra, zde, gp0['LZZ'], ra[sl], dec[sl], Gm[sl], sol[sl])
        A = (W3 @ LZ).reshape(-1, nz3); AtA += A.T @ A; M[sl] = W3 @ gp0['m_Z']; R[sl] = f
    w, V = np.linalg.eigh(AtA); o = np.argsort(w)[::-1]; w, V = np.clip(w[o], 0, None), V[:, o]
    K = int(np.searchsorted(np.cumsum(w) / w.sum(), TRACE_FRAC) + 1)
    W = np.zeros((N, 3, K))
    for s in range(0, N, chunk):                                     # pass 2: the K loadings + dropped modes' per-source part
        sl = slice(s, min(N, s + chunk))
        W3, _ = (gp0['W3'], None) if s == 0 else _project(cal, zra, zde, gp0['LZZ'], ra[sl], dec[sl], Gm[sl], sol[sl])
        Wf = (W3 @ LZ) @ V
        W[sl] = Wf[:, :, :K]; R[sl] += np.einsum('nik,njk->nij', Wf[:, :, K:], Wf[:, :, K:])
    out_npz = Path(out_npz); out_npz.parent.mkdir(parents=True, exist_ok=True)
    qso_excluded = np.array(sorted(anc_ids | (set(exc_in_sample))), dtype=np.int64)
    meta = dict(model=model, modes_version=MODES_VERSION, calibration=KERNEL.name, sha256_calibration=SHA256[KERNEL.name],
                center=list(map(float, center)), inducing_spacing_deg=sp, inducing_keep_deg=KEEP_DEG, n_inducing=len(zra), K=K,
                trace_fraction=float(np.cumsum(w)[K - 1] / w.sum()), n_sources=N, n_table_sources=n_table, chunk=chunk,
                footprint_pad_arcsec=pad_arcsec, n_footprints=None if footprints is None else len(footprints), footprints_dropped=int(footprints_dropped), qso=qst,
                qso_near_deg=QSO_NEAR_DEG, n_qso_used=int(gp0['n_qso']),
                post_mean_center=[float(x) for x in gp0['post_mean_center']], post_sd_center=[float(x) for x in gp0['post_sd_center']],
                field_csvs=[Path(f).name for f in field_csvs], sha256_field_csvs=[_sha(f) for f in field_csvs],
                anchors=[Path(a).name for a in anchors_csvs], qso_sample=Path(qso_csv).name, sha256_qso_sample=_sha(qso_csv),
                milliquas_sha256=_sha(milliquas_path), code={p.name: _sha(p) for p in (Path(__file__), Path(GV3.__file__), Path(QP.__file__))},
                seconds_conditioning=round(t_cond, 1), seconds_total=round(time.time() - t0, 1),
                created=datetime.datetime.now().isoformat(timespec='seconds'))
    build_modes.last_inputs = dict(center=center, z_points=(zra, zde), q=(q_ra, q_dec, q_res, q_S, q_G, q_sol))   # tests
    np.savez(out_npz, source_id=G.source_id.to_numpy(np.int64), M=M, W=W, R=R, qso_excluded=qso_excluded, model=np.array(model),
             z_ra=zra, z_dec=zde,
             calibration=np.array(KERNEL.name), sha256_calibration=np.array(SHA256[KERNEL.name]), created=np.array(meta['created']),
             meta=np.array(json.dumps(meta)))
    Path(str(out_npz) + '.json').write_text(json.dumps(meta, indent=1))
    if verbose:
        print(f'  field modes {out_npz.name}: N {N:,} (of {n_table:,}); QSOs {qst}; {len(zra)} inducing points (spacing {sp:.3f} deg); '
              f'K = {K} ({meta["trace_fraction"]:.4f} of the trace); {meta["seconds_total"]:.0f} s; centre field '
              f'{np.round(1e3 * gp0["post_mean_center"], 2)} +- {np.round(1e3 * gp0["post_sd_center"], 2)} uas(/yr)', flush=True)
    return out_npz


def ensure_modes(field_csvs, model_copies, centre, half_diag_deg, model, lib_dir=None, gaia_tap_server=None, image_files=None,
                 pad_arcsec=20.0) -> Path:
    """The field-mode npz for this field and model (built or reused: inputs sha256, code, MODES_VERSION must match), next to
    the model copies in Gaia/corrected_bp3m/.  Anchors = the model copies' own anchors files (all their ids are excluded)."""
    from .gaia_model import _sidecar
    from .pipeline.download_gaia import find_qso_catalogs
    field_csvs = [Path(f) for f in field_csvs]; model_copies = [Path(p) for p in model_copies]
    gdir = field_csvs[0].parent
    stem = field_csvs[0].name[:-len('_G0.0_gaia.csv')] if field_csvs[0].name.endswith('_G0.0_gaia.csv') else field_csvs[0].stem
    anc = []
    for p in model_copies:
        hits = [a for a in sorted(p.parent.glob(f'*_qso_anchors_{model}.csv')) if _sidecar(a).get('source_csv') == _sidecar(p).get('source_csv')]
        if len(hits) != 1:
            raise SystemExit(f'{p.name}: expected one anchors file for model {model}, found {hits}')
        anc += hits
    mq = find_qso_catalogs(Path(lib_dir) if lib_dir else None).get('milliquas')
    polys = image_footprints(image_files, cache_path=gdir / f'{stem}_image_footprints.json') if image_files else None
    n_drop = 0
    if polys:      # 10-29: stray images (bad header WCS / other targets: 47 Tuc and M31 folders hold images 80-153 deg away) cannot hold
        c0 = _unit([centre[0]], [centre[1]])[0]                     # field sources but inflated the QSO cone to the whole sky
        near = [p for p in polys if np.degrees(np.arccos(np.clip(_unit(p[:, 0], p[:, 1]) @ c0, -1, 1))).min() <= half_diag_deg + FOOTPRINT_MARGIN_DEG]
        n_drop = len(polys) - len(near)
        if n_drop:
            print(f'  field modes: {n_drop} image footprint(s) farther than the field Gaia extent + {FOOTPRINT_MARGIN_DEG} deg dropped', flush=True)
        if not near:
            raise SystemExit('field modes: no image footprint overlaps the field Gaia catalogue')
        polys = near
    r_q = QSO_RADIUS_DEG                                            # QSO cone: must reach QSO_NEAR_DEG beyond every footprint
    if polys:
        V = np.concatenate(polys); c = _unit([centre[0]], [centre[1]])[0]
        far = float(np.degrees(np.arccos(np.clip(_unit(V[:, 0], V[:, 1]) @ c, -1, 1))).max())
        r_q = max(QSO_RADIUS_DEG, round(far + QSO_NEAR_DEG + 0.05, 1))
    from .qso_allsky import ensure_table
    qp = download_qso_sample(gdir, stem, float(centre[0]), float(centre[1]), gaia_tap_server, radius_deg=r_q,
                             qso_table=ensure_table(lib_dir, gaia_tap_server))   # local selection from the all-sky table (10-29)
    import hashlib
    fp_key = (hashlib.sha256(np.round(np.concatenate([p.ravel() for p in polys]), 7).tobytes()).hexdigest()[:16] if polys else 'full-table')
    out = model_copies[0].parent / f'{stem}_gaia_field_modes_{model}.npz'
    side = Path(str(out) + '.json')
    if out.exists() and side.exists():
        m = json.loads(side.read_text())
        if (m.get('modes_version') == MODES_VERSION and m.get('footprint_key') == fp_key and m.get('footprint_pad_arcsec') == (pad_arcsec if polys else None) and m.get('sha256_field_csvs') == [_sha(f) for f in field_csvs]
                and m.get('sha256_qso_sample') == _sha(qp) and m.get('anchors') == [a.name for a in anc]
                and m.get('code') == {p.name: _sha(p) for p in (Path(__file__), Path(GV3.__file__), Path(QP.__file__))}):
            return out
    o = build_modes(field_csvs, anc, qp, mq, np.asarray(centre, float), half_diag_deg, out, model=model, footprints=polys, footprints_dropped=n_drop,
                    pad_arcsec=pad_arcsec if polys else None)
    meta = json.loads(Path(str(o) + '.json').read_text()); meta['footprint_key'] = fp_key
    Path(str(o) + '.json').write_text(json.dumps(meta, indent=1))
    return o
