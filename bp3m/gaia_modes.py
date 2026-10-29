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
QSO_QUERY_COLS = ('g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.pmra_error, g.pmdec_error, g.pmra_pmdec_corr, g.parallax, '
                  'g.parallax_error, g.parallax_pmra_corr, g.parallax_pmdec_corr, g.phot_g_mean_mag, g.phot_g_mean_flux, '
                  'g.phot_g_mean_flux_error, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.ruwe, g.nu_eff_used_in_astrometry, '
                  'g.pseudocolour, g.ecl_lat, g.astrometric_params_solved')
MODES_VERSION = 2       # 2: sources inside the image footprints + padding (10-29)


def qso_sample_path(gaia_dir, stem):
    return Path(gaia_dir) / f'{stem}_qsocond_r{QSO_RADIUS_DEG:.1f}_G{QSO_GMAX:.1f}.csv'


def download_qso_sample(gaia_dir, stem, ra, dec, gaia_tap_server=None) -> Path:
    """Gaia DR3 QSO candidates (pop_fit production cuts) within QSO_RADIUS_DEG; cached with a verified sidecar."""
    from .pipeline.download_gaia import _submit_gaia_async
    out = qso_sample_path(gaia_dir, stem); side = Path(str(out) + '.json')
    if out.exists() and side.exists():
        m = json.loads(side.read_text())
        if m.get('sha256') == _sha(out) and m.get('row_count_verified') and m.get('columns') == QSO_QUERY_COLS:
            return out
    where = (f"CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {QSO_RADIUS_DEG:.6f})) = 1 "
             f"AND g.astrometric_params_solved >= 31 AND g.pmra IS NOT NULL AND g.pmdec IS NOT NULL AND g.parallax IS NOT NULL "
             f"AND g.ruwe < 1.4 AND g.ipd_gof_harmonic_amplitude < 0.2 AND g.phot_g_mean_mag < {QSO_GMAX:.2f}")
    frm = 'FROM gaiadr3.gaia_source AS g JOIN gaiadr3.qso_candidates AS q ON q.source_id = g.source_id'
    df = _submit_gaia_async(f'SELECT {QSO_QUERY_COLS} {frm} WHERE {where}', gaia_tap_server)
    n = int(_submit_gaia_async(f'SELECT COUNT(*) AS n {frm} WHERE {where}', gaia_tap_server).iloc[0, 0])
    df.columns = [c.lower() for c in df.columns]
    if len(df) != n or df.source_id.duplicated().any():
        raise SystemExit(f'QSO sample: {len(df)} rows vs COUNT(*) {n} (or duplicates)')
    df['source_id'] = df.source_id.astype(np.int64)
    tmp = Path(str(out) + '.tmp'); df.to_csv(tmp, index=False); tmp.replace(out)
    side.write_text(json.dumps({'ra': ra, 'dec': dec, 'radius_deg': QSO_RADIUS_DEG, 'gmax': QSO_GMAX, 'where': where,
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


def image_footprints(image_files):
    """Sky polygons (deg) of every science extension of the given HST (_fl?) / JWST (_cal) images, from the header WCS.  Used
    ONLY to select which Gaia sources get a field term (the padding absorbs WCS pointing errors); never for positions."""
    from astropy.io import fits
    from astropy.wcs import WCS
    import warnings
    polys = []
    drop = ('CPDIS', 'DP1', 'DP2', 'D2IM', 'CPERR', 'NPOLEXT', 'D2IMEXT', 'D2IMERR', 'WCSDVARR')
    for f in image_files:
        with fits.open(f, memmap=True, lazy_load_hdus=True) as h:
            for e in h:                                  # headers only: the lookup-table distortions (read from the data) are
                hd = e.header                            # irrelevant for a selection footprint and cost a minute per field
                if hd.get('EXTNAME', '') == 'SCI' and hd.get('NAXIS', 0) == 2:
                    hc = hd.copy()
                    for k in [k for k in hc.keys() if k.startswith(drop)]:
                        del hc[k]
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        polys.append(WCS(hc).calc_footprint(axes=(hd['NAXIS1'], hd['NAXIS2'])))
    if not polys:
        raise SystemExit('field modes: no science-extension footprints in the image list')
    return polys


def in_footprints(ra, dec, polys, pad_arcsec):
    """True for sources inside any footprint polygon or within pad_arcsec of its edges (tangent plane per polygon)."""
    ra = np.asarray(ra, float); dec = np.asarray(dec, float); keep = np.zeros(len(ra), bool); pad = pad_arcsec / 3600.0
    for P in polys:
        c0, d0 = np.mean(P[:, 0]), np.mean(P[:, 1]); cd = np.cos(np.radians(d0))
        px = ((P[:, 0] - c0 + 180) % 360 - 180) * cd; py = P[:, 1] - d0
        x = ((ra - c0 + 180) % 360 - 180) * cd; y = dec - d0
        near = (np.abs(x) < np.ptp(px) + pad + 0.05) & (np.abs(y) < np.ptp(py) + pad + 0.05) & ~keep
        if not near.any():
            continue
        xs, ys = x[near], y[near]; inside = np.zeros(len(xs), bool); dmin = np.full(len(xs), np.inf)
        for k in range(len(px)):
            x1, y1, x2, y2 = px[k], py[k], px[(k + 1) % len(px)], py[(k + 1) % len(px)]
            cr = ((y1 > ys) != (y2 > ys)) & (xs < (x2 - x1) * (ys - y1) / (y2 - y1 + 1e-300) + x1)
            inside ^= cr
            t = np.clip(((xs - x1) * (x2 - x1) + (ys - y1) * (y2 - y1)) / max((x2 - x1) ** 2 + (y2 - y1) ** 2, 1e-300), 0, 1)
            dmin = np.minimum(dmin, np.hypot(xs - (x1 + t * (x2 - x1)), ys - (y1 + t * (y2 - y1))))
        keep[np.where(near)[0][inside | (dmin <= pad)]] = True
    return keep


def build_modes(field_csvs, anchors_csvs, qso_csv, milliquas_path, center, half_diag_deg, out_npz, model='v3real2p',
                chunk=4000, verbose=True, footprints=None, pad_arcsec=None) -> Path:
    """Field-mode npz for the Gaia sources (with a PM) of the field table(s)."""
    check_inputs()
    cal = GV3.load(str(KERNEL))
    G = pd.concat([pd.read_csv(f, dtype={'source_id': 'int64'}, usecols=['source_id', 'ra', 'dec', 'gmag', 'pmra', 'pseudocolour',
                                                                        'astrometric_params_solved']) for f in field_csvs])
    G = G.drop_duplicates('source_id').reset_index(drop=True)
    G = G[G.pmra.notna() & G.astrometric_params_solved.isin([31, 95])].reset_index(drop=True)
    n_table = len(G)
    if footprints is not None:
        # 10-29 (user): only Gaia sources inside the images (+ padding), not the whole Gaia box: every star the solver can see
        # (incl. saturated / unmatched ones), lightweight files; the inducing grid follows the selected sources
        G = G[in_footprints(G.ra.values, G.dec.values, footprints, pad_arcsec)].reset_index(drop=True)
        if len(G) == 0:
            raise SystemExit('field modes: no Gaia source inside the image footprints')
        cd = np.cos(np.radians(G.dec.mean()))
        center = np.array([G.ra.mean() + ((G.ra - G.ra.mean() + 180) % 360 - 180).mean(), G.dec.mean()])   # wrap-safe mean
        half_diag_deg = float(np.max(np.hypot(((G.ra - center[0] + 180) % 360 - 180) * cd, G.dec - center[1]))) + 0.01
    anc_ids = set()
    for a in anchors_csvs:
        anc_ids |= set(pd.read_csv(a, usecols=['source_id'], dtype={'source_id': 'int64'}).source_id.astype(int))
    field_ids = set(G.source_id.astype(int))
    q_ra, q_dec, q_res, q_S, q_G, q_sol, exc_in_sample, qst = prepare_qsos(qso_csv, milliquas_path, anc_ids | field_ids, center)
    extent = max(0.3, float(half_diag_deg)); n_rings = max(3, int(math.ceil(extent / 0.13)))
    sol = G.astrometric_params_solved.to_numpy(); Gm = G.gmag.to_numpy(float)
    ra, dec = G.ra.to_numpy(float), G.dec.to_numpy(float)
    N = len(G)
    # pass 1: accumulate A^T A over star chunks (A = W3 L_Z); the QSO conditioning is identical in every call
    AtA, M, fitc, gp0 = None, np.zeros((N, 3)), np.zeros((N, 3, 3)), None
    for s in range(0, N, chunk):
        sl = slice(s, min(N, s + chunk))
        g = GV3.build_gp_v3(cal, center, extent, ra[sl], dec[sl], Gm[sl], sol[sl], q_ra, q_dec, q_res, q_S, q_G, q_sol,
                            n_rings=n_rings, max_qso=MAX_QSO)
        if gp0 is None:
            gp0 = g; g_first = g
        elif not (np.allclose(g['m_Z'], gp0['m_Z'], rtol=1e-10, atol=1e-14) and np.allclose(g['L_Z'], gp0['L_Z'], rtol=1e-10, atol=1e-14)):
            raise SystemExit('field modes: the QSO conditioning differs between chunks')
        A = (g['W3'] @ gp0['L_Z']).reshape(-1, gp0['L_Z'].shape[1])
        AtA = A.T @ A if AtA is None else AtA + A.T @ A
        M[sl] = g['W3'] @ gp0['m_Z']; fitc[sl] = g['fitc']
    w, V = np.linalg.eigh(AtA); o = np.argsort(w)[::-1]; w, V = np.clip(w[o], 0, None), V[:, o]
    K = int(np.searchsorted(np.cumsum(w) / w.sum(), TRACE_FRAC) + 1)
    # pass 2: loadings of the K modes and the dropped modes' per-source remainder
    W = np.zeros((N, 3, K)); R = fitc.copy()
    for s in range(0, N, chunk):
        sl = slice(s, min(N, s + chunk))
        g = g_first if N <= chunk else GV3.build_gp_v3(cal, center, extent, ra[sl], dec[sl], Gm[sl], sol[sl], q_ra, q_dec, q_res,
                                                       q_S, q_G, q_sol, n_rings=n_rings, max_qso=MAX_QSO)   # one chunk: reuse pass 1
        A3 = g['W3'] @ gp0['L_Z']                                       # (n, 3, 3nz)
        Wf = A3 @ V                                                      # all modes
        W[sl] = Wf[:, :, :K]
        R[sl] += np.einsum('nik,njk->nij', Wf[:, :, K:], Wf[:, :, K:])
    out_npz = Path(out_npz); out_npz.parent.mkdir(parents=True, exist_ok=True)
    qso_excluded = np.array(sorted(anc_ids | (set(exc_in_sample))), dtype=np.int64)
    meta = dict(model=model, modes_version=MODES_VERSION, calibration=KERNEL.name, sha256_calibration=SHA256[KERNEL.name],
                center=list(map(float, center)), extent_deg=extent, n_rings=n_rings, n_inducing=int(gp0['n_z']), K=K,
                trace_fraction=float(np.cumsum(w)[K - 1] / w.sum()), n_sources=N, n_table_sources=n_table,
                footprint_pad_arcsec=pad_arcsec, n_footprints=None if footprints is None else len(footprints), qso=qst, n_qso_used=int(gp0['n_qso']),
                post_mean_center=[float(x) for x in gp0['post_mean_center']], post_sd_center=[float(x) for x in gp0['post_sd_center']],
                field_csvs=[Path(f).name for f in field_csvs], sha256_field_csvs=[_sha(f) for f in field_csvs],
                anchors=[Path(a).name for a in anchors_csvs], qso_sample=Path(qso_csv).name, sha256_qso_sample=_sha(qso_csv),
                milliquas_sha256=_sha(milliquas_path), code={p.name: _sha(p) for p in (Path(__file__), Path(GV3.__file__), Path(QP.__file__))},
                created=datetime.datetime.now().isoformat(timespec='seconds'))
    np.savez(out_npz, source_id=G.source_id.to_numpy(np.int64), M=M, W=W, R=R, qso_excluded=qso_excluded, model=np.array(model),
             calibration=np.array(KERNEL.name), sha256_calibration=np.array(SHA256[KERNEL.name]), created=np.array(meta['created']),
             meta=np.array(json.dumps(meta)))
    Path(str(out_npz) + '.json').write_text(json.dumps(meta, indent=1))
    if verbose:
        print(f'  field modes {out_npz.name}: N {N:,}, QSOs {qst} -> {gp0["n_qso"]} used; extent {extent:.2f} deg, {n_rings} rings, '
              f'{gp0["n_z"]} inducing points; K = {K} modes ({meta["trace_fraction"]:.4f} of the trace); post. mean at the centre '
              f'{np.round(1e3 * gp0["post_mean_center"], 2)} uas(/yr) +- {np.round(1e3 * gp0["post_sd_center"], 2)}')
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
    qp = download_qso_sample(gdir, stem, float(centre[0]), float(centre[1]), gaia_tap_server)
    polys = image_footprints(image_files) if image_files else None
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
    o = build_modes(field_csvs, anc, qp, mq, np.asarray(centre, float), half_diag_deg, out, model=model, footprints=polys,
                    pad_arcsec=pad_arcsec if polys else None)
    meta = json.loads(Path(str(o) + '.json').read_text()); meta['footprint_key'] = fp_key
    Path(str(o) + '.json').write_text(json.dumps(meta, indent=1))
    return o
