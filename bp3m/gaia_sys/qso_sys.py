# Vendored VERBATIM from the pop_fit session (owner of the Gaia model inputs; handoff GAIA_SYSTEMATICS_BP3M_HANDOFF.md 3.4)
# source: pop_fit_dev/qso_sys.py  sha256 0d281c8bae1fe77ada69fc943500e8e8c766644014b3bed82266a17996817bcb  (copied 2026-10-29)
# Only change: see the VENDOR NOTE lines (if any).  Re-vendor, never edit, when pop_fit announces a new version.
"""
QSO-calibrated all-sky Gaia systematics model (LAB_NOTEBOOK §2.18).

PM, magnitude-dependent part: per G bin and solution (5p / 6p), vector spherical harmonics l = 1..L
(spheroidal + toroidal) fitted to the CRF3 QSO PM residuals (drift removed).  The coefficients are
interpolated linearly in G between bin centres (held constant outside), and the correction is taken
RELATIVE to G = 20 — the magnitude of the QSO anchor (delta_pm_sys / the GP), so QSO-anchored PMs of
fainter / brighter stars move into the frame of the G = 20 QSOs:
    D_pm(x, G, sol) = V_L(x) [c(G, sol) - c(20, sol)]
Below the brightest bin centre (G ~ 17.8) the correction is extrapolated flat (QSOs give no information).
Parallax: 6p offset vs pseudocolour (quintile means, interpolated) + scalar spherical harmonics
l = 0..L_plx per solution fitted to what remains (5p: the SH term only):
    D_plx(x, sol, pc)
Corrections are SUBTRACTED from the data (after the L21 zero point).

build(...) writes the calibration npz; correction(...) evaluates D (N, 3) for any sources.
"""
import numpy as np
from scipy.special import sph_harm_y

G_EDGES = np.array([17.0, 18.5, 19.25, 19.75, 20.15, 20.5, 21.0])
L_PM, L_PLX = 3, 3
G_REF = 20.0


def _real_ylm(l, m, th, ph):
    y = sph_harm_y(l, abs(m), th, ph)
    if m > 0:
        return np.sqrt(2) * (-1)**m * y.real
    if m < 0:
        return np.sqrt(2) * (-1)**m * y.imag
    return y.real


def vsh_basis(lmax, ra, dec, h=1e-5):
    """(N, 2, K): spheroidal and toroidal VSH, l = 1..lmax, components (mu_a*, mu_d)."""
    th = np.radians(90.0 - np.asarray(dec, float)); ph = np.radians(np.asarray(ra, float))
    st = np.sin(th); cols = []
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            dth = (_real_ylm(l, m, th + h, ph) - _real_ylm(l, m, th - h, ph)) / (2 * h)
            dph = (_real_ylm(l, m, th, ph + h) - _real_ylm(l, m, th, ph - h)) / (2 * h) / st
            cols.append(np.stack([dph, -dth], 1))
            cols.append(np.stack([dth, dph], 1))
    return np.stack(cols, 2)


def sh_basis(lmax, ra, dec):
    th = np.radians(90.0 - np.asarray(dec, float)); ph = np.radians(np.asarray(ra, float))
    return np.stack([_real_ylm(l, m, th, ph) for l in range(0, lmax + 1) for m in range(-l, l + 1)], 1)


def build(ra, dec, G, sol, pc, pm_resid, pm_w, plx_resid, plx_w, path):
    """Fit and save the calibration.  pm_resid / pm_w (N, 2); plx_resid / plx_w (N,)."""
    six = np.asarray(sol) == 95
    B = vsh_basis(L_PM, ra, dec)
    K = B.shape[2]
    centres = np.zeros((2, len(G_EDGES) - 1)); coef = np.zeros((2, len(G_EDGES) - 1, K)); nbin = np.zeros((2, len(G_EDGES) - 1), int)
    for s6 in (0, 1):
        for b, (lo, hi) in enumerate(zip(G_EDGES[:-1], G_EDGES[1:])):
            sel = (G >= lo) & (G < hi) & (six == bool(s6))
            X = np.vstack([B[sel, 0, :], B[sel, 1, :]]); y = np.concatenate([pm_resid[sel, 0], pm_resid[sel, 1]])
            w = np.concatenate([pm_w[sel, 0], pm_w[sel, 1]])
            coef[s6, b] = np.linalg.solve(X.T @ (w[:, None] * X) + 1e-12 * np.eye(K), X.T @ (w * y))
            centres[s6, b] = np.sum(pm_w[sel, 0] * G[sel]) / np.sum(pm_w[sel, 0])
            nbin[s6, b] = int(sel.sum())
    # parallax: 6p pseudocolour offset, then SH per solution
    q = np.nanpercentile(pc[six], [0, 20, 40, 60, 80, 100])
    pc_c = np.zeros(5); pc_off = np.zeros(5)
    off = np.zeros(len(plx_resid))
    for k, (a, b) in enumerate(zip(q[:-1], q[1:])):
        s = six & (pc >= a) & (pc <= b)
        pc_off[k] = np.sum(plx_w[s] * plx_resid[s]) / np.sum(plx_w[s]); pc_c[k] = np.median(pc[s])
    off[six] = np.interp(pc[six], pc_c, pc_off)
    S = sh_basis(L_PLX, ra, dec)
    sh = np.zeros((2, S.shape[1]))
    for s6 in (0, 1):
        s = six == bool(s6)
        X = S[s]; y = plx_resid[s] - off[s]; w = plx_w[s]
        sh[s6] = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
    np.savez(path, G_edges=G_EDGES, L_pm=L_PM, L_plx=L_PLX, G_ref=G_REF, pm_centres=centres, pm_coef=coef, pm_nbin=nbin,
             plx_pc_centres=pc_c, plx_pc_offsets=pc_off, plx_sh=sh)
    return path


_CACHE = {}


def load(path):
    if path not in _CACHE:
        _CACHE[path] = dict(np.load(path))
    return _CACHE[path]


def _coef_at(cal, s6, g):
    c, x = cal['pm_coef'][s6], cal['pm_centres'][s6]
    gq = np.clip(np.nan_to_num(np.asarray(g, float), nan=G_REF), x[0], x[-1])
    return np.stack([np.interp(gq, x, c[:, k]) for k in range(c.shape[1])], 1)        # (N, K)


def correction(path, ra, dec, G, sol, pc):
    """(N, 3) systematics to SUBTRACT: (pmra, pmdec) relative to G = 20, parallax."""
    cal = load(path)
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    six = np.asarray(sol) == 95
    B = vsh_basis(int(cal['L_pm']), ra, dec)
    D = np.zeros((len(ra), 3))
    for s6 in (0, 1):
        s = six == bool(s6)
        if not s.any():
            continue
        dc = _coef_at(cal, s6, np.asarray(G, float)[s]) - _coef_at(cal, s6, np.full(s.sum(), float(cal['G_ref'])))
        D[s, :2] = np.einsum('nck,nk->nc', B[s], dc)
    S = sh_basis(int(cal['L_plx']), ra, dec)
    pcv = np.asarray(pc, float)
    off = np.where(six, np.interp(np.nan_to_num(pcv, nan=np.nanmedian(cal['plx_pc_centres'])),
                                  cal['plx_pc_centres'], cal['plx_pc_offsets']), 0.0)
    D[:, 2] = off + np.where(six, S @ cal['plx_sh'][1], S @ cal['plx_sh'][0])
    return D
