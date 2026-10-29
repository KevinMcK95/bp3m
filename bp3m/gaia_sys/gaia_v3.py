# Vendored VERBATIM from the pop_fit session (owner of the Gaia model inputs; handoff GAIA_SYSTEMATICS_BP3M_HANDOFF.md 3.4)
# source: pop_fit_dev/gaia_v3.py  sha256 bae58041c75cb2e51e8ce3cd1bf39d9063510375a9131518146bfed8f20d6ca0  (copied 2026-10-29)
# Only change: see the VENDOR NOTE lines (if any).  Re-vendor, never edit, when pop_fit announces a new version.
"""
Kernel v3 Gaia DR3 error / systematics model for pop_fit (LAB_NOTEBOOK 2.38): the joint star + QSO calibration
(pop_fit_review/kernel_v3/joint, fit real2) applied exactly as it was trained.

Per source (production-corrected astrometry: L21, 6p conditioning, qso_sys; pop_fit order pmra, pmdec, plx):
  noise   S_tot = D S D + diag(floor²) + diag(σ_crowd²) + a aᵀ ∘ B_short d²
          D = diag(k_pm, k_pm, k_plx), log k = k0 + k1 (G − 19) + k2 dens + k3 [6p],  dens = log(1 + n10/10),
          n10 = Gaia sources within 10″ in the field's own (full) download,
          σ_crowd = exp(c0 + c1 (G − 19)) dens,  a_c = exp(β_c (G − 19)) × a6 [6p],  d = exp(γ dens).
          The ~3″ short-range term enters as per-source variance (its correlation between distinct members is
          dropped; pairs closer than ~10″ are rare outside crowded cores — report_close_pairs()).
  field   s(x) = a ∘ f(x), f ~ GP(0, B_mid / (1 + θ/θ_mid) + B_large exp(−θ/θ_large)) with the ϖ–PM cross terms;
          represented at inducing points Z (gp_sys.inducing_points), conditioned analytically on the field's QSOs
          (residual = a_q ∘ f(x_q) + noise_q, noise_q = S_tot of the QSO with n10 = 0), projected to the sources
          by kriging (W = K_xZ K_ZZ⁻¹) with the FITC remainder added to S_tot.
  zp      stars' parallax offset zp(G) relative to L21 + qso_sys (piecewise linear), applied to the member class
          only: the QSO class has zp = 0 by construction, and the MW class is trained on the field's own stars
          (its XD model already contains their offset).
"""
import json
import numpy as np

DEFAULT_PATH = None    # VENDOR NOTE: pop_fit's default pointed at the superseded real2 file; bp3m always passes the vendored real2p path
PERM = [1, 2, 0]           # pop_fit component j = calibration component PERM[j]  (calibration order plx, pmra, pmdec)
DEG = np.pi / 180


def load(path=None):
    if not (path or DEFAULT_PATH):
        raise ValueError('gaia_v3.load: pass the calibration path explicitly (VENDOR NOTE)')
    c = json.load(open(path or DEFAULT_PATH))
    for k in ('B_short', 'B_mid', 'B_large'):
        c[k + '_pop'] = np.asarray(c[k], float)[np.ix_(PERM, PERM)]
    c['path'] = path or DEFAULT_PATH
    return c


def dens_from_n10(n10):
    return np.log1p(np.nan_to_num(np.asarray(n10, float), nan=0.0) / 10.0)


def count_n10(ra, dec, all_ra, all_dec, radius_arcsec=10.0):
    """Number of OTHER sources of the full download within radius of each (ra, dec)."""
    from scipy.spatial import cKDTree
    c0 = np.cos(np.radians(np.nanmedian(all_dec)))
    T = cKDTree(np.column_stack([np.asarray(all_ra) * c0, np.asarray(all_dec)]) * 3600)
    q = np.column_stack([np.asarray(ra) * c0, np.asarray(dec)]) * 3600
    return np.array([len(i) - 1 for i in T.query_ball_point(q, radius_arcsec)], float)


def amplitude(cal, G, sol):
    """(N, 3) amplitude factors a_c in pop order (pmra, pmdec, plx)."""
    g = np.where(np.isfinite(G), np.asarray(G, float), cal['G_ref']) - cal['G_ref']
    six = np.asarray(sol) == 95
    a = np.stack([np.exp(cal['beta_pm'] * g), np.exp(cal['beta_pm'] * g), np.exp(cal['beta_plx'] * g)], 1)
    return a * np.where(six, cal['a6'], 1.0)[:, None]


def noise_cov(cal, S, G, n10, sol):
    """(N, 3, 3) total per-source noise covariance (pop order), excluding the member intrinsic dispersion."""
    S = np.asarray(S, float); N = len(S)
    g = np.where(np.isfinite(G), np.asarray(G, float), cal['G_ref']) - cal['G_ref']
    six = (np.asarray(sol) == 95).astype(float); dens = dens_from_n10(n10)
    lk = lambda v: v[0] + v[1] * g + v[2] * dens + v[3] * six
    kpm, kplx = np.exp(lk(cal['k_pm'])), np.exp(lk(cal['k_plx']))
    D = np.stack([kpm, kpm, kplx], 1)
    out = S * D[:, :, None] * D[:, None, :]
    cr = lambda v: np.exp(v[0] + v[1] * g) * dens
    var = np.stack([cal['floor_pm']**2 + cr(cal['crowd_pm'])**2, cal['floor_pm']**2 + cr(cal['crowd_pm'])**2,
                    cal['floor_plx']**2 + cr(cal['crowd_plx'])**2], 1)
    out = out + np.einsum('ni,ij->nij', var, np.eye(3))
    return out + short_cov(cal, G, n10, sol)


def short_cov(cal, G, n10, sol):
    """(N, 3, 3) per-source variance of the ~3″ short-range term, a aᵀ ∘ B_short d² (pop order; part of noise_cov)."""
    a = amplitude(cal, G, sol); d2 = np.exp(2 * cal['gamma_short'] * dens_from_n10(n10))
    return np.einsum('ni,nj->nij', a, a) * cal['B_short_pop'][None] * d2[:, None, None]


def zp_star(cal, G):
    """Stars' parallax offset (mas) to SUBTRACT, relative to L21 + qso_sys."""
    return np.interp(np.where(np.isfinite(G), np.asarray(G, float), cal['G_ref']), cal['zp_nodes'], cal['zp'])


def _unit(ra, dec):
    ra, dec = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _theta(u1, u2):
    ch = np.sqrt(np.clip(((u1[:, None, :] - u2[None, :, :])**2).sum(-1), 0, None))
    return 2 * np.arcsin(np.clip(ch / 2, 0, 1)) / DEG


def smooth_cov(cal, u1, u2):
    """(n1, 3, n2, 3) unit-amplitude smooth-field covariance (pop order)."""
    t = _theta(u1, u2)
    k = (1.0 / (1.0 + t / cal['theta_mid_deg']))[:, :, None, None] * cal['B_mid_pop'] \
        + np.exp(-t / cal['theta_large_deg'])[:, :, None, None] * cal['B_large_pop']
    return np.transpose(k, (0, 2, 1, 3))


def build_gp_v3(cal, center, extent_deg, star_ra, star_dec, star_G, star_sol, q_ra, q_dec, q_resid, q_S, q_G, q_sol,
                n_rings=3, max_qso=1500, jitter=1e-12):
    """
    QSO-conditioned smooth field at inducing points and the projection to the sources.
    q_resid (M, 3): QSO pmra − drift, pmdec − drift, plx (pop order); q_S (M, 3, 3) their raw Gaia covariances
    (the v3 noise model with n10 = 0 is applied here).  Returns dict(mode='v3', m_Z, L_Z, W3 (N, 3, 3nz) incl. the
    sources' amplitudes, Wc (3, 3nz) at the centre for G = 19 5p, fitc (N, 3, 3), q_loo_mean, q_loo_var_s, ...).
    """
    from gp_sys import inducing_points
    zra, zde = inducing_points(center[0], center[1], extent_deg, n_rings=n_rings)
    uZ = _unit(zra, zde); nz = len(zra)
    uq = _unit(q_ra, q_dec)
    if len(uq) > max_qso:                                        # nearest QSOs to the centre
        dq = _theta(_unit([center[0]], [center[1]]), uq)[0]
        keep = np.argsort(dq)[:max_qso]
        uq, q_resid, q_S, q_G, q_sol = uq[keep], q_resid[keep], q_S[keep], q_G[keep], np.asarray(q_sol)[keep]
    M = len(uq)
    aq = amplitude(cal, q_G, q_sol)
    Nq = noise_cov(cal, q_S, q_G, np.zeros(M), q_sol)
    Kqq = smooth_cov(cal, uq, uq) * aq[:, :, None, None] * aq[None, None, :, :]
    Kqq = Kqq.reshape(3 * M, 3 * M)
    Nblk = np.zeros((M, 3, M, 3)); Nblk[np.arange(M), :, np.arange(M), :] = Nq
    Kt = Kqq + Nblk.reshape(3 * M, 3 * M) + jitter * np.eye(3 * M)
    r = np.asarray(q_resid, float).reshape(-1)
    ok = np.isfinite(r)
    Kt = Kt[np.ix_(ok, ok)]; r = r[ok]
    L = np.linalg.cholesky(Kt)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, r))
    KZq = (smooth_cov(cal, uZ, uq) * aq[None, None, :, :]).reshape(3 * nz, 3 * M)[:, ok]
    KZZ = smooth_cov(cal, uZ, uZ).reshape(3 * nz, 3 * nz) + jitter * np.eye(3 * nz)
    m_Z = KZq @ alpha
    V_ = np.linalg.solve(L, KZq.T)
    C_Z = KZZ - V_.T @ V_
    C_Z = 0.5 * (C_Z + C_Z.T) + 1e-14 * np.eye(3 * nz)
    L_Z = np.linalg.cholesky(C_Z)
    # leave-one-out predictive mean / variance of the systematic at each QSO (diagnostics)
    Kinv_diag = np.einsum('ij,ij->j', np.linalg.inv(L), np.linalg.inv(L))
    loo = np.full(3 * M, np.nan); loo[ok] = r - alpha / Kinv_diag
    # projection to the sources
    us = _unit(star_ra, star_dec); N = len(us)
    a = amplitude(cal, star_G, star_sol)
    KxZ = smooth_cov(cal, us, uZ).reshape(3 * N, 3 * nz)
    LZZ = np.linalg.cholesky(KZZ)
    W = np.linalg.solve(LZZ.T, np.linalg.solve(LZZ, KxZ.T)).T          # (3N, 3nz) unit field
    # FITC remainder: Kxx − KxZ KZZ⁻¹ KZx per source (3x3), times amplitudes
    Kxx = np.repeat((cal['B_mid_pop'] + cal['B_large_pop'])[None], N, 0)
    Q = np.einsum('nia,nja->nij', KxZ.reshape(N, 3, 3 * nz), W.reshape(N, 3, 3 * nz))
    fitc = (Kxx - Q) * a[:, :, None] * a[:, None, :]
    W3 = W.reshape(N, 3, 3 * nz) * a[:, :, None]
    uc = _unit([center[0]], [center[1]])
    Wc = np.linalg.solve(LZZ.T, np.linalg.solve(LZZ, smooth_cov(cal, uc, uZ).reshape(3, 3 * nz).T)).T
    # free-amplitude form (--v3-free-amp): prior f_Z = λ chol(K_ZZ) z, the QSOs enter the model's likelihood directly
    # (DTC projection Wq of the inducing values + their noise and FITC remainder) instead of the analytic conditioning
    KqZ = (smooth_cov(cal, uq, uZ) * aq[:, :, None, None]).reshape(3 * M, 3 * nz)
    Wq = np.linalg.solve(LZZ.T, np.linalg.solve(LZZ, KqZ.T)).T                     # (3M, 3nz), amplitudes included
    Qq = np.einsum('mia,mja->mij', KqZ.reshape(M, 3, 3 * nz), Wq.reshape(M, 3, 3 * nz))
    Kqq_d = np.repeat((cal['B_mid_pop'] + cal['B_large_pop'])[None], M, 0) * aq[:, :, None] * aq[:, None, :]
    q_cov = Nq + (Kqq_d - Qq)
    return dict(mode='v3', m_Z=m_Z, L_Z=L_Z, W3=W3, Wc=Wc, fitc=fitc, n_z=nz, n_qso=M,
                LZZ=LZZ, Wq3=Wq.reshape(M, 3, 3 * nz), q_resid=np.asarray(q_resid, float), q_cov=q_cov,
                q_loo_mean=loo.reshape(M, 3), prior_sd_center=np.sqrt(np.diag(cal['B_mid_pop'] + cal['B_large_pop'])),
                post_sd_center=np.sqrt(np.clip(np.diag(Wc @ C_Z @ Wc.T), 0, None)), post_mean_center=Wc @ m_Z)


def report_close_pairs(ra, dec, radius_arcsec=10.0):
    """Fraction of sources with another source within radius (where dropping the short-range correlation matters)."""
    n = count_n10(ra, dec, ra, dec, radius_arcsec)
    return float(np.mean(n > 0)), int(np.sum(n)) // 2
