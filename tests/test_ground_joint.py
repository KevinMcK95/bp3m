"""bp3m.ground.joint (2026-10-30): Schur-PCG joint frame + star solve against a dense direct solve on a synthetic problem."""
import numpy as np

from bp3m.ground import joint as J


def _poly(u, v):
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], 1)


def _problem(seed=1, ns=240, ne=10, nccd=4, p=5):
    r = np.random.default_rng(seed)
    pos = r.uniform(-1, 1, (ns, 2))
    rows = []
    for e in range(ne):
        shift = r.uniform(-0.15, 0.15, 2)
        for j in range(ns):
            if r.random() < 0.75:
                q = pos[j] + shift
                if np.all(np.abs(q) < 1):
                    k = int(q[0] > 0) + 2 * int(q[1] > 0)
                    rows.append((j, e, e * nccd + k, q[0], q[1], 2020.0 + e * 0.9 - 4))
    R = np.array(rows)
    o = np.argsort(R[:, 0], kind='stable'); R = R[o]
    star, ex, ccd = R[:, 0].astype(int), R[:, 1].astype(int), R[:, 2].astype(int)
    n = len(R); u, v, t = R[:, 3], R[:, 4], R[:, 5] - 2016
    A = np.zeros((n, 2, p)); A[:, 0, 0] = 1; A[:, 1, 1] = 1; A[:, 0, 2] = t; A[:, 1, 3] = t
    A[:, 0, 4] = r.normal(0, 0.7, n); A[:, 1, 4] = r.normal(0, 0.7, n)            # parallax-like column
    Bs = _poly(u, v)
    D = np.zeros((n, 2, 3)); cc = r.normal(0, 1, ns)[star]; D[:, 0, 0] = cc; D[:, 1, 1] = cc; D[:, 0, 2] = cc * u
    nsep, nfull = ne * nccd, ne
    sz = np.r_[np.full(nsep, 12), np.full(nfull, 3)]; off = np.r_[0, np.cumsum(sz)[:-1]]; kind = np.r_[np.zeros(nsep), np.ones(nfull)]
    sb, fb = ccd, nsep + ex
    C = np.c_[r.uniform(4, 9, n), r.uniform(4, 9, n), r.uniform(-1, 1, n)]
    Ps = np.zeros((ns, p, p)); ms = np.zeros((ns, p))
    Ps[:, 0, 0] = Ps[:, 1, 1] = 1e-10; Ps[:, 2, 2] = Ps[:, 3, 3] = 1e-6; Ps[:, 4, 4] = 1.0
    gaia = r.random(ns) < 0.2
    G = np.diag([4.0, 4.0, 1.0, 1.0, 2.0]); Ps[gaia] = G; ms[gaia] = r.normal(0, 1, (gaia.sum(), p))
    Pf = J.BlockDiag(off, sz)
    for s in Pf.groups:
        Pf.set_blocks(s, np.tile(np.eye(s) * 1e-3, (len(Pf.groups[s]), 1, 1)))
    mf = r.normal(0, 0.1, int(sz.sum()))
    y = r.normal(0, 3, (n, 2))
    act = r.random(ns) < 0.85
    used = r.random(n) > 0.03
    ptr = np.r_[0, np.cumsum(np.bincount(star, minlength=ns))]
    return dict(ptr=ptr, A=A, y=y, C=C, sb=sb, Bs=Bs, fb=fb, D=D, off=off, sz=sz, block_kind=kind, Ps=Ps, ms=ms, Pf=Pf, mf=mf,
                used=used, act=act), star


def _dense(P, star, act_only):
    """Dense normal equations over (theta for all stars, phi) using the detections of `act_only` stars for the frame rows."""
    A, ptr = P['A'], P['ptr']; ns, p = len(ptr) - 1, A.shape[2]; nphi = int(P['sz'].sum()); n = len(star)
    Jm = np.zeros((2 * n, ns * p + nphi))
    for i in range(n):
        j = star[i]; Jm[2 * i:2 * i + 2, j * p:(j + 1) * p] = A[i]
        b = P['sb'][i]; o = ns * p + P['off'][b]; m = P['sz'][b] // 2
        Jm[2 * i, o:o + m] = P['Bs'][i]; Jm[2 * i + 1, o + m:o + 2 * m] = P['Bs'][i]
        b = P['fb'][i]; o = ns * p + P['off'][b]; Jm[2 * i:2 * i + 2, o:o + P['sz'][b]] = P['D'][i]
    C = P['C']; det = C[:, 0] * C[:, 1] - C[:, 2] ** 2
    Wd = np.zeros((2 * n, 2 * n))
    for i in range(n):
        if P['used'][i] and act_only[star[i]]:
            Wd[2 * i:2 * i + 2, 2 * i:2 * i + 2] = np.array([[C[i, 1], -C[i, 2]], [-C[i, 2], C[i, 0]]]) / det[i]
    N = Jm.T @ Wd @ Jm; b = Jm.T @ Wd @ P['y'].ravel()
    for j in range(ns):
        N[j * p:(j + 1) * p, j * p:(j + 1) * p] += P['Ps'][j]; b[j * p:(j + 1) * p] += P['Ps'][j] @ P['ms'][j]
    Pfd = np.zeros((nphi, nphi))
    for bl in range(len(P['sz'])):
        o, s = P['off'][bl], P['sz'][bl]; f = P['Pf'].foff[bl]; Pfd[o:o + s, o:o + s] = P['Pf'].flat[f:f + s * s].reshape(s, s)
    N[ns * p:, ns * p:] += Pfd; b[ns * p:] += Pfd @ P['mf']
    return N, b, Jm


def test_joint_matches_dense():
    P, star = _problem()
    prob = J.JointProblem(**P)
    th, phi = prob.solve(tol=1e-12)
    ns, p = len(P['ptr']) - 1, P['A'].shape[2]
    # frames from the active stars only; every star conditional on those frames with all its used detections
    N, b, _ = _dense(P, star, P['act'])
    x = np.linalg.solve(N, b); Ninv = np.linalg.inv(N)
    assert np.allclose(phi, x[ns * p:], atol=1e-7 * np.abs(x[ns * p:]).max())
    act = P['act']
    assert np.allclose(th[act], x[:ns * p].reshape(ns, p)[act], atol=1e-6)
    # passive stars: theta = Ninv_j (b_j - N_jf phi) with their own detections
    Nall, ball, _ = _dense(P, star, np.ones(ns, bool))
    for j in np.where(~act)[0][:20]:
        sl = slice(j * p, (j + 1) * p); Njj = Nall[sl, sl]; Njf = Nall[sl, ns * p:]
        assert np.allclose(th[j], np.linalg.solve(Njj, ball[sl] - Njf @ phi), atol=1e-6)
    # frame part of the active stars' marginal covariance from perturb-and-solve samples
    Phi = prob.frame_samples(K=3000, batch=500, seed=5, tol=1e-10)
    Cphi = Ninv[ns * p:, ns * p:]
    emp = np.cov(Phi)
    d = np.sqrt(np.diag(Cphi)); rel = (emp - Cphi) / np.outer(d, d)
    assert np.abs(rel).max() < 0.12                                    # MC error ~ sqrt(2/K) ~ 0.026 per element; max over ~2e4
    fc = prob.frame_cov(Phi)
    for j in np.where(act)[0][:30]:
        sl = slice(j * p, (j + 1) * p)
        exact = Ninv[sl, sl] - np.linalg.inv(N[sl, sl])                 # marginal - conditional = B Cov(phi) B^T
        B = np.linalg.solve(N[sl, sl], N[sl, ns * p:]); bp = B @ Phi
        dd = np.sqrt(np.diag(Ninv[sl, sl]))
        assert np.allclose(fc[j], bp @ bp.T / Phi.shape[1], rtol=1e-8, atol=1e-12 * dd.max() ** 2)   # kernel = B phi* exactly
        assert np.abs((fc[j] - exact) / np.outer(dd, dd)).max() < 0.12                                # Monte Carlo (K = 3000)
        assert np.allclose(prob.Ninv[j], np.linalg.inv(N[sl, sl]), rtol=1e-6, atol=1e-10)


def test_fixed_frames_is_per_star_gls():
    P, star = _problem(seed=2)
    prob = J.JointProblem(**P)
    th, phi = prob.solve(fixed_phi=np.zeros(int(P['sz'].sum())))
    assert np.all(phi == 0)
    A, C, y, used = P['A'], P['C'], P['y'], P['used']
    for j in range(10):
        i = np.arange(P['ptr'][j], P['ptr'][j + 1]); i = i[used[i]]
        Nj = P['Ps'][j].copy(); bj = P['Ps'][j] @ P['ms'][j]
        for k in i:
            Ci = np.array([[C[k, 0], C[k, 2]], [C[k, 2], C[k, 1]]]); Wi = np.linalg.inv(Ci)
            Nj += A[k].T @ Wi @ A[k]; bj += A[k].T @ Wi @ y[k]
        assert np.allclose(th[j], np.linalg.solve(Nj, bj), atol=1e-8)


def test_block_floor_recovers_injected():
    r = np.random.default_rng(0)
    nb, n = 30, 30000
    blk = r.integers(0, nb, n); f2 = r.uniform(0, 9, nb)
    Cm = np.c_[r.uniform(1, 4, n), r.uniform(1, 4, n), np.zeros(n)]
    lev = np.full(n, 0.5)                                                # residual variance scaled by (1 - lev / 2) per axis
    sx = np.sqrt((Cm[:, 0] + f2[blk]) * (1 - lev / 2)); sy = np.sqrt((Cm[:, 1] + f2[blk]) * (1 - lev / 2))
    res = np.c_[r.normal(0, 1, n) * sx, r.normal(0, 1, n) * sy]
    est, cnt = J.block_floor(blk, res, Cm, lev, np.ones(n, bool), nb)
    assert np.all(cnt > 500)
    assert np.abs(est - f2).max() < 1.0                               # ~2000 dof per block: sd ~0.25 mas^2
