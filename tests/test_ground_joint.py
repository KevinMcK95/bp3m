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


# --- 11-02: nested per-star random effects and coupled measurements with a global block ----------------------------------
def _ext_problem(seed=3, Kz=4):
    """_problem + 2-level nested random effects (pairs of exposures = 'nights' inside halves = 'seasons', only on some stars) +
    coupled 'Gaia' measurements of 20 % of the stars sharing a global block z (Kz) on the PM / parallax rows."""
    P, star = _problem(seed=seed)
    r = np.random.default_rng(seed + 100)
    n = len(star); ns = len(P['ptr']) - 1; p = P['A'].shape[2]
    ex = np.zeros(n, int)                                           # exposure index from the full block (fb = nsep + e)
    nsep = int((P['block_kind'] == 0).sum()); ex = P['fb'] - nsep
    re_star = r.random(ns) < 0.5
    on = re_star[star]
    night = np.where(on, star * 100 + ex // 2, -1); season = np.where(on, star * 100 + 50 + ex // 4, -1)      # nested: nights (pairs) inside seasons (quads)
    re_grp = np.c_[night, season].astype(np.int64)
    re_lam = np.stack([np.c_[r.uniform(0.5, 1.5, n), r.uniform(0.5, 1.5, n)], np.c_[r.uniform(0.3, 1.0, n), r.uniform(0.3, 1.0, n)]], 1)
    re_lam *= on[:, None, None]
    # global block appended as one more full-kind block touched by no detection
    off, sz = P['off'], P['sz']; zb = len(sz)
    P['off'] = np.r_[off, off[-1] + sz[-1]]; P['sz'] = np.r_[sz, Kz]; P['block_kind'] = np.r_[P['block_kind'], 1]
    Pf = J.BlockDiag(P['off'], P['sz'])
    Pf.flat[:len(P['Pf'].flat)] = P['Pf'].flat; Pf.set_block(zb, np.eye(Kz)); P['Pf'] = Pf
    P['mf'] = np.r_[P['mf'], np.zeros(Kz)]
    cs = np.where(r.random(ns) < 0.2)[0]
    Pc = np.zeros((len(cs), p, p))
    for k in range(len(cs)):
        Q = r.normal(0, 1, (p, p)); Pc[k] = Q @ Q.T + np.diag([4.0, 4.0, 1.0, 1.0, 2.0])
    G = r.normal(0, 0.5, (len(cs), 3, Kz)); rows = np.array([2, 3, 4])
    cm = r.normal(0, 1, (len(cs), p))
    P.update(re_grp=re_grp, re_lam=re_lam, coupling=dict(star=cs, P=Pc, m=cm, G=G, rows=rows, block=zb))
    return P, star


def _dense_ext(P, star, act_only):
    """Dense normal equations over (theta all stars, phi incl. z, r all random-effect groups) with the frame rows of act_only stars."""
    N0, b0, Jm = _dense({k: v for k, v in P.items() if k not in ('re_grp', 're_lam', 'coupling')}, star, act_only)
    ns, p = len(P['ptr']) - 1, P['A'].shape[2]; nphi = int(P['sz'].sum()); n = len(star)
    gid = P['re_grp']; ug = np.unique(gid[gid >= 0]); gpos = {g: k for k, g in enumerate(ug)}; nr = 2 * len(ug)
    Z = np.zeros((2 * n, nr))
    for i in range(n):
        for l in range(gid.shape[1]):
            if gid[i, l] >= 0:
                k = gpos[gid[i, l]]; Z[2 * i, 2 * k] = P['re_lam'][i, l, 0]; Z[2 * i + 1, 2 * k + 1] = P['re_lam'][i, l, 1]
    C = P['C']; det = C[:, 0] * C[:, 1] - C[:, 2] ** 2
    Wd = np.zeros((2 * n, 2 * n))
    for i in range(n):
        if P['used'][i] and act_only[star[i]]:
            Wd[2 * i:2 * i + 2, 2 * i:2 * i + 2] = np.array([[C[i, 1], -C[i, 2]], [-C[i, 2], C[i, 0]]]) / det[i]
    Jx = np.c_[Jm, Z]
    N = Jx.T @ Wd @ Jx; b = Jx.T @ Wd @ P['y'].ravel()
    N[:ns * p + nphi, :ns * p + nphi] += N0 - Jm.T @ Wd @ Jm       # star / frame priors from _dense
    b[:ns * p + nphi] += b0 - Jm.T @ Wd @ P['y'].ravel()
    N[ns * p + nphi:, ns * p + nphi:] += np.eye(nr)
    cp = P['coupling']; zo = ns * p + int(P['off'][cp['block']]); Kz = cp['G'].shape[2]
    for k, j in enumerate(cp['star']):
        if not act_only[j]:
            continue
        Jc = np.zeros((p, N.shape[0])); Jc[:, j * p:(j + 1) * p] = np.eye(p)
        Gf = np.zeros((p, Kz)); Gf[cp['rows']] = cp['G'][k]; Jc[:, zo:zo + Kz] = Gf
        N += Jc.T @ cp['P'][k] @ Jc; b += Jc.T @ cp['P'][k] @ cp['m'][k]
    return N, b


def test_random_effects_and_coupling_match_dense():
    P, star = _ext_problem()
    prob = J.JointProblem(**P)
    assert prob.reon.sum() > 50 and prob._ncp() > 20
    th, phi = prob.solve(tol=1e-12)
    ns, p = len(P['ptr']) - 1, P['A'].shape[2]; nphi = int(P['sz'].sum())
    N, b = _dense_ext(P, star, P['act'])
    # inactive stars' own parameters are undetermined in the active-only system: give them unit priors there (decoupled)
    for j in np.where(~P['act'])[0]:
        N[j * p:(j + 1) * p, j * p:(j + 1) * p] += np.eye(p)
    x = np.linalg.solve(N, b)
    assert np.allclose(phi, x[ns * p:ns * p + nphi], atol=1e-7 * np.abs(x[ns * p:ns * p + nphi]).max())
    act = P['act']
    assert np.allclose(th[act], x[:ns * p].reshape(ns, p)[act], atol=1e-6)
    # every star given phi (incl. z) from the dense all-star system: theta_j = argmin over (theta_j, r_j) with phi fixed
    Na, ba = _dense_ext(P, star, np.ones(ns, bool))
    gid = P['re_grp']; ug = np.unique(gid[gid >= 0])
    for j in list(np.where(~act)[0][:15]) + list(np.where(act)[0][:15]):
        rr = [ns * p + nphi + 2 * k + c for k, g in enumerate(ug) if (gid[star == j] == g).any() for c in (0, 1)]
        idx = np.r_[np.arange(j * p, (j + 1) * p), np.array(rr, int)]
        Njj = Na[np.ix_(idx, idx)]; Njf = Na[np.ix_(idx, np.arange(ns * p, ns * p + nphi))]
        sol = np.linalg.solve(Njj, ba[idx] - Njf @ phi)
        assert np.allclose(th[j], sol[:p], atol=1e-6)
        # conditional covariance = (theta, r) block inverse marginalised over r
        Ni = np.linalg.inv(Njj)[:p, :p]
        assert np.abs(prob.Ninv[j] - Ni).max() < 1e-6 * np.abs(Ni).max()     # (some blocks ill-conditioned: 1e7 entries)
    # frame samples (incl. z): covariance vs the dense inverse
    Phi = prob.frame_samples(K=2000, batch=500, seed=7, tol=1e-10)
    Cphi = np.linalg.inv(N)[ns * p:ns * p + nphi, ns * p:ns * p + nphi]
    d = np.sqrt(np.diag(Cphi)); rel = (np.cov(Phi) - Cphi) / np.outer(d, d)
    assert np.abs(rel).max() < 0.15
    zs = prob._zslice(); assert np.abs(rel[zs, zs]).max() < 0.15


def test_extensions_off_identical():
    """No random effects / no coupling given: identical to the plain problem (bitwise on the solution)."""
    P, star = _problem(seed=4)
    a = J.JointProblem(**P); ta, pa = a.solve(tol=1e-12)
    n = len(P['y'])
    b = J.JointProblem(**P, re_grp=np.full((n, 2), -1, np.int64), re_lam=np.zeros((n, 2, 2))); tb, pb = b.solve(tol=1e-12)
    assert np.allclose(ta, tb, atol=1e-12) and np.allclose(pa, pb, atol=1e-12)


def test_random_effects_nesting_checked():
    P, star = _ext_problem()
    ex = P['fb'] - int((P['block_kind'] == 0).sum()); ex = np.where(ex < 1000, ex, 0)
    g = P['re_grp'].copy(); on = g[:, 1] >= 0
    g[on, 1] = star[on] * 100 + 50 + (ex[on] + 1) // 4          # seasons shifted by one exposure: night pairs straddle them
    P['re_grp'] = g
    try:
        J.JointProblem(**P)
    except ValueError as e:
        assert 'not nested' in str(e)
    else:
        raise AssertionError('non-nested random effects accepted')
