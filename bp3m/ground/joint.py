"""Joint frame + star astrometric solve: stars eliminated analytically (Schur complement), frame system solved by block-Jacobi
preconditioned conjugate gradients (2026-10-30; design: cfht_sextractor/gd1/JOINT_SOLVE_DESIGN.md).

Model for detection i of star j(i):   y_i = A_i theta_j + F_i phi + eps_i,   eps_i ~ N(0, C_i)   (2-vectors, mas)
  theta_j  star parameters (p per star, e.g. x0, y0, mu_x, mu_y, plx, dc); Gaussian prior with precision Ps_j and mean ms_j
  phi      frame parameters, in blocks.  Each detection touches at most one 'sep' block (one scalar basis Bs_i for both axes,
           coefficients [x part, y part]) and at most one 'full' block (2 x m design D_i).  Block-diagonal prior (Pf, mf).
Normal equations reduced to phi:  S phi = r,   S = Pf + sum_active F^T (W - W A Ninv A^T W) F   (W = C^-1, Ninv = star block inverse).
Only ACTIVE stars' detections enter S; every star (active or passive) is then solved conditional on phi:
  theta_j = Ninv_j (b_j - sum_i A_i^T W_i F_i phi).
Frame uncertainty: perturb-and-solve samples phi* ~ N(0, S^-1) (data noise ~ N(0, C), prior noise ~ N(0, P)) give the frame part of
each star's covariance, B_j Cov(phi) B_j^T with B_j = Ninv_j N_jf (Monte Carlo over K samples); Ninv_j is the conditional part.
Detections are sorted by star (CSR pointer ptr).  Each detection enters once; a detection with used = False enters nothing.

11-02 extensions (both optional; without them every result is identical to 5990721):
* coupled star measurements (`coupling`): for a set of stars c, an extra Gaussian MEASUREMENT of theta_c that shares a global
  random effect z (one frame block, prior N(mf, Pf^-1) like any frame block, touched by no detection):
      m_c = theta_c + G_c z + e_c,   e_c ~ N(0, Pc_c^-1)      (G_c: p x K_z, nonzero only on the rows `rows`)
  e.g. Gaia DR3 (real3 corrected) astrometry of calibrators with the kernel-v3 field modes z ~ N(0, I) (bp3m.gaia_modes): the
  correlated Gaia term enters ONCE, as the shared z, and the per-star e_c carries only the per-star (+ remainder) covariance.
  Normal equations: N_cc += Pc, b_c += Pc m_c, N_cz = Pc G_c, N_zz += G^T Pc G, b_z += G^T Pc m_c.  Passive coupled stars do
  not inform z (like their detections do not inform frames) but are solved given z.
* nested per-star random effects on detections (`re_grp`, `re_lam`): up to L levels of groups (e.g. star-night, star-season,
  star-object), every group belonging to ONE star and every group of level l containing whole groups of the levels below that
  share detections with it (nesting); detection i in group g of level l gets the offset diag(lam_il) r_lg, r_lg ~ N(0, I_2).
  Sigma_star = blockdiag(C_i) + sum_l Z_l Z_l^T is applied exactly by a nested Woodbury recursion per star (O(n_det) per
  application, no stored per-star matrices); C is then the INDEPENDENT part of each detection's covariance.  Equivalent to
  adding the r's to theta_j and eliminating them with the star."""
from __future__ import annotations
import time
import numpy as np
import numba
from numba import njit, prange


@njit(cache=True, inline='always')
def _Fv(i, sb, Bs, fb, D, off, sz, V, k):
    ux = 0.0
    uy = 0.0
    b = sb[i]
    if b >= 0:
        o = off[b]
        m = sz[b] // 2
        for a in range(m):
            ux += Bs[i, a] * V[o + a, k]
            uy += Bs[i, a] * V[o + m + a, k]
    b = fb[i]
    if b >= 0:
        o = off[b]
        for a in range(sz[b]):
            ux += D[i, 0, a] * V[o + a, k]
            uy += D[i, 1, a] * V[o + a, k]
    return ux, uy


@njit(cache=True, inline='always')
def _FTz(i, sb, Bs, fb, D, off, sz, zx, zy, out, k):
    b = sb[i]
    if b >= 0:
        o = off[b]
        m = sz[b] // 2
        for a in range(m):
            out[o + a, k] += Bs[i, a] * zx
            out[o + m + a, k] += Bs[i, a] * zy
    b = fb[i]
    if b >= 0:
        o = off[b]
        for a in range(sz[b]):
            out[o + a, k] += D[i, 0, a] * zx + D[i, 1, a] * zy


@njit(cache=True)
def _level_update(i0, nd, used, rg, rl, m, Ym, Bim, ngm, X):
    """X <- X - Y_m Binv_m (Z_m^T X)   (one level of the nested Woodbury recursion; X (nd, 2, C) over the star's rows)."""
    C = X.shape[2]
    s = np.zeros((ngm, 2, C))
    for t in range(nd):
        i = i0 + t
        g = rg[i, m]
        if g < 0 or not used[i]:
            continue
        for k in range(C):
            s[g, 0, k] += rl[i, m, 0] * X[t, 0, k]
            s[g, 1, k] += rl[i, m, 1] * X[t, 1, k]
    q = np.zeros((ngm, 2, C))
    for g in range(ngm):
        for k in range(C):
            q[g, 0, k] = Bim[g, 0, 0] * s[g, 0, k] + Bim[g, 0, 1] * s[g, 1, k]
            q[g, 1, k] = Bim[g, 1, 0] * s[g, 0, k] + Bim[g, 1, 1] * s[g, 1, k]
    for t in range(nd):
        i = i0 + t
        g = rg[i, m]
        if g < 0 or not used[i]:
            continue
        for k in range(C):
            X[t, 0, k] -= Ym[t, 0, 0] * q[g, 0, k] + Ym[t, 0, 1] * q[g, 1, k]
            X[t, 1, k] -= Ym[t, 1, 0] * q[g, 0, k] + Ym[t, 1, 1] * q[g, 1, k]


@njit(cache=True)
def _sinv(i0, nd, W, used, rg, rl, U):
    """Sigma_star^-1 U for the star's detection rows i0 .. i0 + nd (U (nd, 2, C)); Sigma = blockdiag(C_i over used rows) +
    sum_l Z_l Z_l^T (nested random effects, local group ids rg (n, L), loadings rl (n, L, 2)).  Unused rows: zero output."""
    L = rg.shape[1]
    C = U.shape[2]
    X = np.zeros((nd, 2, C))
    for t in range(nd):
        i = i0 + t
        if not used[i]:
            continue
        for k in range(C):
            X[t, 0, k] = W[i, 0] * U[t, 0, k] + W[i, 2] * U[t, 1, k]
            X[t, 1, k] = W[i, 2] * U[t, 0, k] + W[i, 1] * U[t, 1, k]
    if L == 0:
        return X
    ng = np.zeros(L, np.int64)
    for t in range(nd):
        i = i0 + t
        if not used[i]:
            continue
        for l in range(L):
            if rg[i, l] + 1 > ng[l]:
                ng[l] = rg[i, l] + 1
    mx = 1
    for l in range(L):
        if ng[l] > mx:
            mx = ng[l]
    Y = np.zeros((L, nd, 2, 2))
    Bi = np.zeros((L, mx, 2, 2))
    for l in range(L):
        if ng[l] == 0:
            continue
        Xl = np.zeros((nd, 2, 2))                        # W Z_l (own-group columns)
        for t in range(nd):
            i = i0 + t
            if rg[i, l] < 0 or not used[i]:
                continue
            lx = rl[i, l, 0]
            ly = rl[i, l, 1]
            Xl[t, 0, 0] = W[i, 0] * lx
            Xl[t, 0, 1] = W[i, 2] * ly
            Xl[t, 1, 0] = W[i, 2] * lx
            Xl[t, 1, 1] = W[i, 1] * ly
        for m in range(l):
            if ng[m] > 0:
                _level_update(i0, nd, used, rg, rl, m, Y[m], Bi[m], ng[m], Xl)
        Y[l] = Xl
        B = np.zeros((ng[l], 2, 2))
        for g in range(ng[l]):
            B[g, 0, 0] = 1.0
            B[g, 1, 1] = 1.0
        for t in range(nd):
            i = i0 + t
            g = rg[i, l]
            if g < 0 or not used[i]:
                continue
            for a in range(2):
                for b in range(2):
                    B[g, a, b] += rl[i, l, a] * Xl[t, a, b]
        for g in range(ng[l]):
            a00 = B[g, 0, 0]
            a01 = 0.5 * (B[g, 0, 1] + B[g, 1, 0])
            a11 = B[g, 1, 1]
            det = a00 * a11 - a01 * a01
            Bi[l, g, 0, 0] = a11 / det
            Bi[l, g, 1, 1] = a00 / det
            Bi[l, g, 0, 1] = -a01 / det
            Bi[l, g, 1, 0] = -a01 / det
    for l in range(L):
        if ng[l] > 0:
            _level_update(i0, nd, used, rg, rl, l, Y[l], Bi[l], ng[l], X)
    return X


@njit(parallel=True, cache=True)
def _star_normals(ptr, A, W, used, Y, reon, rg, rl):
    """N_j = sum A^T W A, b_j = sum A^T W Y over used detections (no prior); Sigma_star^-1 instead of W for random-effect stars."""
    ns = len(ptr) - 1
    p = A.shape[2]
    K = Y.shape[2]
    N = np.zeros((ns, p, p))
    b = np.zeros((ns, p, K))
    for j in prange(ns):
        i0 = ptr[j]
        nd = ptr[j + 1] - i0
        if reon[j]:
            U = np.zeros((nd, 2, p + K))
            for t in range(nd):
                for a in range(p):
                    U[t, 0, a] = A[i0 + t, 0, a]
                    U[t, 1, a] = A[i0 + t, 1, a]
                for k in range(K):
                    U[t, 0, p + k] = Y[i0 + t, 0, k]
                    U[t, 1, p + k] = Y[i0 + t, 1, k]
            X = _sinv(i0, nd, W, used, rg, rl, U)
            for t in range(nd):
                i = i0 + t
                if not used[i]:
                    continue
                for a in range(p):
                    for c in range(p):
                        N[j, a, c] += A[i, 0, a] * X[t, 0, c] + A[i, 1, a] * X[t, 1, c]
                    for k in range(K):
                        b[j, a, k] += A[i, 0, a] * X[t, 0, p + k] + A[i, 1, a] * X[t, 1, p + k]
            continue
        for i in range(i0, i0 + nd):
            if not used[i]:
                continue
            for a in range(p):
                wax = W[i, 0] * A[i, 0, a] + W[i, 2] * A[i, 1, a]
                way = W[i, 2] * A[i, 0, a] + W[i, 1] * A[i, 1, a]
                for c in range(p):
                    N[j, a, c] += wax * A[i, 0, c] + way * A[i, 1, c]
                for k in range(K):
                    b[j, a, k] += wax * Y[i, 0, k] + way * Y[i, 1, k]
    return N, b


@njit(parallel=True, cache=True)
def _reduce(ptr, cuts, A, W, used, act, sb, Bs, fb, D, off, sz, Ninv, Y, Bst, buf, reon, rg, rl, cidx, hc):
    """buf[c] = sum over active stars in chunk c of F^T Sigma^-1 (Y_i - A_i Ninv_j Bst_j)  (the reduced right-hand side);
    hc[cidx[j]] = Ninv_j Bst_j for coupled active stars."""
    p = A.shape[2]
    K = Y.shape[2]
    for c in prange(len(cuts) - 1):
        out = buf[c]
        out[:] = 0.0
        for j in range(cuts[c], cuts[c + 1]):
            if not act[j]:
                continue
            h = np.zeros((p, K))
            for a in range(p):
                for e in range(p):
                    for k in range(K):
                        h[a, k] += Ninv[j, a, e] * Bst[j, e, k]
            if cidx[j] >= 0:
                hc[cidx[j]] = h
            i0 = ptr[j]
            nd = ptr[j + 1] - i0
            R = np.zeros((nd, 2, K))
            for t in range(nd):
                i = i0 + t
                for k in range(K):
                    ax = 0.0
                    ay = 0.0
                    for a in range(p):
                        ax += A[i, 0, a] * h[a, k]
                        ay += A[i, 1, a] * h[a, k]
                    R[t, 0, k] = Y[i, 0, k] - ax
                    R[t, 1, k] = Y[i, 1, k] - ay
            if reon[j]:
                X = _sinv(i0, nd, W, used, rg, rl, R)
                for t in range(nd):
                    i = i0 + t
                    if not used[i]:
                        continue
                    for k in range(K):
                        _FTz(i, sb, Bs, fb, D, off, sz, X[t, 0, k], X[t, 1, k], out, k)
                continue
            for t in range(nd):
                i = i0 + t
                if not used[i]:
                    continue
                for k in range(K):
                    rx = R[t, 0, k]
                    ry = R[t, 1, k]
                    _FTz(i, sb, Bs, fb, D, off, sz, W[i, 0] * rx + W[i, 2] * ry, W[i, 2] * rx + W[i, 1] * ry, out, k)


@njit(parallel=True, cache=True)
def _schur_mv(ptr, cuts, A, W, used, act, sb, Bs, fb, D, off, sz, Ninv, V, buf, reon, rg, rl, cidx, gadd, hc):
    """buf[c] = sum over active stars in chunk c of F^T (Sigma^-1 - Sigma^-1 A Ninv A^T Sigma^-1) F V, with the coupled
    measurement's N_jz V_z = gadd[cidx[j]] added to A^T Sigma^-1 F V; hc[cidx[j]] = h_j = Ninv_j (N_jf V) for coupled stars."""
    p = A.shape[2]
    K = V.shape[1]
    for c in prange(len(cuts) - 1):
        out = buf[c]
        out[:] = 0.0
        for j in range(cuts[c], cuts[c + 1]):
            if not act[j]:
                continue
            i0 = ptr[j]
            nd = ptr[j + 1] - i0
            g = np.zeros((p, K))
            if reon[j]:
                U = np.zeros((nd, 2, K))
                for t in range(nd):
                    i = i0 + t
                    if not used[i]:
                        continue
                    for k in range(K):
                        ux, uy = _Fv(i, sb, Bs, fb, D, off, sz, V, k)
                        U[t, 0, k] = ux
                        U[t, 1, k] = uy
                wu = _sinv(i0, nd, W, used, rg, rl, U)
            else:
                wu = np.zeros((nd, 2, K))
                for t in range(nd):
                    i = i0 + t
                    if not used[i]:
                        continue
                    for k in range(K):
                        ux, uy = _Fv(i, sb, Bs, fb, D, off, sz, V, k)
                        wu[t, 0, k] = W[i, 0] * ux + W[i, 2] * uy
                        wu[t, 1, k] = W[i, 2] * ux + W[i, 1] * uy
            for t in range(nd):
                i = i0 + t
                if not used[i]:
                    continue
                for k in range(K):
                    for a in range(p):
                        g[a, k] += A[i, 0, a] * wu[t, 0, k] + A[i, 1, a] * wu[t, 1, k]
            if cidx[j] >= 0:
                for a in range(p):
                    for k in range(K):
                        g[a, k] += gadd[cidx[j], a, k]
            h = np.zeros((p, K))
            for a in range(p):
                for e in range(p):
                    for k in range(K):
                        h[a, k] += Ninv[j, a, e] * g[e, k]
            if cidx[j] >= 0:
                hc[cidx[j]] = h
            if reon[j]:
                Ah = np.zeros((nd, 2, K))
                for t in range(nd):
                    i = i0 + t
                    for k in range(K):
                        for a in range(p):
                            Ah[t, 0, k] += A[i, 0, a] * h[a, k]
                            Ah[t, 1, k] += A[i, 1, a] * h[a, k]
                wah = _sinv(i0, nd, W, used, rg, rl, Ah)
                for t in range(nd):
                    i = i0 + t
                    if not used[i]:
                        continue
                    for k in range(K):
                        _FTz(i, sb, Bs, fb, D, off, sz, wu[t, 0, k] - wah[t, 0, k], wu[t, 1, k] - wah[t, 1, k], out, k)
                continue
            for t in range(nd):
                i = i0 + t
                if not used[i]:
                    continue
                for k in range(K):
                    ax = 0.0
                    ay = 0.0
                    for a in range(p):
                        ax += A[i, 0, a] * h[a, k]
                        ay += A[i, 1, a] * h[a, k]
                    zx = wu[t, 0, k] - (W[i, 0] * ax + W[i, 2] * ay)
                    zy = wu[t, 1, k] - (W[i, 2] * ax + W[i, 1] * ay)
                    _FTz(i, sb, Bs, fb, D, off, sz, zx, zy, out, k)


@njit(parallel=True, cache=True)
def _stars_given(ptr, A, W, used, sb, Bs, fb, D, off, sz, Ninv, Bst, Phi, cov_only, reon, rg, rl, cidx, gadd):
    """theta_j = Ninv_j (Bst_j - N_jf Phi) for every star (K columns), N_jf Phi = sum A^T Sigma^-1 F Phi (+ gadd[cidx[j]] for a
    coupled star).  cov_only: Bst ignored, returns (1/K) sum_k t t^T with t = -Ninv_j N_jf phi_k (the frame part of the covariance)."""
    ns = len(ptr) - 1
    p = A.shape[2]
    K = Phi.shape[1]
    th = np.zeros((ns, p, K)) if not cov_only else np.zeros((1, p, K))
    cov = np.zeros((ns, p, p)) if cov_only else np.zeros((1, p, p))
    for j in prange(ns):
        g = np.zeros((p, K))
        i0 = ptr[j]
        nd = ptr[j + 1] - i0
        if reon[j]:
            U = np.zeros((nd, 2, K))
            for t in range(nd):
                i = i0 + t
                if not used[i]:
                    continue
                for k in range(K):
                    ux, uy = _Fv(i, sb, Bs, fb, D, off, sz, Phi, k)
                    U[t, 0, k] = ux
                    U[t, 1, k] = uy
            X = _sinv(i0, nd, W, used, rg, rl, U)
            for t in range(nd):
                i = i0 + t
                if not used[i]:
                    continue
                for k in range(K):
                    for a in range(p):
                        g[a, k] += A[i, 0, a] * X[t, 0, k] + A[i, 1, a] * X[t, 1, k]
        else:
            for i in range(i0, i0 + nd):
                if not used[i]:
                    continue
                for k in range(K):
                    ux, uy = _Fv(i, sb, Bs, fb, D, off, sz, Phi, k)
                    wx = W[i, 0] * ux + W[i, 2] * uy
                    wy = W[i, 2] * ux + W[i, 1] * uy
                    for a in range(p):
                        g[a, k] += A[i, 0, a] * wx + A[i, 1, a] * wy
        if cidx[j] >= 0:
            for a in range(p):
                for k in range(K):
                    g[a, k] += gadd[cidx[j], a, k]
        t_ = np.zeros((p, K))
        for a in range(p):
            for e in range(p):
                for k in range(K):
                    if cov_only:
                        t_[a, k] -= Ninv[j, a, e] * g[e, k]
                    else:
                        t_[a, k] += Ninv[j, a, e] * (Bst[j, e, k] - g[e, k])
        if cov_only:
            for a in range(p):
                for e in range(p):
                    s = 0.0
                    for k in range(K):
                        s += t_[a, k] * t_[e, k]
                    cov[j, a, e] = s / K
        else:
            th[j] = t_
    return th, cov


@njit(parallel=True, cache=True)
def _resid(ptr, A, y, W, used, sb, Bs, fb, D, off, sz, Ninv, theta, phi):
    """Per detection: residual r = y - A theta - F phi, chi2 = r^T W r, M = A Ninv A^T (xx, yy, xy), leverage tr(W M)."""
    n = A.shape[0]
    p = A.shape[2]
    r = np.zeros((n, 2))
    chi2 = np.zeros(n)
    M = np.zeros((n, 3))
    lev = np.zeros(n)
    for j in prange(len(ptr) - 1):
        for i in range(ptr[j], ptr[j + 1]):
            ux, uy = _Fv(i, sb, Bs, fb, D, off, sz, phi, 0)
            ax = 0.0
            ay = 0.0
            for a in range(p):
                ax += A[i, 0, a] * theta[j, a]
                ay += A[i, 1, a] * theta[j, a]
            rx = y[i, 0] - ax - ux
            ry = y[i, 1] - ay - uy
            r[i, 0] = rx
            r[i, 1] = ry
            chi2[i] = W[i, 0] * rx * rx + 2 * W[i, 2] * rx * ry + W[i, 1] * ry * ry
            mxx = 0.0
            myy = 0.0
            mxy = 0.0
            for a in range(p):
                for e in range(p):
                    q = Ninv[j, a, e]
                    mxx += A[i, 0, a] * q * A[i, 0, e]
                    myy += A[i, 1, a] * q * A[i, 1, e]
                    mxy += A[i, 0, a] * q * A[i, 1, e]
            M[i, 0] = mxx
            M[i, 1] = myy
            M[i, 2] = mxy
            lev[i] = W[i, 0] * mxx + W[i, 1] * myy + 2 * W[i, 2] * mxy
    return r, chi2, M, lev


@njit(parallel=True, cache=True)
def _accum_blocks(blocks, order, bptr, is_sep, Bs, D, H, sz, foff, flat):
    """flat[block] += sum_i F_i^T H_i F_i over the block's detections (dets sorted by block: order, bptr)."""
    for q in prange(len(blocks)):
        b = blocks[q]
        m = sz[b]
        o = foff[b]
        for t in range(bptr[q], bptr[q + 1]):
            i = order[t]
            hxx = H[i, 0]
            hyy = H[i, 1]
            hxy = H[i, 2]
            if hxx == 0.0 and hyy == 0.0 and hxy == 0.0:
                continue
            if is_sep:
                mh = m // 2
                for a in range(mh):
                    for a2 in range(mh):
                        bb = Bs[i, a] * Bs[i, a2]
                        flat[o + a * m + a2] += bb * hxx
                        flat[o + a * m + mh + a2] += bb * hxy
                        flat[o + (mh + a) * m + a2] += bb * hxy
                        flat[o + (mh + a) * m + mh + a2] += bb * hyy
            else:
                for a in range(m):
                    dx = D[i, 0, a]
                    dy = D[i, 1, a]
                    for a2 in range(m):
                        ex = D[i, 0, a2]
                        ey = D[i, 1, a2]
                        flat[o + a * m + a2] += dx * (hxx * ex + hxy * ey) + dy * (hxy * ex + hyy * ey)


def _sym_inv(N):
    """Batched inverse of symmetric positive-definite (n, p, p) via Cholesky (raises on a non-PD block)."""
    L = np.linalg.cholesky(N)
    Li = np.linalg.inv(L)
    return np.einsum('nji,njk->nik', Li, Li)


class BlockDiag:
    """Block-diagonal matrix over the frame parameter vector (blocks at off[b], size sz[b]; flat storage at foff[b])."""
    def __init__(self, off, sz, flat=None):
        self.off, self.sz = off, sz
        self.foff = np.r_[0, np.cumsum(sz.astype(np.int64) ** 2)]
        self.flat = np.zeros(self.foff[-1]) if flat is None else flat
        self.groups = {int(s): np.where(sz == s)[0] for s in np.unique(sz)}

    def _idx(self, s, bl):
        return self.foff[bl][:, None] + np.arange(s * s)[None, :], self.off[bl][:, None] + np.arange(s)[None, :]

    def blocks(self, s):
        fi, _ = self._idx(s, self.groups[s])
        return self.flat[fi].reshape(-1, s, s)

    def set_blocks(self, s, M):
        fi, _ = self._idx(s, self.groups[s])
        self.flat[fi] = M.reshape(len(fi), -1)

    def block(self, b):
        s = int(self.sz[b]); return self.flat[self.foff[b]:self.foff[b] + s * s].reshape(s, s)

    def set_block(self, b, M):
        s = int(self.sz[b]); self.flat[self.foff[b]:self.foff[b] + s * s] = np.asarray(M, float).reshape(-1)

    def matvec(self, V):
        out = np.zeros_like(V)
        for s, bl in self.groups.items():
            fi, vi = self._idx(s, bl)
            out[vi] = np.einsum('bij,bjk->bik', self.flat[fi].reshape(-1, s, s), V[vi])
        return out

    def map_blocks(self, fn):
        out = BlockDiag(self.off, self.sz)
        for s, bl in self.groups.items():
            out.set_blocks(s, fn(self.blocks(s)))
        return out

    def __add__(self, other):
        return BlockDiag(self.off, self.sz, self.flat + other.flat)


def _local_groups(ptr, grp):
    """Global group ids (n, L) -> per-star local ids 0 .. n_lj - 1 (order of first appearance), -1 kept."""
    n, L = grp.shape
    star = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    out = np.full((n, L), -1, np.int32)
    for l in range(L):
        g = grp[:, l]; ok = g >= 0
        if not ok.any():
            continue
        key = np.c_[star[ok], g[ok]]
        u, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        gs = u[:, 0]                                                 # groups sorted by (star, gid); rank within the star
        start = np.searchsorted(gs, gs, side='left')
        out[ok, l] = (np.arange(len(u)) - start)[inv.ravel()]
        # every group id must belong to exactly one star
        if len(np.unique(u[:, 1])) != len(u):
            raise ValueError(f'random-effect level {l}: a group id spans more than one star')
    return out


def _check_nesting(grp):
    """The nested Woodbury needs every group of level l to lie inside ONE group of each higher level m > l (where it has one):
    raises otherwise (e.g. a night straddling a season boundary)."""
    L = grp.shape[1]
    for l in range(L):
        for m in range(l + 1, L):
            ok = (grp[:, l] >= 0) & (grp[:, m] >= 0)
            if not ok.any():
                continue
            pairs = np.unique(np.c_[grp[ok, l], grp[ok, m]], axis=0)
            if len(np.unique(pairs[:, 0])) != len(pairs):
                raise ValueError(f'random effects: a level-{l} group spans several level-{m} groups (not nested)')
            # and a level-l group may not be partly inside / partly outside its level-m group
            hasm = pd_bincount_any(grp[:, l], grp[:, m] >= 0)
            if hasm is not None and hasm:
                raise ValueError(f'random effects: a level-{l} group has detections with and without a level-{m} group')


def pd_bincount_any(g, flag):
    """True if some group id in g (>= 0) has both flag True and flag False rows."""
    ok = g >= 0
    if not ok.any():
        return None
    u, inv = np.unique(g[ok], return_inverse=True)
    t = np.bincount(inv.ravel(), flag[ok].astype(float), minlength=len(u)); c = np.bincount(inv.ravel(), minlength=len(u))
    return bool(((t > 0) & (t < c)).any())


class JointProblem:
    """Arrays (n detections sorted by star; ns stars; nphi frame parameters in nb blocks):
      ptr (ns+1) int64; A (n, 2, p); y (n, 2); C (n, 3) = (cxx, cyy, cxy) measurement covariance (the independent part when
      random effects are given)
      sb (n) sep block (-1 none), Bs (n, mb) basis; fb (n) full block (-1 none), D (n, 2, mf) design
      off (nb), sz (nb) block offsets / sizes in phi; block_kind (nb) 0 sep / 1 full
      Ps (ns, p, p), ms (ns, p) star priors (Ps + coupled precision must make every N_j positive definite); Pf (BlockDiag),
      mf (nphi) frame prior; used (n) bool, act (ns) bool.
      re_grp (n, L) int64 global random-effect group ids (-1 none; each group within one star; nested), re_lam (n, L, 2) loadings.
      coupling: dict(star (nc,) int, P (nc, p, p), m (nc, p), G (nc, r, Kz), rows (r,) int, block (int)) - see module doc."""

    def __init__(self, ptr, A, y, C, sb, Bs, fb, D, off, sz, block_kind, Ps, ms, Pf, mf, used=None, act=None, nchunk=None, log=print,
                 re_grp=None, re_lam=None, coupling=None):
        self.ptr = np.ascontiguousarray(ptr, np.int64)
        self.A, self.y = np.ascontiguousarray(A, float), np.ascontiguousarray(y, float)
        self.sb, self.fb = np.ascontiguousarray(sb, np.int64), np.ascontiguousarray(fb, np.int64)
        self.Bs, self.D = np.ascontiguousarray(Bs, float), np.ascontiguousarray(D, float)
        self.off, self.sz, self.kind = np.asarray(off, np.int64), np.asarray(sz, np.int64), np.asarray(block_kind, np.int64)
        self.nphi = int(self.off[-1] + self.sz[-1]) if len(self.off) else 0
        self.Ps, self.ms, self.Pf, self.mf = Ps, ms, Pf, mf
        n, ns = len(self.y), len(self.ptr) - 1
        self.used = np.ones(n, bool) if used is None else np.asarray(used, bool).copy()
        self.act = np.ones(ns, bool) if act is None else np.asarray(act, bool).copy()
        self.log = log
        p = self.A.shape[2]
        # nested random effects
        if re_grp is None:
            self.rg = np.zeros((n, 0), np.int32); self.rl = np.zeros((n, 0, 2)); self.re_gid = np.zeros((n, 0), np.int64)
            self.reon = np.zeros(ns, bool)
        else:
            self.re_gid = np.ascontiguousarray(re_grp, np.int64)
            self.rg = np.ascontiguousarray(_local_groups(self.ptr, self.re_gid))
            self.rl = np.ascontiguousarray(re_lam, float)
            if self.rl.shape != (n, self.re_gid.shape[1], 2):
                raise ValueError('re_lam must be (n, L, 2)')
            star = np.repeat(np.arange(ns), np.diff(self.ptr))
            self.reon = np.bincount(star, (self.rg >= 0).any(1).astype(float), minlength=ns) > 0
            _check_nesting(self.re_gid)
        # coupled measurements
        self.cp = coupling
        self.cidx = np.full(ns, -1, np.int64)
        if coupling is not None:
            cs = np.asarray(coupling['star'], np.int64)
            if len(np.unique(cs)) != len(cs):
                raise ValueError('coupling: duplicate stars')
            self.cidx[cs] = np.arange(len(cs))
            self.cG = np.ascontiguousarray(coupling['G'], float); self.crows = np.asarray(coupling['rows'], np.int64)
            self.cP = np.ascontiguousarray(coupling['P'], float); self.cm = np.ascontiguousarray(coupling['m'], float)
            self.cz = int(coupling['block']); self.cKz = int(self.sz[self.cz])
            if self.cG.shape != (len(cs), len(self.crows), self.cKz) or self.cP.shape != (len(cs), p, p) or self.cm.shape != (len(cs), p):
                raise ValueError('coupling: shapes')
            for k in (0, 1):
                bi = self.sb if k == 0 else self.fb
                if np.any(bi == self.cz):
                    raise ValueError('coupling: the global block must not be touched by any detection')
            self.cact = self.act[cs]
        self.set_cov(C)
        nchunk = nchunk or 4 * numba.get_num_threads()
        cut_at = np.linspace(0, n, nchunk + 1)
        self.cuts = np.unique(np.searchsorted(self.ptr, cut_at)).astype(np.int64)
        self.cuts[0], self.cuts[-1] = 0, ns
        self._border = {}
        for kind in (0, 1):
            bi = self.sb if kind == 0 else self.fb
            o = np.argsort(bi, kind='stable'); o = o[bi[o] >= 0]
            bl, cnt = np.unique(bi[o], return_counts=True)
            self._border[kind] = (bl.astype(np.int64), o.astype(np.int64), np.r_[0, np.cumsum(cnt)].astype(np.int64))

    def set_cov(self, C):
        self.C = np.ascontiguousarray(C, float)
        det = self.C[:, 0] * self.C[:, 1] - self.C[:, 2] ** 2
        if not np.all(det > 0):
            raise ValueError(f'{int((det <= 0).sum())} detections with a non-positive-definite covariance')
        self.W = np.ascontiguousarray(np.c_[self.C[:, 1] / det, self.C[:, 0] / det, -self.C[:, 2] / det])

    # --- coupled-measurement helpers (z = the global block)
    def _ncp(self):
        return 0 if self.cp is None else len(self.cidx[self.cidx >= 0])

    def _zslice(self):
        o = int(self.off[self.cz]); return slice(o, o + self.cKz)

    def _Gfull_times(self, Vz):
        """G_c V_z as (nc, p, K)."""
        nc, r, Kz = self.cG.shape
        out = np.zeros((nc, self.A.shape[2], Vz.shape[1]))
        out[:, self.crows] = (self.cG.reshape(nc * r, Kz) @ Vz).reshape(nc, r, -1)
        return out

    def _GT_times(self, T):
        """sum_c G_c^T T_c[rows]  (T (nc, p, K)) -> (Kz, K)."""
        nc, r, Kz = self.cG.shape
        return self.cG.reshape(nc * r, Kz).T @ np.ascontiguousarray(T[:, self.crows]).reshape(nc * r, -1)

    def _gadd(self, V):
        if self.cp is None:
            return np.zeros((1, self.A.shape[2], V.shape[1]))
        return np.ascontiguousarray(np.einsum('cij,cjk->cik', self.cP, self._Gfull_times(V[self._zslice()])))

    def _hc(self, K):
        return np.zeros((max(self._ncp(), 1), self.A.shape[2], K))

    # --- star blocks
    def star_blocks(self):
        N, b = _star_normals(self.ptr, self.A, self.W, self.used, self.y[:, :, None], self.reon, self.rg, self.rl)
        N += self.Ps
        b = b[:, :, 0] + np.einsum('nij,nj->ni', self.Ps, self.ms)
        if self.cp is not None:
            cs = np.where(self.cidx >= 0)[0]
            N[cs] += self.cP; b[cs] += np.einsum('cij,cj->ci', self.cP, self.cm)
        self.Nss = N
        self.Ninv = _sym_inv(N)
        self.bs = b

    def _buf(self, K):
        return np.zeros((len(self.cuts) - 1, self.nphi, K))

    def schur_mv(self, V):
        V = np.ascontiguousarray(V)
        buf = self._buf(V.shape[1]); hc = self._hc(V.shape[1]); gadd = self._gadd(V)
        _schur_mv(self.ptr, self.cuts, self.A, self.W, self.used, self.act, self.sb, self.Bs, self.fb, self.D, self.off, self.sz,
                  self.Ninv, V, buf, self.reon, self.rg, self.rl, self.cidx, gadd, hc)
        out = buf.sum(0) + self.Pf.matvec(V)
        if self.cp is not None:                         # z part: sum_c G^T Pc (G V_z - h_c) over ACTIVE coupled stars
            T = np.einsum('cij,cjk->cik', self.cP, self._Gfull_times(V[self._zslice()]) - hc) * self.cact[:, None, None]
            out[self._zslice()] += self._GT_times(T)
        return out

    def reduced_rhs(self, Y, Bst, prior_term, cdata=None):
        """cdata (nc, p, K): the coupled measurements' Pc m (MAP) or their noise draws (samples); Bst must already contain them."""
        buf = self._buf(Y.shape[2]); hc = self._hc(Y.shape[2])
        _reduce(self.ptr, self.cuts, self.A, self.W, self.used, self.act, self.sb, self.Bs, self.fb, self.D, self.off, self.sz,
                self.Ninv, np.ascontiguousarray(Y), np.ascontiguousarray(Bst), buf, self.reon, self.rg, self.rl, self.cidx, hc)
        out = buf.sum(0) + prior_term
        if self.cp is not None:
            T = (cdata - np.einsum('cij,cjk->cik', self.cP, hc)) * self.cact[:, None, None]
            out[self._zslice()] += self._GT_times(T)
        return out

    def preconditioner(self):
        """Exact Schur diagonal blocks (sep and full blocks separately; cross terms ignored; random effects ignored) + prior,
        inverted.  The global coupled block: sum_c G^T (Pc - Pc Ninv_c Pc) G over active coupled stars."""
        Ninv = self.Ninv
        if self.reon.any():                            # random-effect stars: H = W - W A N0^-1 A^T W with the independent-noise
            N0, _ = _star_normals(self.ptr, self.A, self.W, self.used, np.zeros((len(self.y), 2, 1)),     # star block N0 (PSD H;
                                  np.zeros(len(self.ptr) - 1, bool), self.rg, self.rl)                     # the RE Ninv is larger)
            N0 = N0 + self.Ps
            if self.cp is not None:
                cs = np.where(self.cidx >= 0)[0]; N0[cs] += self.cP
            Ninv = np.where(self.reon[:, None, None], _sym_inv(N0), self.Ninv); del N0
        _, _, M, _ = _resid(self.ptr, self.A, self.y, self.W, self.used, self.sb, self.Bs, self.fb, self.D, self.off, self.sz,
                            Ninv, np.zeros((len(self.ptr) - 1, self.A.shape[2])), np.zeros((self.nphi, 1)))
        del Ninv
        w = self.W; on = (self.used & np.repeat(self.act, np.diff(self.ptr))).astype(float)
        # H = W - W M W  (2x2 symmetric per detection), zero for detections outside S
        wm = np.stack([np.stack([w[:, 0] * M[:, 0] + w[:, 2] * M[:, 2], w[:, 0] * M[:, 2] + w[:, 2] * M[:, 1]], -1),
                       np.stack([w[:, 2] * M[:, 0] + w[:, 1] * M[:, 2], w[:, 2] * M[:, 2] + w[:, 1] * M[:, 1]], -1)], -2)
        Wm = np.stack([np.stack([w[:, 0], w[:, 2]], -1), np.stack([w[:, 2], w[:, 1]], -1)], -2)
        WMW = wm @ Wm
        H = np.c_[w[:, 0] - WMW[:, 0, 0], w[:, 1] - WMW[:, 1, 1], w[:, 2] - WMW[:, 0, 1]] * on[:, None]
        P = BlockDiag(self.off, self.sz)
        for kind in (0, 1):
            bl, o, bp = self._border[kind]
            if len(bl):
                _accum_blocks(bl, o, bp, kind == 0, self.Bs, self.D, np.ascontiguousarray(H), self.sz, P.foff, P.flat)
        if self.cp is not None:
            cs = np.where(self.cidx >= 0)[0]
            Hc = (self.cP - np.einsum('cij,cjk,ckl->cil', self.cP, self.Ninv[cs], self.cP)) * self.cact[:, None, None]
            Hr = Hc[:, self.crows][:, :, self.crows]
            nc, r, Kz = self.cG.shape
            X = np.einsum('cij,cjk->cik', Hr, self.cG).reshape(nc * r, Kz)
            P.set_block(self.cz, P.block(self.cz) + self.cG.reshape(nc * r, Kz).T @ X)
        return (P + self.Pf).map_blocks(_sym_inv)

    def pcg(self, rhs, X0=None, tol=1e-8, maxit=2000, Minv=None, label=""):
        Minv = Minv or self.preconditioner()
        X = np.zeros_like(rhs) if X0 is None else X0.copy()
        R = rhs - self.schur_mv(X) if X0 is not None else rhs.copy()
        Z = Minv.matvec(R); P = Z.copy(); rz = (R * Z).sum(0)
        nr0 = np.linalg.norm(rhs, axis=0) + 1e-300; t0 = time.time()
        for it in range(maxit):
            Q = self.schur_mv(P)
            alpha = rz / (P * Q).sum(0)
            X += alpha * P; R -= alpha * Q
            rel = np.linalg.norm(R, axis=0) / nr0
            if it % 10 == 0 or rel.max() < tol:
                self.log(f'  pcg{label} it {it + 1}: max rel residual {rel.max():.2e} ({time.time() - t0:.0f} s)')
            if rel.max() < tol:
                break
            Z = Minv.matvec(R); rzn = (R * Z).sum(0); P = Z + (rzn / rz) * P; rz = rzn
        else:
            raise RuntimeError(f'pcg{label}: not converged after {maxit} iterations (max rel residual {rel.max():.2e})')
        return X, it + 1

    # --- solves
    def _cdata_map(self):
        return None if self.cp is None else np.einsum('cij,cj->ci', self.cP, self.cm)[:, :, None]

    def solve(self, phi0=None, tol=1e-8, fixed_phi=None):
        """Joint MAP solve.  fixed_phi: frames held at that value (no frame solve; e.g. zeros = the two-step reproduction)."""
        self.star_blocks()
        if fixed_phi is not None:
            self.phi = np.asarray(fixed_phi, float).reshape(-1, 1)
            self.pcg_iters = 0
        else:
            rhs = self.reduced_rhs(self.y[:, :, None], self.bs[:, :, None], self.Pf.matvec(self.mf[:, None]), self._cdata_map())
            self.Minv = self.preconditioner()
            self.phi, self.pcg_iters = self.pcg(rhs, X0=None if phi0 is None else phi0.reshape(-1, 1), tol=tol, Minv=self.Minv)
        th, _ = _stars_given(self.ptr, self.A, self.W, self.used, self.sb, self.Bs, self.fb, self.D, self.off, self.sz, self.Ninv,
                             self.bs[:, :, None], self.phi, False, self.reon, self.rg, self.rl, self.cidx, self._gadd(self.phi))
        self.theta = th[:, :, 0]
        return self.theta, self.phi[:, 0]

    def residuals(self):
        """r, chi2, M, lev per detection.  With random effects chi2 uses each detection's MARGINAL covariance C + sum_l diag(lam^2)."""
        r, chi2, M, lev = _resid(self.ptr, self.A, self.y, self.W, self.used, self.sb, self.Bs, self.fb, self.D, self.off, self.sz,
                                 self.Ninv, self.theta, self.phi)
        if self.rg.shape[1]:
            add = ((self.rg >= 0)[:, :, None] * self.rl ** 2).sum(1)
            a, d, c = self.C[:, 0] + add[:, 0], self.C[:, 1] + add[:, 1], self.C[:, 2]
            chi2 = (d * r[:, 0] ** 2 - 2 * c * r[:, 0] * r[:, 1] + a * r[:, 1] ** 2) / (a * d - c * c)
        return r, chi2, M, lev

    def frame_samples(self, K=64, batch=16, seed=0, tol=1e-6):
        """phi* ~ N(0, S^-1) by perturb-and-solve (active stars' data noise incl. their random effects + their prior noise +
        coupled-measurement noise + frame prior noise)."""
        rng = np.random.default_rng(seed)
        n = len(self.y); l11 = np.sqrt(self.C[:, 0]); l21 = self.C[:, 2] / l11; l22 = np.sqrt(self.C[:, 1] - l21 ** 2)
        Ls = np.linalg.cholesky(self.Ps); Lf = self.Pf.map_blocks(np.linalg.cholesky)
        Lc = np.linalg.cholesky(self.cP) if self.cp is not None else None
        on = self.used & np.repeat(self.act, np.diff(self.ptr))
        Minv = getattr(self, 'Minv', None) or self.preconditioner()
        L = self.re_gid.shape[1]
        out = []
        for k0 in range(0, K, batch):
            k = min(batch, K - k0)
            xi = rng.standard_normal((n, 2, k)) * on[:, None, None]
            Y = np.stack([l11[:, None] * xi[:, 0], l21[:, None] * xi[:, 0] + l22[:, None] * xi[:, 1]], 1)
            del xi
            for l in range(L):                                       # random effects: one N(0, I_2) draw per group
                g = self.re_gid[:, l]; ok = (g >= 0) & on
                if not ok.any():
                    continue
                ug, inv = np.unique(g[ok], return_inverse=True)
                rr = rng.standard_normal((len(ug), 2, k))[inv.ravel()]
                Y[ok] += self.rl[ok, l, :, None] * rr
            _, b = _star_normals(self.ptr, self.A, self.W, self.used, np.ascontiguousarray(Y), self.reon, self.rg, self.rl)
            b += np.einsum('nij,njk->nik', Ls, rng.standard_normal((len(Ls), Ls.shape[1], k)))
            cdata = None
            if self.cp is not None:
                cs = np.where(self.cidx >= 0)[0]
                cdata = np.einsum('cij,cjk->cik', Lc, rng.standard_normal((len(cs), Lc.shape[1], k)))   # Pc e,  e ~ N(0, Pc^-1)
                b[cs] += cdata
            prior = Lf.matvec(rng.standard_normal((self.nphi, k)))
            rhs = self.reduced_rhs(Y, b, prior, cdata)
            del Y, b
            X, it = self.pcg(rhs, tol=tol, Minv=Minv, label=f' samples {k0}-{k0 + k}')
            out.append(X)
        return np.concatenate(out, 1)

    def frame_cov(self, Phi):
        """Per-star frame part of the covariance, (1/K) sum_k (B_j phi_k)(B_j phi_k)^T, for every star (incl. the coupled block)."""
        Phi = np.ascontiguousarray(Phi)
        _, cov = _stars_given(self.ptr, self.A, self.W, self.used, self.sb, self.Bs, self.fb, self.D, self.off, self.sz, self.Ninv,
                              np.zeros((1, self.A.shape[2], 1)), Phi, True, self.reon, self.rg, self.rl, self.cidx, self._gadd(Phi))
        return cov


def block_floor(blk, r, Cm, lev, sel, nblk, f2_max=400.0, n_min=10, iters=50):
    """Per-block additive variance f^2 (mas^2, per axis) such that the selected detections' residuals are consistent:
    sum r^T (Cm + f^2 I)^-1 r = sum (2 - lev).  lev = star-parameter leverage tr(W A Ninv A^T) at the current weights (the frame
    leverage, ~ block size / detections per block, is neglected).  Blocks with fewer than n_min selected detections -> NaN."""
    b = blk[sel]; rx, ry = r[sel, 0], r[sel, 1]; cxx, cyy, cxy = Cm[sel, 0], Cm[sel, 1], Cm[sel, 2]
    target = np.bincount(b, np.clip(2.0 - lev[sel], 0.0, 2.0), minlength=nblk)
    cnt = np.bincount(b, minlength=nblk)
    lo, hi = np.zeros(nblk), np.full(nblk, f2_max)
    def g(f2):
        f = f2[b]; a, d, c = cxx + f, cyy + f, cxy; det = a * d - c * c
        return np.bincount(b, (d * rx * rx - 2 * c * rx * ry + a * ry * ry) / det, minlength=nblk)
    at0 = g(lo) <= target                              # already consistent without a floor
    for _ in range(iters):
        mid = 0.5 * (lo + hi); over = g(mid) > target
        lo = np.where(over, mid, lo); hi = np.where(over, hi, mid)
    f2 = np.where(at0, 0.0, 0.5 * (lo + hi))
    return np.where(cnt >= n_min, f2, np.nan), cnt
