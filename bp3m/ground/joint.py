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
Detections are sorted by star (CSR pointer ptr).  Each detection enters once; a detection with used = False enters nothing."""
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


@njit(parallel=True, cache=True)
def _star_normals(ptr, A, W, used, Y):
    """N_j = sum A^T W A, b_j = sum A^T W Y over used detections (no prior)."""
    ns = len(ptr) - 1
    p = A.shape[2]
    K = Y.shape[2]
    N = np.zeros((ns, p, p))
    b = np.zeros((ns, p, K))
    for j in prange(ns):
        for i in range(ptr[j], ptr[j + 1]):
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
def _reduce(ptr, cuts, A, W, used, act, sb, Bs, fb, D, off, sz, Ninv, Y, Bst, buf):
    """buf[c] = sum over active stars in chunk c of F^T (W Y_i - W A_i Ninv_j Bst_j)  (the reduced right-hand side)."""
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
            for i in range(ptr[j], ptr[j + 1]):
                if not used[i]:
                    continue
                for k in range(K):
                    ax = 0.0
                    ay = 0.0
                    for a in range(p):
                        ax += A[i, 0, a] * h[a, k]
                        ay += A[i, 1, a] * h[a, k]
                    rx = Y[i, 0, k] - ax
                    ry = Y[i, 1, k] - ay
                    _FTz(i, sb, Bs, fb, D, off, sz, W[i, 0] * rx + W[i, 2] * ry, W[i, 2] * rx + W[i, 1] * ry, out, k)


@njit(parallel=True, cache=True)
def _schur_mv(ptr, cuts, A, W, used, act, sb, Bs, fb, D, off, sz, Ninv, V, buf):
    """buf[c] = sum over active stars in chunk c of F^T (W - W A Ninv A^T W) F V."""
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
            wu = np.zeros((nd, 2, K))
            g = np.zeros((p, K))
            for t in range(nd):
                i = i0 + t
                if not used[i]:
                    continue
                for k in range(K):
                    ux, uy = _Fv(i, sb, Bs, fb, D, off, sz, V, k)
                    wx = W[i, 0] * ux + W[i, 2] * uy
                    wy = W[i, 2] * ux + W[i, 1] * uy
                    wu[t, 0, k] = wx
                    wu[t, 1, k] = wy
                    for a in range(p):
                        g[a, k] += A[i, 0, a] * wx + A[i, 1, a] * wy
            h = np.zeros((p, K))
            for a in range(p):
                for e in range(p):
                    for k in range(K):
                        h[a, k] += Ninv[j, a, e] * g[e, k]
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
def _stars_given(ptr, A, W, used, sb, Bs, fb, D, off, sz, Ninv, Bst, Phi, cov_only):
    """theta_j = Ninv_j (Bst_j - sum A^T W F Phi) for every star (K columns).  cov_only: Bst ignored, returns
    (1/K) sum_k t t^T with t = -Ninv_j sum A^T W F phi_k (the frame part of the covariance) in the first output."""
    ns = len(ptr) - 1
    p = A.shape[2]
    K = Phi.shape[1]
    th = np.zeros((ns, p, K)) if not cov_only else np.zeros((1, p, K))
    cov = np.zeros((ns, p, p)) if cov_only else np.zeros((1, p, p))
    for j in prange(ns):
        g = np.zeros((p, K))
        for i in range(ptr[j], ptr[j + 1]):
            if not used[i]:
                continue
            for k in range(K):
                ux, uy = _Fv(i, sb, Bs, fb, D, off, sz, Phi, k)
                wx = W[i, 0] * ux + W[i, 2] * uy
                wy = W[i, 2] * ux + W[i, 1] * uy
                for a in range(p):
                    g[a, k] += A[i, 0, a] * wx + A[i, 1, a] * wy
        t = np.zeros((p, K))
        for a in range(p):
            for e in range(p):
                for k in range(K):
                    if cov_only:
                        t[a, k] -= Ninv[j, a, e] * g[e, k]
                    else:
                        t[a, k] += Ninv[j, a, e] * (Bst[j, e, k] - g[e, k])
        if cov_only:
            for a in range(p):
                for e in range(p):
                    s = 0.0
                    for k in range(K):
                        s += t[a, k] * t[e, k]
                    cov[j, a, e] = s / K
        else:
            th[j] = t
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


class JointProblem:
    """Arrays (n detections sorted by star; ns stars; nphi frame parameters in nb blocks):
      ptr (ns+1) int64; A (n, 2, p); y (n, 2); C (n, 3) = (cxx, cyy, cxy) measurement covariance
      sb (n) sep block (-1 none), Bs (n, mb) basis; fb (n) full block (-1 none), D (n, 2, mf) design
      off (nb), sz (nb) block offsets / sizes in phi; block_kind (nb) 0 sep / 1 full
      Ps (ns, p, p), ms (ns, p) star priors (Ps must make every N_j positive definite); Pf (BlockDiag), mf (nphi) frame prior
      used (n) bool, act (ns) bool."""

    def __init__(self, ptr, A, y, C, sb, Bs, fb, D, off, sz, block_kind, Ps, ms, Pf, mf, used=None, act=None, nchunk=None, log=print):
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

    # --- star blocks
    def star_blocks(self):
        N, b = _star_normals(self.ptr, self.A, self.W, self.used, self.y[:, :, None])
        N += self.Ps
        self.Nss = N
        self.Ninv = _sym_inv(N)
        self.bs = b[:, :, 0] + np.einsum('nij,nj->ni', self.Ps, self.ms)

    def _buf(self, K):
        return np.zeros((len(self.cuts) - 1, self.nphi, K))

    def schur_mv(self, V):
        buf = self._buf(V.shape[1])
        _schur_mv(self.ptr, self.cuts, self.A, self.W, self.used, self.act, self.sb, self.Bs, self.fb, self.D, self.off, self.sz,
                  self.Ninv, np.ascontiguousarray(V), buf)
        return buf.sum(0) + self.Pf.matvec(V)

    def reduced_rhs(self, Y, Bst, prior_term):
        buf = self._buf(Y.shape[2])
        _reduce(self.ptr, self.cuts, self.A, self.W, self.used, self.act, self.sb, self.Bs, self.fb, self.D, self.off, self.sz,
                self.Ninv, np.ascontiguousarray(Y), np.ascontiguousarray(Bst), buf)
        return buf.sum(0) + prior_term

    def preconditioner(self):
        """Exact Schur diagonal blocks (sep and full blocks separately; cross terms ignored) + prior, inverted."""
        _, _, M, _ = _resid(self.ptr, self.A, self.y, self.W, self.used, self.sb, self.Bs, self.fb, self.D, self.off, self.sz,
                            self.Ninv, np.zeros((len(self.ptr) - 1, self.A.shape[2])), np.zeros((self.nphi, 1)))
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
    def solve(self, phi0=None, tol=1e-8, fixed_phi=None):
        """Joint MAP solve.  fixed_phi: frames held at that value (no frame solve; e.g. zeros = the two-step reproduction)."""
        self.star_blocks()
        if fixed_phi is not None:
            self.phi = np.asarray(fixed_phi, float).reshape(-1, 1)
            self.pcg_iters = 0
        else:
            rhs = self.reduced_rhs(self.y[:, :, None], self.bs[:, :, None], self.Pf.matvec(self.mf[:, None]))
            self.Minv = self.preconditioner()
            self.phi, self.pcg_iters = self.pcg(rhs, X0=None if phi0 is None else phi0.reshape(-1, 1), tol=tol, Minv=self.Minv)
        th, _ = _stars_given(self.ptr, self.A, self.W, self.used, self.sb, self.Bs, self.fb, self.D, self.off, self.sz, self.Ninv,
                             self.bs[:, :, None], self.phi, False)
        self.theta = th[:, :, 0]
        return self.theta, self.phi[:, 0]

    def residuals(self):
        return _resid(self.ptr, self.A, self.y, self.W, self.used, self.sb, self.Bs, self.fb, self.D, self.off, self.sz,
                      self.Ninv, self.theta, self.phi)

    def frame_samples(self, K=64, batch=16, seed=0, tol=1e-6):
        """phi* ~ N(0, S^-1) by perturb-and-solve (active stars' data noise + their prior noise + frame prior noise)."""
        rng = np.random.default_rng(seed)
        n = len(self.y); l11 = np.sqrt(self.C[:, 0]); l21 = self.C[:, 2] / l11; l22 = np.sqrt(self.C[:, 1] - l21 ** 2)
        Ls = np.linalg.cholesky(self.Ps); Lf = self.Pf.map_blocks(np.linalg.cholesky)
        on = self.used & np.repeat(self.act, np.diff(self.ptr))
        Minv = getattr(self, 'Minv', None) or self.preconditioner()
        out = []
        for k0 in range(0, K, batch):
            k = min(batch, K - k0)
            xi = rng.standard_normal((n, 2, k)) * on[:, None, None]
            Y = np.stack([l11[:, None] * xi[:, 0], l21[:, None] * xi[:, 0] + l22[:, None] * xi[:, 1]], 1)
            _, b = _star_normals(self.ptr, self.A, self.W, self.used, np.ascontiguousarray(Y))
            b += np.einsum('nij,njk->nik', Ls, rng.standard_normal((len(Ls), Ls.shape[1], k)))
            prior = Lf.matvec(rng.standard_normal((self.nphi, k)))
            rhs = self.reduced_rhs(Y, b, prior)
            X, it = self.pcg(rhs, tol=tol, Minv=Minv, label=f' samples {k0}-{k0 + k}')
            out.append(X)
        return np.concatenate(out, 1)

    def frame_cov(self, Phi):
        """Per-star frame part of the covariance, (1/K) sum_k (B_j phi_k)(B_j phi_k)^T, for every star."""
        _, cov = _stars_given(self.ptr, self.A, self.W, self.used, self.sb, self.Bs, self.fb, self.D, self.off, self.sz, self.Ninv,
                              np.zeros((1, self.A.shape[2], 1)), np.ascontiguousarray(Phi), True)
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
