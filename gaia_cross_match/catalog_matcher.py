import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from itertools import combinations, permutations

def get_inv_2x2(C):
    """Vectorized 2x2 matrix inversion."""
    det = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] * C[:, 1, 0]
    inv = np.zeros_like(C)
    inv[:, 0, 0] = C[:, 1, 1] / det
    inv[:, 1, 1] = C[:, 0, 0] / det
    inv[:, 0, 1] = -C[:, 0, 1] / det
    inv[:, 1, 0] = -C[:, 1, 0] / det
    return inv, det

def fit_affine_weighted(x_src, y_src, x_tgt, y_tgt, cov_src, cov_tgt, initial_M=None, skew_prior=1e-4):
    """
    Weighted 6-parameter affine transform with skew priors.
    skew_prior: expected size of (A-D) and (B+C).
    Returns (A, B, C, D, xs_o, ys_o, xt_o, yt_o), p_err, inv_lhs, chi2.
    """
    n = len(x_src)
    if n < 3: return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), np.zeros(6), np.eye(6), 1e10
    
    M = initial_M if initial_M is not None else np.eye(2)
    C_proj = np.einsum('ij,njk,lk->nil', M, cov_src, M)
    C = cov_tgt + C_proj
    inv_C, _ = get_inv_2x2(C)
    
    xs_o, ys_o = np.mean(x_src), np.mean(y_src)
    xt_o, yt_o = np.mean(x_tgt), np.mean(y_tgt)
    dxs, dys = x_src - xs_o, y_src - ys_o
    dxt, dyt = x_tgt - xt_o, y_tgt - yt_o

    wxx, wyy, wxy = inv_C[:, 0, 0], inv_C[:, 1, 1], inv_C[:, 0, 1]
    
    # 6x6 Normal equations: [A, B, xt_fit, C, D, yt_fit]
    lhs = np.zeros((6, 6))
    rhs = np.zeros(6)
    
    g = np.column_stack([dxs, dys, np.ones(n)])
    
    # Block diagonal terms
    m_tl = np.einsum('ni,nj,n->ij', g, g, wxx)
    m_br = np.einsum('ni,nj,n->ij', g, g, wyy)
    m_tr = np.einsum('ni,nj,n->ij', g, g, wxy)
    
    lhs[:3, :3] = m_tl
    lhs[3:, 3:] = m_br
    lhs[:3, 3:] = m_tr
    lhs[3:, :3] = m_tr.T
    
    rhs[:3] = np.sum(g * (wxx * dxt + wxy * dyt)[:, np.newaxis], axis=0)
    rhs[3:] = np.sum(g * (wxy * dxt + wyy * dyt)[:, np.newaxis], axis=0)
    
    # --- Add Skew Priors ---
    # We prioritize similarity: A=D and B=-C.
    # Regularization penalty = 0.5 * (A-D)^2 / sig^2 + 0.5 * (B+C)^2 / sig^2
    # d/dA = (A-D)/sig^2  => Add 1/sig^2 to lhs[0,0], -1/sig^2 to lhs[0,4]
    # d/dD = (D-A)/sig^2  => Add 1/sig^2 to lhs[4,4], -1/sig^2 to lhs[4,0]
    # d/dB = (B+C)/sig^2  => Add 1/sig^2 to lhs[1,1], 1/sig^2 to lhs[1,3]
    # d/dC = (C+B)/sig^2  => Add 1/sig^2 to lhs[3,3], 1/sig^2 to lhs[3,1]
    if skew_prior > 0:
        w_prior = 1.0 / (skew_prior**2)
        # A - D term
        lhs[0, 0] += w_prior; lhs[4, 4] += w_prior
        lhs[0, 4] -= w_prior; lhs[4, 0] -= w_prior
        # B + C term
        lhs[1, 1] += w_prior; lhs[3, 3] += w_prior
        lhs[1, 3] += w_prior; lhs[3, 1] += w_prior

    try:
        inv_lhs = np.linalg.inv(lhs)
        p = np.einsum('ij,j->i', inv_lhs, rhs)
        p_err = np.sqrt(np.diag(inv_lhs))
    except np.linalg.LinAlgError:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), np.zeros(6), np.eye(6), 1e10
        
    A, B, xt_fit, C, D, yt_fit = p
    xt_p = xt_o + xt_fit + A*dxs + B*dys
    yt_p = yt_o + yt_fit + C*dxs + D*dys
    dx, dy = x_tgt - xt_p, y_tgt - yt_p
    chi2 = np.sum(wxx*dx**2 + wyy*dy**2 + 2*wxy*dx*dy)
    
    return (A, B, C, D, xs_o, ys_o, xt_o + xt_fit, yt_o + yt_fit), p_err, inv_lhs, chi2

def fit_4p_weighted(x_src, y_src, x_tgt, y_tgt, cov_src, cov_tgt, initial_M=None, 
                    scale_prior=None, scale_sigma=0.02, rot_sigma=0.0035):
    """
    Similarity transform fit with scale and rotation priors.
    scale_prior: default initial_scale.
    scale_sigma: uncertainty in scale (~2% -> 0.02).
    rot_sigma: uncertainty in rotation (~0.2 deg -> 0.0035 rad).
    """
    n = len(x_src)
    if n < 2: return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), np.zeros(4), np.eye(4), 1e10
    
    M = initial_M if initial_M is not None else np.eye(2)
    C_proj = np.einsum('ij,njk,lk->nil', M, cov_src, M)
    C = cov_tgt + C_proj
    inv_C, _ = get_inv_2x2(C)
    
    xs_o, ys_o = np.mean(x_src), np.mean(y_src)
    xt_o, yt_o = np.mean(x_tgt), np.mean(y_tgt)
    dxs, dys = x_src - xs_o, y_src - ys_o
    dxt, dyt = x_tgt - xt_o, y_tgt - yt_o

    wxx, wyy, wxy = inv_C[:, 0, 0], inv_C[:, 1, 1], inv_C[:, 0, 1]
    lhs, rhs = np.zeros((4, 4)), np.zeros(4)
    
    lhs[0,0] = np.sum(wxx*dxs**2 + wyy*dys**2 + 2*wxy*dxs*dys)
    lhs[0,1] = np.sum(wyy*dxs*dys - wxx*dxs*dys + wxy*(dxs**2 - dys**2))
    lhs[0,2] = np.sum(wxx*dxs + wxy*dys)
    lhs[0,3] = np.sum(wxy*dxs + wyy*dys)
    lhs[1,1] = np.sum(wxx*dys**2 + wyy*dxs**2 - 2*wxy*dxs*dys)
    lhs[1,2] = np.sum(wxy*dxs - wxx*dys)
    lhs[1,3] = np.sum(wyy*dxs - wxy*dys)
    lhs[2,2], lhs[2,3], lhs[3,3] = np.sum(wxx), np.sum(wxy), np.sum(wyy)
    lhs[1,0], lhs[2,0], lhs[3,0], lhs[2,1], lhs[3,1], lhs[3,2] = lhs[0,1], lhs[0,2], lhs[0,3], lhs[1,2], lhs[1,3], lhs[2,3]
    
    rhs[0] = np.sum(dxt*(wxx*dxs + wxy*dys) + dyt*(wxy*dxs + wyy*dys))
    rhs[1] = np.sum(dxt*(wxy*dxs - wxx*dys) + dyt*(wyy*dxs - wxy*dys))
    rhs[2] = np.sum(wxx*dxt + wxy*dyt); rhs[3] = np.sum(wxy*dxt + wyy*dyt)

    # --- Add 4P Priors ---
    # p = [a, b, xt, yt] where a = s*cos(th), b = s*sin(th)
    # Rotation prior (th=0): b=0. Scale prior: a=scale_prior.
    if scale_prior is not None:
        wa = 1.0 / (scale_sigma**2)
        wb = 1.0 / (rot_sigma**2) # assuming small angles, sin(th) ~ th
        lhs[0, 0] += wa
        rhs[0] += wa * scale_prior
        lhs[1, 1] += wb
        # rhs[1] += wb * 0.0

    try:
        inv_lhs = np.linalg.inv(lhs)
        p = np.einsum('ij,j->i', inv_lhs, rhs)
        p_err = np.sqrt(np.diag(inv_lhs))
    except np.linalg.LinAlgError:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), np.zeros(4), np.eye(4), 1e10
        
    a, b, dxt_fit, dyt_fit = p
    xt_p = xt_o + dxt_fit + a*dxs - b*dys
    yt_p = yt_o + dyt_fit + b*dxs + a*dys
    dx, dy = x_tgt - xt_p, y_tgt - yt_p
    chi2 = np.sum(wxx*dx**2 + wyy*dy**2 + 2*wxy*dx*dy)
    
    return (a, -b, b, a, xs_o, ys_o, xt_o + dxt_fit, yt_o + dyt_fit), p_err, inv_lhs, chi2

def apply_affine(x, y, A, B, C, D, xs_o, ys_o, xt_o, yt_o):
    return xt_o + A * (x - xs_o) + B * (y - ys_o), yt_o + C * (x - xs_o) + D * (y - ys_o)

def find_offset(x1, y1, m1, x2, y2, m2, cov1=None, cov2=None,
                max_offset=500, bin_size=2, top_n=1, return_histogram=False):
    """Find the top_n translation offset peaks between catalog 1 and catalog 2.

    When cov1/cov2 (shape [n, 2, 2]) are provided each pair is weighted by
    1/(sig_x * sig_y) where sig_x = sqrt(cov2_xx + cov1_xx), and the histogram
    is smoothed with a Gaussian of width = median combined sigma.  This makes
    well-measured pairs dominate the peak.

    Returns a list of (dx, dy, score) tuples ordered by peak height (length == top_n).
    dx, dy satisfy x2 ≈ x1 + dx.  Falls back to [(0.0, 0.0, 0.0)] * top_n on failure.
    If return_histogram=True, returns (peaks, hist, xed, yed) instead.
    """
    n1_u, n2_u = min(len(x1), 1000), min(len(x2), 1000)
    idx1, idx2 = np.argsort(m1)[:n1_u], np.argsort(m2)[:n2_u]
    x1s, y1s, x2s, y2s = x1[idx1], y1[idx1], x2[idx2], y2[idx2]
    dx = x2s[:, None] - x1s[None, :]  # [n2, n1]
    dy = y2s[:, None] - y1s[None, :]
    mask = (np.abs(dx) <= max_offset) & (np.abs(dy) <= max_offset)

    n_bins = int(2 * max_offset / bin_size) + 1
    bin_range = [[-max_offset, max_offset], [-max_offset, max_offset]]
    empty_edges = np.linspace(-max_offset, max_offset, n_bins + 1)
    empty = [(0.0, 0.0, 0.0)] * top_n

    if not np.any(mask):
        if return_histogram:
            return empty, np.zeros((n_bins, n_bins)), empty_edges, empty_edges
        return empty

    xed = np.linspace(-max_offset, max_offset, n_bins + 1)
    yed = np.linspace(-max_offset, max_offset, n_bins + 1)

    if cov1 is not None and cov2 is not None:
        # Monte Carlo KDE: draw n_draws samples per pair from its own Gaussian.
        # Each pair contributes exactly 1 unit of total weight regardless of sigma,
        # so tight (well-measured) pairs naturally build a taller, narrower peak.
        sig_x = np.sqrt(cov2[idx2, 0, 0][:, None] + cov1[idx1, 0, 0][None, :])  # [n2, n1]
        sig_y = np.sqrt(cov2[idx2, 1, 1][:, None] + cov1[idx1, 1, 1][None, :])

        dx_v, dy_v = dx[mask], dy[mask]
        sx_v, sy_v = sig_x[mask], sig_y[mask]

        # Weight each pair by 1/(sig_x * sig_y) so tight pairs dominate.
        # Pairs with large sigma (poorly constrained Gaia PMs) get tiny weight
        # and barely contribute to the peak even though they spread over many bins.
        pair_weight = 1.0 / (sx_v * sy_v)

        n_draws = 10
        rng = np.random.default_rng()
        samp_x = (dx_v + sx_v * rng.standard_normal((n_draws, len(dx_v)))).ravel()
        samp_y = (dy_v + sy_v * rng.standard_normal((n_draws, len(dy_v)))).ravel()
        in_range = (np.abs(samp_x) <= max_offset) & (np.abs(samp_y) <= max_offset)
        w = np.repeat(pair_weight / n_draws, n_draws)
        hist, _, _ = np.histogram2d(samp_x[in_range], samp_y[in_range],
                                    bins=[xed, yed], weights=w[in_range])

        suppress = max(10, int(5 * float(np.median(sx_v[sx_v < 2.0])) / bin_size) if np.any(sx_v < 2.0) else 10)
    else:
        hist, _, _ = np.histogram2d(dx[mask], dy[mask], bins=[xed, yed])
        hist = gaussian_filter(hist, sigma=1.5)
        suppress = 10
    hist_work = hist.copy()
    peaks = []
    for _ in range(top_n):
        i, j = np.unravel_index(hist_work.argmax(), hist_work.shape)
        peaks.append(((xed[i]+xed[i+1])/2, (yed[j]+yed[j+1])/2, float(hist_work[i, j])))
        i0, i1 = max(0, i-suppress), min(hist_work.shape[0], i+suppress+1)
        j0, j1 = max(0, j-suppress), min(hist_work.shape[1], j+suppress+1)
        hist_work[i0:i1, j0:j1] = 0

    if return_histogram:
        return peaks, hist, xed, yed
    return peaks

def find_scale_and_offset(x1, y1, m1, x2, y2, m2, cov1=None, cov2=None,
                          x_cen=0.0, y_cen=0.0,
                          max_offset=100, bin_size=1, top_n=1,
                          ds_range=(-0.01, 0.01), n_scales=21,
                          return_histogram=False):
    """Find translation (dx, dy) and residual scale ds simultaneously.

    Model: x2 ≈ x1 + ds*(x1 - x_cen) + dx  (scale applied around (x_cen, y_cen)).

    For each ds in the sweep grid the scale-corrected offsets are histogrammed; the
    ds that produces the tallest peak — and that peak's (dx, dy) — are returned.

    Returns (best_ds, peaks) where peaks is a list of (dx, dy, score) tuples.
    If return_histogram=True, returns (best_ds, peaks, hist, xed, yed).
    Falls back to ds=0 and [(0,0,0)]*top_n on failure.
    """
    n1_u = min(len(x1), 1000)
    n2_u = min(len(x2), 1000)
    idx1 = np.argsort(m1)[:n1_u]
    idx2 = np.argsort(m2)[:n2_u]
    x1s, y1s = x1[idx1], y1[idx1]
    x2s, y2s = x2[idx2], y2[idx2]

    # Raw offsets [n2, n1] and scale-correction factors [1, n1]
    dx_raw = x2s[:, None] - x1s[None, :]
    dy_raw = y2s[:, None] - y1s[None, :]
    dx1_cen = (x1s - x_cen)[None, :]
    dy1_cen = (y1s - y_cen)[None, :]

    has_cov = (cov1 is not None) and (cov2 is not None)
    if has_cov:
        sig_x = np.sqrt(cov2[idx2, 0, 0][:, None] + cov1[idx1, 0, 0][None, :])
        sig_y = np.sqrt(cov2[idx2, 1, 1][:, None] + cov1[idx1, 1, 1][None, :])

    n_bins = int(2 * max_offset / bin_size) + 1
    xed = np.linspace(-max_offset, max_offset, n_bins + 1)
    yed = np.linspace(-max_offset, max_offset, n_bins + 1)
    empty = [(0.0, 0.0, 0.0)] * top_n

    best_score = -1.0
    best_ds = 0.0
    best_hist = np.zeros((n_bins, n_bins))
    rng = np.random.default_rng()

    for ds in np.linspace(ds_range[0], ds_range[1], n_scales):
        dx_corr = dx_raw - ds * dx1_cen
        dy_corr = dy_raw - ds * dy1_cen
        mask = (np.abs(dx_corr) <= max_offset) & (np.abs(dy_corr) <= max_offset)
        if not np.any(mask):
            continue

        if has_cov:
            dx_v, dy_v = dx_corr[mask], dy_corr[mask]
            sx_v, sy_v = sig_x[mask], sig_y[mask]
            pair_weight = 1.0 / (sx_v * sy_v)
            n_draws = 5
            samp_x = (dx_v + sx_v * rng.standard_normal((n_draws, len(dx_v)))).ravel()
            samp_y = (dy_v + sy_v * rng.standard_normal((n_draws, len(dy_v)))).ravel()
            in_range = (np.abs(samp_x) <= max_offset) & (np.abs(samp_y) <= max_offset)
            w = np.repeat(pair_weight / n_draws, n_draws)
            hist, _, _ = np.histogram2d(samp_x[in_range], samp_y[in_range],
                                        bins=[xed, yed], weights=w[in_range])
        else:
            hist, _, _ = np.histogram2d(dx_corr[mask], dy_corr[mask], bins=[xed, yed])
            hist = gaussian_filter(hist, sigma=1.5)

        score = hist.max()
        if score > best_score:
            best_score = score
            best_ds = float(ds)
            best_hist = hist.copy()

    if best_score < 0:
        if return_histogram:
            return 0.0, empty, np.zeros((n_bins, n_bins)), xed, yed
        return 0.0, empty

    hist_work = best_hist.copy()
    suppress = 10
    peaks = []
    for _ in range(top_n):
        i, j = np.unravel_index(hist_work.argmax(), hist_work.shape)
        peaks.append(((xed[i] + xed[i+1]) / 2, (yed[j] + yed[j+1]) / 2, float(hist_work[i, j])))
        i0 = max(0, i - suppress); i1 = min(hist_work.shape[0], i + suppress + 1)
        j0 = max(0, j - suppress); j1 = min(hist_work.shape[1], j + suppress + 1)
        hist_work[i0:i1, j0:j1] = 0

    if return_histogram:
        return best_ds, peaks, best_hist, xed, yed
    return best_ds, peaks


def compute_mahalanobis(dx, dy, cov):
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
    inv_cxx = cov[:, 1, 1] / det
    inv_cyy = cov[:, 0, 0] / det
    inv_cxy = -cov[:, 0, 1] / det
    return np.sqrt(inv_cxx * dx**2 + inv_cyy * dy**2 + 2 * inv_cxy * dx * dy)

def compute_logprob_cost(dx, dy, cov):
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
    mahal = (compute_mahalanobis(dx, dy, cov))**2
    return np.log(det) + mahal
