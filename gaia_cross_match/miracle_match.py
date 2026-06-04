"""
miracle_match.py - Robust tiered geometric matching (V/VMAX, SNS, Offset).
"""

import numpy as np
from scipy.spatial import KDTree
from itertools import combinations, permutations
from scipy.signal import convolve2d

# Sigma-clip radii used for progressive tightening
DARRAY = np.array([99.0, 50.00, 25.00, 15.0, 10.00,
                    8.00,  5.00,  3.50,  2.00,  1.50,
                    1.20,  1.00,  0.80,  0.60,  0.45,
                    0.35,  0.25,  0.20,  0.16,  0.12,
                    0.09,  0.07,  0.05,  0.04,  0.03])

# ---------------------------------------------------------------------------
# Coordinate projection
# ---------------------------------------------------------------------------

def rd2x(r, d, r0, d0):
    to_rad = np.pi / 180.0
    cosra, sinra = np.cos((r - r0) * to_rad), np.sin((r - r0) * to_rad)
    cosde, sinde = np.cos(d * to_rad), np.sin(d * to_rad)
    cosd0, sind0 = np.cos(d0 * to_rad), np.sin(d0 * to_rad)
    rrrr = sind0 * sinde + cosd0 * cosde * cosra
    res = np.degrees(cosde * sinra / rrrr)
    x, y, z = cosde * np.cos(r * to_rad), cosde * np.sin(r * to_rad), sinde
    xx, yy, zz = cosd0 * np.cos(r0 * to_rad), cosd0 * np.sin(r0 * to_rad), sind0
    if np.ndim(res) > 0: res[x * xx + y * yy + z * zz < 0] = 90.0
    elif x * xx + y * yy + z * zz < 0: res = 90.0
    return res

def rd2y(r, d, r0, d0):
    to_rad = np.pi / 180.0
    cosra = np.cos((r - r0) * to_rad)
    cosde, sinde = np.cos(d * to_rad), np.sin(d * to_rad)
    cosd0, sind0 = np.cos(d0 * to_rad), np.sin(d0 * to_rad)
    rrrr = sind0 * sinde + cosd0 * cosde * cosra
    res = np.degrees((cosd0 * sinde - sind0 * cosde * cosra) / rrrr)
    x, y, z = cosde * np.cos(r * to_rad), cosde * np.sin(r * to_rad), sinde
    xx, yy, zz = cosd0 * np.cos(r0 * to_rad), cosd0 * np.sin(r0 * to_rad), sind0
    if np.ndim(res) > 0: res[x * xx + y * yy + z * zz < 0] = 90.0
    elif x * xx + y * yy + z * zz < 0: res = 90.0
    return res

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ord_brite(x, y, m, N):
    N = min(N, len(x))
    idx = np.argsort(m)[:N]
    return x[idx], y[idx], m[idx], idx

def glob_fit6(x1, y1, x2, y2):
    if len(x1) < 3: return 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
    x1o, y1o, x2o, y2o = x1.mean(), y1.mean(), x2.mean(), y2.mean()
    dx1, dy1, dx2, dy2 = x1 - x1o, y1 - y1o, x2 - x2o, y2 - y2o
    sxx, syy, sxy = (dx1**2).sum(), (dy1**2).sum(), (dx1*dy1).sum()
    det = sxx * syy - sxy ** 2
    if det == 0: return 1.0, 0.0, 0.0, 1.0, x1o, y1o, x2o, y2o
    A = ((dx2*dx1).sum() * syy - (dx2*dy1).sum() * sxy) / det
    B = ((dx2*dy1).sum() * sxx - (dx2*dx1).sum() * sxy) / det
    C = ((dy2*dx1).sum() * syy - (dy2*dy1).sum() * sxy) / det
    D = ((dy2*dy1).sum() * sxx - (dy2*dx1).sum() * sxy) / det
    return A, B, C, D, x1o, y1o, x2o, y2o

def _apply(x1, y1, A, B, C, D, x1o, y1o, x2o, y2o):
    return x2o + A*(x1-x1o) + B*(y1-y1o), y2o + C*(x1-x1o) + D*(y1-y1o)

def verify_priors(x1, y1, x2, y2, scale_range, rot_range):
    if len(x1) < 2: return False
    x1o, y1o, x2o, y2o = x1.mean(), y1.mean(), x2.mean(), y2.mean()
    dx1, dy1, dx2, dy2 = x1 - x1o, y1 - y1o, x2 - x2o, y2 - y2o
    den = np.sum(dx1**2 + dy1**2)
    if den == 0: return False
    a, b = np.sum(dx1*dx2 + dy1*dy2)/den, np.sum(dx1*dy2 - dy1*dx2)/den
    scale, rot = np.sqrt(a**2 + b**2), np.degrees(np.arctan2(b, a))
    return (scale_range[0] <= scale <= scale_range[1]) and (rot_range[0] <= rot <= rot_range[1])

def purify_buoy(x1b, y1b, x2b, y2b, dmax):
    x1, y1, x2, y2 = np.copy(x1b), np.copy(y1b), np.copy(x2b), np.copy(y2b)
    while len(x1) >= 3:
        A, B, C, D, x1o, y1o, x2o, y2o = glob_fit6(x1, y1, x2, y2)
        x2g, y2g = _apply(x1, y1, A, B, C, D, x1o, y1o, x2o, y2o)
        dd = np.sqrt((x2 - x2g)**2 + (y2 - y2g)**2)
        dlim = max(0.9 * dd.max(), dmax)
        keep = dd < dlim
        if keep.all(): break
        x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
    return x1, y1, x2, y2

# ---------------------------------------------------------------------------
# Matching Algorithms
# ---------------------------------------------------------------------------

def _match_offset(x1, y1, m1, x2, y2, m2, lim=500):
    dx_all = x2[:, None] - x1[None, :]
    dy_all = y2[:, None] - y1[None, :]
    mask = (np.abs(dx_all) <= lim) & (np.abs(dy_all) <= lim)
    if not np.any(mask): return None
    hist, xed, yed = np.histogram2d(dx_all[mask], dy_all[mask], bins=200, range=[[-lim, lim], [-lim, lim]])
    hist = convolve2d(hist, np.ones((3,3)), mode='same')
    idx = np.unravel_index(hist.argmax(), hist.shape)
    dx, dy = (xed[idx[0]]+xed[idx[0]+1])/2, (yed[idx[1]]+yed[idx[1]+1])/2
    tree = KDTree(np.column_stack([x2, y2]))
    d, i = tree.query(np.column_stack([x1+dx, y1+dy]), distance_upper_bound=5.0)
    k = d < 5.0
    return (x1[k], y1[k], x2[i[k]], y2[i[k]]) if k.sum() >= 3 else None

def _match_v_vmax(x1, y1, m1, x2, y2, m2, n_use=25):
    n_use = min(n_use, len(x1), len(x2))
    x1s, y1s, _, _ = ord_brite(x1, y1, m1, n_use)
    x2s, y2s, _, _ = ord_brite(x2, y2, m2, n_use)
    hash_tab = {}
    for na, nb, nc in combinations(range(n_use), 3):
        dab, dac = np.hypot(x1s[na]-x1s[nb], y1s[na]-y1s[nb]), np.hypot(x1s[na]-x1s[nc], y1s[na]-y1s[nc])
        if dab == 0 or dac == 0: continue
        vvmax = (min(dab, dac)**2)/(max(dab, dac)**2)
        if dab < dac: vvmax *= -1
        vd = (x1s[nb]-x1s[na])*(x1s[nc]-x1s[na]) + (y1s[nb]-y1s[na])*(y1s[nc]-y1s[na])
        vc = (x1s[nb]-x1s[na])*(y1s[nc]-y1s[na]) - (y1s[nb]-y1s[na])*(x1s[nc]-x1s[na])
        ang = np.degrees(np.arctan2(vc, vd)) % 360
        key = (int(100+100*vvmax), int(ang))
        if key not in hash_tab: hash_tab[key] = []
        hash_tab[key].append((na, nb, nc))
    vote = np.zeros((n_use, n_use))
    for na, nb, nc in combinations(range(n_use), 3):
        dab, dac = np.hypot(x2s[na]-x2s[nb], y2s[na]-y2s[nb]), np.hypot(x2s[na]-x2s[nc], y2s[na]-y2s[nc])
        if dab == 0 or dac == 0: continue
        vvmax = (min(dab, dac)**2)/(max(dab, dac)**2)
        if dab < dac: vvmax *= -1
        vd = (x2s[nb]-x2s[na])*(x2s[nc]-x2s[na]) + (y2s[nb]-y2s[na])*(y2s[nc]-y2s[na])
        vc = (x2s[nb]-x2s[na])*(y2s[nc]-y2s[na]) - (y2s[nb]-y2s[na])*(x2s[nc]-x2s[na])
        ang = np.degrees(np.arctan2(vc, vd)) % 360
        key = (int(100+100*vvmax), int(ang))
        if key in hash_tab:
            for t in hash_tab[key]:
                vote[t[0], na]+=1; vote[t[1], nb]+=1; vote[t[2], nc]+=1
    matches = []
    for i in range(n_use):
        if vote[:, i].max() > 0:
            best = vote[:, i].argmax()
            if vote[best, i] > 1.5 * np.partition(vote[:, i], -2)[-2]: matches.append((best, i))
    if len(matches) < 3: return None
    idx1, idx2 = zip(*matches)
    return x1s[list(idx1)], y1s[list(idx1)], x2s[list(idx2)], y2s[list(idx2)]

def _match_sns(x1, y1, m1, x2, y2, m2, n_use=25):
    n_use = min(n_use, len(x1), len(x2))
    x1s, y1s, _, _ = ord_brite(x1, y1, m1, n_use)
    x2s, y2s, _, _ = ord_brite(x2, y2, m2, n_use)
    tree2 = KDTree(np.column_stack([x2s, y2s]))
    best_n, best_res = 0, None
    for i1a, i1b in combinations(range(n_use), 2):
        dx1, dy1 = x1s[i1b]-x1s[i1a], y1s[i1b]-y1s[i1a]
        d1sq = dx1**2 + dy1**2
        if d1sq == 0: continue
        fx, fy = (dx1*(x1s-x1s[i1a]) + dy1*(y1s-y1s[i1a]))/d1sq, (-dx1*(y1s-y1s[i1a]) + dy1*(x1s-x1s[i1a]))/d1sq
        for i2a, i2b in permutations(range(n_use), 2):
            dx2, dy2 = x2s[i2b]-x2s[i2a], y2s[i2b]-y2s[i2a]
            dists, idxs = tree2.query(np.column_stack([x2s[i2a] + fx*dx2 + fy*dy2, y2s[i2a] + fx*dy2 - fy*dx2]), distance_upper_bound=5.0)
            k = dists < 5.0
            if k.sum() > best_n:
                best_n, best_res = k.sum(), (x1s[k], y1s[k], x2s[idxs[k]], y2s[idxs[k]])
                if best_n > 15: return best_res
    return best_res if best_n >= 3 else None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def miracle_match(x1, y1, m1, x2, y2, m2, scale_range=(0.98, 1.02), rot_range=(-180, 180), min_matches=3, verbose=True):
    def _log(m): 
        if verbose: print(m)
    x1, y1, m1, x2, y2, m2 = [np.asarray(a, float) for a in [x1, y1, m1, x2, y2, m2]]
    _log("  SEARCHING FOR INITIAL MATCH (Gaia -> HST):")
    tests = [("1-Offset", _match_offset), ("2-MM25", lambda *a: _match_v_vmax(*a, n_use=25)), 
             ("3-SNS", _match_sns), ("4-MM50", lambda *a: _match_v_vmax(*a, n_use=50))]
    for name, func in tests:
        _log(f"    {name}: ")
        res = func(x1, y1, m1, x2, y2, m2)
        if res:
            px1, py1, px2, py2 = purify_buoy(*res, 10.0)
            if len(px1) >= min_matches and verify_priors(px1, py1, px2, py2, scale_range, rot_range):
                _log(f"succeeded with {len(px1)} seeds!")
                t1, t2 = KDTree(np.column_stack([x1, y1])), KDTree(np.column_stack([x2, y2]))
                return px1, py1, m1[t1.query(np.column_stack([px1, py1]))[1]], px2, py2, m2[t2.query(np.column_stack([px2, py2]))[1]]
        _log("failed!")
    return (np.array([]),)*6
