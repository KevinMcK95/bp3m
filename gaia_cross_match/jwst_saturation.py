"""JWST cross-match guards against saturated Gaia stars (2026-10-12).

In dense NIRCam/NIRISS frames (47 Tuc, Omega Cen, NGC 6397, M4, M92, Draco) the cores of Gaia stars brighter
than G ~ 15-19.7 are DQ-SATURATED.  Up to pypass a4a4226 the cores were masked, and only their diffraction-spike
and wing detections reached the catalogue: 5-14 px from each core, at a fixed fraction of the star's flux.  The
offset search then locked onto that pattern 10-23 px off, with AB - G 5-8 mag too faint.  The chance test passed
those solutions decisively, since shifted copies of the pattern find nothing.  This affected 1,139 of 4,929 dev
frames (xmatch_failures, 2026-10-12).  Guards:

1. Saturated-core anchors: the header offset is measured from the centroids of DQ-SATURATED clusters against
   bright Gaia stars, independently of the catalogue.  The match is seeded there, and a solution that does not
   put the anchors on their cores is rejected.
2. Catalogue sources within max(JWST_SAT_EXCL_PX, 3 r_eq) of a saturated cluster are dropped from discovery and
   matching.  The core detection itself is kept: since a4a4226, pypass fits the partially saturated core.
3. A per-filter absolute AB - G gate on the solution's zero point (jwst_abg_window).
4. MIRI: detections with position errors above JWST_MAX_POS_SIG_PX (unconverged fits) are dropped, and the chance
   test uses a wider magnitude window (mid-IR minus G spans several mag).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import poisson

JWST_DQ_SATURATED = 2
JWST_SAT_MIN_PIX = 4          # a saturated cluster has >= this many connected SATURATED pixels
JWST_SAT_EXCL_PX = 20.0       # sources within max(this, JWST_SAT_EXCL_REQ x r_eq) of a cluster centroid are dropped ...
JWST_SAT_EXCL_REQ = 3.0
JWST_SAT_CORE_PX = 2.0        # ... except the core detection, within max(this, r_eq / 2) of the centroid
JWST_ANCHOR_GMAX = 18.0       # Gaia stars used as saturated-core anchors
JWST_ANCHOR_BIN_PX = 2.0
JWST_ANCHOR_MIN = 3           # stars in the anchor offset peak
JWST_ANCHOR_FAP = 1e-6        # peak significance (Poisson x number of cells)
JWST_ANCHOR_TOL_PX = 3.0      # an anchor matches its core within this; a solution agrees within this (median)
JWST_ANCHOR_MIN_FRAC = 0.5    # ... and puts at least this fraction of the anchors on their cores

# AB - G (pypass mag_ab_gdc minus Gaia G) of the matched stars: median over the frames whose solution the
# saturated-core test confirmed (xmatch_failures/frame_classes.csv 'ok', 2,797 frames; 1-99% within -0.3..+2.5).
# The artefact locks sit at +3.0..+8.4.  Window: centre - JWST_ABG_LO .. centre + JWST_ABG_HI.  The blue side is
# wide because no failure lives there, while the 1% tails reach -1.9 (F200W).
JWST_ABG_CENTRE = {
    ('NIRCAM', 'F070W'): 0.53, ('NIRCAM', 'F090W'): 0.66, ('NIRCAM', 'F115W'): 0.86, ('NIRCAM', 'F140M'): 1.09,
    ('NIRCAM', 'F150W'): 1.28, ('NIRCAM', 'F182M'): 1.55, ('NIRCAM', 'F200W'): 1.62, ('NIRCAM', 'F210M'): 1.63,
    ('NIRCAM', 'F212N'): 1.40, ('NIRCAM', 'F277W'): 2.06,
    ('NIRISS', 'F090W'): -0.11, ('NIRISS', 'F150W'): -0.09, ('NIRISS', 'F200W'): -0.16,
}
JWST_ABG_DEFAULT = {'NIRCAM_SW': 1.0, 'NIRCAM_LW': 2.0, 'NIRISS': 0.0}   # filters without a measured centre
JWST_ABG_LO = 4.0
JWST_ABG_HI = 2.0
MIRI_CHANCE_MAG_WIN = 3.0     # chance-test magnitude half-window for MIRI (optical/NIR: 1.5)
# Position-error cap: MIRI pypass catalogues carry unconverged fits with sigma_x of 10^3-10^4 px (Sextans A F770W:
# > 10% of 5,030 detections).  Inside the 5-sigma association window they pair with any Gaia star: the refinement
# drifted to 20-30 px residuals and the chance test counted 45 such pairs for every shifted copy, so even the
# correct solution (15 seeds at 0.1-0.2 px) failed.  Such detections cannot constrain a match: dropped.
JWST_MAX_POS_SIG_PX = 2.0


def jwst_filter_name(h0) -> str:
    """Filter of a JWST exposure: the PUPIL element when it carries the filter (NIRCam F162M/F164N/F323N/F405N/
    F466N/F470N, NIRISS imaging), else FILTER."""
    filt = str(h0.get('FILTER', '') or '').strip().upper()
    pupil = str(h0.get('PUPIL', '') or '').strip().upper()
    if pupil.startswith('F') and pupil[1:4].isdigit():
        return pupil
    return filt


def jwst_abg_window(instrument: str, detector: str, filt: str):
    """(lo, hi) allowed AB - G of a solution's matched stars, or None (MIRI / unknown instrument: no gate)."""
    inst = str(instrument or '').strip().upper()
    if inst not in ('NIRCAM', 'NIRISS'):
        return None
    c = JWST_ABG_CENTRE.get((inst, str(filt).upper()))
    if c is None:
        if inst == 'NIRISS':
            c = JWST_ABG_DEFAULT['NIRISS']
        else:
            c = JWST_ABG_DEFAULT['NIRCAM_LW' if 'LONG' in str(detector).upper() else 'NIRCAM_SW']
    return c - JWST_ABG_LO, c + JWST_ABG_HI


def saturated_clusters(flc_path):
    """Centroids (0-based raw pixel x, y, the pypass catalogue convention) and equivalent radii of the connected
    DQ-SATURATED clusters of >= JWST_SAT_MIN_PIX pixels.  Returns (x, y, r_eq); empty arrays if there is no DQ."""
    from astropy.io import fits
    try:
        with fits.open(flc_path, memmap=False) as h:
            dq = np.asarray(h['DQ'].data)
    except Exception:
        e = np.zeros(0)
        return e, e, e
    sat = (dq & JWST_DQ_SATURATED) != 0
    lab, n = ndimage.label(sat)
    if n == 0:
        e = np.zeros(0)
        return e, e, e
    sz = np.asarray(ndimage.sum(sat, lab, range(1, n + 1)))
    big = np.where(sz >= JWST_SAT_MIN_PIX)[0] + 1
    if len(big) == 0:
        e = np.zeros(0)
        return e, e, e
    cy, cx = np.array(ndimage.center_of_mass(sat, lab, big)).T
    return cx.astype(float), cy.astype(float), np.sqrt(sz[big - 1] / np.pi)


def near_saturation(x_raw, y_raw, cx, cy, r_eq):
    """(drop, core) masks over the catalogue: drop = within the exclusion radius of a saturated cluster and not
    its core detection; core = within the core radius of a cluster centroid."""
    x_raw, y_raw = np.asarray(x_raw, float), np.asarray(y_raw, float)
    drop = np.zeros(len(x_raw), bool); core = np.zeros(len(x_raw), bool)
    ok = np.isfinite(x_raw) & np.isfinite(y_raw)
    if len(cx) == 0 or not ok.any():
        return drop, core
    idx = np.where(ok)[0]
    tree = cKDTree(np.column_stack([x_raw[idx], y_raw[idx]]))
    r_ex = np.maximum(JWST_SAT_EXCL_PX, JWST_SAT_EXCL_REQ * r_eq)
    r_co = np.maximum(JWST_SAT_CORE_PX, 0.5 * r_eq)
    for k in range(len(cx)):
        near = tree.query_ball_point([cx[k], cy[k]], float(r_ex[k]))
        if not near:
            continue
        near = np.asarray(near, int)
        d = np.hypot(x_raw[idx[near]] - cx[k], y_raw[idx[near]] - cy[k])
        drop[idx[near]] = True
        core[idx[near[d <= r_co[k]]]] = True
    drop &= ~core
    return drop, core


def raw_to_gdc(x_raw, y_raw, x_gdc, y_gdc):
    """Cubic polynomial raw -> GDC pixel map fitted to the catalogue's own (x, y) / (x_gdc, y_gdc) pairs
    (sub-0.01 px on NIRCam), or None when the catalogue is too small."""
    x_raw, y_raw = np.asarray(x_raw, float), np.asarray(y_raw, float)
    x_gdc, y_gdc = np.asarray(x_gdc, float), np.asarray(y_gdc, float)
    ok = np.isfinite(x_raw) & np.isfinite(y_raw) & np.isfinite(x_gdc) & np.isfinite(y_gdc)
    if ok.sum() < 30:
        return None
    s = 1024.0

    def _A(x, y):
        u, v = (np.asarray(x, float) - s) / s, (np.asarray(y, float) - s) / s
        return np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3])

    A = _A(x_raw[ok], y_raw[ok])
    cx_, *_ = np.linalg.lstsq(A, x_gdc[ok], rcond=None)
    cy_, *_ = np.linalg.lstsq(A, y_gdc[ok], rcond=None)
    return lambda x, y: (_A(x, y) @ cx_, _A(x, y) @ cy_)


def anchor_offset(cx_g, cy_g, gx, gy, gmag, window_px, footprint):
    """Header offset (HST - Gaia guess, GDC px) from saturated-cluster centroids (cx_g, cy_g; GDC frame) vs
    the guess positions of Gaia stars with G < JWST_ANCHOR_GMAX.  footprint = (x0, x1, y0, y1) in GDC px.
    Returns dict(dx, dy, n_peak, fap, anchors=(gi, cx, cy) arrays), or None if there is no significant peak."""
    if len(cx_g) < JWST_ANCHOR_MIN:
        return None
    gx, gy, gmag = np.asarray(gx, float), np.asarray(gy, float), np.asarray(gmag, float)
    x0, x1, y0, y1 = footprint
    W = float(window_px)
    b = np.where(np.isfinite(gx) & np.isfinite(gy) & (gmag < JWST_ANCHOR_GMAX)
                 & (gx > x0 - W) & (gx < x1 + W) & (gy > y0 - W) & (gy < y1 + W))[0]
    if len(b) < JWST_ANCHOR_MIN:
        return None
    tree = cKDTree(np.column_stack([cx_g, cy_g]))
    lists = tree.query_ball_point(np.column_stack([gx[b], gy[b]]), r=W * 1.415)
    gi = np.repeat(b, [len(l) for l in lists])
    ci = np.fromiter((j for l in lists for j in l), int, count=len(gi))
    if len(gi) < JWST_ANCHOR_MIN:
        return None
    dx, dy = cx_g[ci] - gx[gi], cy_g[ci] - gy[gi]
    sel = (np.abs(dx) <= W) & (np.abs(dy) <= W)
    dx, dy, gi, ci = dx[sel], dy[sel], gi[sel], ci[sel]
    if len(dx) < JWST_ANCHOR_MIN:
        return None
    nb = max(int(np.ceil(2 * W / JWST_ANCHOR_BIN_PX)), 2)
    e = np.linspace(-W, W, nb + 1)
    H, _, _ = np.histogram2d(dx, dy, bins=[e, e])
    S = H.copy()
    S[:-1] += H[1:]
    S[:, :-1] += S[:, 1:].copy()          # 2x2-bin sums: a 4 px box, robust to a peak on a bin edge
    k = np.unravel_index(int(np.argmax(S)), S.shape)
    n_pk = int(S[k])
    lam = len(dx) * (2 * JWST_ANCHOR_BIN_PX) ** 2 / (2 * W) ** 2
    fap = float(poisson.sf(n_pk - 1, lam)) * S.size
    if n_pk < JWST_ANCHOR_MIN or fap > JWST_ANCHOR_FAP:
        return None
    px, py = e[k[0]] + JWST_ANCHOR_BIN_PX, e[k[1]] + JWST_ANCHOR_BIN_PX
    m = np.hypot(dx - px, dy - py) < JWST_ANCHOR_TOL_PX
    ox, oy = float(np.median(dx[m])), float(np.median(dy[m]))
    # one anchor per Gaia star: its nearest cluster at the offset, inside the footprint
    inside = b[(gx[b] + ox > x0) & (gx[b] + ox < x1) & (gy[b] + oy > y0) & (gy[b] + oy < y1)]
    d, j = tree.query(np.column_stack([gx[inside] + ox, gy[inside] + oy]), distance_upper_bound=JWST_ANCHOR_TOL_PX)
    ok = np.isfinite(d)
    if ok.sum() < JWST_ANCHOR_MIN:
        return None
    # a cluster anchors one star only (blended bright pairs)
    a_g, a_c = inside[ok], j[ok]
    _, first = np.unique(a_c, return_index=True)
    a_g, a_c = a_g[first], a_c[first]
    return dict(dx=ox, dy=oy, n_peak=n_pk, n_pairs=int(len(dx)), lam=lam, fap=fap,
                anchors=(a_g, cx_g[a_c], cy_g[a_c]))


def anchor_agreement(A, B, C, D, xs_o, ys_o, xt_o, yt_o, x_g, y_g, anchors):
    """(median residual px, fraction within JWST_ANCHOR_TOL_PX) of a solution at the anchors: the saturated
    core centroids mapped into the Gaia frame vs their Gaia stars."""
    gi, cx, cy = anchors
    if len(gi) == 0:
        return np.inf, 0.0
    px = A * (cx - xs_o) + B * (cy - ys_o) + xt_o
    py = C * (cx - xs_o) + D * (cy - ys_o) + yt_o
    r = np.hypot(px - np.asarray(x_g)[gi], py - np.asarray(y_g)[gi])
    return float(np.median(r)), float(np.mean(r < JWST_ANCHOR_TOL_PX))
