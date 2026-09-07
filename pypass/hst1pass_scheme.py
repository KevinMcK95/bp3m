"""Anderson/hst1pass-native PSF evaluation scheme for pypass.

Activated with ``PYPASS_PSF_SCHEME=hst1pass``.  Replaces pypass's smooth
interpolants with faithful ports of the scheme the STDPSF library is
calibrated under (hst1pass.2025.02.14_v1h.F):

  sub-pixel (``rpsf_phot``): r <= 4 det px — four local quadratic Taylor
    patches from central differences, blended bilinearly; 4 < r <= 12 —
    bilinear on the supersampled raster; r > 12 — zero.
  spatial (``rpsf_phot_ij_STDPSF``): bilinear between the 4 nearest fiducial
    PSFs with weights from the star's INTEGER pixel; linear extrapolation
    beyond the outermost fiducials.

In this mode all "coefficient" arrays are the RAW supersampled rasters
(spline_filter is bypassed), and position gradients are numeric central
differences over one supersample pixel — the same convention pypass's
B-spline kernel uses for its analytic-equivalent derivative step.

Measured motivation (2026-09-07, tests_pypass/scheme_compare.py): the
B-spline scheme differs from this one by ~2.9 mpx median / 40 mpx p99 per
star on M31 data — above the 1 mpx systematics-floor target — because the
STDPSF rasters were fit to data THROUGH rpsf_phot.
"""

import os
import numpy as np

_SCHEME_ENV = 'PYPASS_PSF_SCHEME'


def psf_scheme() -> str:
    """'bspline' (default) or 'hst1pass'."""
    v = os.environ.get(_SCHEME_ENV, 'bspline').strip().lower()
    return 'hst1pass' if v in ('hst1pass', 'anderson', 'fortran') else 'bspline'


# ---------------------------------------------------------------------------
# Sub-pixel evaluation on a (cropped) raw supersampled raster
# ---------------------------------------------------------------------------

def rpsf_eval(raster, off_x, off_y, psf_scale, center=None):
    """Vectorized rpsf_phot on *raster* at detector offsets (off_x, off_y).

    raster : 2D raw supersampled PSF (full 101x101 or a centred crop)
    off_x, off_y : broadcastable arrays — pixel offset from the star centre
                   (Fortran's x, y arguments)
    center : supersampled index of the raster centre (default (n-1)//2)

    Needs the raster to extend >= 2 supersample px beyond the evaluated
    coordinates (quadratic-patch stencil); callers crop accordingly.
    """
    off_x, off_y = np.broadcast_arrays(np.asarray(off_x, dtype=np.float64),
                                       np.asarray(off_y, dtype=np.float64))
    c = (raster.shape[0] - 1) // 2 if center is None else center
    rx = c + off_x * psf_scale
    ry = c + off_y * psf_scale
    ix = np.floor(rx).astype(np.intp)
    iy = np.floor(ry).astype(np.intp)
    fx = rx - ix
    fy = ry - iy
    dd = np.hypot(off_x, off_y)

    ny, nx = raster.shape
    ixc = np.clip(ix, 2, nx - 4)
    iyc = np.clip(iy, 2, ny - 4)

    def P(jx, jy):
        return raster[iyc + jy, ixc + jx]

    bl = ((1 - fx) * (1 - fy) * P(0, 0) + fx * (1 - fy) * P(1, 0)
          + (1 - fx) * fy * P(0, 1) + fx * fy * P(1, 1))

    def patch(jx, jy, u, v, esign, ex, ey):
        A = P(jx, jy)
        B = (P(jx + 1, jy) - P(jx - 1, jy)) / 2
        C = (P(jx, jy + 1) - P(jx, jy - 1)) / 2
        D = (P(jx + 1, jy) + P(jx - 1, jy) - 2 * A) / 2
        F = (P(jx, jy + 1) + P(jx, jy - 1) - 2 * A) / 2
        E = esign * (P(ex, ey) - A)
        return A + B * u + C * v + D * u * u + E * u * v + F * v * v

    V1 = patch(0, 0, fx,     fy,     +1.0, 1, 1)
    V2 = patch(1, 0, fx - 1, fy,     -1.0, 0, 1)
    V3 = patch(0, 1, fx,     fy - 1, -1.0, 1, 0)
    V4 = patch(1, 1, fx - 1, fy - 1, +1.0, 0, 0)
    qd = ((1 - fx) * (1 - fy) * V1 + fx * (1 - fy) * V2
          + (1 - fx) * fy * V3 + fx * fy * V4)

    return np.where(dd <= 4.0, qd, np.where(dd <= 12.0, bl, 0.0))


def eval_psf_grad_hst1(raster, dx, dy, dix, diy, psf_scale, center=None):
    """(P, dPdx, dPdy) matching _eval_psf_grad_fast's contract.

    P evaluated at pixel offsets (dix - dx, diy - dy); gradients are central
    differences of one supersample pixel in the STAR coordinate — identical
    convention to the B-spline kernel's derivative step.
    """
    dix_b, diy_b = np.broadcast_arrays(np.asarray(dix), np.asarray(diy))
    h = 1.0 / psf_scale
    P   = rpsf_eval(raster, dix_b - dx,       diy_b - dy,       psf_scale, center)
    Pxp = rpsf_eval(raster, dix_b - (dx + h), diy_b - dy,       psf_scale, center)
    Pxm = rpsf_eval(raster, dix_b - (dx - h), diy_b - dy,       psf_scale, center)
    Pyp = rpsf_eval(raster, dix_b - dx,       diy_b - (dy + h), psf_scale, center)
    Pym = rpsf_eval(raster, dix_b - dx,       diy_b - (dy - h), psf_scale, center)
    dPdx = (Pxp - Pxm) / (2 * h)
    dPdy = (Pyp - Pym) / (2 * h)
    return P, dPdx, dPdy


# ---------------------------------------------------------------------------
# Spatial variation: bilinear between the 4 nearest fiducials
# ---------------------------------------------------------------------------

def spatial_weights_h1(x_det, y_det, xs, ys):
    """(W (n,4), K (n,4)) hst1pass spatial weights for 0-based positions.

    Fortran: iloc = star's integer pixel (1-based) against ilist (1-based
    fiducial positions); the +1s cancel in the differences, so 0-based
    round(x) against 0-based xs gives identical weights.  Segment search
    clamps to the last interior segment, leaving fx outside [0,1] at the
    edges (linear extrapolation) exactly like the Fortran.
    """
    x_det = np.asarray(x_det, dtype=np.float64)
    y_det = np.asarray(y_det, dtype=np.float64)
    n = len(x_det)
    nxg, nyg = len(xs), len(ys)
    iloc = np.round(x_det).astype(np.int64)
    jloc = np.round(y_det).astype(np.int64)

    def seg(loc, grid):
        ng = len(grid)
        if ng == 1:
            return np.zeros(len(loc), dtype=np.int64), np.zeros(len(loc))
        # Fortran loop: advance while loc > grid[k+1] and k <= ng-3.
        # Grids have <= ~10 nodes, so an explicit scan replicates the
        # strict '>' condition exactly.
        k = np.zeros(len(loc), dtype=np.int64)
        for j in range(1, ng - 1):
            k = np.where(loc > grid[j], j, k)
        f = (loc - grid[k]) / (grid[k + 1] - grid[k])
        return k, f

    kx, fx = seg(iloc, np.asarray(xs, dtype=np.float64))
    ky, fy = seg(jloc, np.asarray(ys, dtype=np.float64))

    K = np.empty((n, 4), dtype=np.int32)
    W = np.empty((n, 4), dtype=np.float64)
    kx1 = np.minimum(kx + 1, nxg - 1)
    ky1 = np.minimum(ky + 1, nyg - 1)
    K[:, 0] = ky  * nxg + kx
    K[:, 1] = ky  * nxg + kx1
    K[:, 2] = ky1 * nxg + kx
    K[:, 3] = ky1 * nxg + kx1
    W[:, 0] = (1 - fx) * (1 - fy)
    W[:, 1] = fx * (1 - fy)
    W[:, 2] = (1 - fx) * fy
    W[:, 3] = fx * fy
    return W, K


class Hst1passBlender:
    """ExactTileBlender-compatible provider for the hst1pass scheme.

    Tiles are RAW raster crops (margin covers the quadratic stencil plus
    sub-pixel drift); spatial blending is the 4-fiducial bilinear.  The
    blended tile is exact: bilinear mixing commutes with rpsf_phot's linear
    dependence on raster values.
    """

    def __init__(self, psf_cube, psf_coeffs_cube_unused, xs, ys, psf_scale,
                 hw, x_offset=0.0, y_offset=0.0):
        self.xs = np.asarray(xs, dtype=np.float64)
        self.ys = np.asarray(ys, dtype=np.float64)
        self.psf_scale = int(psf_scale)
        self.hw = int(hw)
        self.x_offset = float(x_offset)
        self.y_offset = float(y_offset)
        # margin: fit window + 0.5 px drift + numeric-gradient step (1 ss px)
        # + quadratic stencil reach (2 ss px) + floor slack (1)
        self.tr = hw * self.psf_scale + self.psf_scale // 2 + 4
        self.ts = 2 * self.tr + 1

        n_psf, ny_psf, nx_psf = psf_cube.shape
        hy, hx = (ny_psf - 1) // 2, (nx_psf - 1) // 2
        tr = self.tr
        self.tile_cube = np.ascontiguousarray(
            psf_cube[:, hy - tr:hy + tr + 1, hx - tr:hx + tr + 1]
            .astype(np.float64))
        self.peak_cube = np.ascontiguousarray(
            psf_cube[:, hy, hx].astype(np.float64))
        self.n_psf = n_psf

    def weights(self, x_arr, y_arr):
        xd = np.asarray(x_arr, dtype=np.float64) + self.x_offset
        yd = np.asarray(y_arr, dtype=np.float64) + self.y_offset
        if self.n_psf == 1 or len(self.xs) < 2 or len(self.ys) < 2:
            n = len(xd)
            W = np.zeros((n, 4)); W[:, 0] = 1.0
            K = np.zeros((n, 4), dtype=np.int32)
            return W, K
        return spatial_weights_h1(xd, yd, self.xs, self.ys)

    def blend(self, W, K, out=None):
        m = len(W)
        if out is None:
            out = np.zeros((m, self.ts, self.ts), dtype=np.float64)
        else:
            out[:] = 0.0
        for j in range(W.shape[1]):
            out += W[:, j, None, None] * self.tile_cube[K[:, j]]
        return out

    def blend_one(self, w, k):
        return np.einsum('f,fab->ab', w, self.tile_cube[k])

    def peaks(self, W, K):
        return np.einsum('nf,nf->n', W, self.peak_cube[K])


def prefilter_cube(psf_cube):
    """Scheme-aware replacement for the per-model spline_filter loop.

    bspline mode: cubic B-spline coefficient cube (legacy behaviour).
    hst1pass mode: the raw cube itself — Anderson evaluation works on raw
    rasters, and every downstream evaluator routes on the scheme.
    """
    if psf_scheme() == 'hst1pass':
        return np.asarray(psf_cube, dtype=np.float64)
    from scipy.ndimage import spline_filter
    return np.array([spline_filter(p, order=3, output=np.float64)
                     for p in psf_cube])
