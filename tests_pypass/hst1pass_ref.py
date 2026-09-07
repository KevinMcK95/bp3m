"""Faithful port of hst1pass's PSF evaluation (the STDPSF-native scheme).

Ported from GaiaHub-master/fortran_codes/hst1pass.2025.02.14_v1h.F:

  * ``rpsf_phot`` (line ~4726): sub-pixel evaluation on the 4x-supersampled
    101x101 raster.  r <= 4 det px: four local quadratic Taylor patches
    (central differences at the 4 surrounding supersample nodes) blended
    bilinearly; 4 < r <= 12: bilinear; r > 12: zero.
  * ``rpsf_phot_ij_STDPSF`` (line ~9923): spatial variation by BILINEAR
    interpolation between the 4 nearest fiducial PSFs, with weights computed
    from the star's INTEGER pixel (iloc/jloc) against the ilist/jlist grid
    positions; the segment search allows linear extrapolation beyond the
    outermost fiducials.

This module is the ground-truth reference for interpolation-scheme
comparisons; it is vectorized over pixel offsets but deliberately mirrors
the Fortran arithmetic exactly (including the 1-indexed centre at raster
index 51 -> 0-based 50, and integer-pixel spatial weights).
"""
import numpy as np


def rpsf_phot(psf, dx, dy, psf_scale=4):
    """Evaluate one supersampled PSF raster at detector offsets (dx, dy).

    psf : (101, 101) raster (y, x ordering as stored by load_stdpsf)
    dx, dy : arrays of detector-pixel offsets from the star centre
    Returns array of PSF values, matching hst1pass's rpsf_phot.
    """
    dx = np.asarray(dx, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    n_half = (psf.shape[0] - 1) // 2          # 50 for 101
    rx = n_half + dx * psf_scale              # 0-based (Fortran: 51 + x*4)
    ry = n_half + dy * psf_scale
    ix = np.floor(rx).astype(int)
    iy = np.floor(ry).astype(int)
    fx = rx - ix
    fy = ry - iy
    dd = np.hypot(dx, dy)

    out = np.zeros(np.broadcast(dx, dy).shape, dtype=np.float64)
    ixc = np.clip(ix, 2, psf.shape[1] - 4)    # keep +/-2 stencils in-bounds
    iyc = np.clip(iy, 2, psf.shape[0] - 4)

    def P(jx, jy):
        return psf[iyc + jy, ixc + jx]

    # ---- bilinear regime: 4 < r <= 12 -----------------------------------
    bl = ((1 - fx) * (1 - fy) * P(0, 0) + fx * (1 - fy) * P(1, 0)
          + (1 - fx) * fy * P(0, 1) + fx * fy * P(1, 1))

    # ---- quadratic-patch regime: r <= 4 ---------------------------------
    # Patch k at node (jx, jy): value + gradient + curvature from central
    # differences, cross term from the diagonal neighbour (Fortran E terms).
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

    out = np.where(dd <= 4.0, qd, np.where(dd <= 12.0, bl, 0.0))
    return out


def spatial_weights_hst1pass(iloc, jloc, ilist, jlist):
    """(NX, NY, fx, fy) exactly as rpsf_phot_ij_STDPSF computes them.

    iloc, jloc : integer detector pixel of the star (1-indexed combined
                 frame, matching the Fortran; caller converts).
    ilist, jlist : fiducial grid positions (as read from the STDPSF header).
    Segment search allows fx/fy outside [0,1] (linear extrapolation).
    """
    nxg, nyg = len(ilist), len(jlist)
    nx = 0
    while nx <= nxg - 3 and iloc > ilist[nx + 1]:
        nx += 1
    ny = 0
    while ny <= nyg - 3 and jloc > jlist[ny + 1]:
        ny += 1
    fx = (iloc - ilist[nx]) / (ilist[nx + 1] - ilist[nx]) if nxg > 1 else 0.0
    fy = (jloc - jlist[ny]) / (jlist[ny + 1] - jlist[ny]) if nyg > 1 else 0.0
    if nxg == 1:
        nx = 0
    if nyg == 1:
        ny = 0
    return nx, ny, float(fx), float(fy)


def eval_psf_hst1pass(psf_cube, xs, ys, x_det, y_det, dx, dy, psf_scale=4,
                      grid_shape=None):
    """Full hst1pass evaluation: bilinear-of-4-fiducials of rpsf_phot values.

    psf_cube  : (n_psf, 101, 101) with k = iy_g * nx_g + ix_g
    xs, ys    : fiducial grid detector positions (ilist / jlist)
    x_det, y_det : star position in detector coords (0-based; converted to
                 the Fortran's 1-based integer pixel internally)
    dx, dy    : arrays of pixel offsets from the star centre
    """
    nx_g = len(xs)
    ny_g = len(ys)
    # Fortran iloc = integer (1-based) central pixel; ilist is 1-based too.
    iloc = int(round(x_det)) + 1
    jloc = int(round(y_det)) + 1
    NX, NY, fx, fy = spatial_weights_hst1pass(iloc, jloc, xs + 1, ys + 1)

    def kk(ix_g, iy_g):
        return int(np.clip(iy_g, 0, ny_g - 1) * nx_g
                   + np.clip(ix_g, 0, nx_g - 1))

    p00 = rpsf_phot(psf_cube[kk(NX,     NY)],     dx, dy, psf_scale)
    p10 = rpsf_phot(psf_cube[kk(NX + 1, NY)],     dx, dy, psf_scale)
    p01 = rpsf_phot(psf_cube[kk(NX,     NY + 1)], dx, dy, psf_scale)
    p11 = rpsf_phot(psf_cube[kk(NX + 1, NY + 1)], dx, dy, psf_scale)
    return ((1 - fx) * (1 - fy) * p00 + fx * (1 - fy) * p10
            + (1 - fx) * fy * p01 + fx * fy * p11)
