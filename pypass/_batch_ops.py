"""Vectorized (batched-over-stars) implementations of the NumPy machinery
around the JAX Newton kernel: pixel-window extraction, flux/sky
initialisation support, post-fit sigma clipping, and record building.

Phase 2 of the pypass optimization plan.  Profiling on a dense M31 chip
showed the JAX kernel is only ~17% of wall time; the per-star Python loops
in prepare/clip/records dominate the rest.  Every function here reproduces
the corresponding per-star reference arithmetic exactly:

  * window extraction and the hst1pass evaluators are element-for-element
    the same operations, so results are bit-identical;
  * the batched B-spline stencil evaluation performs the same gathers and
    einsum contraction as ``eval_psf_on_tile`` (verified bit-identical in
    tests_pypass/test_batch_ops.py);
  * sigma clipping runs the identical per-star 4x4 re-solves (only the PSF
    evaluations are batched), so positions are bit-identical; final-metric
    reductions (cov/qfit/chi2) are computed with zero-weight full-window
    sums instead of masked BLAS products, which differs only in fp
    summation order (~1e-15 relative).

Set ``PYPASS_BATCH_OPS=0`` to fall back to the per-star reference paths.
"""

import os
import numpy as np

from ._jax_kernel import (
    tile_radius, eval_psf_on_tile, _bspline3_weights, _bspline3_dweights,
)


def use_batch_ops() -> bool:
    return os.environ.get('PYPASS_BATCH_OPS', '1').strip() != '0'


_WIN_GRIDS: dict = {}


def _win_grid(hw: int):
    """(dix, diy) flat int window-offset grids, cached per hw."""
    g = _WIN_GRIDS.get(hw)
    if g is None:
        diy_g, dix_g = np.mgrid[-hw:hw + 1, -hw:hw + 1]
        g = (dix_g.ravel().copy(), diy_g.ravel().copy())
        _WIN_GRIDS[hw] = g
    return g


# ---------------------------------------------------------------------------
# Batched pixel-window extraction (mirrors extract_pixel_window)
# ---------------------------------------------------------------------------

def extract_pixel_windows_batch(data, xs_stars, ys_stars, skies, hw,
                                mask, noise_map, gain, read_noise):
    """Vectorized extract_pixel_window over all stars.

    Returns (pixel_vals, pixel_var_rn, valid, dx0, dy0, xi, yi) with leading
    star axis; arithmetic identical to the per-star reference.
    """
    ny, nx = data.shape
    xs_stars = np.asarray(xs_stars, dtype=np.float64)
    ys_stars = np.asarray(ys_stars, dtype=np.float64)
    skies    = np.asarray(skies,    dtype=np.float64)
    dix, diy = _win_grid(hw)

    xi = np.round(xs_stars).astype(np.int64)   # banker's rounding == round()
    yi = np.round(ys_stars).astype(np.int64)
    dx0 = xs_stars - xi
    dy0 = ys_stars - yi

    px = xi[:, None] + dix[None, :]            # (n, n_pix)
    py = yi[:, None] + diy[None, :]
    in_bounds = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
    px_c = np.clip(px, 0, nx - 1)
    py_c = np.clip(py, 0, ny - 1)

    pixel_vals = np.where(in_bounds, data[py_c, px_c],
                          skies[:, None]).astype(np.float64)

    if noise_map is not None:
        pixel_var_rn = np.where(in_bounds, noise_map[py_c, px_c],
                                1e6).astype(np.float64)
    else:
        rn_per_dn = read_noise / gain
        pixel_var_rn = np.full(px.shape, rn_per_dn ** 2, dtype=np.float64)

    valid = in_bounds.copy()
    if mask is not None:
        valid &= ~mask[py_c, px_c]

    return pixel_vals, pixel_var_rn, valid, dx0, dy0, xi, yi


# ---------------------------------------------------------------------------
# Batched PSF evaluation on per-star tiles
# ---------------------------------------------------------------------------

def _eval_bspline_tiles_at(tiles, offx, offy, psf_scale, tr, want_grad=True):
    """Batched B-spline stencil evaluation at detector offsets.

    tiles : (n, ts, ts) float64 coefficient tiles
    offx, offy : (n, m) detector-pixel offsets from the PSF centre
                 (= dix - dx in eval_psf_on_tile's convention)
    Same gathers, weights and einsum contraction as ``eval_psf_on_tile``.
    """
    n, ts, _ = tiles.shape
    x_t = tr + offx * psf_scale                # (n, m)
    y_t = tr + offy * psf_scale
    ix = np.floor(x_t).astype(np.intp)
    iy = np.floor(y_t).astype(np.intp)
    tx = x_t - np.floor(x_t)
    ty = y_t - np.floor(y_t)

    kv = np.array([-1, 0, 1, 2], dtype=np.intp)
    ix_g = np.clip(ix[..., None] + kv, 0, ts - 1)    # (n, m, 4)
    iy_g = np.clip(iy[..., None] + kv, 0, ts - 1)
    ridx = np.arange(n, dtype=np.intp)[:, None, None, None]
    C = tiles[ridx, iy_g[:, :, :, None], ix_g[:, :, None, :]]  # (n, m, 4, 4)

    wx = _bspline3_weights(tx)
    wy = _bspline3_weights(ty)
    P = np.einsum('npi,npij,npj->np', wy, C, wx)
    if not want_grad:
        return P, None, None
    dwx = _bspline3_dweights(tx)
    dwy = _bspline3_dweights(ty)
    dP_dxt = np.einsum('npi,npij,npj->np', wy, C, dwx)
    dP_dyt = np.einsum('npi,npij,npj->np', dwy, C, wx)
    return P, dP_dxt * (-float(psf_scale)), dP_dyt * (-float(psf_scale))


def _rpsf_eval_tiles(tiles, offx, offy, psf_scale, center):
    """Batched hst1pass rpsf_phot over per-star raw tiles.

    Same arithmetic as hst1pass_scheme.rpsf_eval with a leading star axis:
    bit-identical per element.
    """
    n, ny_t, nx_t = tiles.shape
    rx = center + offx * psf_scale
    ry = center + offy * psf_scale
    ix = np.floor(rx).astype(np.intp)
    iy = np.floor(ry).astype(np.intp)
    fx = rx - ix
    fy = ry - iy
    dd = np.hypot(offx, offy)

    ixc = np.clip(ix, 2, nx_t - 4)
    iyc = np.clip(iy, 2, ny_t - 4)
    ridx = np.arange(n, dtype=np.intp)[:, None]

    def P(jx, jy):
        return tiles[ridx, iyc + jy, ixc + jx]

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


def eval_psf_tiles_batch(tiles, dxs, dys, hw, psf_scale, tr, scheme,
                         offx=None, offy=None, want_grad=True):
    """Batched (P, dPdx, dPdy) for stars on their tiles.

    Default evaluation points are the full fit window; pass explicit
    ``offx/offy`` (n, m) detector offsets *before* the star shift (i.e. the
    dix/diy values) to evaluate elsewhere (e.g. concentration boxes).
    """
    tiles = np.asarray(tiles, dtype=np.float64)
    dxs = np.asarray(dxs, dtype=np.float64)
    dys = np.asarray(dys, dtype=np.float64)
    if offx is None:
        dix, diy = _win_grid(hw)
        DIX = np.broadcast_to(dix.astype(np.float64), (len(dxs), dix.size))
        DIY = np.broadcast_to(diy.astype(np.float64), (len(dys), diy.size))
    else:
        DIX = np.asarray(offx, dtype=np.float64)
        DIY = np.asarray(offy, dtype=np.float64)
    offx = DIX - dxs[:, None]
    offy = DIY - dys[:, None]

    if scheme == 'hst1pass':
        P = _rpsf_eval_tiles(tiles, offx, offy, psf_scale, tr)
        if not want_grad:
            return P, None, None
        # eval_psf_grad_hst1 convention: offsets built as dix - (dx ± h),
        # in that association order, so results are bit-identical.
        h = 1.0 / psf_scale
        Pxp = _rpsf_eval_tiles(tiles, DIX - (dxs + h)[:, None], offy,
                               psf_scale, tr)
        Pxm = _rpsf_eval_tiles(tiles, DIX - (dxs - h)[:, None], offy,
                               psf_scale, tr)
        Pyp = _rpsf_eval_tiles(tiles, offx, DIY - (dys + h)[:, None],
                               psf_scale, tr)
        Pym = _rpsf_eval_tiles(tiles, offx, DIY - (dys - h)[:, None],
                               psf_scale, tr)
        return P, (Pxp - Pxm) / (2 * h), (Pyp - Pym) / (2 * h)

    return _eval_bspline_tiles_at(tiles, offx, offy, psf_scale, tr,
                                  want_grad=want_grad)


# ---------------------------------------------------------------------------
# Batched sigma clipping (public body of _sigma_clip_jax_results)
# ---------------------------------------------------------------------------

_CHUNK = 4096


def sigma_clip_results_batch(jax_res, inputs_dict, gain,
                             sigma_clip_sigma, sigma_clip_iter,
                             scheme):
    """Vectorized twin of the per-star sigma-clip reference.

    PSF evaluations are batched; the 4x4 re-solves run per star only for the
    (typically few) stars that actually have outliers, using the identical
    reference arithmetic — positions are bit-identical to the reference.
    Final metrics use zero-weight full-window reductions (fp summation
    order differs at ~1e-15 relative).
    """
    has_noise_map = inputs_dict.get('has_noise_map', False)
    hw        = inputs_dict['hw']
    psf_scale = inputs_dict['psf_scale']
    tr        = inputs_dict.get('tile_radius', tile_radius(hw, psf_scale))
    n_pix     = (2 * hw + 1) ** 2
    center    = n_pix // 2
    n_stars   = len(jax_res['flux'])

    flux_arr        = jax_res['flux'].astype(np.float64).copy()
    dx_arr          = jax_res['dx'].astype(np.float64).copy()
    dy_arr          = jax_res['dy'].astype(np.float64).copy()
    sky_arr         = jax_res['sky'].astype(np.float64).copy()
    cov_arr         = jax_res['cov'].copy()
    qfit_arr        = jax_res['qfit'].copy()
    chi2_arr        = jax_res['chi2'].copy()
    psf_frac_arr    = jax_res['psf_frac'].copy()
    central_res_arr = jax_res['central_res'].copy()
    clipped_masks   = np.zeros((n_stars, n_pix), dtype=bool)

    coeff_tiles = inputs_dict['psf_coeff_tiles']
    pixel_vals  = inputs_dict['pixel_vals']
    pixel_var   = inputs_dict['pixel_var_rn']
    valid_masks = inputs_dict['valid_masks']

    def _var_batch(flux, P, sky, pvar):
        if has_noise_map:
            return np.maximum(pvar, 1e-10)
        return np.maximum(
            (np.maximum(flux[:, None] * P, 0.0)
             + np.maximum(sky, 0.0)[:, None]) / gain + pvar,
            1e-10,
        )

    for a in range(0, n_stars, _CHUNK):
        b = min(a + _CHUNK, n_stars)
        m = b - a
        tiles = np.asarray(coeff_tiles[a:b], dtype=np.float64)
        pv    = pixel_vals[a:b]
        pvar  = pixel_var[a:b]
        valid = valid_masks[a:b].copy()
        flux  = flux_arr[a:b].copy()
        dx    = dx_arr[a:b].copy()
        dy    = dy_arr[a:b].copy()
        sky   = sky_arr[a:b].copy()

        active = np.ones(m, dtype=bool)
        for _round in range(sigma_clip_iter):
            idx = np.nonzero(active)[0]
            if idx.size == 0:
                break
            P, dPdx, dPdy = eval_psf_tiles_batch(
                tiles[idx], dx[idx], dy[idx], hw, psf_scale, tr, scheme)
            var = _var_batch(flux[idx], P, sky[idx], pvar[idx])
            r = pv[idx] - sky[idx, None] - flux[idx, None] * P
            outlier = np.abs(r) / np.sqrt(var) > sigma_clip_sigma
            new_valid = valid[idx] & ~outlier
            has_out = (outlier & valid[idx]).any(axis=1)
            nv_sum = new_valid.sum(axis=1)
            # reference break: keep current valid, stop
            stop_keep = (nv_sum < 5) | ~has_out
            active[idx[stop_keep]] = False

            for j in np.nonzero(~stop_keep)[0]:
                i = idx[j]
                g = new_valid[j]
                n_g = int(g.sum())
                w_g = 1.0 / var[j][g]
                A = np.column_stack([P[j][g], flux[i] * dPdx[j][g],
                                     flux[i] * dPdy[j][g], np.ones(n_g)])
                AtWA = A.T @ (w_g[:, None] * A) + 1e-10 * np.eye(4)
                AtWr = A.T @ (w_g * r[j][g])
                try:
                    delta = np.linalg.solve(AtWA, AtWr)
                except np.linalg.LinAlgError:
                    valid[i] = g
                    active[i] = False
                    continue
                if abs(delta[1]) > 0.5 or abs(delta[2]) > 0.5:
                    valid[i] = g
                    active[i] = False
                    continue
                flux[i] = max(flux[i] + delta[0], 1.0)
                dx[i]  += delta[1]
                dy[i]  += delta[2]
                sky[i] += delta[3]
                valid[i] = g

        # --- Final evaluation at the post-clipping positions (all stars) ---
        P, dPdx, dPdy = eval_psf_tiles_batch(
            tiles, dx, dy, hw, psf_scale, tr, scheme)
        var = _var_batch(flux, P, sky, pvar)
        r = pv - sky[:, None] - flux[:, None] * P
        w = np.where(valid, 1.0 / var, 0.0)
        n_g = valid.sum(axis=1)

        A = np.empty((m, n_pix, 4), dtype=np.float64)
        A[:, :, 0] = P
        A[:, :, 1] = flux[:, None] * dPdx
        A[:, :, 2] = flux[:, None] * dPdy
        A[:, :, 3] = 1.0
        AtWA = np.einsum('npk,np,npl->nkl', A, w, A) + 1e-6 * np.eye(4)
        few = n_g < 4
        if few.any():
            AtWA[few] = np.eye(4)          # placeholder; cov overwritten below
        cov = np.linalg.inv(AtWA)
        if few.any():
            cov[few] = np.eye(4) * 1e6

        sum_abs_res  = np.sum(np.abs(r) * valid, axis=1)
        sum_abs_data = np.sum(np.abs(pv - sky[:, None]) * valid, axis=1)
        qfit = sum_abs_res / np.maximum(sum_abs_data, 1e-10)
        dof  = np.maximum(n_g - 4, 1)
        chi2 = np.sqrt(np.sum(r ** 2 / var * valid, axis=1) / dof)

        psf_frac = P[:, center]
        central_res = np.clip(
            (pv[:, center] - sky - flux * psf_frac)
            / np.maximum(flux, 1e-10),
            -0.999, 0.999,
        )

        clipped_masks[a:b] = valid_masks[a:b] & ~valid
        flux_arr[a:b] = flux
        dx_arr[a:b]   = dx
        dy_arr[a:b]   = dy
        sky_arr[a:b]  = sky
        cov_arr[a:b]  = cov
        qfit_arr[a:b] = qfit
        chi2_arr[a:b] = chi2
        psf_frac_arr[a:b]    = psf_frac
        central_res_arr[a:b] = central_res

    return dict(
        flux          = flux_arr,
        dx            = dx_arr,
        dy            = dy_arr,
        sky           = sky_arr,
        cov           = cov_arr,
        n_iter        = jax_res['n_iter'],
        converged     = jax_res['converged'],
        delta_max     = jax_res['delta_max'],
        qfit          = qfit_arr,
        chi2          = chi2_arr,
        psf_frac      = psf_frac_arr,
        central_res   = central_res_arr,
        clipped_masks = clipped_masks,
    )


# ---------------------------------------------------------------------------
# Batched record building (public body of _jax_results_to_records)
# ---------------------------------------------------------------------------

def records_from_jax_batch(jax_res, inputs_dict, pass_number, gain,
                           zero_point, sat_threshold, scheme,
                           star_record_cls):
    """Vectorized twin of the per-star record builder.

    Scalar per-star fields are computed as arrays; the concentration boxes
    are evaluated with the batched tile evaluators.  In bspline mode the
    2x2/3x3 boxes use the exact separable stencil instead of
    ``map_coordinates`` (identical B-spline sum; interior values agree to
    fp rounding).  The final StarRecord construction is a plain loop over
    precomputed scalars.
    """
    hw        = inputs_dict['hw']
    tr        = inputs_dict['tile_radius']
    psf_scale = int(inputs_dict.get('psf_scale', 4))
    n_pix     = (2 * hw + 1) ** 2
    nx_win    = 2 * hw + 1
    center    = n_pix // 2
    n_stars   = len(jax_res['flux'])

    flux_arr = jax_res['flux'].astype(np.float64)
    sky_arr  = jax_res['sky'].astype(np.float64)
    dx_a     = jax_res['dx'].astype(np.float64)
    dy_a     = jax_res['dy'].astype(np.float64)
    cov_a    = jax_res['cov']
    chi2_a   = jax_res['chi2'].astype(np.float64)
    pf_a     = jax_res['psf_frac'].astype(np.float64)
    pixel_vals = inputs_dict['pixel_vals']

    x_a = inputs_dict['xi'].astype(np.float64) + dx_a
    y_a = inputs_dict['yi'].astype(np.float64) + dy_a

    psf_peaks = inputs_dict['psf_peak'].astype(np.float64)
    peaks     = pixel_vals[:, center] - sky_arr
    n_sat_arr = np.sum(pixel_vals > sat_threshold, axis=1)

    flux_err_a = np.sqrt(np.maximum(cov_a[:, 0, 0], 0.0))
    sky_err_a  = np.sqrt(np.maximum(cov_a[:, 3, 3], 0.0))

    # mag_from_flux, vectorized (flux clamped > 0 exactly as the reference)
    _fl = np.maximum(flux_arr, 1e-10)
    mag_a = zero_point - 2.5 * np.log10(_fl)
    with np.errstate(invalid='ignore'):
        mag_err_a = np.where(np.isfinite(flux_err_a) & (flux_err_a >= 0),
                             (2.5 / np.log(10.0)) * flux_err_a / _fl,
                             np.inf)

    eps_psf_a = chi2_a / np.sqrt(
        np.maximum(flux_arr * np.maximum(pf_a, 1e-6) * gain, 1.0))

    vm = inputs_dict['valid_masks']
    cm_all = jax_res.get('clipped_masks', None)
    good = vm & ~cm_all if cm_all is not None else vm.copy()

    cen_good = good[:, center]
    conc_denom = flux_arr * np.maximum(pf_a, 1e-10)
    with np.errstate(divide='ignore', invalid='ignore'):
        conc1 = np.where((conc_denom > 0) & cen_good,
                         peaks / conc_denom, np.nan)
    n_conc1 = cen_good.astype(int)

    # --- 2x2 / 3x3 concentration boxes, batched --------------------------
    coeff_tiles = inputs_dict.get('psf_coeff_tiles', None)
    conc2 = np.full(n_stars, np.nan)
    conc3 = np.full(n_stars, np.nan)
    n_conc2 = np.zeros(n_stars, dtype=int)
    n_conc3 = np.zeros(n_stars, dtype=int)
    if coeff_tiles is not None and n_stars > 0:
        lo_x = np.where(dx_a >= 0, 0, -1).astype(np.int64)
        lo_y = np.where(dy_a >= 0, 0, -1).astype(np.int64)
        OFFX2 = lo_x[:, None] + np.array([0, 1, 0, 1])
        OFFY2 = lo_y[:, None] + np.array([0, 0, 1, 1])
        off3 = np.array([-1, 0, 1])
        DIX3, DIY3 = np.meshgrid(off3, off3)
        OFFX3 = np.broadcast_to(DIX3.ravel(), (n_stars, 9))
        OFFY3 = np.broadcast_to(DIY3.ravel(), (n_stars, 9))

        def _box(OFFX, OFFY):
            idx = np.clip((OFFY + hw) * nx_win + (OFFX + hw), 0, n_pix - 1)
            g = np.take_along_axis(good, idx, axis=1)
            d = np.take_along_axis(pixel_vals, idx, axis=1).astype(np.float64) \
                - sky_arr[:, None]
            return idx, g, d

        idx2, g2, d2 = _box(OFFX2, OFFY2)
        idx3, g3, d3 = _box(OFFX3, OFFY3)

        P2 = np.empty((n_stars, 4))
        P3 = np.empty((n_stars, 9))
        for a in range(0, n_stars, _CHUNK):
            b = min(a + _CHUNK, n_stars)
            tiles = np.asarray(coeff_tiles[a:b], dtype=np.float64)
            P2[a:b], _, _ = eval_psf_tiles_batch(
                tiles, dx_a[a:b], dy_a[a:b], hw, psf_scale, tr, scheme,
                offx=OFFX2[a:b].astype(np.float64),
                offy=OFFY2[a:b].astype(np.float64), want_grad=False)
            P3[a:b], _, _ = eval_psf_tiles_batch(
                tiles, dx_a[a:b], dy_a[a:b], hw, psf_scale, tr, scheme,
                offx=OFFX3[a:b].astype(np.float64),
                offy=OFFY3[a:b].astype(np.float64), want_grad=False)

        pos_flux = flux_arr > 0
        P2_sum = np.sum(P2 * g2, axis=1)
        n_conc2 = np.where(pos_flux, g2.sum(axis=1), 0)
        ok2 = pos_flux & (n_conc2 >= 2) & (P2_sum > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            conc2 = np.where(ok2, np.sum(d2 * g2, axis=1)
                             / (flux_arr * P2_sum), np.nan)
        P3_sum = np.sum(P3 * g3, axis=1)
        n_conc3 = np.where(pos_flux, g3.sum(axis=1), 0)
        ok3 = pos_flux & (n_conc3 >= 5) & (P3_sum > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            conc3 = np.where(ok3, np.sum(d3 * g3, axis=1)
                             / (flux_arr * P3_sum), np.nan)

    qfit_a  = jax_res['qfit']
    cres_a  = jax_res['central_res']
    niter_a = jax_res['n_iter']
    conv_a  = jax_res['converged']
    dmax_a  = jax_res['delta_max']

    records = []
    for i in range(n_stars):
        cm = cm_all[i] if cm_all is not None else None
        _rec_append(records, star_record_cls, i, x_a, y_a, flux_arr,
                    flux_err_a, sky_arr, sky_err_a, mag_a, mag_err_a,
                    qfit_a, chi2_a, cres_a, n_sat_arr, pf_a, psf_peaks,
                    peaks, cov_a, pass_number, niter_a, conv_a, dmax_a, cm,
                    eps_psf_a, conc1, conc2, conc3, n_conc1, n_conc2,
                    n_conc3)

    return records


def _rec_append(records, star_record_cls, i, x_a, y_a, flux_arr, flux_err_a,
                sky_arr, sky_err_a, mag_a, mag_err_a, qfit_a, chi2_a, cres_a,
                n_sat_arr, pf_a, psf_peaks, peaks, cov_a, pass_number,
                niter_a, conv_a, dmax_a, cm, eps_psf_a, conc1, conc2, conc3,
                n_conc1, n_conc2, n_conc3):
    records.append(star_record_cls(
        x=float(x_a[i]), y=float(y_a[i]),
        flux=float(flux_arr[i]), flux_err=float(flux_err_a[i]),
        sky=float(sky_arr[i]), sky_err=float(sky_err_a[i]),
        mag=float(mag_a[i]), mag_err=float(mag_err_a[i]),
        qfit=float(qfit_a[i]),
        chi2=float(chi2_a[i]),
        central_res=float(cres_a[i]),
        n_sat=int(n_sat_arr[i]),
        psf_frac=float(pf_a[i]),
        psf_peak=float(psf_peaks[i]),
        peak=float(peaks[i]),
        cov=cov_a[i].copy(),
        pass_number=pass_number,
        n_neighbors=0,
        dist_nearest=np.inf,
        dist_nearest_brighter=np.inf,
        n_iter=int(niter_a[i]),
        converged=bool(conv_a[i]),
        delta_max=float(dmax_a[i]),
        clipped_mask=cm,
        chi2_scale=1.0,
        eps_psf=float(eps_psf_a[i]),
        concentration=float(conc1[i]) if np.isfinite(conc1[i]) else np.nan,
        concentration_2x2=float(conc2[i]) if np.isfinite(conc2[i]) else np.nan,
        concentration_3x3=float(conc3[i]) if np.isfinite(conc3[i]) else np.nan,
        n_conc_1x1=int(n_conc1[i]),
        n_conc_2x2=int(n_conc2[i]),
        n_conc_3x3=int(n_conc3[i]),
    ))


# ---------------------------------------------------------------------------
# Phase 3: batched image-plane star-model application
# ---------------------------------------------------------------------------

def spatial_blend_weights(x_arr, y_arr, xs, ys, scheme,
                          x_offset=0.0, y_offset=0.0, n_psf=None):
    """(W, K) per-star spatial blend weights over the full PSF cube.

    hst1pass: bilinear of the 4 nearest fiducials with integer-pixel weights
    (identical to spatial_weights_h1 — bit-exact vs the per-record path).
    bspline: 16-weight Catmull-Rom contraction replicating interpolate_psf
    exactly (the per-record path additionally reuses cached tiles within
    5-px cells; this path always blends at the true position).
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    xd = np.asarray(x_arr, dtype=np.float64) + x_offset
    yd = np.asarray(y_arr, dtype=np.float64) + y_offset
    n = len(xd)
    nx_g, ny_g = len(xs), len(ys)
    single = (n_psf == 1) or nx_g < 2 or ny_g < 2

    if scheme == 'hst1pass':
        if single:
            W = np.zeros((n, 4)); W[:, 0] = 1.0
            K = np.zeros((n, 4), dtype=np.int32)
            return W, K
        from .hst1pass_scheme import spatial_weights_h1
        return spatial_weights_h1(xd, yd, xs, ys)

    from .tile_provider import _catmull_rom_weights_vec
    if single:
        W = np.zeros((n, 16)); W[:, 0] = 1.0
        K = np.zeros((n, 16), dtype=np.int32)
        return W, K
    gx = np.arange(nx_g, dtype=np.float64)
    gy = np.arange(ny_g, dtype=np.float64)
    tx_full = np.interp(xd, xs, gx)
    ty_full = np.interp(yd, ys, gy)
    ix = np.minimum(np.floor(tx_full), nx_g - 2).astype(np.int64)
    iy = np.minimum(np.floor(ty_full), ny_g - 2).astype(np.int64)
    wx = _catmull_rom_weights_vec(tx_full - ix)
    wy = _catmull_rom_weights_vec(ty_full - iy)
    off = np.arange(-1, 3)
    ix_idx = np.clip(ix[:, None] + off[None, :], 0, nx_g - 1)
    iy_idx = np.clip(iy[:, None] + off[None, :], 0, ny_g - 1)
    K = (iy_idx[:, :, None] * nx_g + ix_idx[:, None, :]) \
        .reshape(n, 16).astype(np.int32)
    W = (wy[:, :, None] * wx[:, None, :]).reshape(n, 16)
    return W, K


_APPLY_CHUNK = 512


def apply_star_models_batch(target, xs_stars, ys_stars, fluxes, full_cube,
                            xs, ys, psf_scale, shw, x_offset, y_offset,
                            scheme, mode='add'):
    """Accumulate each star's PSF model over its full footprint into *target*.

    mode='add'      : target += flux * P        (restore)
    mode='subtract' : target -= flux * P
    mode='poisson'  : target += max(flux*P, 0) / `fluxes2` ... (not used; see
                      add_poisson_variance_batch)

    full_cube is the coefficient cube (bspline) or raw cube (hst1pass).
    Batched replacement of the per-record _psf_window loops in multipass.
    """
    _accumulate_models(target, xs_stars, ys_stars, fluxes, full_cube, xs, ys,
                       psf_scale, shw, x_offset, y_offset, scheme,
                       sign=-1.0 if mode == 'subtract' else 1.0,
                       poisson_gain=None)


def add_poisson_variance_batch(var_image, xs_stars, ys_stars, fluxes,
                               full_cube, xs, ys, psf_scale, shw,
                               x_offset, y_offset, scheme, gain):
    """var_image += max(flux * P, 0) / gain over each star's footprint."""
    _accumulate_models(var_image, xs_stars, ys_stars, fluxes, full_cube, xs,
                       ys, psf_scale, shw, x_offset, y_offset, scheme,
                       sign=1.0, poisson_gain=gain)


def _accumulate_models(target, xs_stars, ys_stars, fluxes, full_cube, xs, ys,
                       psf_scale, shw, x_offset, y_offset, scheme, sign,
                       poisson_gain):
    n = len(xs_stars)
    if n == 0:
        return
    ny, nx = target.shape
    xs_stars = np.asarray(xs_stars, dtype=np.float64)
    ys_stars = np.asarray(ys_stars, dtype=np.float64)
    fluxes   = np.asarray(fluxes,   dtype=np.float64)
    full_cube = np.asarray(full_cube, dtype=np.float64)
    n_psf, S, _ = full_cube.shape
    tr_full = (S - 1) // 2
    side = 2 * shw + 1

    xi = np.round(xs_stars).astype(np.int64)
    yi = np.round(ys_stars).astype(np.int64)
    dxs = xs_stars - xi
    dys = ys_stars - yi

    W, K = spatial_blend_weights(xs_stars, ys_stars, xs, ys, scheme,
                                 x_offset=x_offset, y_offset=y_offset,
                                 n_psf=n_psf)

    for a in range(0, n, _APPLY_CHUNK):
        b = min(a + _APPLY_CHUNK, n)
        m = b - a
        tiles = np.zeros((m, S, S), dtype=np.float64)
        for j in range(W.shape[1]):
            tiles += W[a:b, j, None, None] * full_cube[K[a:b, j]]
        P, _, _ = eval_psf_tiles_batch(tiles, dxs[a:b], dys[a:b], shw,
                                       psf_scale, tr_full, scheme,
                                       want_grad=False)
        for j in range(m):
            i = a + j
            y_lo = max(0, yi[i] - shw); y_hi = min(ny, yi[i] + shw + 1)
            x_lo = max(0, xi[i] - shw); x_hi = min(nx, xi[i] + shw + 1)
            if y_lo >= y_hi or x_lo >= x_hi:
                continue
            P2 = P[j].reshape(side, side)[
                y_lo - (yi[i] - shw): y_hi - (yi[i] - shw),
                x_lo - (xi[i] - shw): x_hi - (xi[i] - shw)]
            if poisson_gain is not None:
                target[y_lo:y_hi, x_lo:x_hi] += \
                    np.maximum(fluxes[i] * P2, 0.0) / poisson_gain
            else:
                target[y_lo:y_hi, x_lo:x_hi] += sign * fluxes[i] * P2
