"""Sibling-seeded cross-match retry (user 2026-10-01).

Exposures that share one guide-star acquisition (``bp3m.visit_groups``: same proposal, visit,
dominant + roll guide stars, PA_V3, contiguous in time) share the error of their a-priori header
WCS: the guide-star catalogue position and roll error.  A sibling that cross-matched therefore
measures that error, and it can be transferred to an image that failed:

    E      = W_S,fit  o  W_S,ref^-1          (sky-plane similarity, measured on the sibling)
    W_F,pred = E  o  W_F,ref                 (prediction for the failed image)

W_X,ref is the header-only mapping used to seed the normal discovery
(pixel h -> Gaia-frame g = c + (h - c) * initial_scale -> sky through the header frame), and
W_S,fit is the sibling's fitted affine (transformation.csv) taken through the same frame.  Both
use the same reference definition, so whatever the reference gets wrong (nominal scale,
ORIENTAT, CRVAL) cancels in E.

The prediction is returned as the Gaia "guess" positions in the failed image's HST pixel frame
-- exactly the quantity the 4P discovery histograms -- so discovery, refinement and every
validation gate run unchanged, just inside a narrower offset window around a better centre.
The recorded header frame (ra_cen/dec_cen/orientat in transformation.csv, which BP3M uses as
the tangent point and rotation-prior centre) is NOT modified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .miracle_match import rd2x, rd2y

SIBLING_SEED_VERSION = 1


# ---------------------------------------------------------------------------
# Header frame: the exact forward projection used by process_single_image, and its inverse
# ---------------------------------------------------------------------------

def _rot(orientat_deg):
    th = np.radians(-orientat_deg)
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])   # init_rot_mat
    return R, np.linalg.inv(R)


def sky_to_frame(ra, dec, fr):
    """(ra, dec) deg -> Gaia-frame pixel coords g, identical to process_single_image."""
    sdeg = fr['pixel_scale'] / 3600.0
    dx = rd2x(np.asarray(ra, float), np.asarray(dec, float), fr['ra_cen'], fr['dec_cen'])
    dy = rd2y(np.asarray(ra, float), np.asarray(dec, float), fr['ra_cen'], fr['dec_cen'])
    xp = fr['x_cen'] - dx / sdeg
    yp = fr['y_cen'] + dy / sdeg
    _, Rinv = _rot(fr['orientat'])
    c = np.array([fr['x_cen'], fr['y_cen']])
    g = np.einsum('ij,nj->ni', Rinv, np.column_stack([xp, yp]) - c) + c
    return g[:, 0], g[:, 1]


def frame_to_sky(gx, gy, fr):
    """Inverse of sky_to_frame (inverse gnomonic)."""
    sdeg = fr['pixel_scale'] / 3600.0
    R, _ = _rot(fr['orientat'])
    c = np.array([fr['x_cen'], fr['y_cen']])
    p = np.einsum('ij,nj->ni', R, np.column_stack([gx, gy]) - c) + c
    xi = np.radians((fr['x_cen'] - p[:, 0]) * sdeg)
    eta = np.radians((p[:, 1] - fr['y_cen']) * sdeg)
    a0, d0 = np.radians(fr['ra_cen']), np.radians(fr['dec_cen'])
    den = np.cos(d0) - eta * np.sin(d0)
    ra = a0 + np.arctan2(xi, den)
    dec = np.arctan2((np.sin(d0) + eta * np.cos(d0)) * np.cos(ra - a0), den)
    return np.degrees(ra) % 360.0, np.degrees(dec)


def _tangent(ra, dec, ra0, dec0):
    return rd2x(ra, dec, ra0, dec0) * 3.6e6, rd2y(ra, dec, ra0, dec0) * 3.6e6   # mas


def _untangent(u, v, ra0, dec0):
    fr = dict(pixel_scale=1.0, ra_cen=ra0, dec_cen=dec0, x_cen=0.0, y_cen=0.0, orientat=0.0)
    # x = x_cen - dx/sdeg, y = y_cen + dy/sdeg with sdeg = 1/3600 deg  ->  g = (-dx*3600, dy*3600) [arcsec]
    return frame_to_sky(-np.asarray(u) / 1000.0, np.asarray(v) / 1000.0, fr)


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------

def read_transformation(path) -> dict:
    t = pd.read_csv(path).set_index('parameter')['value']
    return {k: float(t[k]) for k in t.index}


def _grid(xr, yr, n=9):
    gx, gy = np.meshgrid(np.linspace(*xr, n), np.linspace(*yr, n))
    return gx.ravel(), gy.ravel()


def measure_header_error(sib_params: dict, sib_trans: dict, sib_xr, sib_yr):
    """Sky-plane similarity E (u' = a u - b v + tx, v' = b u + a v + ty, mas about the sibling's
    tangent point) mapping the sibling's header-only reference onto its fitted solution."""
    fr = {k: sib_trans[k] for k in ('ra_cen', 'dec_cen', 'x_cen', 'y_cen', 'pixel_scale', 'orientat')}
    hx, hy = _grid(sib_xr, sib_yr)
    s = sib_params['initial_scale']
    gxr, gyr = fr['x_cen'] + (hx - fr['x_cen']) * s, fr['y_cen'] + (hy - fr['y_cen']) * s
    A, B, C, D = sib_trans['A'], sib_trans['B'], sib_trans['C'], sib_trans['D']
    gxf = A * (hx - sib_trans['xs_o']) + B * (hy - sib_trans['ys_o']) + sib_trans['xt_o']
    gyf = C * (hx - sib_trans['xs_o']) + D * (hy - sib_trans['ys_o']) + sib_trans['yt_o']
    ra_r, de_r = frame_to_sky(gxr, gyr, fr)
    ra_f, de_f = frame_to_sky(gxf, gyf, fr)
    ra0, de0 = fr['ra_cen'], fr['dec_cen']
    ur, vr = _tangent(ra_r, de_r, ra0, de0)
    uf, vf = _tangent(ra_f, de_f, ra0, de0)
    # linear LSQ for (a, b, tx, ty)
    n = len(ur)
    M = np.zeros((2 * n, 4)); y = np.zeros(2 * n)
    M[:n, 0], M[:n, 1], M[:n, 2] = ur, -vr, 1.0
    M[n:, 0], M[n:, 1], M[n:, 3] = vr, ur, 1.0
    y[:n], y[n:] = uf, vf
    p, *_ = np.linalg.lstsq(M, y, rcond=None)
    rms = float(np.sqrt(np.mean((M @ p - y) ** 2)))
    return dict(a=p[0], b=p[1], tx=p[2], ty=p[3], ra0=ra0, dec0=de0, fit_rms_mas=rms,
                rot_arcsec=float(np.degrees(np.arctan2(p[1], p[0])) * 3600.0),
                scale_minus1=float(np.hypot(p[0], p[1]) - 1.0),
                offset_mas=float(np.hypot(p[2], p[3])))


def apply_header_error(E: dict, ra, dec):
    u, v = _tangent(np.asarray(ra, float), np.asarray(dec, float), E['ra0'], E['dec0'])
    u2 = E['a'] * u - E['b'] * v + E['tx']
    v2 = E['b'] * u + E['a'] * v + E['ty']
    return _untangent(u2, v2, E['ra0'], E['dec0'])


def predicted_affine(target_params: dict, E: dict, xr, yr):
    """Affine h -> g (g = M (h - c) + t) predicted for the target image in its own header frame."""
    fr = {k: target_params[k] for k in ('ra_cen', 'dec_cen', 'x_cen', 'y_cen', 'pixel_scale', 'orientat')}
    hx, hy = _grid(xr, yr)
    s = target_params['initial_scale']
    gxr, gyr = fr['x_cen'] + (hx - fr['x_cen']) * s, fr['y_cen'] + (hy - fr['y_cen']) * s
    ra, de = frame_to_sky(gxr, gyr, fr)
    ra2, de2 = apply_header_error(E, ra, de)
    gx, gy = sky_to_frame(ra2, de2, fr)
    X = np.column_stack([hx - fr['x_cen'], hy - fr['y_cen'], np.ones_like(hx)])
    px, *_ = np.linalg.lstsq(X, gx, rcond=None)
    py, *_ = np.linalg.lstsq(X, gy, rcond=None)
    M = np.array([[px[0], px[1]], [py[0], py[1]]]); t = np.array([px[2], py[2]])
    resid = np.hypot(X @ px - gx, X @ py - gy)
    return M, t, float(resid.max())


def sibling_guess(target_params: dict, sibling: dict, xr, yr, gx, gy):
    """Gaia guess positions (HST pixel frame of the target) from one sibling.

    sibling: {'params': get_hst_params(sibling), 'trans': read_transformation(...),
              'xr': (xmin, xmax), 'yr': (ymin, ymax)}  (sibling catalogue extent)
    gx, gy : the target's Gaia-frame positions (x_g_in, y_g_in).
    Returns (xguess, yguess, info).
    """
    E = measure_header_error(sibling['params'], sibling['trans'], sibling['xr'], sibling['yr'])
    M, t, lin_err = predicted_affine(target_params, E, xr, yr)
    c = np.array([target_params['x_cen'], target_params['y_cen']])
    h = np.einsum('ij,nj->ni', np.linalg.inv(M), np.column_stack([gx, gy]) - t) + c
    # header-only guess for comparison: c + (g - c) / initial_scale
    s = target_params['initial_scale']
    h0x = c[0] + (np.asarray(gx) - c[0]) / s; h0y = c[1] + (np.asarray(gy) - c[1]) / s
    shift = np.hypot(h[:, 0] - h0x, h[:, 1] - h0y)
    info = dict(E_offset_mas=E['offset_mas'], E_rot_arcsec=E['rot_arcsec'], E_scale_minus1=E['scale_minus1'],
                E_fit_rms_mas=E['fit_rms_mas'], affine_lin_err_px=lin_err,
                guess_shift_px_median=float(np.median(shift)) if len(shift) else float('nan'))
    return h[:, 0], h[:, 1], info


def load_sibling(flc, catalog, transformation_csv, get_hst_params):
    """Everything sibling_guess needs, from a successfully matched sibling's files."""
    from astropy.io import fits
    p = get_hst_params(flc, catalog_file=catalog)
    tr = read_transformation(transformation_csv)
    cat = fits.getdata(catalog)
    x = np.asarray(cat['x_gdc'], float); y = np.asarray(cat['y_gdc'], float)
    ok = np.isfinite(x) & np.isfinite(y)
    return {'params': p, 'trans': tr,
            'xr': (float(np.percentile(x[ok], 1)), float(np.percentile(x[ok], 99))),
            'yr': (float(np.percentile(y[ok], 1)), float(np.percentile(y[ok], 99)))}
