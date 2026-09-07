#!/usr/bin/env python3
"""Direct (isolated) PSF-quantization position error.

For a sample of real detections, fit the SAME pixel windows twice with
fit_batch_jax: once with exact-position coefficient tiles, once with
quantized tiles — piecewise-constant (cell-centre) and bilinear corner-blend
schemes, at several cell sizes.  Any dx/dy difference is purely the tile
approximation: no residual/variance coupling, no detection/dedup chaos.
Tolerance: max |dx,dy| < 1e-4 px (user requirement).
"""
import os, json, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np

IMGDIR = ('/home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results/M31/'
          'HST/mastDownload/HST/j6d508ngq')
IMG = f'{IMGDIR}/j6d508ngq_flc.fits'
BASE = ('/home/jupyter-kmckinnon/data_bootes/hst_dist_corr/pypass_opt/'
        'dense_jax_baseline.npz')
N_SAMPLE = 4000

from pypass.io import load_stdpsf, load_image, find_psf
from pypass.core import interpolate_psf
from pypass._jax_kernel import (prepare_jax_inputs, fit_batch_jax,
                                 tile_radius)
from astropy.io import fits

pp = json.load(open(f'{IMGDIR}/psf_params.json'))
hdr = fits.getheader(IMG, 0)
psf_path = find_psf(os.path.join(pp['lib_dir'], 'STDPSFs', 'ACSWFC'), hdr)
psf_cube, xs, ys, psf_scale, _ = load_stdpsf(psf_path)
data, gain, rn, mask, x_off, y_off = load_image(IMG, sci_ext=1, dq_ext=3)
hw = pp['half_width']
from scipy.ndimage import spline_filter
coeffs_cube = np.array([spline_filter(p, order=3, output=np.float64)
                        for p in psf_cube])

b = np.load(BASE)
conv = b['converged']
xs_b, ys_b, sky_b = b['x'][conv], b['y'][conv], b['sky'][conv]
rng = np.random.default_rng(1)
pick = rng.choice(len(xs_b), size=min(N_SAMPLE, len(xs_b)), replace=False)
X, Y, SKY = xs_b[pick], ys_b[pick], sky_b[pick]
print(f'{len(X)} stars sampled from baseline')

tr = tile_radius(hw, psf_scale)


def blended_coeffs(x_det, y_det):
    return interpolate_psf(coeffs_cube, xs, ys, x_det, y_det)


def crop(c):
    ny_, nx_ = c.shape
    hy, hx = (ny_ - 1) // 2, (nx_ - 1) // 2
    return np.asarray(c[hy - tr:hy + tr + 1, hx - tr:hx + tr + 1],
                      dtype=np.float64)


class FixedTiles:
    """IndexedTiles-alike backed by an explicit per-star tile array."""
    def __init__(self, arr): self.arr = arr
    def __len__(self): return len(self.arr)
    def __getitem__(self, k): return self.arr[k]


def fit_with_tiles(inputs, tiles):
    d = dict(inputs)
    d['psf_coeff_tiles'] = FixedTiles(tiles)
    res = fit_batch_jax(d, gain=gain, tol=pp['tol'],
                        max_iter=pp['max_iter_fit'])
    return res['dx'], res['dy'], res['converged']


# Base inputs (pixel windows etc.) — provider-free exact prepare
inputs = prepare_jax_inputs(
    data, X, Y, SKY, psf_cube, xs, ys, psf_scale, hw,
    mask=mask, noise_map=None, gain=gain, read_noise=rn,
    x_offset=x_off, y_offset=y_off, psf_coeffs_cube=coeffs_cube)

exact_tiles = np.asarray(inputs['psf_coeff_tiles'])
dx0, dy0, conv0 = fit_with_tiles(inputs, exact_tiles)

xd = X + x_off
yd = Y + y_off

for cell in (10.0, 25.0, 50.0, 100.0, 200.0):
    # piecewise-constant: tile at cell centre
    cxs = (np.floor(xd / cell) + 0.5) * cell
    cys = (np.floor(yd / cell) + 0.5) * cell
    cache = {}
    t_pc = np.empty_like(exact_tiles)
    for i in range(len(X)):
        key = (round(float(cxs[i]), 3), round(float(cys[i]), 3))
        t = cache.get(key)
        if t is None:
            t = crop(blended_coeffs(cxs[i], cys[i]))
            cache[key] = t
        t_pc[i] = t
    dx1, dy1, conv1 = fit_with_tiles(inputs, t_pc)

    # bilinear corner blend: 4 corner tiles weighted by in-cell fraction
    x0c = np.floor(xd / cell) * cell
    y0c = np.floor(yd / cell) * cell
    fx = (xd - x0c) / cell
    fy = (yd - y0c) / cell
    ccache = {}
    def corner(cx, cy):
        key = (round(float(cx), 3), round(float(cy), 3))
        t = ccache.get(key)
        if t is None:
            t = crop(blended_coeffs(cx, cy))
            ccache[key] = t
        return t
    t_bl = np.empty_like(exact_tiles)
    for i in range(len(X)):
        c00 = corner(x0c[i],        y0c[i])
        c10 = corner(x0c[i] + cell, y0c[i])
        c01 = corner(x0c[i],        y0c[i] + cell)
        c11 = corner(x0c[i] + cell, y0c[i] + cell)
        t_bl[i] = ((1 - fx[i]) * (1 - fy[i]) * c00 + fx[i] * (1 - fy[i]) * c10
                   + (1 - fx[i]) * fy[i] * c01 + fx[i] * fy[i] * c11)
    dx2, dy2, conv2 = fit_with_tiles(inputs, t_bl)

    both1 = conv0 & conv1
    both2 = conv0 & conv2
    e1 = np.hypot(dx1[both1] - dx0[both1], dy1[both1] - dy0[both1])
    e2 = np.hypot(dx2[both2] - dx0[both2], dy2[both2] - dy0[both2])
    print(f"cell={cell:5.0f}px  const: max={e1.max():.2e} p99={np.percentile(e1,99):.2e} "
          f"(n_cells={len(cache)})   bilinear: max={e2.max():.2e} "
          f"p99={np.percentile(e2,99):.2e} (n_corners={len(ccache)})", flush=True)
print('DIRECT QUANT TEST DONE')
