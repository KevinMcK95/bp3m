#!/usr/bin/env python3
"""Validate the h1 sigma-clip routing fix on real Leo_I data.

Runs prep -> fit_batch_jax -> _sigma_clip_jax_results in hst1pass mode on a
subset of stars and reports how far the clip step moves positions relative
to the (correct) JAX kernel result.  Under the OLD code the mis-centred
B-spline evaluation drags mid-brightness stars by ~0.2 px; under the FIX
the clip should only touch genuine outliers (shifts ~ mpx, few stars).
"""
import os, json, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ['PYPASS_PSF_SCHEME'] = 'hst1pass'
import numpy as np

IMGDIR = ('/home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results/'
          'Leo_I_schemeB/HST/mastDownload/HST/ibgr01syq')
IMG = f'{IMGDIR}/ibgr01syq_flc.fits'
N = 3000

import pypass
print('pypass from:', pypass.__file__, flush=True)
from pypass.io import load_stdpsf, load_image, find_psf
from pypass._jax_kernel import (prepare_jax_inputs, fit_batch_jax,
                                _sigma_clip_jax_results)
from pypass.hst1pass_scheme import Hst1passBlender
from astropy.io import fits

pp = json.load(open(f'{IMGDIR}/psf_params.json'))
hdr = fits.getheader(IMG, 0)
det = 'WFC3UV' if hdr.get('INSTRUME', '').startswith('WFC3') else 'ACSWFC'
psf_path = find_psf(os.path.join(pp['lib_dir'], 'STDPSFs', det), hdr)
psf_cube, xs, ys, psf_scale, _ = load_stdpsf(psf_path)
data, gain, rn, mask, x_off, y_off = load_image(IMG, sci_ext=1, dq_ext=3)
hw = pp['half_width']

cat = fits.open(f'{IMGDIR}/ibgr01syq_flc_catalog.fits')[1].data
sel = np.arange(len(cat))
rng = np.random.default_rng(11)
pick = rng.choice(sel, size=min(N, len(sel)), replace=False)
X = np.asarray(cat['x'][pick], dtype=np.float64)
Y = np.asarray(cat['y'][pick], dtype=np.float64)
S = np.asarray(cat['sky'][pick], dtype=np.float64)
inchip = (Y < data.shape[0] - hw - 2) & (Y > hw + 2) \
         & (X < data.shape[1] - hw - 2) & (X > hw + 2)
X, Y, S = X[inchip], Y[inchip], S[inchip]
print(f'{len(X)} stars', flush=True)

blender = Hst1passBlender(psf_cube, None, xs, ys, psf_scale, hw,
                          x_offset=x_off, y_offset=y_off)
raw_cube = np.asarray(psf_cube, dtype=np.float64)
inputs = prepare_jax_inputs(
    data, X, Y, S, raw_cube, xs, ys, psf_scale, hw,
    mask=mask, noise_map=None, gain=gain, read_noise=rn,
    x_offset=x_off, y_offset=y_off, psf_coeffs_cube=raw_cube,
    tile_provider=blender)
res = fit_batch_jax(inputs, gain=gain, tol=1e-3, max_iter=pp['max_iter_fit'])
dx_fit = res['dx'].copy(); dy_fit = res['dy'].copy()
conv = res['converged'].copy()
print(f'converged {conv.sum()}/{len(X)}', flush=True)

res2 = _sigma_clip_jax_results(res, inputs, gain=gain,
                               sigma_clip_sigma=4.0, sigma_clip_iter=2)
ddx = res2['dx'] - dx_fit
ddy = res2['dy'] - dy_fit
dr = np.hypot(ddx, ddy)[conv]
fl = res2['flux'][conv]
nclip = res2['clipped_masks'].sum(axis=1)[conv]
print(f"clip moved positions: median={np.median(dr)*1e3:.3f} mpx  "
      f"p90={np.percentile(dr,90)*1e3:.3f}  p99={np.percentile(dr,99)*1e3:.3f}  "
      f"max={dr.max()*1e3:.1f} mpx")
print(f"stars with >=1 clipped pixel: {(nclip>0).sum()}/{conv.sum()} "
      f"({(nclip>0).mean()*100:.1f}%)")
mid = (fl > 5e3) & (fl < 5e4)
print(f"mid-flux stars: n={mid.sum()} med |shift|={np.median(dr[mid])*1e3:.3f} mpx  "
      f"med ddx={np.median(ddx[conv][mid])*1e3:+.3f}  "
      f"med ddy={np.median(ddy[conv][mid])*1e3:+.3f}")
print(f"qfit: med={np.median(res2['qfit'][conv]):.3f}  "
      f"chi2 med={np.median(res2['chi2'][conv]):.3f}")
print('CLIP CHECK DONE')
