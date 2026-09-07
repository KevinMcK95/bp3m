"""Unit check: production hst1pass scheme vs the reference Fortran port."""
import os, json
import numpy as np
from pypass.io import load_stdpsf, find_psf
from pypass.hst1pass_scheme import Hst1passBlender, eval_psf_grad_hst1
from hst1pass_ref import eval_psf_hst1pass
from astropy.io import fits

IMGDIR = ('/home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results/M31/'
          'HST/mastDownload/HST/j6d508ngq')
pp = json.load(open(f'{IMGDIR}/psf_params.json'))
hdr = fits.getheader(f'{IMGDIR}/j6d508ngq_flc.fits', 0)
psf_path = find_psf(os.path.join(pp['lib_dir'], 'STDPSFs', 'ACSWFC'), hdr)
psf_cube, xs, ys, psf_scale, _ = load_stdpsf(psf_path)
hw = pp['half_width']

rng = np.random.default_rng(3)
bl = Hst1passBlender(psf_cube, None, xs, ys, psf_scale, hw)
diy, dix = np.mgrid[-hw:hw + 1, -hw:hw + 1]

worst = 0.0
for _ in range(60):
    x_det = rng.uniform(50, 4040)
    y_det = rng.uniform(50, 2000)
    dx = rng.uniform(-0.5, 0.5)
    dy = rng.uniform(-0.5, 0.5)
    W, K = bl.weights(np.array([x_det]), np.array([y_det]))
    tile = bl.blend_one(W[0], K[0])
    P_new, _, _ = eval_psf_grad_hst1(tile, dx, dy, dix, diy, psf_scale)
    P_ref = eval_psf_hst1pass(psf_cube, xs, ys, x_det, y_det,
                              (dix - dx).ravel(), (diy - dy).ravel(),
                              psf_scale).reshape(P_new.shape)
    worst = max(worst, float(np.abs(P_new - P_ref).max()))
print(f"max |P_new - P_ref| over 60 random stars: {worst:.3e}  "
      f"({'PASS' if worst < 1e-12 else 'FAIL'})")
