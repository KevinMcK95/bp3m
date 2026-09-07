#!/usr/bin/env python3
"""Position systematics of pypass's interpolation scheme vs hst1pass's.

For a sample of real detections (same pixel windows, same Poisson noise
model, same Newton configuration), fit each star twice:

  A. pypass model  — prefiltered cubic B-spline sub-pixel evaluation on the
     Catmull-Rom spatially-blended coefficient tile (the production path,
     via fit_batch_jax on exact tiles);
  B. hst1pass model — Anderson's rpsf_phot quadratic-patch/bilinear scheme
     with bilinear-of-4-fiducials spatial weights (tests_pypass.hst1pass_ref),
     fit with an equivalent 4-parameter Newton (numeric gradients, one
     supersample-pixel central differences — the same convention pypass's
     numba kernel uses).

The Δ(x, y) between converged A and B fits is the systematic offset the
pypass scheme carries relative to the scheme the STDPSF library is built
for.  Also reports the raw model disagreement max|ΔP|/P_peak per window.
"""
import os, json
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np

IMGDIR = ('/home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results/M31/'
          'HST/mastDownload/HST/j6d508ngq')
IMG = f'{IMGDIR}/j6d508ngq_flc.fits'
BASE = ('/home/jupyter-kmckinnon/data_bootes/hst_dist_corr/pypass_opt/'
        'dense_jax_baseline.npz')
N_SAMPLE = 1200

from pypass.io import load_stdpsf, load_image, find_psf
from pypass._jax_kernel import prepare_jax_inputs, fit_batch_jax
from pypass.tile_provider import ExactTileBlender
from hst1pass_ref import eval_psf_hst1pass
from astropy.io import fits
from scipy.ndimage import spline_filter

pp = json.load(open(f'{IMGDIR}/psf_params.json'))
hdr = fits.getheader(IMG, 0)
psf_path = find_psf(os.path.join(pp['lib_dir'], 'STDPSFs', 'ACSWFC'), hdr)
psf_cube, xs, ys, psf_scale, _ = load_stdpsf(psf_path)
data, gain, rn, mask, x_off, y_off = load_image(IMG, sci_ext=1, dq_ext=3)
hw = pp['half_width']
coeffs_cube = np.array([spline_filter(p, order=3, output=np.float64)
                        for p in psf_cube])

b = np.load(BASE)
ok = b['converged'] & (b['qfit'] < 0.3) & (b['chi2'] < 3)
xs_b, ys_b, sky_b, fl_b = (b['x'][ok], b['y'][ok], b['sky'][ok],
                           b['flux'][ok])
rng = np.random.default_rng(7)
pick = rng.choice(len(xs_b), size=min(N_SAMPLE, len(xs_b)), replace=False)
X, Y, SKY = xs_b[pick], ys_b[pick], sky_b[pick]
print(f'{len(X)} well-fit stars sampled', flush=True)

# ---- Fit A: pypass model -------------------------------------------------
blender = ExactTileBlender(psf_cube, coeffs_cube, xs, ys, psf_scale, hw,
                           x_offset=x_off, y_offset=y_off)
inputs = prepare_jax_inputs(
    data, X, Y, SKY, psf_cube, xs, ys, psf_scale, hw,
    mask=mask, noise_map=None, gain=gain, read_noise=rn,
    x_offset=x_off, y_offset=y_off, psf_coeffs_cube=coeffs_cube,
    tile_provider=blender)
TOL = 1e-5   # well below the comparison scale; prod tol=1e-3 would floor it
resA = fit_batch_jax(inputs, gain=gain, tol=TOL,
                     max_iter=pp['max_iter_fit'])
xA = inputs['xi'] + resA['dx']
yA = inputs['yi'] + resA['dy']
convA = resA['converged']
print('fit A (pypass model) done', flush=True)

# ---- Fit B: hst1pass model ----------------------------------------------
diy_g, dix_g = np.mgrid[-hw:hw + 1, -hw:hw + 1]
dix_f = dix_g.ravel().astype(np.float64)
diy_f = diy_g.ravel().astype(np.float64)
h_gr = 1.0 / psf_scale        # one supersample pixel, pypass's convention
rn2 = (rn / gain) ** 2

xB = np.full(len(X), np.nan)
yB = np.full(len(X), np.nan)
fB = np.full(len(X), np.nan)
convB = np.zeros(len(X), dtype=bool)
model_dev = np.full(len(X), np.nan)

for i in range(len(X)):
    pv    = inputs['pixel_vals'][i]
    valid = inputs['valid_masks'][i]
    xi_i  = float(inputs['xi'][i]); yi_i = float(inputs['yi'][i])
    dxs   = float(inputs['dx0'][i]); dys = float(inputs['dy0'][i])
    flux  = float(inputs['flux0'][i]); sky = float(inputs['sky0'][i])
    xd    = xi_i + x_off; yd = yi_i + y_off

    def model_P(ddx, ddy):
        return eval_psf_hst1pass(psf_cube, xs, ys, xd, yd,
                                 dix_f - ddx, diy_f - ddy, psf_scale)

    conv = False
    for _ in range(pp['max_iter_fit']):
        P   = model_P(dxs, dys)
        # d/d(star x) of P(dix - dxs): central difference in the star coord
        dPdx = (model_P(dxs + h_gr, dys) - model_P(dxs - h_gr, dys)) / (2 * h_gr)
        dPdy = (model_P(dxs, dys + h_gr) - model_P(dxs, dys - h_gr)) / (2 * h_gr)
        var = np.maximum(np.maximum(pv, flux * np.maximum(P, 0) + max(sky, 0))
                         / gain + rn2, 1e-10)
        w = np.where(valid, 1.0 / var, 0.0)
        r = pv - sky - flux * P
        A = np.column_stack([P, flux * dPdx, flux * dPdy,
                             np.ones(len(P))])
        Aw = A * w[:, None]
        try:
            delta = np.linalg.solve(Aw.T @ A + 1e-6 * np.eye(4), Aw.T @ r)
        except np.linalg.LinAlgError:
            break
        flux = max(flux + delta[0], 1.0)
        dxs += delta[1]; dys += delta[2]; sky += delta[3]
        if max(abs(delta[1]), abs(delta[2])) < TOL:
            conv = True
            break
    xB[i] = xi_i + dxs
    yB[i] = yi_i + dys
    fB[i] = flux
    convB[i] = conv

    # raw model disagreement at the pypass-fit position
    Pp = np.asarray(inputs['psf_coeff_tiles'][i])
    from pypass._jax_kernel import eval_psf_on_tile
    P_py, _, _ = eval_psf_on_tile(Pp, float(resA['dx'][i]),
                                  float(resA['dy'][i]), hw, psf_scale)
    P_h1 = model_P(float(resA['dx'][i]), float(resA['dy'][i]))
    model_dev[i] = np.abs(P_py - P_h1).max() / max(P_py.max(), 1e-10)
    if (i + 1) % 200 == 0:
        print(f'  {i+1}/{len(X)}', flush=True)

both = convA & convB
dx = xA[both] - xB[both]
dy = yA[both] - yB[both]
dr = np.hypot(dx, dy)
frel = np.abs(resA['flux'][both] - fB[both]) / np.maximum(fB[both], 1)
print(f"\nconverged in both: {both.sum()}/{len(X)}")
print(f"POSITION pypass-vs-hst1pass:  median={np.median(dr):.2e}  "
      f"p99={np.percentile(dr, 99):.2e}  max={dr.max():.2e} px")
print(f"  dx: median={np.median(dx):+.2e}  dy: median={np.median(dy):+.2e} "
      f"(coherent bias if nonzero)")
print(f"FLUX  rel: median={np.median(frel):.2e}  p99={np.percentile(frel, 99):.2e}")
md = model_dev[np.isfinite(model_dev)]
print(f"MODEL max|dP|/P_peak per window: median={np.median(md):.2e}  "
      f"p99={np.percentile(md, 99):.2e}")
print('SCHEME COMPARE DONE')
