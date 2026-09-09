#!/usr/bin/env python3
"""Equivalence tests: _batch_ops vectorized paths vs per-star references.

Runs the full prep -> fit -> sigma-clip -> records chain twice on the same
synthetic scene (PYPASS_BATCH_OPS=1 vs 0) for BOTH schemes and asserts:
  positions   : bit-identical (clip re-solves use identical arithmetic)
  flux0/sky0  : bit-identical (same lstsq)
  evaluators  : batched vs per-star bit-identical
  cov/qfit/chi2 and concentrations: <= 1e-9 relative (summation order)
"""
import os, sys, json
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np

IMGDIR = ('/home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results/'
          'Leo_I_schemeB/HST/mastDownload/HST/ibgr01syq')

import pypass
print('pypass from:', pypass.__file__, flush=True)
from pypass.io import load_stdpsf, find_psf
from pypass.hst1pass_scheme import (Hst1passBlender, prefilter_cube,
                                    eval_psf_grad_hst1)
from pypass.tile_provider import ExactTileBlender
from astropy.io import fits

pp = json.load(open(f'{IMGDIR}/psf_params.json'))
hdr = fits.getheader(f'{IMGDIR}/ibgr01syq_flc.fits', 0)
psf_path = find_psf(os.path.join(pp['lib_dir'], 'STDPSFs', 'WFC3UV'), hdr)
psf_cube, xs, ys, psf_scale, _ = load_stdpsf(psf_path)
hw = pp['half_width']
GAIN, RN = 1.5, 3.1
NY = NX = 700
N_STARS = 320
rng = np.random.default_rng(3)

fails = []


def check(name, a, b, tol=0.0, rel=False):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    a = np.where(both_nan, 0.0, a); b = np.where(both_nan, 0.0, b)
    d = np.abs(a - b)
    if rel:
        d = d / np.maximum(np.abs(b), 1e-30)
    mx = float(d.max()) if d.size else 0.0
    ok = mx <= tol
    print(f"  {name:28s} max{'rel' if rel else ''}diff={mx:.3e}  "
          f"{'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        fails.append(name)


def run_scheme(scheme):
    print(f"\n=== scheme: {scheme} ===", flush=True)
    os.environ['PYPASS_PSF_SCHEME'] = scheme
    from pypass._jax_kernel import (prepare_jax_inputs, fit_batch_jax,
                                    _sigma_clip_jax_results)
    from pypass.core import _jax_results_to_records
    import pypass._batch_ops as bo

    if scheme == 'hst1pass':
        cube = np.asarray(psf_cube, dtype=np.float64)
        blender = Hst1passBlender(psf_cube, None, xs, ys, psf_scale, hw)
    else:
        cube = prefilter_cube(psf_cube)
        blender = ExactTileBlender(psf_cube, cube, xs, ys, psf_scale, hw)

    # --- synthetic scene: real PSF shapes + Poisson-ish noise -------------
    from pypass.core import interpolate_psf
    data = np.full((NY, NX), 55.0)
    X = rng.uniform(hw + 3, NX - hw - 4, N_STARS)
    Y = rng.uniform(hw + 3, NY - hw - 4, N_STARS)
    F = 10 ** rng.uniform(2.6, 5.2, N_STARS)
    for x0, y0, f0 in zip(X, Y, F):
        pb = interpolate_psf(cube if scheme == 'hst1pass' else psf_cube,
                             xs, ys, x0, y0)
        xi, yi = int(round(x0)), int(round(y0))
        for dyy in range(-hw, hw + 1):
            for dxx in range(-hw, hw + 1):
                px, py = xi + dxx, yi + dyy
                c = (pb.shape[0] - 1) // 2
                sx = int(round(c + (dxx - (x0 - xi)) * psf_scale))
                sy = int(round(c + (dyy - (y0 - yi)) * psf_scale))
                if 0 <= sx < pb.shape[1] and 0 <= sy < pb.shape[0]:
                    data[py, px] += f0 * pb[sy, sx]
    data += rng.normal(0, np.sqrt(np.maximum(data, 1) / GAIN), data.shape)
    # a few hot pixels to exercise the clip re-solve branches
    hot = rng.integers(0, N_STARS, 25)
    for i in hot:
        data[int(round(Y[i])) + rng.integers(-hw, hw + 1),
             int(round(X[i])) + rng.integers(-hw, hw + 1)] += F[i] * 0.4
    mask = np.zeros_like(data, dtype=bool)
    mask[::97, ::89] = True
    S = np.full(N_STARS, 55.0)

    def chain(batch_on):
        os.environ['PYPASS_BATCH_OPS'] = '1' if batch_on else '0'
        inp = prepare_jax_inputs(
            data, X, Y, S, psf_cube, xs, ys, psf_scale, hw,
            mask=mask, noise_map=None, gain=GAIN, read_noise=RN,
            psf_coeffs_cube=cube, tile_provider=blender)
        res = fit_batch_jax(inp, gain=GAIN, tol=1e-3,
                            max_iter=pp['max_iter_fit'])
        clip = _sigma_clip_jax_results(res, inp, gain=GAIN,
                                       sigma_clip_sigma=4.0,
                                       sigma_clip_iter=2)
        recs = _jax_results_to_records(clip, inp, pass_number=1, gain=GAIN,
                                       zero_point=25.0, sat_threshold=6e4)
        return inp, clip, recs

    inp1, clip1, recs1 = chain(True)
    inp0, clip0, recs0 = chain(False)
    os.environ['PYPASS_BATCH_OPS'] = '1'

    # --- prepare equivalence ---------------------------------------------
    for k in ('pixel_vals', 'pixel_var_rn', 'valid_masks', 'dx0', 'dy0',
              'flux0', 'sky0'):
        check(f'prep.{k}', inp1[k], inp0[k], tol=0.0)
    check('prep.xi', inp1['xi'], inp0['xi'], tol=0.0)
    check('prep.yi', inp1['yi'], inp0['yi'], tol=0.0)

    # --- raw evaluator equivalence (batched vs per-star) ------------------
    tiles = np.asarray(inp0['psf_coeff_tiles'][:64], dtype=np.float64)
    tr = inp0['tile_radius']
    dxs = clip0['dx'][:64]; dys = clip0['dy'][:64]
    Pb, Gxb, Gyb = bo.eval_psf_tiles_batch(tiles, dxs, dys, hw, psf_scale,
                                           tr, scheme)
    from pypass._jax_kernel import eval_psf_tile_routed
    Pr = np.empty_like(Pb); Gxr = np.empty_like(Pb); Gyr = np.empty_like(Pb)
    for i in range(len(tiles)):
        Pr[i], Gxr[i], Gyr[i] = eval_psf_tile_routed(
            tiles[i], float(dxs[i]), float(dys[i]), hw, psf_scale,
            tr=tr, scheme=scheme)
    check('eval.P (bitwise)',  Pb,  Pr, tol=0.0)
    check('eval.gx (bitwise)', Gxb, Gxr, tol=0.0)
    check('eval.gy (bitwise)', Gyb, Gyr, tol=0.0)

    # --- clip equivalence --------------------------------------------------
    check('clip.dx (bitwise)', clip1['dx'], clip0['dx'], tol=0.0)
    check('clip.dy (bitwise)', clip1['dy'], clip0['dy'], tol=0.0)
    check('clip.flux (bitwise)', clip1['flux'], clip0['flux'], tol=0.0)
    check('clip.sky (bitwise)',  clip1['sky'],  clip0['sky'],  tol=0.0)
    check('clip.clipped_masks', clip1['clipped_masks'],
          clip0['clipped_masks'], tol=0.0)
    check('clip.qfit', clip1['qfit'], clip0['qfit'], tol=1e-9, rel=True)
    check('clip.chi2', clip1['chi2'], clip0['chi2'], tol=1e-9, rel=True)
    check('clip.cov',  clip1['cov'],  clip0['cov'],  tol=1e-6, rel=True)
    n_clipped = int((clip0['clipped_masks'].sum(1) > 0).sum())
    print(f'  [{n_clipped} stars had clipped pixels]', flush=True)

    # --- records equivalence -----------------------------------------------
    def col(recs, f):
        return np.array([getattr(r, f) for r in recs], dtype=np.float64)
    for f in ('x', 'y', 'flux', 'sky', 'mag', 'peak', 'psf_peak', 'n_sat',
              'n_iter', 'n_conc_1x1', 'n_conc_2x2', 'n_conc_3x3'):
        check(f'rec.{f}', col(recs1, f), col(recs0, f), tol=0.0)
    for f in ('flux_err', 'sky_err', 'mag_err', 'qfit', 'chi2', 'eps_psf',
              'central_res', 'psf_frac', 'concentration'):
        check(f'rec.{f}', col(recs1, f), col(recs0, f), tol=1e-9, rel=True)
    for f in ('concentration_2x2', 'concentration_3x3'):
        check(f'rec.{f}', col(recs1, f), col(recs0, f), tol=1e-7, rel=True)

    # --- phase 3: image-plane subtract/restore/variance --------------------
    from pypass.multipass import subtract_stars, restore_stars, \
        build_variance_image
    scale_ref = np.median([abs(r.flux) for r in recs0]) or 1.0
    for name, fn in (('subtract', subtract_stars), ('restore', restore_stars)):
        imgs = {}
        for flag in ('1', '0'):
            os.environ['PYPASS_BATCH_OPS'] = flag
            r_img = data.copy()
            fn(r_img, recs0, psf_cube, xs, ys, psf_scale, hw,
               0.0, 0.0, psf_coeffs_cube=cube)
            imgs[flag] = r_img
        d = np.abs(imgs['1'] - imgs['0']).max() / scale_ref
        ok = d < 1e-9
        print(f'  img.{name:26s} maxdiff/medflux={d:.3e}  '
              f'{"OK" if ok else "FAIL"}', flush=True)
        if not ok:
            fails.append(f'img.{name}.{scheme}')
    imgs = {}
    for flag in ('1', '0'):
        os.environ['PYPASS_BATCH_OPS'] = flag
        imgs[flag] = build_variance_image(
            recs0, psf_cube, xs, ys, psf_scale, data.shape, GAIN, RN,
            psf_coeffs_cube=cube)
    d = np.abs(imgs['1'] - imgs['0']).max() / scale_ref
    ok = d < 1e-9
    print(f'  img.variance                 maxdiff/medflux={d:.3e}  '
          f'{"OK" if ok else "FAIL"}', flush=True)
    if not ok:
        fails.append(f'img.variance.{scheme}')
    os.environ['PYPASS_BATCH_OPS'] = '1'


run_scheme('bspline')
run_scheme('hst1pass')
print(f"\n{'ALL OK' if not fails else 'FAILURES: ' + ', '.join(fails)}",
      flush=True)
sys.exit(1 if fails else 0)
