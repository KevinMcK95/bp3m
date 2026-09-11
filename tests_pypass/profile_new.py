#!/usr/bin/env python3
"""cProfile the phase-2/3 code on a dense bspline chip at production fmin.

Answers: where do the ~9 ms/star go now that the per-star machinery is
vectorized?  Prints top functions by cumulative and total time.
"""
import os, sys, json, time, cProfile, pstats, io, resource
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np

IMGDIR = ('/home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results/M31/'
          'HST/mastDownload/HST/j6d508ngq')
IMG = f'{IMGDIR}/j6d508ngq_flc.fits'

import pypass
print('pypass from:', pypass.__file__, flush=True)
from pypass.io import load_stdpsf, load_image, find_psf
from pypass.core import run_photometry
from astropy.io import fits

pp = json.load(open(f'{IMGDIR}/psf_params.json'))
hdr = fits.getheader(IMG, 0)
psf_path = find_psf(os.path.join(pp['lib_dir'], 'STDPSFs', 'ACSWFC'), hdr)
psf_cube, xs, ys, psf_scale, _ = load_stdpsf(psf_path)
data, gain, rn, mask, x_off, y_off = load_image(IMG, sci_ext=1, dq_ext=3)
fmin = pp['fmin_thresh']
print(f'production fmin={fmin}', flush=True)

pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable()
records = run_photometry(
    data=data, psf_models=psf_cube, psf_positions=(xs, ys),
    psf_scale=psf_scale, half_width=pp['half_width'],
    sky_inner=pp['sky_inner'], sky_outer=pp['sky_outer'], hmin=pp['hmin'],
    fmin=fmin, max_iter_fit=pp['max_iter_fit'], tol=pp['tol'],
    n_passes=pp['n_passes'], n_discovery_passes=pp['n_discovery_passes'],
    gain=gain, read_noise=rn, zero_point=0.0, mask=mask, peak_mask=mask,
    verbose=True, x_offset=x_off, y_offset=y_off,
    sat_threshold=pp['sat_threshold'], sigma_clip=pp['sigma_clip'],
    sigma_clip_sigma=pp['sigma_clip_sigma'], n_jobs=1, backend='auto',
    conc_limit=pp['conc_limit'])
pr.disable()
dt = time.perf_counter() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
print(f'\nTOTAL: {dt:.1f}s  n_records={len(records)}  peakRSS={peak:.2f}GB',
      flush=True)

for sort in ('cumulative', 'tottime'):
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sort)
    ps.print_stats(28)
    txt = s.getvalue()
    # trim path noise
    txt = txt.replace('/home/jupyter-kmckinnon/.conda/envs/bp3m-test/'
                      'lib/python3.11/site-packages/', '')
    print(f'\n===== sorted by {sort} =====')
    print('\n'.join(txt.splitlines()[:42]), flush=True)
