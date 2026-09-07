#!/usr/bin/env python3
"""Regression + performance harness for the pypass optimization campaign.

Runs pypass.core.run_photometry on fixed reference chips (API only — writes
nothing into any pipeline directory), records the catalog as flat arrays plus
wall-time / peak-RSS, and diffs a run against a stored baseline.

Cases
-----
dense_jax    : M31 j6d508ngq chip ext1, fmin=2000  -> JAX batch path
sparse_numpy : AM_4 ib2d01ibq chip ext1, production fmin -> NumPy path

Usage
-----
  python harness.py baseline <case>          # write <case>_baseline.npz
  python harness.py run <case> [tag]         # write <case>_<tag>.npz
  python harness.py diff <case> <tag>        # compare vs baseline
Outputs under /home/jupyter-kmckinnon/data_bootes/hst_dist_corr/pypass_opt/.
"""
import os, sys, time, json, resource
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np

OUT = '/home/jupyter-kmckinnon/data_bootes/hst_dist_corr/pypass_opt'

CASES = {
    'dense_jax': dict(
        img='/home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results/M31/'
            'HST/mastDownload/HST/j6d508ngq/j6d508ngq_flc.fits',
        sci_ext=1, dq_ext=3, fmin=1000.0, prefix='ACSWFC'),
    'numpy_small': dict(
        img='/home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results/M31/'
            'HST/mastDownload/HST/j6d508ngq/j6d508ngq_flc.fits',
        sci_ext=1, dq_ext=3, fmin=20000.0, prefix='ACSWFC'),
    'sparse_numpy': dict(
        img='/home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results/AM_4/'
            'HST/mastDownload/HST/ib2d01ibq/ib2d01ibq_flc.fits',
        sci_ext=1, dq_ext=3, fmin=None, prefix='WFC3UV'),
}

FIELDS = ('x', 'y', 'flux', 'flux_err', 'sky', 'qfit', 'chi2', 'psf_frac',
          'central_res', 'concentration', 'concentration_2x2',
          'concentration_3x3', 'eps_psf', 'n_iter')


def _peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576


def run_case(case):
    from pypass.io import load_stdpsf, load_image, find_psf
    from pypass.core import run_photometry
    from astropy.io import fits

    c = CASES[case]
    pp_path = os.path.join(os.path.dirname(c['img']), 'psf_params.json')
    pp = json.load(open(pp_path))
    hdr = fits.getheader(c['img'], 0)
    psf_path = find_psf(os.path.join(pp['lib_dir'], 'STDPSFs', c['prefix']), hdr)
    psf_cube, xs, ys, psf_scale, _ = load_stdpsf(psf_path)
    data, gain, rn, mask, x_off, y_off = load_image(
        c['img'], sci_ext=c['sci_ext'], dq_ext=c['dq_ext'])
    fmin = c['fmin'] if c['fmin'] is not None else pp['fmin_thresh']

    t0 = time.perf_counter()
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
    dt = time.perf_counter() - t0

    out = {f: np.array([getattr(r, f) for r in records], dtype=float)
           for f in FIELDS}
    out['converged'] = np.array([r.converged for r in records], dtype=bool)
    cov = np.array([r.cov for r in records], dtype=float)   # (n, 4, 4)
    out['cov'] = cov
    out['wall_s'] = np.array([dt])
    out['peak_gb'] = np.array([_peak_gb()])
    return out


def save(case, tag):
    out = run_case(case)
    path = os.path.join(OUT, f'{case}_{tag}.npz')
    np.savez_compressed(path, **out)
    print(f"saved {path}: n={len(out['x'])}  wall={out['wall_s'][0]:.1f}s  "
          f"peakRSS={out['peak_gb'][0]:.2f}GB")


def diff(case, tag):
    b = np.load(os.path.join(OUT, f'{case}_baseline.npz'))
    n = np.load(os.path.join(OUT, f'{case}_{tag}.npz'))
    print(f"baseline n={len(b['x'])}  {tag} n={len(n['x'])}")
    print(f"wall: {b['wall_s'][0]:.1f}s -> {n['wall_s'][0]:.1f}s   "
          f"peakRSS: {b['peak_gb'][0]:.2f} -> {n['peak_gb'][0]:.2f} GB")

    # Match rows by position (0.5 px), baseline -> new
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([n['x'], n['y']]))
    d, j = tree.query(np.column_stack([b['x'], b['y']]),
                      distance_upper_bound=0.5)
    ok = np.isfinite(d)
    print(f"matched {ok.sum()}/{len(b['x'])} baseline rows "
          f"({(~ok).sum()} unmatched)")
    conv = ok & b['converged'] & (n['converged'][np.where(ok, j, 0)])
    ji = j[conv]
    dx = n['x'][ji] - b['x'][conv]
    dy = n['y'][ji] - b['y'][conv]
    print(f"POSITION  max|dx|={np.abs(dx).max():.2e}  "
          f"max|dy|={np.abs(dy).max():.2e}  "
          f"p99|dxy|={np.percentile(np.hypot(dx, dy), 99):.2e} px   "
          f"[tolerance: 1e-4 extreme max]")
    for f in ('flux', 'sky', 'qfit', 'chi2'):
        bb, nn = b[f][conv], n[f][ji]
        denom = np.maximum(np.abs(bb), 1e-10)
        rel = np.abs(nn - bb) / denom
        print(f"{f:>6s}  max rel diff = {rel.max():.2e}")
    cb = b['cov'][conv].reshape(conv.sum(), -1)
    cn = n['cov'][ji].reshape(len(ji), -1)
    crel = np.abs(cn - cb) / np.maximum(np.abs(cb), 1e-12)
    print(f"   cov  max rel diff = {crel.max():.2e}")


if __name__ == '__main__':
    cmd, case = sys.argv[1], sys.argv[2]
    if cmd == 'baseline':
        save(case, 'baseline')
    elif cmd == 'run':
        save(case, sys.argv[3] if len(sys.argv) > 3 else 'test')
    elif cmd == 'diff':
        diff(case, sys.argv[3])
