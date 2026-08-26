"""
6P affine refinement and final pass — instrument-independent.

Both stages query the FULL Gaia catalogue for the image, not a quality-filtered
subset.  That is what puts 2p stars into the output: they never survive the
discovery seed (no reliable offset vote without proper motions), but once a good
transform exists they match on position like anything else.

One-to-one matching is enforced here, at the point where candidate pairs are
turned into matches, by `.drop_duplicates` on BOTH the Gaia index and the source
index after sorting by log-probability cost.  Do not relax this: a Gaia source
matched to two detections (or vice versa) double-counts that star in the
alignment likelihood.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from gaia_cross_match.catalog_matcher import (
    fit_affine_weighted, apply_affine, compute_mahalanobis, compute_logprob_cost,
)

from .discovery import GaiaField
from .instruments.base import InstrumentConfig, SourceCatalog

SIGMA_CUT = 5.0


def _model_covariance(cat, h_v, M, xs_o, ys_o, C_params):
    """Transform-parameter covariance propagated to each source position."""
    C_proj = np.einsum('ij,njk,lk->nil', M, cat.C_src[h_v], M)
    dxh, dyh = cat.xi[h_v] - xs_o, cat.eta[h_v] - ys_o
    J = np.zeros((len(h_v), 2, 6))
    J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = dxh, dyh, 1.0
    J[:, 1, 3], J[:, 1, 4], J[:, 1, 5] = dxh, dyh, 1.0
    return C_proj + np.einsum('nij,jk,nlk->nil', J, C_params, J)


def _candidate_pairs(cat, gaia, tree, params, radius):
    """Every (source, Gaia) pair within `radius` of the predicted position."""
    A, B, C, D, xs_o, ys_o, xt_o, yt_o = params
    xi_p, eta_p = apply_affine(cat.xi, cat.eta, A, B, C, D, xs_o, ys_o, xt_o, yt_o)
    ds, g_idx = tree.query(np.column_stack([xi_p, eta_p]), k=5,
                           distance_upper_bound=radius)
    h_all = np.repeat(np.arange(len(cat.xi)), 5)
    ok = ds.ravel() < radius
    return h_all[ok], g_idx.ravel()[ok], xi_p, eta_p


def _score_pairs(cat, gaia, h_v, g_v, xi_p, eta_p, M, xs_o, ys_o,
                 C_params, resid_cov, zp, config):
    """Mahalanobis sigma and log-prob cost for candidate pairs."""
    dx = gaia.xi[g_v] - xi_p[h_v]
    dy = gaia.eta[g_v] - eta_p[h_v]
    C_tot = gaia.C[g_v] + _model_covariance(cat, h_v, M, xs_o, ys_o, C_params) + resid_cov

    sig = compute_mahalanobis(dx, dy, C_tot)
    cost = compute_logprob_cost(dx, dy, C_tot)
    mag_diff = gaia.mag[g_v] - cat.mag[h_v]
    cost = cost + ((mag_diff - zp) / 1.0) ** 2
    cost[np.abs(mag_diff - zp) > config.max_mag_diff] = np.inf

    return pd.DataFrame({
        'h': h_v, 'g': g_v, 's': sig, 'c': cost,
        'dx': dx, 'dy': dy, 'md': mag_diff,
        'cxx': C_tot[:, 0, 0], 'cyy': C_tot[:, 1, 1],
    })


def _dedup(df):
    """Greedy one-to-one: best cost wins, each source and each Gaia used once."""
    return (df.sort_values('c')
              .drop_duplicates('g')
              .drop_duplicates('h'))


def _spread(values):
    """Half the 16-84 percentile range — a robust 1-sigma."""
    return 0.5 * float(np.diff(np.nanpercentile(values, [16, 84]))[0])


def run_affine_refinement(cat: SourceCatalog, gaia: GaiaField,
                          config: InstrumentConfig, best: dict,
                          use_resid_floor=True, verbose=True):
    """
    Upgrade the 4P discovery seed to a 6P affine transform and iterate to
    convergence over the full Gaia catalogue.

    Returns a dict of the converged transform plus the matched index arrays,
    or None if the initial residual sanity gate fails.
    """
    h_b, g_b = best['h_v'], best['g_v']
    zp = best['zp']
    M = np.array([[best['A'], best['B']], [best['C'], best['D']]])

    res, _, C_params, _ = fit_affine_weighted(
        cat.xi[h_b], cat.eta[h_b], gaia.xi[g_b], gaia.eta[g_b],
        cat.C_src[h_b], gaia.C[g_b], initial_M=M,
        sigma_rot_deg=config.sigma_rot_deg, sigma_scale=config.sigma_scale,
        sigma_skew=None, skew_prior=config.sigma_skew)
    A, B, C, D, xs_o, ys_o, xt_o, yt_o = res
    M = np.array([[A, B], [C, D]])

    xi_p, eta_p = apply_affine(cat.xi[h_b], cat.eta[h_b], A, B, C, D,
                               xs_o, ys_o, xt_o, yt_o)
    init_rx = _spread(gaia.xi[g_b] - xi_p)
    init_ry = _spread(gaia.eta[g_b] - eta_p)
    resid_cov = (np.diag([init_rx**2, init_ry**2]) if use_resid_floor
                 else np.zeros((2, 2)))

    if verbose:
        scale = np.sqrt(A * D - B * C)
        rot = np.degrees(np.arctan2(B - C, A + D))
        print(f'  Init 6P: {len(h_b)} seeds, scale={scale:.6f}, rot={rot:.4f}deg, '
              f'on_skew={0.5*(A-D):.2e}, off_skew={0.5*(B+C):.2e}, '
              f'resid=[{init_rx:.2f},{init_ry:.2f}]mas, zp={zp:.3f}')

    tree = KDTree(np.column_stack([gaia.xi, gaia.eta]))
    h_f, g_f = h_b, g_b

    for it in range(10):
        params = (A, B, C, D, xs_o, ys_o, xt_o, yt_o)
        h_v, g_v, xi_p, eta_p = _candidate_pairs(cat, gaia, tree, params,
                                                 config.refine_radius)
        if len(h_v) < config.min_matches:
            break

        scored = _score_pairs(cat, gaia, h_v, g_v, xi_p, eta_p, M, xs_o, ys_o,
                              C_params, resid_cov, zp, config)
        mdf = _dedup(scored)
        good = mdf[(mdf['s'] < SIGMA_CUT)
                   & (np.abs(mdf['md'] - zp) < config.max_mag_diff)]
        if len(good) < config.min_matches:
            break

        h_f, g_f = good['h'].values, good['g'].values
        res_new, _, C_params, _ = fit_affine_weighted(
            cat.xi[h_f], cat.eta[h_f], gaia.xi[g_f], gaia.eta[g_f],
            cat.C_src[h_f], gaia.C[g_f], initial_M=M,
            sigma_rot_deg=config.sigma_rot_deg, sigma_scale=config.sigma_scale,
            sigma_skew=None, skew_prior=config.sigma_skew)
        change = abs(res_new[0] - A) + abs(res_new[1] - B)
        A, B, C, D, xs_o, ys_o, xt_o, yt_o = res_new
        M = np.array([[A, B], [C, D]])

        rx, ry = _spread(good['dx']), _spread(good['dy'])
        resid_cov = (np.diag([rx**2, ry**2]) if use_resid_floor
                     else np.zeros((2, 2)))
        zp = float(np.median(good['md']))

        if verbose:
            scale = np.sqrt(A * D - B * C)
            rot = np.degrees(np.arctan2(B - C, A + D))
            print(f'  Iter {it}: {len(h_f)} matches, scale={scale:.6f}, '
                  f'rot={rot:.4f}deg, resid=[{rx:.2f},{ry:.2f}]mas, zp={zp:.3f}')
        if it > 3 and change < 1e-10:
            break

    return {
        'A': A, 'B': B, 'C': C, 'D': D,
        'xs_o': xs_o, 'ys_o': ys_o, 'xt_o': xt_o, 'yt_o': yt_o,
        'C_params': C_params, 'resid_cov': resid_cov, 'zp': zp,
        'h_f': h_f, 'g_f': g_f,
        'init_resid_xi': init_rx, 'init_resid_eta': init_ry,
    }


def final_pass(cat: SourceCatalog, gaia: GaiaField, config: InstrumentConfig,
               fit: dict, verbose=True):
    """
    One last full-catalogue pass with the converged transform.

    Returns (matched, rejected) DataFrames of scored pairs, both already
    deduplicated to one-to-one.
    """
    M = np.array([[fit['A'], fit['B']], [fit['C'], fit['D']]])
    params = (fit['A'], fit['B'], fit['C'], fit['D'],
              fit['xs_o'], fit['ys_o'], fit['xt_o'], fit['yt_o'])

    tree = KDTree(np.column_stack([gaia.xi, gaia.eta]))
    h_v, g_v, xi_p, eta_p = _candidate_pairs(cat, gaia, tree, params,
                                             config.refine_radius)
    if len(h_v) < config.min_matches:
        return None, None

    scored = _score_pairs(cat, gaia, h_v, g_v, xi_p, eta_p, M,
                          fit['xs_o'], fit['ys_o'], fit['C_params'],
                          fit['resid_cov'], fit['zp'], config)
    allp = _dedup(scored)
    keep = ((allp['s'] < SIGMA_CUT)
            & (np.abs(allp['md'] - fit['zp']) < config.max_mag_diff))
    matched, rejected = allp[keep], allp[~keep]
    if len(matched) < config.min_matches:
        return None, None
    if verbose:
        n2p = int((~gaia.has_pms[matched['g'].values]).sum())
        print(f'  Final matches found: {len(matched)} ({n2p} Gaia 2p)')
    return matched, rejected


__all__ = ['run_affine_refinement', 'final_pass', 'SIGMA_CUT']
