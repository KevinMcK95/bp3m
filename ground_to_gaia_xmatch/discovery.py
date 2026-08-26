"""
Instrument-independent cross-match: 4P discovery, 6P affine refinement, final pass.

This is the algorithm that used to be duplicated between cfht_unions_gaia_xmatch.py
and lsst_gaia_xmatch.py.  It operates purely on a normalised SourceCatalog (mas on
the tangent plane) plus a GaiaField, so it never needs to know which telescope it
is looking at.  Instrument scale constants arrive via InstrumentConfig; extra
quality axes for the tier-walk arrive via a source_tiers callable.

Structure (unchanged from the CFHT implementation this was hoisted from):

  run_4p_discovery      tier-walk source quality x magnitude, vote on the offset,
                        seed-match, greedy 1-to-1 dedup, iterative 4P fit with
                        distance rejection, scale/rotation sanity gate.  Returns
                        the minimum-red_cost tier.

  GAIA_TIERS            clean 5p -> clean 5p+2p -> all Gaia, each tried with
                        stars-only sources then all sources.  The Gaia seed mask
                        restricts ONLY the offset-vote histogram; the seed match
                        that follows always uses the full seed subset, so 2p
                        stars contribute pairs in every tier.

  run_affine_refinement upgrade the 4P seed to 6P and iterate over the full Gaia
                        catalogue.  Enforces 1-to-1 via drop_duplicates on both
                        sides.

  final_pass            one more full-catalogue pass, keeping sigma < 5.

Two invariants that are easy to break and expensive to debug:
  * Matching is strictly one-to-one.  Every stage that produces pairs ends with
    .drop_duplicates on the Gaia index AND the source index.
  * Gaia quality is a seed mask, never a filter.  2p stars have no RUWE, so any
    hard clean_label cut removes all of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from gaia_cross_match.catalog_matcher import (
    fit_4p_weighted, fit_affine_weighted, apply_affine,
    compute_mahalanobis, compute_logprob_cost, find_scale_and_offset,
)

from .instruments.base import InstrumentConfig, SourceCatalog

# Cap on the Gaia seed subset used for the offset vote (CFHT n_subset).
SEED_SUBSET = 1000


@dataclass
class GaiaField:
    """Gaia sources near one image, projected to the same tangent plane [mas]."""

    xi: np.ndarray
    eta: np.ndarray
    C: np.ndarray             # (n, 2, 2) propagated position covariance [mas^2]
    mag: np.ndarray
    err: np.ndarray           # det(C)^(1/4) [mas], used to rank seed quality
    has_pms: np.ndarray       # bool — 5p/6p (finite pmra); False for 2p
    clean: np.ndarray         # bool — RUWE-based clean_label.  SEED MASK ONLY.
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.xi)


# ── Source tier-walk ─────────────────────────────────────────────────────────

def magnitude_tiers(cat: SourceCatalog, config: InstrumentConfig,
                    stars_only: bool) -> Iterator[tuple[str, np.ndarray]]:
    """
    Default source tier-walk: magnitude only.

    Adapters with extra quality axes (CFHT's qfit/chi2) supply their own
    generator with this signature.
    """
    base = cat.is_star if stars_only else np.ones(len(cat), dtype=bool)
    base = base & np.isfinite(cat.mag)
    if not base.any():
        return
    mags = cat.mag[base]
    limits = np.arange(mags.min() + 1.0, mags.max() + 0.5, config.mag_step)
    if len(limits) == 0:
        limits = np.array([mags.max()])
    limits[-1] = mags.max()
    for mlim in limits:
        yield f'm<{mlim:.1f}', base & (cat.mag <= mlim)


# ── Gaia quality tiers ───────────────────────────────────────────────────────

def gaia_tiers(gaia: GaiaField) -> list[tuple[str, np.ndarray]]:
    """
    Gaia seed-quality tiers, loosest last.

    These masks gate the offset-vote histogram only.  Note that 'clean 5p+2p'
    is genuinely different from 'clean 5p' only because clean is not applied as
    a filter upstream — if it were, every 2p star would already be gone and
    these tiers would collapse into each other.
    """
    return [
        ('clean 5p',    gaia.clean & gaia.has_pms),
        ('clean 5p+2p', gaia.clean),
        ('all Gaia',    np.ones(len(gaia), dtype=bool)),
    ]


# ── 4P discovery ─────────────────────────────────────────────────────────────

def run_4p_discovery(cat: SourceCatalog, gaia: GaiaField, config: InstrumentConfig,
                     seed_quality_mask=None, stars_only=True,
                     source_tiers: Callable = magnitude_tiers,
                     verbose=True):
    """
    Tier-walk to find a physically plausible 4P similarity seed.

    seed_quality_mask restricts ONLY the offset-vote histogram.  The KDTree seed
    match afterwards uses the full seed subset, so 2p stars contribute pairs
    even in the 5p-seeded tiers.

    Returns the minimum-red_cost tier dict, or None if every tier failed.
    """
    n_gaia = len(gaia)
    if n_gaia > SEED_SUBSET:
        seed_idx = np.argsort(gaia.err)[:SEED_SUBSET]
    else:
        seed_idx = np.arange(n_gaia)

    xi_seed, eta_seed = gaia.xi[seed_idx], gaia.eta[seed_idx]
    C_seed, err_seed = gaia.C[seed_idx], gaia.err[seed_idx]
    min_n = config.min_matches

    discovered = []
    dbg = {'seed_fail': 0, 'dedup': 0, 'reject': 0, 'scale_rot': 0}

    for tier_label, h_mask in source_tiers(cat, config, stars_only):
        if h_mask.sum() < min_n:
            continue
        h_idx = np.where(h_mask)[0]

        if seed_quality_mask is not None:
            hist_keep = seed_quality_mask[seed_idx]
        else:
            hist_keep = np.ones(len(seed_idx), dtype=bool)
            if gaia.has_pms[seed_idx].sum() >= min_n:
                hist_keep = gaia.has_pms[seed_idx].copy()
        if hist_keep.sum() < min_n:
            continue

        _, peaks, hist, xed, yed = find_scale_and_offset(
            xi_seed[hist_keep], eta_seed[hist_keep], err_seed[hist_keep],
            cat.xi[h_idx], cat.eta[h_idx], cat.mag[h_idx],
            cov1=C_seed[hist_keep], cov2=cat.C_src[h_idx],
            x_cen=0.0, y_cen=0.0,
            max_offset=config.disc_max, bin_size=config.disc_bin, top_n=3,
            ds_range=(0.0, 0.0), n_scales=1, return_histogram=True,
        )
        dx_off, dy_off, _ = peaks[0]

        # Shift ALL seeds, not just hist_keep — this is what lets 2p stars into
        # the seed match during the strict 5p tiers.
        xi_tier, eta_tier = xi_seed + dx_off, eta_seed + dy_off

        tree = KDTree(np.column_stack([cat.xi[h_idx], cat.eta[h_idx]]))
        cur_rad, valid = None, None
        for rad in config.disc_seed_radii:
            dists, near = tree.query(np.column_stack([xi_tier, eta_tier]),
                                     k=1, distance_upper_bound=rad)
            if (dists < rad).sum() >= min_n:
                cur_rad, valid = rad, dists < rad
                break
        if cur_rad is None:
            dbg['seed_fail'] += 1
            continue

        # Greedy one-to-one cleanup by log-probability cost.
        h_full = h_idx[near[valid]]
        g_seed = np.where(valid)[0]
        dx = cat.xi[h_full] - xi_tier[g_seed]
        dy = cat.eta[h_full] - eta_tier[g_seed]
        costs = compute_logprob_cost(dx, dy, C_seed[g_seed] + cat.C_src[h_full])
        mdf = (pd.DataFrame({'g': g_seed, 'h': h_full, 'c': costs})
               .sort_values('c').drop_duplicates('h').drop_duplicates('g'))
        if len(mdf) < min_n:
            dbg['dedup'] += 1
            continue

        h_b = mdf['h'].values
        g_b = seed_idx[mdf['g'].values]
        mag_diffs = gaia.mag[g_b] - cat.mag[h_b]

        # Prefer 5p/6p for the fit and zero-point when enough are present.
        keep = np.ones(len(h_b), dtype=bool)
        if gaia.has_pms[g_b].sum() >= min_n:
            keep = gaia.has_pms[g_b].copy()
        zp = np.median(mag_diffs[keep])
        M = np.eye(2)
        good = keep.copy()

        for _ in range(5):
            res, _, C_params, _ = fit_4p_weighted(
                cat.xi[h_b[keep]], cat.eta[h_b[keep]],
                gaia.xi[g_b[keep]], gaia.eta[g_b[keep]],
                cat.C_src[h_b[keep]], gaia.C[g_b[keep]],
                initial_M=M, scale_prior=1.0,
                scale_sigma=config.sigma_scale,
                rot_sigma=np.radians(config.sigma_rot_deg),
            )
            A, B, C, D, xs_o, ys_o, xt_o, yt_o = res
            M = np.array([[A, B], [C, D]])

            xp, yp = apply_affine(cat.xi[h_b], cat.eta[h_b], A, B, C, D,
                                  xs_o, ys_o, xt_o, yt_o)
            dxv, dyv = gaia.xi[g_b] - xp, gaia.eta[g_b] - yp

            C_proj = np.einsum('ij,njk,lk->nil', M, cat.C_src[h_b], M)
            dxh, dyh = cat.xi[h_b] - xs_o, cat.eta[h_b] - ys_o
            J = np.zeros((len(dxh), 2, 4))
            J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = dxh, -dyh, 1.0
            J[:, 1, 0], J[:, 1, 1], J[:, 1, 3] = dyh, dxh, 1.0
            C_model = np.einsum('nij,jk,nlk->nil', J, C_params, J)
            # One-pixel floor so cost is on a consistent scale regardless of
            # Gaia quality (5p vs 2p) or source brightness.
            floor = np.eye(2)[np.newaxis] * config.disc_floor**2
            C_tot = gaia.C[g_b] + C_proj + C_model + floor
            chi2 = np.sum(compute_mahalanobis(dxv, dyv, C_tot)[keep])
            cost = np.sum(compute_logprob_cost(dxv, dyv, C_tot)[keep])

            ds = np.sqrt(dxv**2 + dyv**2)
            fin = np.isfinite(ds)
            p16, p50 = np.nanpercentile(ds[fin & keep], [16, 50])
            thresh = min(max(p50 + 3 * (p50 - p16), config.disc_floor), cur_rad)
            if not np.isfinite(thresh):
                thresh = cur_rad

            # Distance-only rejection: Gaia formal errors are far below the
            # residuals of a noisy few-pair 4P fit, so a Mahalanobis cut here
            # would reject everything.  The scale/rotation gate below is what
            # catches spurious transforms.
            good = (ds < thresh) & (np.abs(mag_diffs - zp) < config.max_mag_diff)
            if good.sum() < min_n or np.all(keep == good):
                break
            keep[:] = good
            zp = np.median(mag_diffs[keep])

        if good.sum() < min_n:
            dbg['reject'] += 1
            continue
        g_b, h_b = g_b[keep], h_b[keep]

        scale = np.sqrt(A * D - B * C)
        rot = np.degrees(np.arctan2(B - C, A + D))
        if not (0.98 <= scale <= 1.02 and abs(rot) < 0.2):
            dbg['scale_rot'] += 1
            continue

        red_cost = cost / len(h_b)
        zp_tier = np.median(gaia.mag[g_b] - cat.mag[h_b])
        discovered.append({
            'A': A, 'B': B, 'C': C, 'D': D,
            'xs_o': xs_o, 'ys_o': ys_o, 'xt_o': xt_o, 'yt_o': yt_o,
            'C_params': C_params, 'zp': zp_tier,
            'h_v': h_b, 'g_v': g_b, 'n_match': len(h_b),
            'red_chi2': chi2 / max(2 * len(h_b) - 4, 1), 'red_cost': red_cost,
            'tier': tier_label,
            'offset_peaks': list(peaks),
            'offset_hist': hist, 'offset_xed': xed, 'offset_yed': yed,
        })
        if verbose:
            n2p = int((~gaia.has_pms[g_b]).sum())
            print(f'      {tier_label}: {len(h_b)} pairs ({n2p} 2p), '
                  f'red_cost={red_cost:.2f}, zp={zp_tier:.3f}, '
                  f'scale={scale:.6f}, rot={rot:.4f}deg')

    if not discovered:
        if verbose:
            print(f'      no tier succeeded: {dbg}')
        return None
    return min(discovered, key=lambda d: d['red_cost'])


def discover(cat: SourceCatalog, gaia: GaiaField, config: InstrumentConfig,
             source_tiers: Callable = magnitude_tiers, verbose=True):
    """
    Full tiered 4P discovery: Gaia quality tiers x (stars-only, all sources).

    Returns (best, tier_name) or (None, None).
    """
    for gaia_label, seed_mask in gaia_tiers(gaia):
        for stars_only in (True, False):
            label = f"{gaia_label} / {'stars' if stars_only else 'all sources'}"
            if seed_mask.sum() < config.min_matches:
                if verbose:
                    print(f"  Skipping tier '{label}': {int(seed_mask.sum())} Gaia available.")
                continue
            n_src = int((cat.is_star if stars_only else np.ones(len(cat), bool)).sum())
            if n_src < config.min_matches:
                continue
            if verbose:
                print(f'  Trying 4P discovery [{label}] '
                      f'({int(seed_mask.sum())} Gaia, {n_src} sources)...')
            best = run_4p_discovery(cat, gaia, config,
                                    seed_quality_mask=seed_mask,
                                    stars_only=stars_only,
                                    source_tiers=source_tiers, verbose=verbose)
            if best is not None:
                return best, label
            if verbose:
                print(f'  4P Discovery failed [{label}] — trying next tier...')
    return None, None


__all__ = ['GaiaField', 'magnitude_tiers', 'gaia_tiers',
           'run_4p_discovery', 'discover', 'SEED_SUBSET']
