"""
bp3m-pop-fit-v2: population PM fit over the v2 master catalog, including
HST-only stars far fainter than Gaia.

DEFAULT (joint) mode delegates to the full bp3m-pop-fit machinery with
--use_master_v2: the alignment r, every master-catalog star (Gaia + HST-only
with synthetic negative ids), and mu_pop are solved JOINTLY, exactly analogous
to the original pop fit — see run_pop_fit.py.  All bp3m-pop-fit options apply.

--catalog_only instead runs the fast catalog-level approximation: it takes
every source in hst_xmatch/master_combined_v2.csv — including HST-only stars
far fainter than Gaia — with its per-source PM measurement and covariance
(conditional on the frozen v1 alignment), and iterates

    membership (2D Mahalanobis vs mu_pop, using C_i + sigma_pm^2 I)
    <-> mu_pop  (inverse-variance mean over members + Gaussian hyperprior)

exactly like pop-fit Phases 1-2.  Gaia-matched sources can take their
BP3M_v2_results posterior PMs (Gaia-informed) in place of the raw xmatch fit.

Outputs (in <field>/BP3M_pop_fit_v2_results/):
    pop_astrometry_v2.csv  — per-source membership + population-shrunk PMs
    mu_pop.json            — same schema as bp3m-pop-fit
    plots/                 — VPD + CMD membership diagnostics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bp3m.pipeline.run_pop_fit import _lookup_lvd, _estimate_mu_pop_v1


# ── data assembly ─────────────────────────────────────────────────────────────

def _load_catalog(field_dir: Path, gaia_source: str = 'v2') -> pd.DataFrame:
    """master_combined_v2 + (optionally) v2 posterior PMs for Gaia stars.

    Returns a DataFrame with columns pm_ra, pm_dec, sig_ra, sig_dec, corr,
    pm_origin ('xmatch' or 'v2'), source_index (stable row id into
    master_combined_v2.csv) plus the photometry/id columns.
    """
    mc_path = field_dir / 'hst_xmatch' / 'master_combined_v2.csv'
    if not mc_path.exists():
        raise FileNotFoundError(f"{mc_path} not found — run bp3m-v2 first")
    mc = pd.read_csv(mc_path, low_memory=False)
    mc['source_index'] = np.arange(len(mc))
    # gaia ids must never round-trip through float64
    if 'gaia_source_id' in mc.columns:
        gid = pd.to_numeric(mc['gaia_source_id'], errors='coerce')
        mc['gaia_source_id'] = gid.fillna(0).astype(np.int64)
    else:
        mc['gaia_source_id'] = np.int64(0)

    mc['pm_ra']  = pd.to_numeric(mc.get('pmra_xmatch'),  errors='coerce')
    mc['pm_dec'] = pd.to_numeric(mc.get('pmdec_xmatch'), errors='coerce')
    mc['sig_ra']  = pd.to_numeric(mc.get('sigma_pmra_xmatch'),  errors='coerce')
    mc['sig_dec'] = pd.to_numeric(mc.get('sigma_pmdec_xmatch'), errors='coerce')
    mc['corr'] = pd.to_numeric(
        mc.get('corr_pmra_pmdec_xmatch'), errors='coerce').fillna(0.0)
    mc['pm_origin'] = 'xmatch'

    if gaia_source == 'v2':
        sa_path = field_dir / 'BP3M_v2_results' / 'stellar_astrometry.csv'
        if sa_path.exists():
            sa = pd.read_csv(sa_path, dtype={'Gaia_id': np.int64})
            sa = sa[np.isfinite(sa['pmra_bp3m'])
                    & (sa['sigma_pmra_bp3m'] > 0)
                    & ~sa.get('prior_fallback', False)]
            sub = sa.set_index('Gaia_id')
            hit = mc['gaia_source_id'].isin(sub.index) & (mc['gaia_source_id'] != 0)
            idx = mc.loc[hit, 'gaia_source_id']
            mc.loc[hit, 'pm_ra']   = sub.loc[idx, 'pmra_bp3m'].to_numpy()
            mc.loc[hit, 'pm_dec']  = sub.loc[idx, 'pmdec_bp3m'].to_numpy()
            mc.loc[hit, 'sig_ra']  = sub.loc[idx, 'sigma_pmra_bp3m'].to_numpy()
            mc.loc[hit, 'sig_dec'] = sub.loc[idx, 'sigma_pmdec_bp3m'].to_numpy()
            mc.loc[hit, 'corr']    = sub.loc[idx, 'corr_pmra_pmdec'].to_numpy()
            mc.loc[hit, 'pm_origin'] = 'v2'
            print(f"  Gaia-matched sources using v2 posterior PMs: {hit.sum()}")
        else:
            print("  WARNING: BP3M_v2_results/stellar_astrometry.csv not found "
                  "— all sources use xmatch PMs")
    return mc


def _build_cov(df: pd.DataFrame, pm_sys_floor: float) -> np.ndarray:
    """(n, 2, 2) measurement covariance incl. systematic floor."""
    s1 = df['sig_ra'].to_numpy(float)
    s2 = df['sig_dec'].to_numpy(float)
    rho = np.clip(df['corr'].to_numpy(float), -0.99, 0.99)
    C = np.zeros((len(df), 2, 2))
    C[:, 0, 0] = s1 ** 2 + pm_sys_floor ** 2
    C[:, 1, 1] = s2 ** 2 + pm_sys_floor ** 2
    C[:, 0, 1] = C[:, 1, 0] = rho * s1 * s2
    return C


# ── the fit ───────────────────────────────────────────────────────────────────

def _fit_population(pm, C, eligible, mu_init, sigma_pm,
                    mu_prior, mu_prior_sigma,
                    sigma_clip=3.0, n_iter=20, min_members=5,
                    seed_member=None, freeze_seed=False,
                    mu_ref=None, mu_ref_label=''):
    """EM iteration: membership <-> mu_pop.  Returns (mu, C_mu, member_mask,
    chi2_member)."""
    n = len(pm)
    mu = np.asarray(mu_init, float).copy()
    Lam_prior = np.zeros((2, 2))
    if mu_prior_sigma and np.isfinite(mu_prior_sigma) and mu_prior_sigma > 0:
        Lam_prior = np.eye(2) / mu_prior_sigma ** 2
    C_tot = C.copy()
    C_tot[:, 0, 0] += sigma_pm ** 2
    C_tot[:, 1, 1] += sigma_pm ** 2
    Ci_tot = np.linalg.inv(C_tot)

    member = np.zeros(n, bool)
    C_mu = np.full((2, 2), np.nan)
    for it in range(n_iter):
        d = pm - mu
        chi2 = np.einsum('ni,nij,nj->n', d, Ci_tot, d)
        new_member = eligible & np.isfinite(chi2) & (chi2 < sigma_clip ** 2)
        if seed_member is not None:
            if freeze_seed:
                # the seed can only LOSE members, never gain
                new_member &= seed_member
            elif it == 0:
                new_member = seed_member & eligible
        if new_member.sum() < min_members:
            print(f"    iter {it:2d}: only {new_member.sum()} members — stopping")
            break
        Lam = Lam_prior + Ci_tot[new_member].sum(axis=0)
        h = (Lam_prior @ np.asarray(mu_prior, float)
             + np.einsum('nij,nj->i', Ci_tot[new_member], pm[new_member]))
        C_mu = np.linalg.inv(Lam)
        mu_new = C_mu @ h
        sig_mu = np.sqrt(np.diag(C_mu))
        line = (f"    iter {it:2d}: mu_pop=({mu_new[0]:+.4f} ± {sig_mu[0]:.4f}, "
                f"{mu_new[1]:+.4f} ± {sig_mu[1]:.4f})  n_members={new_member.sum()}")
        if mu_ref is not None:
            dd = mu_new - np.asarray(mu_ref, float)
            ns = np.abs(dd) / np.maximum(sig_mu, 1e-12)
            line += (f"  Nσ_init=({ns[0]:.1f}, {ns[1]:.1f}) [{mu_ref_label}]")
        print(line)
        converged = (np.array_equal(new_member, member)
                     and np.all(np.abs(mu_new - mu) < 1e-6))
        mu, member = mu_new, new_member
        if converged:
            print(f"    converged after {it + 1} iterations")
            break
    d = pm - mu
    chi2 = np.einsum('ni,nij,nj->n', d, Ci_tot, d)
    return mu, C_mu, member, chi2


def _shrink_members(pm, C, member, mu, sigma_pm):
    """Population-prior-shrunk PM posterior for members."""
    n = len(pm)
    pm_out = np.full((n, 2), np.nan)
    sig_out = np.full((n, 2), np.nan)
    Lam_pop = np.eye(2) / sigma_pm ** 2
    for i in np.where(member)[0]:
        Ci = np.linalg.inv(C[i])
        Cp = np.linalg.inv(Ci + Lam_pop)
        pm_out[i] = Cp @ (Ci @ pm[i] + Lam_pop @ mu)
        sig_out[i] = np.sqrt(np.diag(Cp))
    return pm_out, sig_out


# ── plots ─────────────────────────────────────────────────────────────────────

def _mag_bands(df: pd.DataFrame) -> list:
    return sorted(c.replace('mag_wmean_', '')
                  for c in df.columns if c.startswith('mag_wmean_'))


def _plot_results(df, member, eligible, mu, plots_dir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. VPD
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, lim in zip(axes, (12.0, 2.0)):
        bg = eligible & ~member
        ax.plot(df.loc[bg, 'pm_ra'], df.loc[bg, 'pm_dec'], '.',
                ms=2, alpha=0.3, color='0.5', label='field (eligible)')
        ax.plot(df.loc[member, 'pm_ra'], df.loc[member, 'pm_dec'], '.',
                ms=3, alpha=0.6, color='C0', label='members')
        ax.plot(*mu, 'r*', ms=15, label=r'$\mu_{\rm pop}$')
        ax.set_xlim(mu[0] - lim, mu[0] + lim)
        ax.set_ylim(mu[1] - lim, mu[1] + lim)
        ax.set_xlabel(r'$\mu_{\alpha*}$ [mas/yr]')
        ax.set_ylabel(r'$\mu_\delta$ [mas/yr]')
        ax.set_aspect('equal')
    axes[0].legend(loc='upper left', fontsize=9)
    fig.suptitle('pop-fit v2: vector-point diagram (xmatch sources)')
    fig.tight_layout()
    fig.savefig(plots_dir / 'vpd_membership_v2.png', dpi=150)
    plt.close(fig)

    # 2. CMDs: every band pair with enough joint coverage
    bands = _mag_bands(df)
    pairs = []
    for i in range(len(bands)):
        for j in range(len(bands)):
            if i == j:
                continue
            b1, b2 = bands[i], bands[j]
            if b1 >= b2:
                continue
            m1 = df[f'mag_wmean_{b1}'].to_numpy(float)
            m2 = df[f'mag_wmean_{b2}'].to_numpy(float)
            if np.sum(np.isfinite(m1) & np.isfinite(m2)) >= 100:
                pairs.append((b1, b2))
    if pairs:
        fig, axes = plt.subplots(1, len(pairs),
                                 figsize=(5.5 * len(pairs), 7), squeeze=False)
        for ax, (b1, b2) in zip(axes[0], pairs):
            m1 = df[f'mag_wmean_{b1}'].to_numpy(float)
            m2 = df[f'mag_wmean_{b2}'].to_numpy(float)
            fin = np.isfinite(m1) & np.isfinite(m2)
            ax.plot((m1 - m2)[fin & ~member], m2[fin & ~member], '.',
                    ms=1.5, alpha=0.2, color='0.5')
            ax.plot((m1 - m2)[fin & member], m2[fin & member], '.',
                    ms=2.5, alpha=0.6, color='C0')
            ax.invert_yaxis()
            ax.set_xlabel(f'{b1} − {b2}')
            ax.set_ylabel(b2)
        fig.suptitle('pop-fit v2: CMD membership (blue = members)')
        fig.tight_layout()
        fig.savefig(plots_dir / 'cmd_membership_v2.png', dpi=150)
        plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def _catalog_main(argv=None):
    p = argparse.ArgumentParser(
        description='Catalog-level population PM fit over hst_xmatch outputs')
    p.add_argument('--name', required=True, help='field name')
    p.add_argument('--output_dir', type=str, default='.',
                   help='GaiaHub_results root (default: cwd)')
    p.add_argument('--sigma_pm', type=float, default=None,
                   help='intrinsic PM dispersion [mas/yr]; default from LVD')
    p.add_argument('--lvd_key', type=str, default=None,
                   help='LVD key for automatic population parameters')
    p.add_argument('--lvd_dir', type=str, default=None,
                   help='directory holding dwarf_all.csv / gc_harris.csv')
    p.add_argument('--mu_pop_init', type=float, nargs=2, default=None,
                   metavar=('PMRA', 'PMDEC'),
                   help='initial mu_pop [mas/yr]; default LVD or sigma-clip')
    p.add_argument('--mu_pop_prior_sigma', type=float, default=None,
                   help='Gaussian hyperprior width on mu_pop [mas/yr]')
    p.add_argument('--freeze_mu_pop_init', action='store_true',
                   help='centre the hyperprior on --mu_pop_init')
    p.add_argument('--n_iter', type=int, default=20,
                   help='EM iterations (default 20)')
    p.add_argument('--member_sigma_clip', type=float, default=3.0,
                   help='Mahalanobis clip for membership (default 3)')
    p.add_argument('--max_sigma_free_pm', type=float, default=1.0,
                   help='max RMS PM sigma for membership eligibility '
                        '[mas/yr] (default 1.0)')
    p.add_argument('--pm_sys_floor', type=float, default=0.2,
                   help='systematic PM floor added in quadrature [mas/yr]')
    p.add_argument('--min_detections', type=int, default=3,
                   help='min n_detect_fit for eligibility (default 3)')
    p.add_argument('--gaia_source', choices=['v2', 'xmatch'], default='v2',
                   help='PM source for Gaia-matched stars (default v2 '
                        'posteriors; xmatch = uniform treatment)')
    p.add_argument('--use_member_seed', action='store_true',
                   help='seed membership from <field>/member_seed_v2.csv')
    p.add_argument('--member_seed_csv', type=str, default=None,
                   help='explicit member-seed CSV (source_index and/or '
                        'gaia_source_id columns)')
    p.add_argument('--freeze_member_seed', action='store_true',
                   help='seed can only lose members, never gain')
    p.add_argument('--no_plots', action='store_true')
    args = p.parse_args(argv)

    field_dir = Path(args.output_dir).resolve() / args.name
    out_dir = field_dir / 'BP3M_pop_fit_v2_results'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"bp3m-pop-fit-v2: {args.name}")
    print("=" * 60)

    # LVD-derived population parameters
    lvd = {}
    if args.lvd_key:
        import os
        lvd_dir = Path(args.lvd_dir or os.environ.get('BP3M_LVD_DIR')
                       or (Path.home() / 'data_bootes' / 'bp3m'
                           / 'local_volume_database' / 'data'))
        lvd = _lookup_lvd(lvd_dir, args.lvd_key)
        print(f"  LVD[{args.lvd_key}]: " + ", ".join(
            f"{k}={v}" for k, v in lvd.items() if not isinstance(v, tuple)))
    sigma_pm = args.sigma_pm if args.sigma_pm is not None else lvd.get('sigma_pm')
    if sigma_pm is None:
        raise SystemExit("--sigma_pm required (no LVD value available)")
    print(f"  sigma_pm = {sigma_pm:.5f} mas/yr")

    df = _load_catalog(field_dir, gaia_source=args.gaia_source)
    C = _build_cov(df, args.pm_sys_floor)

    pm = df[['pm_ra', 'pm_dec']].to_numpy(float)
    sig_rms = np.sqrt((df['sig_ra'] ** 2 + df['sig_dec'] ** 2) / 2)
    n_det_fit = pd.to_numeric(df.get('n_detect_fit'), errors='coerce').fillna(0)
    eligible = (np.isfinite(pm).all(axis=1)
                & np.isfinite(sig_rms) & (sig_rms > 0)
                & (sig_rms < args.max_sigma_free_pm)
                & (n_det_fit >= args.min_detections)).to_numpy()
    print(f"  Sources: {len(df)} total, {eligible.sum()} eligible "
          f"(sigma_rms < {args.max_sigma_free_pm}, "
          f"n_detect_fit >= {args.min_detections}); "
          f"{(df['gaia_source_id'] != 0).sum()} Gaia-matched")

    # initial mu_pop + provenance (mirrors bp3m-pop-fit conventions)
    if args.mu_pop_init is not None:
        mu_init = np.array(args.mu_pop_init, float)
        ref_label = '--mu_pop_init [user]'
    elif 'mu_pop_init' in lvd:
        mu_init = np.array(lvd['mu_pop_init'], float)
        ref_label = f'--mu_pop_init [LVD:{args.lvd_key}]'
    else:
        mu_init = _estimate_mu_pop_v1(pm[eligible, 0], pm[eligible, 1])
        ref_label = 'empirical xmatch sigma-clip bootstrap'
    print(f"  mu_init = ({mu_init[0]:+.4f}, {mu_init[1]:+.4f}) mas/yr "
          f"({ref_label})")

    mu_prior_sigma = (args.mu_pop_prior_sigma
                      if args.mu_pop_prior_sigma is not None
                      else lvd.get('mu_pop_prior_sigma', 0.0))
    mu_prior = mu_init.copy()
    if args.freeze_mu_pop_init and args.mu_pop_init is None:
        print("  WARNING: --freeze_mu_pop_init without --mu_pop_init — "
              "freezing to the derived init above")
    if mu_prior_sigma:
        print(f"  hyperprior: mu_pop ~ N(({mu_prior[0]:+.4f}, "
              f"{mu_prior[1]:+.4f}), {mu_prior_sigma}^2)")

    # member seed
    seed_member = None
    seed_path = None
    if args.member_seed_csv:
        seed_path = Path(args.member_seed_csv)
    elif args.use_member_seed:
        for cand in (field_dir / 'member_seed_v2.csv',
                     field_dir / 'member_seed.csv'):
            if cand.exists():
                seed_path = cand
                break
    if seed_path is not None:
        if not seed_path.exists():
            raise SystemExit(f"member seed not found: {seed_path}")
        seed = pd.read_csv(seed_path)
        seed_member = np.zeros(len(df), bool)
        if 'source_index' in seed.columns:
            si = pd.to_numeric(seed['source_index'], errors='coerce').dropna()
            seed_member[np.intersect1d(si.astype(int),
                                       df['source_index'])] = True
        if 'gaia_source_id' in seed.columns:
            gids = pd.to_numeric(seed['gaia_source_id'],
                                 errors='coerce').fillna(0).astype(np.int64)
            seed_member |= df['gaia_source_id'].isin(
                gids[gids != 0]).to_numpy()
        print(f"  member seed: {seed_path.name} -> {seed_member.sum()} of "
              f"{len(seed)} seed rows matched"
              + (' [FROZEN: remove-only]' if args.freeze_member_seed else ''))

    print(f"\n  Population fit ({args.n_iter} iterations)...")
    mu, C_mu, member, chi2 = _fit_population(
        pm, C, eligible, mu_init, sigma_pm,
        mu_prior, mu_prior_sigma,
        sigma_clip=args.member_sigma_clip, n_iter=args.n_iter,
        seed_member=seed_member, freeze_seed=args.freeze_member_seed,
        mu_ref=mu_init, mu_ref_label=ref_label)

    sig_mu = np.sqrt(np.diag(C_mu))
    print(f"\n  FINAL: mu_pop = ({mu[0]:+.4f} ± {sig_mu[0]:.4f}, "
          f"{mu[1]:+.4f} ± {sig_mu[1]:.4f}) mas/yr,  "
          f"{member.sum()} members "
          f"({(member & (df['gaia_source_id'] != 0)).sum()} Gaia, "
          f"{(member & (df['gaia_source_id'] == 0)).sum()} HST-only)")

    pm_pop, sig_pop = _shrink_members(pm, C, member, mu, sigma_pm)

    # ── outputs ───────────────────────────────────────────────────────────────
    keep_cols = (['source_index', 'gaia_source_id', 'ra0', 'dec0',
                  'ra_xmatch', 'dec_xmatch', 'n_detect', 'n_detect_fit',
                  'n_filters', 'pm_origin']
                 + [c for c in df.columns if c.startswith('mag_wmean_')])
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    out['pmra_meas'] = pm[:, 0]
    out['pmdec_meas'] = pm[:, 1]
    out['sigma_pmra_meas'] = df['sig_ra']
    out['sigma_pmdec_meas'] = df['sig_dec']
    out['eligible'] = eligible
    out['is_member'] = member
    out['chi2_member'] = chi2
    out['pmra_pop'] = pm_pop[:, 0]
    out['pmdec_pop'] = pm_pop[:, 1]
    out['sigma_pmra_pop'] = sig_pop[:, 0]
    out['sigma_pmdec_pop'] = sig_pop[:, 1]
    out.to_csv(out_dir / 'pop_astrometry_v2.csv', index=False)
    print(f"  Saved: pop_astrometry_v2.csv ({len(out)} sources)")

    corr_mu = (float(C_mu[0, 1] / (sig_mu[0] * sig_mu[1]))
               if np.all(sig_mu > 0) else 0.0)
    mu_result = {
        'mu_pop_ra_masyr': float(mu[0]),
        'mu_pop_dec_masyr': float(mu[1]),
        'sigma_mu_pop_ra': float(sig_mu[0]),
        'sigma_mu_pop_dec': float(sig_mu[1]),
        'corr_mu_pop_ra_dec': corr_mu,
        'n_members': int(member.sum()),
        'n_members_gaia': int((member & (df['gaia_source_id'] != 0)).sum()),
        'n_members_hst_only': int((member & (df['gaia_source_id'] == 0)).sum()),
        'sigma_pm_masyr': float(sigma_pm),
        'mu_pop_prior_ra': float(mu_prior[0]),
        'mu_pop_prior_dec': float(mu_prior[1]),
        'mu_pop_prior_sigma': float(mu_prior_sigma or 0.0),
        'pm_sys_floor': float(args.pm_sys_floor),
        'member_sigma_clip': float(args.member_sigma_clip),
        'max_sigma_free_pm': float(args.max_sigma_free_pm),
        'min_detections': int(args.min_detections),
        'gaia_source': args.gaia_source,
        'input': 'hst_xmatch/master_combined_v2.csv',
    }
    with open(out_dir / 'mu_pop_catalog.json', 'w') as f:
        json.dump(mu_result, f, indent=2)
    print("  Saved: mu_pop_catalog.json")

    if not args.no_plots:
        _plot_results(df, member, eligible, mu, out_dir / 'plots')
        print("  Saved: plots/")

    from datetime import datetime
    with open(out_dir / 'bp3m_pop_fit_v2_command.txt', 'a') as f:
        f.write(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                + " ".join(sys.argv) + "\n")
    return 0


def main(argv=None):
    """bp3m-pop-fit-v2 entry point.

    Default: the JOINT solve — delegates to bp3m-pop-fit with --use_master_v2
    (all bp3m-pop-fit options are accepted).  Pass --catalog_only for the fast
    catalog-level fit implemented in this module.
    """
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if '--catalog_only' in argv:
        argv.remove('--catalog_only')
        return _catalog_main(argv)
    from bp3m.pipeline.run_pop_fit import main as _joint_main
    extra = ['--use_master_v2']
    # SAFE DEFAULT: keep the alignment FROZEN (mu-only Phase 1).  With
    # HST-only-dominated membership the joint r+mu phases have a runaway
    # mu_pop <-> alignment soft mode (Leo_I 2026-09-01: joint drifted to
    # (-0.41,+0.23) while the Gaia PMs of the same members give
    # (-0.06,-0.12)).  Pass --n_iter_joint/--n_iter_alpha explicitly to
    # opt in to the joint phases regardless.
    if not any(a.startswith('--n_iter_joint') for a in argv):
        extra += ['--n_iter_joint', '0']
        if not any(a.startswith('--n_iter_alpha') for a in argv):
            extra += ['--n_iter_alpha', '0']
        print('bp3m-pop-fit-v2: alignment frozen (mu-only solve; '
              'pass --n_iter_joint to enable joint phases)')
    return _joint_main(argv + extra)


if __name__ == '__main__':
    raise SystemExit(main())
