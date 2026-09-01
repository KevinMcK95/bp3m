"""
Diagnostic plots comparing BP3M astrometry to Gaia.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from collections import defaultdict
from pathlib import Path


PLOT_DIR_NAME = "plots"
RESID_PLOT_DIR_NAME = "residuals"

_MEM_MARKER = 'o'   # circle  — member stars
_NON_MARKER = 'v'   # triangle-down — non-member stars
_MEM_SIZE   = 8
_NON_SIZE   = 5


def _scatter_mem(ax, x, y, c, is_member, *, norm, cmap, alpha=0.8, zorder=2, **kw):
    """Scatter with circles for members and downward-triangles for non-members.
    Falls back to a single scatter (circle, s=6) when is_member is None.
    Returns the last ScalarMappable created (use as colorbar source)."""
    if is_member is None:
        return ax.scatter(x, y, c=c, s=6, norm=norm, cmap=cmap,
                          alpha=alpha, zorder=zorder, **kw)
    sc = None
    nm = ~is_member
    if nm.any():
        sc = ax.scatter(x[nm], y[nm], c=c[nm], marker=_NON_MARKER, s=_NON_SIZE,
                        norm=norm, cmap=cmap, alpha=0.6, zorder=zorder, **kw)
    if is_member.any():
        sc = ax.scatter(x[is_member], y[is_member], c=c[is_member],
                        marker=_MEM_MARKER, s=_MEM_SIZE,
                        norm=norm, cmap=cmap, alpha=0.9, zorder=zorder + 0.1, **kw)
    return sc


def _add_mu_pop_lines(ax, mu_pop):
    """Dashed lines at the population PM (μ_α*, μ_δ)."""
    if mu_pop is None:
        return
    ax.axvline(mu_pop[0], c='0.35', lw=1.1, ls='--', zorder=1e9)
    ax.axhline(mu_pop[1], c='0.35', lw=1.1, ls='--', zorder=1e9)


def _mem_legend_handles():
    from matplotlib.lines import Line2D
    return [
        Line2D([0], [0], marker=_MEM_MARKER, color='w', markerfacecolor='0.5',
               markeredgecolor='0.5', markersize=6, label='Member'),
        Line2D([0], [0], marker=_NON_MARKER, color='w', markerfacecolor='0.5',
               markeredgecolor='0.5', markersize=5, label='Non-member'),
    ]


# ── Public entry point ────────────────────────────────────────────────────────

def make_plots(solver, images, gaia_catalog,
               r_hat, v_hat, v_mean, v_cov, C_vT, C_r,
               output_dir,
               plot_residuals: bool = False,
               member_sidx=None,
               mu_pop=None,
               v_mean_free=None,
               v_cov_free=None,
               C_vT_free=None,
               qso_anchor_ids=None):
    """
    Generate all diagnostic plots.

    Parameters
    ----------
    solver        : BP3MSolver  (fitted, for compute_residuals)
    images        : dict        (image metadata from data_loader)
    gaia_catalog  : pd.DataFrame
    r_hat         : (n_r,) MAP image transformation vector
    v_hat         : (n_stars, 5) MAP stellar astrometry
    v_mean        : (n_stars, 5) posterior mean from sample_posteriors
    v_cov         : (n_stars, 5, 5) posterior covariance from sample_posteriors
                    (r-propagation only — C_vT is added internally for full marginal)
    C_vT          : (n_stars, 5, 5) conditional covariance
    C_r           : (n_r, n_r) posterior covariance of image transformations
    output_dir    : Path-like, parent results directory
    plot_residuals : generate per-image XY residual maps (slow for large fields)
    """
    plot_dir = Path(output_dir) / PLOT_DIR_NAME
    plot_dir.mkdir(parents=True, exist_ok=True)
    resid_plot_dir = Path(plot_dir) / RESID_PLOT_DIR_NAME
    resid_plot_dir.mkdir(parents=True, exist_ok=True)

    # Full marginal covariance = r-propagation + conditional, with prior fallback.
    from bp3m.pipeline.run_alignment import _apply_prior_fallback
    _failed_prior = ~getattr(solver, 'ok_star', np.ones(solver.n_stars, bool))
    v_cov_full, v_mean, _ = _apply_prior_fallback(
        v_cov + C_vT, v_mean, solver.C_prior, solver.v_prior,
        failed_prior_test=_failed_prior)

    # ── Member mask and free (diffuse-prior) data ─────────────────────────────
    is_member = None
    if member_sidx is not None and len(member_sidx) > 0:
        is_member = np.zeros(v_mean.shape[0], dtype=bool)
        is_member[member_sidx] = True

    has_free = (v_mean_free is not None
                and v_cov_free is not None
                and C_vT_free is not None)
    if has_free:
        _vcov_full_free    = v_cov_free + C_vT_free
        _pmra_free         = v_mean_free[:, 2]
        _pmdec_free        = v_mean_free[:, 3]
        _C_pm_free         = _vcov_full_free[:, 2:4, 2:4]
        _det_free          = np.linalg.det(_C_pm_free)
        _sig_pm_free       = np.where(_det_free > 0, _det_free ** 0.25, np.nan)
        _bp3m_conv_free    = _sig_pm_free < 90
        _sig_pmra_free     = np.sqrt(_C_pm_free[:, 0, 0])
        _sig_pmdec_free    = np.sqrt(_C_pm_free[:, 1, 1])
        _rho_free          = (_C_pm_free[:, 0, 1]
                              / np.where(_sig_pmra_free * _sig_pmdec_free > 0,
                                         _sig_pmra_free * _sig_pmdec_free, np.nan))

    pmra_bp3m   = v_mean[:, 2]
    pmdec_bp3m  = v_mean[:, 3]

    pmra_gaia   = solver.v_survey[:,2]
    pmdec_gaia  = solver.v_survey[:,3]
    gmag        = solver.gaia_cat["gmag"].to_numpy(float)

    has_gaia = solver.full_gaia_astrometry   # bool (n_stars,)
    C_pm_gaia = solver.C_survey[:,2:4,2:4]
    sig_pmra_gaia = np.sqrt(C_pm_gaia[:,0,0])
    sig_pmdec_gaia = np.sqrt(C_pm_gaia[:,1,1])
    rho_gaia = C_pm_gaia[:,0,1]/(sig_pmra_gaia*sig_pmdec_gaia)

    sig_pm_gaia = _pm_geom_unc(sig_pmra_gaia, sig_pmdec_gaia, rho_gaia)

    C_pm_bp3m   = v_cov_full[:, 2:4, 2:4]
    det_bp3m    = np.linalg.det(C_pm_bp3m)
    sig_pm_bp3m = np.where(det_bp3m > 0, det_bp3m ** 0.25, np.nan)
    bp3m_converged = (sig_pm_bp3m < 90)

    sig_pmra_bp3m = np.sqrt(C_pm_bp3m[:,0,0])
    sig_pmdec_bp3m = np.sqrt(C_pm_bp3m[:,1,1])
    rho_bp3m = C_pm_bp3m[:,0,1]/(sig_pmra_bp3m*sig_pmdec_bp3m)

    # ── Figure 1: 1:1 PM comparison (top) + PM uncertainty vs mag (bottom) ───
    print("  Plotting 1:1 PM comparison + uncertainty vs magnitude...")

    fig = plt.figure(figsize=(13, 11/2*3), layout="constrained")
    gs  = fig.add_gridspec(3, 2)
    ax_pmra  = fig.add_subplot(gs[0, 0])
    ax_pmdec = fig.add_subplot(gs[0, 1])
    ax_unc   = fig.add_subplot(gs[1, :])
    ax_unc_improve   = fig.add_subplot(gs[2, :])

    _gc = solver.gaia_cat
    _gaia_ids = _gc["Gaia_id"].to_numpy(dtype=np.int64, na_value=0)
    hst_only  = ~has_gaia

    # QSO anchors have prior-dominated (near-zero) BP3M uncertainties that are
    # not astrometrically meaningful — exclude them from the uncertainty and
    # improvement-factor panels so they don't collapse the y-axis.
    _is_qso = np.zeros(len(_gaia_ids), dtype=bool)
    if qso_anchor_ids is not None and len(qso_anchor_ids) > 0:
        _qso_set = set(int(q) for q in qso_anchor_ids)
        for _i, _gid in enumerate(_gaia_ids):
            if _gid in _qso_set:
                _is_qso[_i] = True
    _not_qso = ~_is_qso
    _gaia_not_qso = has_gaia & _not_qso
    _n_gc = len(_gc)
    if "pmra_xmatch" in _gc.columns:
        _pmra_xmatch  = np.array(
            [float(v) if v is not None and str(v) not in ('nan', 'None', '') else np.nan
             for v in _gc["pmra_xmatch"].values], dtype=float)
        _pmdec_xmatch = np.array(
            [float(v) if v is not None and str(v) not in ('nan', 'None', '') else np.nan
             for v in _gc["pmdec_xmatch"].values], dtype=float)
    else:
        _pmra_xmatch  = np.full(_n_gc, np.nan)
        _pmdec_xmatch = np.full(_n_gc, np.nan)
    hst_has_xmatch = hst_only & np.isfinite(_pmra_xmatch) & np.isfinite(_pmdec_xmatch)

    # ── DELVE source categorisation ───────────────────────────────────────────
    # _delve_only:  negative Gaia_id = synthetic row with no real Gaia data.
    # has_delve_pm: any star with a real Gaia ID (5p or 2p) that also has a
    #               full DELVE PM covariance.
    # _gaia_delve_nq: all real-Gaia+DELVE sources (5p and 2p) for PM panels.
    # _gaia5p_delve_nq: 5p-Gaia+DELVE only — used for improvement ratios vs
    #               Gaia 5p prior (2p prior is ~100 mas/yr diffuse; ratio would
    #               be huge and dominate the improvement-factor y-axis).
    # _gaia_only_nq: 5p Gaia without DELVE.
    def _gc_col(col):
        if col in _gc.columns:
            return pd.to_numeric(_gc[col], errors='coerce').to_numpy(float)
        return np.full(len(_gc), np.nan)
    _d_pmra_err  = _gc_col("delve_pmra_error")
    _d_pmdec_err = _gc_col("delve_pmdec_error")
    _d_pmra_val  = _gc_col("delve_pmra")
    _d_pmdec_val = _gc_col("delve_pmdec")
    _has_real_gaia = (_gaia_ids > 0)          # any real Gaia ID (5p or 2p)
    _delve_only    = (_gaia_ids < 0)          # synthetic rows only
    # Use solver._has_delve_pm (the post-veto truth) rather than raw catalog column
    # presence.  Gaia 2p+DELVE stars whose DELVE was vetoed by the solver have
    # C_prior = diffuse prior (1000 mas/yr), not the DELVE prior, so they must not
    # appear in the DELVE-anchored improvement group.
    has_delve_pm = getattr(solver, '_has_delve_pm',
                           (np.isfinite(_d_pmra_err) & (_d_pmra_err > 0) &
                            np.isfinite(_d_pmdec_err) & (_d_pmdec_err > 0) &
                            _has_real_gaia))
    _gaia_delve_nq   = _has_real_gaia & _not_qso & has_delve_pm   # all Gaia+DELVE
    _gaia5p_delve_nq = _gaia_not_qso & has_delve_pm               # 5p Gaia+DELVE
    _gaia_only_nq    = _gaia_not_qso & ~has_delve_pm              # 5p Gaia only

    for ax, gaia_pm, bp3m_pm_g, sig_g, sig_b_g, d_pm, d_sig, comp in zip(
            [ax_pmra, ax_pmdec],
            [pmra_gaia,   pmdec_gaia],
            [pmra_bp3m,   pmdec_bp3m],
            [sig_pmra_gaia, sig_pmdec_gaia],
            [sig_pmra_bp3m, sig_pmdec_bp3m],
            [_d_pmra_val,   _d_pmdec_val],
            [_d_pmra_err,   _d_pmdec_err],
            [r"$\mu_{\alpha*}$",    r"$\mu_\delta$"]):
        # DELVE-only stars (background, zorder=2)
        _do_pm = _delve_only & np.isfinite(d_pm) & np.isfinite(d_sig) & (d_sig > 0)
        if _do_pm.any():
            ax.errorbar(d_pm[_do_pm], bp3m_pm_g[_do_pm],
                        xerr=d_sig[_do_pm], yerr=sig_b_g[_do_pm],
                        fmt='^', ms=4, lw=0.5, alpha=0.6, color='darkorange',
                        label='DELVE only', zorder=2)
        # Gaia+DELVE stars (middle, zorder=3)
        if _gaia_delve_nq.any():
            ax.errorbar(gaia_pm[_gaia_delve_nq], bp3m_pm_g[_gaia_delve_nq],
                        xerr=sig_g[_gaia_delve_nq], yerr=sig_b_g[_gaia_delve_nq],
                        fmt='o', ms=3, lw=0.5, alpha=0.6, color='steelblue',
                        label='Gaia+DELVE', zorder=3)
            # Overlay DELVE PM as diamond markers with DELVE error bars
            _dm = _gaia_delve_nq & np.isfinite(d_pm)
            if _dm.any():
                ax.errorbar(d_pm[_dm], bp3m_pm_g[_dm],
                            xerr=d_sig[_dm], yerr=sig_b_g[_dm],
                            fmt='D', ms=4, lw=0.5, alpha=0.5, color='dodgerblue',
                            label='DELVE PM (for Gaia+DELVE)', zorder=3)
        # Gaia-only stars (foreground, zorder=4)
        if _gaia_only_nq.any():
            ax.errorbar(gaia_pm[_gaia_only_nq], bp3m_pm_g[_gaia_only_nq],
                        xerr=sig_g[_gaia_only_nq], yerr=sig_b_g[_gaia_only_nq],
                        fmt='o', ms=3, lw=0.5, alpha=0.5, color='grey',
                        label='Gaia only', zorder=4)

        # Axis limits from Gaia-prior stars only (DELVE-only can blow up the axes)
        gaia_x = np.concatenate([
            gaia_pm[_gaia_not_qso],
            d_pm[_gaia_delve_nq][np.isfinite(d_pm[_gaia_delve_nq])],
        ])
        gaia_y = np.concatenate([
            bp3m_pm_g[_gaia_not_qso],
            bp3m_pm_g[_gaia_delve_nq],
        ])
        lim = _padded_lim(gaia_x, gaia_y)
        ax.plot(lim, lim, 'k--', lw=1, zorder=4)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"{comp} prior [mas/yr]")
        ax.set_ylabel(f"{comp} BP3M [mas/yr]")
        ax.set_title(f"{comp}: BP3M vs prior PM")
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc='upper left')
        _style_ax(ax)

    _bp3m_gaia_conv = bp3m_converged & has_gaia
    _bp3m_hst_conv  = bp3m_converged & _delve_only   # true DELVE-only (negative Gaia_id)
    _bp3m_gaia_conv_nq = _bp3m_gaia_conv & _not_qso
    gm_nq = gmag[_gaia_not_qso]
    ax_unc.scatter(gm_nq, sig_pm_gaia[_gaia_not_qso],
                   s=6, alpha=0.7, color='#444444', label='Gaia prior', zorder=2)
    # DELVE-alone PM uncertainty (geometric mean of delve_pmra_error, delve_pmdec_error)
    _d_sig_pm = np.where(
        np.isfinite(_d_pmra_err) & (_d_pmra_err > 0) &
        np.isfinite(_d_pmdec_err) & (_d_pmdec_err > 0),
        np.sqrt(_d_pmra_err * _d_pmdec_err), np.nan)
    _has_delve_unc = np.isfinite(_d_sig_pm) & _not_qso
    if _has_delve_unc.any():
        ax_unc.scatter(gmag[_has_delve_unc], _d_sig_pm[_has_delve_unc],
                       s=6, alpha=0.6, color='dodgerblue', marker='D',
                       label='DELVE prior', zorder=2)
    if (_bp3m_gaia_conv_nq & ~has_delve_pm).any():
        ax_unc.scatter(gmag[_bp3m_gaia_conv_nq & ~has_delve_pm],
                       sig_pm_bp3m[_bp3m_gaia_conv_nq & ~has_delve_pm],
                       s=6, alpha=0.85, color='mediumseagreen', label='BP3M Gaia only', zorder=3)
    if (_bp3m_gaia_conv_nq & has_delve_pm).any():
        ax_unc.scatter(gmag[_bp3m_gaia_conv_nq & has_delve_pm],
                       sig_pm_bp3m[_bp3m_gaia_conv_nq & has_delve_pm],
                       s=6, alpha=0.7, color='steelblue', label='BP3M Gaia+DELVE', zorder=3)
    _bp3m_gaia2p_conv = bp3m_converged & (_gaia_ids > 0) & ~has_gaia
    if _bp3m_gaia2p_conv.any():
        ax_unc.scatter(gmag[_bp3m_gaia2p_conv], sig_pm_bp3m[_bp3m_gaia2p_conv],
                       s=8, alpha=0.8, color='mediumpurple', marker='s',
                       label='BP3M Gaia 2p', zorder=4)
    if _bp3m_hst_conv.any():
        ax_unc.scatter(gmag[_bp3m_hst_conv], sig_pm_bp3m[_bp3m_hst_conv],
                       s=10, alpha=0.8, color='darkorange', marker='^',
                       label='BP3M DELVE only', zorder=4)
    ax_unc.set_xlabel("G [mag]")
    ax_unc.set_ylabel(r"$(\det\,C_{\mu})^{1/4}$ [mas/yr]")
    ax_unc.set_title(r"Geometric-mean PM uncertainty $(\det\,C_{\mu})^{1/4}$ vs magnitude")
    ax_unc.legend()
    ax_unc.set_yscale("log")
    xlim = ax_unc.get_xlim()
    _style_ax(ax_unc)

    ax_unc_improve.scatter(gmag[_gaia_only_nq], sig_pm_gaia[_gaia_only_nq]/sig_pm_bp3m[_gaia_only_nq],
                   s=6, alpha=0.6, color='grey', label='Gaia only', zorder=2)
    # 5p Gaia+DELVE improvement vs Gaia 5p prior (2p prior ~100 mas/yr would dominate axis)
    if _gaia5p_delve_nq.any():
        ax_unc_improve.scatter(gmag[_gaia5p_delve_nq],
                               sig_pm_gaia[_gaia5p_delve_nq]/sig_pm_bp3m[_gaia5p_delve_nq],
                               s=6, alpha=0.6, color='steelblue', label='Gaia+DELVE', zorder=3)
    # DELVE-anchored: true DELVE-only + Gaia 2p+DELVE — improvement vs DELVE prior sigma
    _gaia2p_delve = _has_real_gaia & ~has_gaia & has_delve_pm  # 2p Gaia + DELVE
    _do_conv = (_delve_only | _gaia2p_delve) & bp3m_converged & np.isfinite(_d_sig_pm) & (_d_sig_pm > 0)
    if _do_conv.any():
        _delve_improve = _d_sig_pm[_do_conv] / sig_pm_bp3m[_do_conv]
        ax_unc_improve.scatter(gmag[_do_conv], _delve_improve,
                               s=10, alpha=0.7, color='darkorange', marker='^',
                               label='DELVE-anchored', zorder=4)
    ax_unc_improve.set_xlabel("G [mag]")
    ax_unc_improve.set_ylabel(r"PM Improvement Factor")
    ax_unc_improve.set_title(r"PM uncertainty Improvement vs magnitude compared to prior (Gaia or DELVE)")
    ax_unc_improve.set_xlim(xlim)
    ax_unc_improve.axhline(1.0,c='k',lw=2,ls='--',zorder=-1e10)
    ax_unc_improve.legend(fontsize=7)
    _style_ax(ax_unc_improve)

    fig.suptitle("Proper motion comparison", fontsize=13)
    _save(fig, plot_dir / "pm_one_to_one.png")

    if has_free:
        # Free version: BP3M column uses diffuse-prior PMs
        pmra_bp3m_free_gnq   = _pmra_free[_gaia_not_qso]
        pmdec_bp3m_free_gnq  = _pmdec_free[_gaia_not_qso]
        sig_pmra_free_gnq    = _sig_pmra_free[_gaia_not_qso]
        sig_pmdec_free_gnq   = _sig_pmdec_free[_gaia_not_qso]
        sig_pm_free_gnq      = _pm_geom_unc(sig_pmra_free_gnq, sig_pmdec_free_gnq,
                                            _rho_free[_gaia_not_qso])
        _bp3m_gaia_conv_f    = _bp3m_conv_free & has_gaia
        _bp3m_gaia_conv_f_nq = _bp3m_conv_free & _gaia_not_qso
        _bp3m_hst_conv_f     = _bp3m_conv_free & hst_only

        fig = plt.figure(figsize=(13, 11/2*3), layout="constrained")
        gs  = fig.add_gridspec(3, 2)
        ax_pmra  = fig.add_subplot(gs[0, 0])
        ax_pmdec = fig.add_subplot(gs[0, 1])
        ax_unc   = fig.add_subplot(gs[1, :])
        ax_unc_improve = fig.add_subplot(gs[2, :])

        for ax, gaia_pm, bp3m_pm_g, sig_g, sig_b_g, comp in zip(
                [ax_pmra, ax_pmdec],
                [pmra_gaia[_gaia_not_qso],   pmdec_gaia[_gaia_not_qso]],
                [pmra_bp3m_free_gnq,          pmdec_bp3m_free_gnq],
                [sig_pmra_gaia[_gaia_not_qso], sig_pmdec_gaia[_gaia_not_qso]],
                [sig_pmra_free_gnq,            sig_pmdec_free_gnq],
                [r"$\mu_{\alpha*}$",    r"$\mu_\delta$"]):
            ax.errorbar(gaia_pm, bp3m_pm_g, xerr=sig_g, yerr=sig_b_g,
                        fmt='o', ms=3, lw=0.5, alpha=0.5, color='steelblue',
                        label='Gaia-matched', zorder=2)
            lim = _padded_lim(gaia_pm, bp3m_pm_g)
            ax.plot(lim, lim, 'k--', lw=1, zorder=4)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel(f"{comp} Gaia [mas/yr]")
            ax.set_ylabel(f"{comp} BP3M diffuse prior [mas/yr]")
            ax.set_title(f"{comp}: BP3M (diffuse prior) vs Gaia")
            ax.set_aspect("equal")
            ax.legend(fontsize=7, loc='upper left')
            _style_ax(ax)

        ax_unc.scatter(gm_nq, sig_pm_gaia[_gaia_not_qso],
                       s=6, alpha=0.7, color='#444444', label='Gaia 5p', zorder=2)
        ax_unc.scatter(gmag[_bp3m_gaia_conv_f_nq], _sig_pm_free[_bp3m_gaia_conv_f_nq],
                       s=6, alpha=0.7, color='steelblue',
                       label='BP3M Gaia 5p (diffuse prior)', zorder=3)
        if _bp3m_hst_conv_f.any():
            ax_unc.scatter(gmag[_bp3m_hst_conv_f], _sig_pm_free[_bp3m_hst_conv_f],
                           s=10, alpha=0.8, color='darkorange', marker='^',
                           label='BP3M Gaia 2p + HST (diffuse prior)', zorder=4)
        ax_unc.set_xlabel("G [mag]")
        ax_unc.set_ylabel(r"$(\det\,C_{\mu})^{1/4}$ [mas/yr]")
        ax_unc.set_title(r"Geometric-mean PM uncertainty (diffuse prior) vs magnitude")
        ax_unc.legend()
        ax_unc.set_yscale("log")
        _style_ax(ax_unc)

        ax_unc_improve.scatter(gm_nq, sig_pm_gaia[_gaia_not_qso] / _sig_pm_free[_gaia_not_qso],
                               s=6, alpha=0.6, color='steelblue', zorder=2)
        ax_unc_improve.set_xlabel("Gaia G [mag]")
        ax_unc_improve.set_ylabel("PM Improvement Factor (diffuse prior)")
        ax_unc_improve.set_title(
            "PM uncertainty improvement vs magnitude (diffuse prior vs Gaia-alone)")
        ax_unc_improve.axhline(1.0, c='k', lw=2, ls='--', zorder=-1e10)
        _style_ax(ax_unc_improve)

        fig.suptitle("Proper motion comparison (diffuse prior)", fontsize=13)
        _save(fig, plot_dir / "pm_one_to_one_diffuse_prior.png")

    # ── Figure 2: PM vector diagrams coloured by geometric-mean uncertainty ───
    print("  Plotting PM vector diagrams...")

    gaia_pmra_h  = pmra_gaia[has_gaia]
    gaia_pmdec_h = pmdec_gaia[has_gaia]
    bp3m_pmra_h  = pmra_bp3m[bp3m_converged]
    bp3m_pmdec_h = pmdec_bp3m[bp3m_converged]

    full_xlim = _padded_lim(gaia_pmra_h)
    full_ylim = _padded_lim(gaia_pmdec_h)

    zoom_xcen = np.nanmedian(gaia_pmra_h)
    zoom_ycen = np.nanmedian(gaia_pmdec_h)
    zoom_xhw  = max(np.abs(np.nanpercentile(gaia_pmra_h,  [16, 84]) - zoom_xcen))
    zoom_yhw  = max(np.abs(np.nanpercentile(gaia_pmdec_h, [16, 84]) - zoom_ycen))
    zoom_hw   = max(zoom_xhw, zoom_yhw) * 1.15
    zoom_xlim = (zoom_xcen - zoom_hw, zoom_xcen + zoom_hw)
    zoom_ylim = (zoom_ycen - zoom_hw, zoom_ycen + zoom_hw)

    c_gaia = sig_pm_gaia[has_gaia]
    c_bp3m = sig_pm_bp3m[bp3m_converged]
    all_unc = np.concatenate([c_gaia[np.isfinite(c_gaia)],
                              c_bp3m[np.isfinite(c_bp3m)]])
    vmin = np.nanpercentile(all_unc, 2)
    vmax = np.nanpercentile(all_unc, 98)
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
    cmap = "plasma"

    _is_mem_gaia_h = is_member[has_gaia]       if is_member is not None else None
    _is_mem_bp3m_h = is_member[bp3m_converged] if is_member is not None else None
    _has_members   = is_member is not None

    def _render_vd(axes, pmra_g, pmdec_g, c_g, pmra_b, pmdec_b, c_b,
                   is_m_g, is_m_b, vd_norm, title_bp3m='BP3M'):
        sc_last = None
        for col, pmra, pmdec, c_vals, label, is_m in zip(
                [0, 1],
                [pmra_g,  pmra_b],
                [pmdec_g, pmdec_b],
                [c_g,     c_b],
                ["Gaia",  title_bp3m],
                [is_m_g,  is_m_b]):
            for row, xlim, ylim, suffix in zip(
                    [0, 1],
                    [full_xlim, zoom_xlim],
                    [full_ylim, zoom_ylim],
                    ["full range", "zoom (68% CI)"]):
                ax = axes[row, col]
                sc = _scatter_mem(ax, pmra, pmdec, c_vals, is_m,
                                  norm=vd_norm, cmap=cmap, zorder=2)
                ax.axhline(0, c='k', lw=1, ls='--', zorder=1e10)
                ax.axvline(0, c='k', lw=1, ls='--', zorder=1e10)
                _add_mu_pop_lines(ax, mu_pop)
                ax.set_xlim(xlim); ax.set_ylim(ylim)
                ax.set_xlabel(r"$\mu_{\alpha*}$ [mas/yr]")
                ax.set_ylabel(r"$\mu_\delta$ [mas/yr]")
                ax.set_title(f"{label} — {suffix}")
                ax.set_aspect("equal")
                _style_ax(ax)
                sc_last = sc
                if row == 0 and col == 0 and is_m is not None:
                    ax.legend(handles=_mem_legend_handles(), fontsize=7,
                              loc='upper left', framealpha=0.7)
        return sc_last

    def _add_mem_row_vd(axes_row, pmra_g, pmdec_g, c_g, is_m_g,
                        pmra_b, pmdec_b, c_b, is_m_b, vd_norm,
                        labels=("Gaia", "BP3M")):
        """Scatter members only into a 2-element axes row with natural scaling."""
        for col, pmra, pmdec, c_vals, is_m, label in zip(
                [0, 1],
                [pmra_g, pmra_b], [pmdec_g, pmdec_b], [c_g, c_b],
                [is_m_g, is_m_b], labels):
            ax = axes_row[col]
            m = is_m if is_m is not None else np.ones(len(pmra), bool)
            ax.scatter(pmra[m], pmdec[m], c=c_vals[m], norm=vd_norm,
                       cmap=cmap, s=15, alpha=0.85, zorder=2)
            ax.axhline(0, c='k', lw=1, ls='--', zorder=1e10)
            ax.axvline(0, c='k', lw=1, ls='--', zorder=1e10)
            _add_mu_pop_lines(ax, mu_pop)
            ax.set_xlabel(r"$\mu_{\alpha*}$ [mas/yr]")
            ax.set_ylabel(r"$\mu_\delta$ [mas/yr]")
            ax.set_title(f"{label} — members only")
            ax.set_aspect("equal")
            _style_ax(ax)

    _vd_nrows = 3 if _has_members else 2
    fig, axes = plt.subplots(_vd_nrows, 2, figsize=(13, 6 * _vd_nrows), layout="constrained")
    sc_last = _render_vd(axes[:2, :], gaia_pmra_h, gaia_pmdec_h, c_gaia,
                         bp3m_pmra_h, bp3m_pmdec_h, c_bp3m,
                         _is_mem_gaia_h, _is_mem_bp3m_h, norm)
    if _has_members:
        _add_mem_row_vd(axes[2, :], gaia_pmra_h, gaia_pmdec_h, c_gaia, _is_mem_gaia_h,
                        bp3m_pmra_h, bp3m_pmdec_h, c_bp3m, _is_mem_bp3m_h, norm)
    cbar = fig.colorbar(sc_last, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label(r"$(\det\,C_{\mu})^{1/4}$ [mas/yr]")
    fig.suptitle("PM vector diagrams coloured by geometric-mean uncertainty", fontsize=13)
    _save(fig, plot_dir / "pm_vector_diagram.png")

    if has_free:
        _bp3m_pmra_free_h  = _pmra_free[_bp3m_conv_free]
        _bp3m_pmdec_free_h = _pmdec_free[_bp3m_conv_free]
        _c_bp3m_free       = _sig_pm_free[_bp3m_conv_free]
        _is_m_free_h       = is_member[_bp3m_conv_free] if is_member is not None else None
        _all_unc_free = np.concatenate([c_gaia[np.isfinite(c_gaia)],
                                        _c_bp3m_free[np.isfinite(_c_bp3m_free)]])
        _norm_free = mcolors.LogNorm(
            vmin=max(float(np.nanpercentile(_all_unc_free, 2)), 1e-6),
            vmax=float(np.nanpercentile(_all_unc_free, 98)))
        fig, axes = plt.subplots(_vd_nrows, 2, figsize=(13, 6 * _vd_nrows), layout="constrained")
        sc_last = _render_vd(axes[:2, :], gaia_pmra_h, gaia_pmdec_h, c_gaia,
                             _bp3m_pmra_free_h, _bp3m_pmdec_free_h, _c_bp3m_free,
                             _is_mem_gaia_h, _is_m_free_h, _norm_free,
                             title_bp3m='BP3M (diffuse prior)')
        if _has_members:
            _add_mem_row_vd(axes[2, :], gaia_pmra_h, gaia_pmdec_h, c_gaia, _is_mem_gaia_h,
                            _bp3m_pmra_free_h, _bp3m_pmdec_free_h, _c_bp3m_free,
                            _is_m_free_h, _norm_free, labels=("Gaia", "BP3M (diffuse prior)"))
        cbar = fig.colorbar(sc_last, ax=axes, shrink=0.6, pad=0.02, aspect=30)
        cbar.set_label(r"$(\det\,C_{\mu})^{1/4}$ [mas/yr]")
        fig.suptitle("PM vector diagrams (diffuse prior) coloured by geometric-mean uncertainty",
                     fontsize=13)
        _save(fig, plot_dir / "pm_vector_diagram_diffuse_prior.png")

    # ── Figure 2b: PM vector diagrams with covariance error bars ─────────────
    print("  Plotting PM vector diagrams with error bars...")

    def _render_vd_eb(axes, pmra_g, pmdec_g, c_g, C_pm_g,
                      pmra_b, pmdec_b, c_b, C_pm_b,
                      is_m_g, is_m_b, vd_norm, title_bp3m='BP3M'):
        sc_last = None
        for col, pmra, pmdec, c_vals, C_pm_col, label, is_m in zip(
                [0, 1],
                [pmra_g,  pmra_b],
                [pmdec_g, pmdec_b],
                [c_g,     c_b],
                [C_pm_g,  C_pm_b],
                ["Gaia",  title_bp3m],
                [is_m_g,  is_m_b]):
            for row, xlim, ylim, suffix in zip(
                    [0, 1],
                    [full_xlim, zoom_xlim],
                    [full_ylim, zoom_ylim],
                    ["full range", "zoom (68% CI)"]):
                ax = axes[row, col]
                _pm_error_bars(ax, pmra, pmdec, C_pm_col)
                sc = _scatter_mem(ax, pmra, pmdec, c_vals, is_m,
                                  norm=vd_norm, cmap=cmap, zorder=2)
                ax.axhline(0, c='k', lw=1, ls='--', zorder=1e10)
                ax.axvline(0, c='k', lw=1, ls='--', zorder=1e10)
                _add_mu_pop_lines(ax, mu_pop)
                ax.set_xlim(xlim); ax.set_ylim(ylim)
                ax.set_xlabel(r"$\mu_{\alpha*}$ [mas/yr]")
                ax.set_ylabel(r"$\mu_\delta$ [mas/yr]")
                ax.set_title(f"{label} — {suffix}")
                ax.set_aspect("equal")
                _style_ax(ax)
                sc_last = sc
                if row == 0 and col == 0 and is_m is not None:
                    ax.legend(handles=_mem_legend_handles(), fontsize=7,
                              loc='upper left', framealpha=0.7)
        return sc_last

    C_pm_gaia_h = solver.C_survey[has_gaia, 2:4, 2:4]
    C_pm_bp3m_h = C_pm_bp3m[bp3m_converged]

    fig, axes = plt.subplots(_vd_nrows, 2, figsize=(13, 6 * _vd_nrows), layout="constrained")
    sc_last = _render_vd_eb(axes[:2, :],
                            gaia_pmra_h, gaia_pmdec_h, c_gaia, C_pm_gaia_h,
                            bp3m_pmra_h, bp3m_pmdec_h, c_bp3m, C_pm_bp3m_h,
                            _is_mem_gaia_h, _is_mem_bp3m_h, norm)
    if _has_members:
        _add_mem_row_vd(axes[2, :], gaia_pmra_h, gaia_pmdec_h, c_gaia, _is_mem_gaia_h,
                        bp3m_pmra_h, bp3m_pmdec_h, c_bp3m, _is_mem_bp3m_h, norm)
    cbar = fig.colorbar(sc_last, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label(r"$(\det\,C_{\mu})^{1/4}$ [mas/yr]")
    fig.suptitle(
        "PM vector diagrams with 1σ principal-axis error bars\n"
        r"(coloured by $(\det\,C_{\mu})^{1/4}$)",
        fontsize=13)
    _save(fig, plot_dir / "pm_vector_diagram_errorbars.png")

    if has_free:
        _C_pm_bp3m_free_h = _C_pm_free[_bp3m_conv_free]
        fig, axes = plt.subplots(_vd_nrows, 2, figsize=(13, 6 * _vd_nrows), layout="constrained")
        sc_last = _render_vd_eb(axes[:2, :],
                                gaia_pmra_h, gaia_pmdec_h, c_gaia, C_pm_gaia_h,
                                _bp3m_pmra_free_h, _bp3m_pmdec_free_h,
                                _c_bp3m_free, _C_pm_bp3m_free_h,
                                _is_mem_gaia_h, _is_m_free_h, _norm_free,
                                title_bp3m='BP3M (diffuse prior)')
        if _has_members:
            _add_mem_row_vd(axes[2, :], gaia_pmra_h, gaia_pmdec_h, c_gaia, _is_mem_gaia_h,
                            _bp3m_pmra_free_h, _bp3m_pmdec_free_h, _c_bp3m_free,
                            _is_m_free_h, _norm_free, labels=("Gaia", "BP3M (diffuse prior)"))
        cbar = fig.colorbar(sc_last, ax=axes, shrink=0.6, pad=0.02, aspect=30)
        cbar.set_label(r"$(\det\,C_{\mu})^{1/4}$ [mas/yr]")
        fig.suptitle(
            "PM vector diagrams (diffuse prior) with 1σ principal-axis error bars\n"
            r"(coloured by $(\det\,C_{\mu})^{1/4}$)",
            fontsize=13)
        _save(fig, plot_dir / "pm_vector_diagram_errorbars_diffuse_prior.png")

    # ── Figure 2c: BP3M PM coloured by detector position ─────────────────────
    print("  Plotting BP3M PM vector diagram coloured by detector position...")

    n_stars_global = len(solver.gaia_cat)
    _xo_sum = np.zeros(n_stars_global)
    _yo_sum = np.zeros(n_stars_global)
    _det_cnt = np.zeros(n_stars_global)
    for _img in solver.image_names:
        _df = solver.stars_per_image[_img]
        _gids = _df["Gaia_id"].to_numpy()
        _sidx_img = np.array([solver.star_id_to_idx[int(g)]
                               for g in _gids
                               if int(g) in solver.star_id_to_idx])
        _valid = np.array([int(g) in solver.star_id_to_idx for g in _gids])
        _xcol = "X_orig" if "X_orig" in _df.columns else "X"
        _ycol = "Y_orig" if "Y_orig" in _df.columns else "Y"
        _xo_sum[_sidx_img] += _df[_xcol].to_numpy(float)[_valid]
        _yo_sum[_sidx_img] += _df[_ycol].to_numpy(float)[_valid]
        _det_cnt[_sidx_img] += 1

    _obs = _det_cnt > 0
    x_orig_star = np.where(_obs, _xo_sum / np.maximum(_det_cnt, 1), np.nan)[bp3m_converged]
    y_orig_star = np.where(_obs, _yo_sum / np.maximum(_det_cnt, 1), np.nan)[bp3m_converged]

    bp3m_full_xlim = _padded_lim(bp3m_pmra_h)
    bp3m_full_ylim = _padded_lim(bp3m_pmdec_h)

    _bx_cen = np.nanmedian(bp3m_pmra_h)
    _by_cen = np.nanmedian(bp3m_pmdec_h)
    _bx_hw  = max(np.abs(np.nanpercentile(bp3m_pmra_h,  [16, 84]) - _bx_cen))
    _by_hw  = max(np.abs(np.nanpercentile(bp3m_pmdec_h, [16, 84]) - _by_cen))
    _b_hw   = max(_bx_hw, _by_hw) * 1.15
    bp3m_zoom_xlim = (_bx_cen - _b_hw, _bx_cen + _b_hw)
    bp3m_zoom_ylim = (_by_cen - _b_hw, _by_cen + _b_hw)

    def _lin_norm(vals):
        fin = vals[np.isfinite(vals)]
        vlo, vhi = np.nanpercentile(fin, [2, 98])
        return mcolors.Normalize(vmin=vlo, vmax=vhi)

    norm_xo = _lin_norm(x_orig_star)
    norm_yo = _lin_norm(y_orig_star)

    fig, axes = plt.subplots(_vd_nrows, 2, figsize=(13, 6 * _vd_nrows), layout="constrained")

    sc_xo = sc_yo = None
    for row, xlim, ylim, row_label in zip(
            [0, 1],
            [bp3m_full_xlim, bp3m_zoom_xlim],
            [bp3m_full_ylim, bp3m_zoom_ylim],
            ["full range", "zoom (68% CI)"]):

        for col, c_vals, norm_c, cmap_c, coord_label in zip(
                [0, 1],
                [x_orig_star, y_orig_star],
                [norm_xo,     norm_yo],
                ["plasma",    "plasma"],
                ["X_orig",    "Y_orig"]):

            ax = axes[row, col]
            sc = _scatter_mem(ax, bp3m_pmra_h, bp3m_pmdec_h,
                              c_vals, _is_mem_bp3m_h,
                              norm=norm_c, cmap=cmap_c, zorder=2)
            ax.axhline(0, c='k', lw=1, ls='--', zorder=1e10)
            ax.axvline(0, c='k', lw=1, ls='--', zorder=1e10)
            _add_mu_pop_lines(ax, mu_pop)
            if row == 0 and col == 0 and _is_mem_bp3m_h is not None:
                ax.legend(handles=_mem_legend_handles(), fontsize=7,
                          loc='upper left', framealpha=0.7)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(r"$\mu_{\alpha*}$ [mas/yr]")
            ax.set_ylabel(r"$\mu_\delta$ [mas/yr]")
            ax.set_title(f"BP3M — {row_label}  (colour: {coord_label})")
            ax.set_aspect("equal")
            _style_ax(ax)

            if row == 0 and col == 0:
                sc_xo = sc
            if row == 0 and col == 1:
                sc_yo = sc

    if _has_members and _is_mem_bp3m_h is not None:
        _m_b = _is_mem_bp3m_h
        for col, c_vals, norm_c, coord_label in zip(
                [0, 1],
                [x_orig_star[_m_b], y_orig_star[_m_b]],
                [norm_xo, norm_yo],
                ["X_orig", "Y_orig"]):
            ax = axes[2, col]
            ax.scatter(bp3m_pmra_h[_m_b], bp3m_pmdec_h[_m_b],
                       c=c_vals, norm=norm_c, cmap="plasma", s=15, alpha=0.85, zorder=2)
            ax.axhline(0, c='k', lw=1, ls='--', zorder=1e10)
            ax.axvline(0, c='k', lw=1, ls='--', zorder=1e10)
            _add_mu_pop_lines(ax, mu_pop)
            ax.set_xlabel(r"$\mu_{\alpha*}$ [mas/yr]")
            ax.set_ylabel(r"$\mu_\delta$ [mas/yr]")
            ax.set_title(f"BP3M — members only  (colour: {coord_label})")
            ax.set_aspect("equal")
            _style_ax(ax)

    cbar_xo = fig.colorbar(sc_xo, ax=axes[:, 0], shrink=0.6, pad=0.02, aspect=30)
    cbar_xo.set_label("X_orig [pixels]")
    cbar_yo = fig.colorbar(sc_yo, ax=axes[:, 1], shrink=0.6, pad=0.02, aspect=30)
    cbar_yo.set_label("Y_orig [pixels]")
    fig.suptitle("BP3M proper motions coloured by HST detector position", fontsize=13)
    _save(fig, plot_dir / "pm_vector_diagram_detector_pos.png")

    if has_free:
        _bp3m_conv_free_obs = _obs & _bp3m_conv_free
        _x_orig_free  = np.where(_bp3m_conv_free_obs,
                                 _xo_sum / np.maximum(_det_cnt, 1), np.nan)[_bp3m_conv_free]
        _y_orig_free  = np.where(_bp3m_conv_free_obs,
                                 _yo_sum / np.maximum(_det_cnt, 1), np.nan)[_bp3m_conv_free]
        _norm_xo_free = _lin_norm(_x_orig_free)
        _norm_yo_free = _lin_norm(_y_orig_free)

        _bx_cen_f = np.nanmedian(_bp3m_pmra_free_h)
        _by_cen_f = np.nanmedian(_bp3m_pmdec_free_h)
        _bx_hw_f  = max(np.abs(np.nanpercentile(_bp3m_pmra_free_h,  [16, 84]) - _bx_cen_f))
        _by_hw_f  = max(np.abs(np.nanpercentile(_bp3m_pmdec_free_h, [16, 84]) - _by_cen_f))
        _b_hw_f   = max(_bx_hw_f, _by_hw_f) * 1.15
        _free_full_xlim = _padded_lim(_bp3m_pmra_free_h)
        _free_full_ylim = _padded_lim(_bp3m_pmdec_free_h)
        _free_zoom_xlim = (_bx_cen_f - _b_hw_f, _bx_cen_f + _b_hw_f)
        _free_zoom_ylim = (_by_cen_f - _b_hw_f, _by_cen_f + _b_hw_f)

        fig, axes = plt.subplots(_vd_nrows, 2, figsize=(13, 6 * _vd_nrows), layout="constrained")
        sc_xo_f = sc_yo_f = None
        for row, xlim, ylim, row_label in zip(
                [0, 1],
                [_free_full_xlim, _free_zoom_xlim],
                [_free_full_ylim, _free_zoom_ylim],
                ["full range", "zoom (68% CI)"]):
            for col, c_vals, norm_c, cmap_c, coord_label in zip(
                    [0, 1],
                    [_x_orig_free, _y_orig_free],
                    [_norm_xo_free, _norm_yo_free],
                    ["plasma",      "plasma"],
                    ["X_orig",      "Y_orig"]):
                ax = axes[row, col]
                sc = _scatter_mem(ax, _bp3m_pmra_free_h, _bp3m_pmdec_free_h,
                                  c_vals, _is_m_free_h,
                                  norm=norm_c, cmap=cmap_c, zorder=2)
                ax.axhline(0, c='k', lw=1, ls='--', zorder=1e10)
                ax.axvline(0, c='k', lw=1, ls='--', zorder=1e10)
                _add_mu_pop_lines(ax, mu_pop)
                ax.set_xlim(xlim); ax.set_ylim(ylim)
                ax.set_xlabel(r"$\mu_{\alpha*}$ [mas/yr]")
                ax.set_ylabel(r"$\mu_\delta$ [mas/yr]")
                ax.set_title(f"BP3M (diffuse prior) — {row_label}  (colour: {coord_label})")
                ax.set_aspect("equal")
                _style_ax(ax)
                if row == 0 and col == 0:
                    sc_xo_f = sc
                    if _is_m_free_h is not None:
                        ax.legend(handles=_mem_legend_handles(), fontsize=7,
                                  loc='upper left', framealpha=0.7)
                if row == 0 and col == 1:
                    sc_yo_f = sc

        if _has_members and _is_m_free_h is not None:
            _m_f = _is_m_free_h
            for col, c_vals, norm_c, coord_label in zip(
                    [0, 1],
                    [_x_orig_free[_m_f], _y_orig_free[_m_f]],
                    [_norm_xo_free, _norm_yo_free],
                    ["X_orig", "Y_orig"]):
                ax = axes[2, col]
                ax.scatter(_bp3m_pmra_free_h[_m_f], _bp3m_pmdec_free_h[_m_f],
                           c=c_vals, norm=norm_c, cmap="plasma", s=15, alpha=0.85, zorder=2)
                ax.axhline(0, c='k', lw=1, ls='--', zorder=1e10)
                ax.axvline(0, c='k', lw=1, ls='--', zorder=1e10)
                _add_mu_pop_lines(ax, mu_pop)
                ax.set_xlabel(r"$\mu_{\alpha*}$ [mas/yr]")
                ax.set_ylabel(r"$\mu_\delta$ [mas/yr]")
                ax.set_title(f"BP3M (diffuse prior) — members only  (colour: {coord_label})")
                ax.set_aspect("equal")
                _style_ax(ax)

        cbar_xo = fig.colorbar(sc_xo_f, ax=axes[:, 0], shrink=0.6, pad=0.02, aspect=30)
        cbar_xo.set_label("X_orig [pixels]")
        cbar_yo = fig.colorbar(sc_yo_f, ax=axes[:, 1], shrink=0.6, pad=0.02, aspect=30)
        cbar_yo.set_label("Y_orig [pixels]")
        fig.suptitle("BP3M (diffuse prior) proper motions coloured by HST detector position",
                     fontsize=13)
        _save(fig, plot_dir / "pm_vector_diagram_detector_pos_diffuse_prior.png")

    # ── Figure: HST chi2 distributions ───────────────────────────────────────
    print("  Plotting HST chi2 distributions...")
    _plot_chi2_distributions(solver, r_hat, v_hat, plot_dir)

    # ── Figures: sky map + CMDs coloured by PM size / uncertainty ────────────
    print("  Plotting sky distribution and colour-magnitude diagrams...")
    _gc = solver.gaia_cat
    ra   = _gc["ra"].to_numpy(float)
    dec  = _gc["dec"].to_numpy(float)
    bp_rp = _gc["bp_rp"].to_numpy(float) if "bp_rp" in _gc.columns else np.full(len(_gc), np.nan)
    pm_size = np.sqrt(pmra_bp3m**2 + pmdec_bp3m**2)
    pm_unc  = np.sqrt(sig_pmra_bp3m**2 + sig_pmdec_bp3m**2)
    ok = bp3m_converged & np.isfinite(gmag) & np.isfinite(pm_size)
    _is_mem_sky = is_member if is_member is not None else None
    _plot_sky_and_cmd(ra, dec, gmag, bp_rp, pm_size, pm_unc, ok, plot_dir,
                      is_member=_is_mem_sky)
    if has_free:
        pm_size_free = np.sqrt(_pmra_free**2 + _pmdec_free**2)
        pm_unc_free  = np.sqrt(_sig_pmra_free**2 + _sig_pmdec_free**2)
        ok_free = _bp3m_conv_free & np.isfinite(gmag) & np.isfinite(pm_size_free)
        _plot_sky_and_cmd(ra, dec, gmag, bp_rp, pm_size_free, pm_unc_free, ok_free,
                          plot_dir, is_member=_is_mem_sky,
                          fname='sky_cmd_pm_diffuse_prior.png')

    # ── DELVE sky CMD plots (one per available DELVE colour) ─────────────────
    # Use bp3m_converged & finite pm only — do NOT require finite Gaia G,
    # because DELVE-only stars (negative Gaia_id) have gmag=NaN and would
    # be excluded entirely if we reused the Gaia `ok` mask.
    _ok_delve_base = bp3m_converged & np.isfinite(pm_size)
    _DELVE_COLORS = [
        ('delve_gmag', 'delve_rmag', 'DELVE g − r (mag)', 'DELVE r (mag)', 'sky_cmd_pm_delve_gr.png'),
        ('delve_rmag', 'delve_imag', 'DELVE r − i (mag)', 'DELVE i (mag)', 'sky_cmd_pm_delve_ri.png'),
    ]
    for _d_blue, _d_red, _clabel, _mlabel, _fname in _DELVE_COLORS:
        if _d_blue not in _gc.columns or _d_red not in _gc.columns:
            continue
        _sentinel_lo, _sentinel_hi = -90.0, 50.0
        _vb = pd.to_numeric(_gc[_d_blue], errors='coerce').to_numpy(float).copy()
        _vr = pd.to_numeric(_gc[_d_red],  errors='coerce').to_numpy(float).copy()
        _vb[(_vb < _sentinel_lo) | (_vb > _sentinel_hi)] = np.nan
        _vr[(_vr < _sentinel_lo) | (_vr > _sentinel_hi)] = np.nan
        _d_color = _vb - _vr
        _d_mag   = _vr
        _ok_d = _ok_delve_base & np.isfinite(_d_color) & np.isfinite(_d_mag)
        if _ok_d.sum() < 5:
            continue
        _plot_sky_and_cmd(ra, dec, _d_mag, _d_color, pm_size, pm_unc, _ok_d,
                          plot_dir, is_member=_is_mem_sky, fname=_fname,
                          color_label=_clabel, mag_label=_mlabel)

    # ── Figure: HST XY residuals + BP3M proper motions on detector ───────────
    if not plot_residuals:
        print(f"  All plots saved to {plot_dir}/")
        return
    print("  Plotting detector residuals and proper motion maps...")
    resid_dict = solver.compute_residuals(r_hat, v_hat, C_r=C_r, C_vT=C_vT)

    _AMP_SUFFIXES = ('_llo', '_rlo', '_lhi', '_rhi')
    _CCD_SUFFIXES = ('_lo', '_hi')

    img_groups = defaultdict(list)
    for img in resid_dict:
        if img.endswith(_AMP_SUFFIXES):
            base = img[:-4]
        elif img.endswith(_CCD_SUFFIXES):
            base = img[:-3]
        else:
            base = img
        img_groups[base].append(img)

    for base_name, img_list in sorted(img_groups.items()):
        img_list = sorted(img_list)

        def _cat(key):
            return np.concatenate([resid_dict[img][key] for img in img_list])

        X_c   = _cat("X_c")
        Y_c   = _cat("Y_c")
        res_x = _cat("resid_x")
        res_y = _cat("resid_y")
        sr_x  = _cat("sigma_resid_x")
        sr_y  = _cat("sigma_resid_y")
        use   = _cat("use")
        sidx  = _cat("sidx")

        if np.sum(use) == 0:
            print(f'  SKIPPING {base_name}: no usable stars')
            continue

        use &= bp3m_converged[sidx]

        pscale    = images[img_list[0]]["orig_pixel_scale"]
        res_x_mas = res_x * pscale
        res_y_mas = res_y * pscale

        pmra_img  = pmra_bp3m[sidx]
        pmdec_img = pmdec_bp3m[sidx]

        n_split = len(img_list)
        if n_split == 4:
            split_note = "  (4 amp quadrants combined)"
        elif n_split > 1:
            split_note = f"  ({n_split} CCD halves combined)"
        else:
            split_note = ""

        fig, axes = plt.subplots(3, 2, figsize=(13, 15), layout="constrained")

        for ax, res_mas, comp in zip(axes[0], [res_x_mas, res_y_mas], ["x", "y"]):
            vmax = np.nanpercentile(np.abs(res_mas[use]), 95)
            sc = ax.scatter(
                X_c[use], Y_c[use], c=res_mas[use],
                s=10, cmap="RdYlBu_r", vmin=-vmax, vmax=vmax, alpha=0.8, zorder=2)
            fig.colorbar(sc, ax=ax, label=f"residual {comp} [mas]")
            ax.set_xlabel("X − Xo [pixels]")
            ax.set_ylabel("Y − Yo [pixels]")
            ax.set_title(f"{base_name}  residual {comp}  (n={use.sum()}){split_note}")
            ax.set_aspect("equal")
            _style_ax(ax)

        for ax, sr, comp in zip(axes[1], [sr_x, sr_y], ["x", "y"]):
            finite = np.isfinite(sr[use])
            vmax_s = np.nanpercentile(np.abs(sr[use][finite]), 95) if finite.any() else 3.
            sc = ax.scatter(
                X_c[use], Y_c[use], c=sr[use],
                s=10, cmap="RdYlBu_r", vmin=-vmax_s, vmax=vmax_s, alpha=0.8, zorder=2)
            fig.colorbar(sc, ax=ax, label=f"residual {comp} / σ_HST  [σ]")
            ax.set_xlabel("X − Xo [pixels]")
            ax.set_ylabel("Y − Yo [pixels]")
            ax.set_title(f"{base_name}  σ-residual {comp}  "
                         f"(RMS = {np.nanstd(sr[use]):.2f} σ){split_note}")
            ax.set_aspect("equal")
            _style_ax(ax)

        for ax, pm_vals, comp_tex in zip(
                axes[2],
                [pmra_img,  pmdec_img],
                [r"$\mu_{\alpha*}$", r"$\mu_\delta$"]):
            pm_use = pm_vals[use]
            p16, p84 = np.nanpercentile(pm_use, [16, 84])
            sc = ax.scatter(
                X_c[use], Y_c[use], c=pm_use,
                s=10, cmap="viridis", vmin=p16, vmax=p84, alpha=0.8, zorder=2)
            fig.colorbar(sc, ax=ax, label=f"{comp_tex} BP3M [mas/yr]")
            ax.set_xlabel("X − Xo [pixels]")
            ax.set_ylabel("Y − Yo [pixels]")
            ax.set_title(f"{base_name}  {comp_tex} BP3M  "
                         f"(clim: [{p16:.2f}, {p84:.2f}] mas/yr){split_note}")
            ax.set_aspect("equal")
            _style_ax(ax)

        fig.suptitle(f"HST detector residuals & proper motions — {base_name}{split_note}",
                     fontsize=12)
        _save(fig, resid_plot_dir / f"residuals_{base_name}.png")

    print(f"  All plots saved to {plot_dir}/")


def _plot_chi2_distributions(solver, r_hat, v_hat, plot_dir):
    """Three-panel diagnostic for the HST-only chi2 per star per image."""
    from scipy.stats import chi2 as chi2_dist

    resid_hst = solver.compute_residuals(r_hat, v_hat)

    per_img_chi2   = {}
    per_img_all    = {}
    per_img_alpha  = {}
    _MEDIAN_CHI2_2 = 2.0 * np.log(2.0)

    for img, rd in resid_hst.items():
        use    = solver._img_data[img]["use_for_fit"]
        chi2_v = rd["sigma_resid"] ** 2

        per_img_all[img]   = chi2_v
        per_img_chi2[img]  = chi2_v[use]

        med = np.median(chi2_v[use]) if use.sum() >= 2 else np.nan
        per_img_alpha[img] = float(max(1.0, np.sqrt(med / _MEDIAN_CHI2_2)))

    all_accepted = np.concatenate(list(per_img_chi2.values()))
    all_vals     = np.concatenate(list(per_img_all.values()))

    thresholds = {
        "0.99":   chi2_dist.ppf(0.99,   df=2),
        "0.999":  chi2_dist.ppf(0.999,  df=2),
        "0.9999": chi2_dist.ppf(0.9999, df=2),
    }
    thresh_colors = {"0.99": "royalblue", "0.999": "darkorange", "0.9999": "crimson"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("HST-only chi2 distributions at convergence", fontsize=13)

    ax = axes[0]
    clip = np.percentile(all_accepted, 99.5)
    bins = np.linspace(0, max(clip, 20), 80)
    ax.hist(all_accepted, bins=bins, density=True, color="steelblue",
            alpha=0.6, label="accepted stars")
    xx = np.linspace(0.01, bins[-1], 400)
    ax.plot(xx, chi2_dist.pdf(xx, df=2), "k-", lw=1.5, label="χ²(2) theory")
    for label, thr in thresholds.items():
        ax.axvline(thr, color=thresh_colors[label], lw=1.2, ls="--",
                   label=f"q={label}  ({thr:.1f})")
    ax.set_xlabel("σ_resid² (HST-only chi2)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution (accepted stars)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, bins[-1])

    ax = axes[1]
    sorted_chi2 = np.sort(all_accepted)
    cdf = np.arange(1, len(sorted_chi2) + 1) / len(sorted_chi2)
    ax.plot(sorted_chi2, cdf, color="steelblue", lw=1.5)
    ax.plot(np.sort(all_vals), np.arange(1, len(all_vals) + 1) / len(all_vals),
            color="gray", lw=1, ls=":", alpha=0.7, label="all (incl. excluded)")
    for label, thr in thresholds.items():
        frac_survive = float((all_accepted < thr).mean())
        ax.axvline(thr, color=thresh_colors[label], lw=1.2, ls="--",
                   label=f"q={label}: {100*frac_survive:.1f}% survive")
    ax.set_xlabel("σ_resid² threshold")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF — surviving fraction vs. threshold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, max(thresholds["0.9999"] * 1.3, np.percentile(all_accepted, 98)))
    ax.set_ylim(0, 1)

    ax = axes[2]
    img_names = list(per_img_chi2.keys())
    medians   = [np.median(per_img_chi2[im]) for im in img_names]
    alphas    = [per_img_alpha[im] for im in img_names]

    order   = np.argsort(medians)[::-1]
    names_s = [img_names[i] for i in order]
    meds_s  = [medians[i]   for i in order]
    alps_s  = [alphas[i]    for i in order]

    y = np.arange(len(names_s))
    bar_h = max(0.3, min(0.8, 12.0 / max(len(names_s), 1)))

    bars = ax.barh(y, meds_s, height=bar_h, color="steelblue", alpha=0.7,
                   label="median chi2")
    for bar, alp in zip(bars, alps_s):
        bar.set_facecolor("tomato" if alp > 2 else "steelblue")

    for i, (med, alp) in enumerate(zip(meds_s, alps_s)):
        ax.text(med + 0.05, i, f"α={alp:.2f}", va="center", fontsize=6)

    for label, thr in thresholds.items():
        ax.axvline(thr, color=thresh_colors[label], lw=1.0, ls="--", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(names_s, fontsize=max(5, min(8, 200 // max(len(names_s), 1))))
    ax.set_xlabel("Median σ_resid² (HST-only chi2, accepted stars)")
    ax.set_title("Per-image: median chi2 & alpha\n(red = α > 2)")
    finite_meds = [m for m in meds_s if np.isfinite(m)]
    xlim_right = max(max(finite_meds) * 1.25 if finite_meds else 0,
                     thresholds["0.999"] * 1.1)
    ax.set_xlim(0, xlim_right)
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, plot_dir / "chi2_hst_distributions.png")
    plt.close(fig)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pm_error_bars(ax, pmra, pmdec, C_pm, color="gray", alpha=0.18, lw=0.6):
    """Draw 1σ principal-axis error bars for each point in a PM vector diagram."""
    eigvals, eigvecs = np.linalg.eigh(C_pm)
    half_axes = np.sqrt(np.maximum(eigvals, 0.))

    centers = np.stack([pmra, pmdec], axis=1)
    delta = eigvecs * half_axes[:, np.newaxis, :]

    segs = np.concatenate([
        np.stack([centers - delta[:, :, 0], centers + delta[:, :, 0]], axis=1),
        np.stack([centers - delta[:, :, 1], centers + delta[:, :, 1]], axis=1),
    ], axis=0)

    lc = LineCollection(segs, colors=color, alpha=alpha, linewidths=lw, zorder=1)
    ax.add_collection(lc)


def _pm_geom_unc(sig_pmra, sig_pmdec, rho):
    """Geometric-mean PM uncertainty: (det C_pm)^(1/4)."""
    rho  = np.clip(np.nan_to_num(rho), -0.9999, 0.9999)
    det  = sig_pmra**2 * sig_pmdec**2 * (1.0 - rho**2)
    return np.where((sig_pmra > 0) & (sig_pmdec > 0), det**0.25, np.nan)


def _padded_lim(*arrays, pad=0.04):
    """Return (lo, hi) spanning all values in *arrays with a fractional pad."""
    lo = min(np.nanmin(a) for a in arrays)
    hi = max(np.nanmax(a) for a in arrays)
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


def _style_ax(ax):
    """Apply consistent grid + minor-tick style to an axis."""
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-",  linewidth=0.5, alpha=0.6)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.4)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def _plot_sky_and_cmd(ra, dec, gmag, bp_rp, pm_size, pm_unc, ok, plot_dir,
                      is_member=None, fname='sky_cmd_pm.png',
                      color_label="Gaia BP − RP (mag)", mag_label="Gaia G (mag)"):
    """Three panels: sky map coloured by |PM|, CMD coloured by |PM|, CMD coloured by σ_PM."""
    from matplotlib.colors import LogNorm

    # --- common colour scales (log-normalised) ---
    pm_vals  = pm_size[ok & (pm_size > 0)]
    unc_vals = pm_unc[ok & (pm_unc > 0)]
    vmin_pm  = float(np.nanpercentile(pm_vals,  1))
    vmax_pm  = float(np.nanpercentile(pm_vals, 99))
    vmin_unc = float(np.nanpercentile(unc_vals,  1))
    vmax_unc = float(np.nanpercentile(unc_vals, 99))

    norm_pm  = LogNorm(vmin=max(vmin_pm,  1e-3), vmax=vmax_pm)
    norm_unc = LogNorm(vmin=max(vmin_unc, 1e-3), vmax=vmax_unc)
    cmap_pm  = "plasma"
    cmap_unc = "viridis"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")

    # ── Panel 1: sky distribution coloured by |PM| ──────────────────────────
    ax = axes[0]
    _is_m_ok = is_member[ok] if is_member is not None else None
    sc = _scatter_mem(ax, ra[ok], dec[ok], pm_size[ok], _is_m_ok,
                      norm=norm_pm, cmap=cmap_pm, zorder=2,
                      linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=ax, label="|PM| (mas/yr)")
    ax.set_xlabel("R.A. (deg)")
    ax.set_ylabel("Dec. (deg)")
    ax.set_title("Sky distribution  (colour = |PM|)")
    ax.invert_xaxis()
    _style_ax(ax)

    # ── Panel 2: CMD coloured by |PM| ───────────────────────────────────────
    ax = axes[1]
    has_cmd = ok & np.isfinite(bp_rp)
    _is_m_cmd = is_member[has_cmd] if is_member is not None else None
    sc = _scatter_mem(ax, bp_rp[has_cmd], gmag[has_cmd], pm_size[has_cmd], _is_m_cmd,
                      norm=norm_pm, cmap=cmap_pm, zorder=2,
                      linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=ax, label="|PM| (mas/yr)")
    ax.set_xlabel(color_label)
    ax.set_ylabel(mag_label)
    ax.set_title("CMD  (colour = |PM|)")
    ax.invert_yaxis()
    _style_ax(ax)

    # ── Panel 3: CMD coloured by PM uncertainty ──────────────────────────────
    ax = axes[2]
    _is_m_cmd2 = is_member[has_cmd] if is_member is not None else None
    sc = _scatter_mem(ax, bp_rp[has_cmd], gmag[has_cmd], pm_unc[has_cmd], _is_m_cmd2,
                      norm=norm_unc, cmap=cmap_unc, zorder=2,
                      linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=ax, label="σ_PM (mas/yr)")
    ax.set_xlabel(color_label)
    ax.set_ylabel(mag_label)
    ax.set_title("CMD  (colour = σ_PM)")
    ax.invert_yaxis()
    _style_ax(ax)

    _save(fig, plot_dir / fname)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")
