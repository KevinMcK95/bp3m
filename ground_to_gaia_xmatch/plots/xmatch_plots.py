"""
Cross-match diagnostic figures.

This is the established 5x2 diagnostic layout, unchanged.  It is driven entirely
by the match table, so the one implementation serves every instrument — that is
the only thing the refactor changed about it.

Series are split by src_is_star (rejected / matched non-star / matched star)
with the original red / orange / blue colouring.  Do not restyle or re-split
these panels: they are read side-by-side against historical runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_xmatch_plots(result: dict, out_dir: Path) -> Path:
    """Write the diagnostic figure set for one image."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _diagnostic_plots(result, out_dir)
    _offset_histogram(result, out_dir)
    return out_dir / 'diagnostic_plots.png'


def _diagnostic_plots(result: dict, det_dir: Path) -> None:
    """5x2 diagnostic figure."""
    matched = result['matched']
    rejected = result['rejected']
    if rejected is None:
        rejected = matched.iloc[0:0]
    tr = result['params']
    image_name = result['meta'].image_id
    INST = result['meta'].instrument.upper()

    all_df = pd.concat([matched, rejected], ignore_index=True)

    m_stars = matched[matched['src_is_star'].astype(bool)]
    m_nonstars = matched[~matched['src_is_star'].astype(bool)]

    fig, axes = plt.subplots(5, 2, figsize=(14, 24 / 4 * 5))
    fig.suptitle(f"Match Diagnostics: {image_name}", fontsize=18)

    # ── 0,0: Field Map (pixels) ──────────────────────────────────────────────
    ax = axes[0, 0]
    if len(rejected) > 0:
        ax.scatter(rejected['src_x'], rejected['src_y'],
                   c='red', alpha=0.3, s=5, label='Rejected')
    if len(m_nonstars) > 0:
        ax.scatter(m_nonstars['src_x'], m_nonstars['src_y'],
                   c='orange', alpha=0.6, s=12, label='Matched non-star')
    if len(m_stars) > 0:
        ax.scatter(m_stars['src_x'], m_stars['src_y'],
                   c='blue', alpha=0.6, s=10, label='Matched star')
    ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)')
    ax.set_title('Field Map (pixels)'); ax.legend(fontsize=7)

    # ── 0,1: Gaia Proper Motions ─────────────────────────────────────────────
    ax = axes[0, 1]
    if len(rejected) > 0:
        ax.scatter(rejected['gaia_pmra'], rejected['gaia_pmdec'],
                   c='red', alpha=0.15, s=5, label='Rejected')
    if len(m_nonstars) > 0:
        ax.scatter(m_nonstars['gaia_pmra'], m_nonstars['gaia_pmdec'],
                   c='orange', alpha=0.6, s=12, label='Matched non-star')
    if len(m_stars) > 0:
        ax.scatter(m_stars['gaia_pmra'], m_stars['gaia_pmdec'],
                   c='blue', alpha=0.6, s=10, label='Matched star')
    ax.set_xlabel('PMRA (mas/yr)'); ax.set_ylabel('PMDec (mas/yr)')
    ax.set_title('Gaia Proper Motions'); ax.legend(fontsize=7)

    # ── 1,0: Gaia CMD (G vs BP-RP) ───────────────────────────────────────────
    ax = axes[1, 0]
    if 'gaia_bprp' in all_df.columns:
        if len(rejected) > 0:
            ax.scatter(rejected['gaia_bprp'], rejected['gaia_gmag'],
                       c='red', alpha=0.15, s=5, label='Rejected')
        if len(m_nonstars) > 0:
            ax.scatter(m_nonstars['gaia_bprp'], m_nonstars['gaia_gmag'],
                       c='orange', alpha=0.6, s=12, label='Matched non-star')
        if len(m_stars) > 0:
            ax.scatter(m_stars['gaia_bprp'], m_stars['gaia_gmag'],
                       c='blue', alpha=0.6, s=10, label='Matched star')
        ax.invert_yaxis()
        ax.set_xlabel('BP − RP (mag)'); ax.set_ylabel('Gaia G (mag)')
        ax.set_title('Gaia Color-Magnitude Diagram'); ax.legend(fontsize=7)

    # ── 1,1: G vs G-instrumental (photometric calibration) ───────────────────
    ax = axes[1, 1]
    zp = float(tr['zp'])
    for sub, col, lbl in [(rejected, 'red', 'Rejected'),
                          (m_nonstars, 'orange', 'Matched non-star'),
                          (m_stars, 'blue', 'Matched star')]:
        if len(sub) > 0:
            color_inst = sub['gaia_gmag'] - sub['src_mag']
            ax.scatter(color_inst, sub['gaia_gmag'],
                       c=col, alpha=0.5 if col != 'red' else 0.15,
                       s=12 if col != 'red' else 5, label=lbl)
    ax.axvline(zp, color='black', ls='--', lw=1, label=f'ZP={zp:.3f}')
    ax.invert_yaxis()
    ax.set_xlabel(f'G − {INST} (mag)'); ax.set_ylabel('Gaia G (mag)')
    ax.set_title(f'Gaia G − {INST} Color-Magnitude'); ax.legend(fontsize=7)

    # ── 2,0: (xi,eta) Residual Scatter ───────────────────────────────────────
    ax = axes[2, 0]
    if len(rejected) > 0:
        ax.scatter(rejected['residual_xi_mas'], rejected['residual_eta_mas'],
                   c='red', alpha=0.2, s=8)
    if len(m_nonstars) > 0:
        ax.scatter(m_nonstars['residual_xi_mas'], m_nonstars['residual_eta_mas'],
                   c='orange', alpha=0.6, s=12, label='Matched non-star')
    if len(m_stars) > 0:
        ax.scatter(m_stars['residual_xi_mas'], m_stars['residual_eta_mas'],
                   c='blue', alpha=0.6, s=10, label='Matched star')
    ax.axhline(0, color='black', ls='--', alpha=0.5)
    ax.axvline(0, color='black', ls='--', alpha=0.5)
    if len(matched) > 0:
        lim = max(matched['residual_xi_mas'].abs().max(),
                  matched['residual_eta_mas'].abs().max()) * 2.5
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('Δξ (mas)'); ax.set_ylabel('Δη (mas)')
    ax.set_title('(ξ,η) Residuals'); ax.legend(fontsize=7)

    # ── 2,1: Normalized Residuals ────────────────────────────────────────────
    ax = axes[2, 1]
    if len(rejected) > 0:
        sx_r = rejected['sigma_xi_mas'].values
        sy_r = rejected['sigma_eta_mas'].values
        ax.scatter(rejected['residual_xi_mas'] / sx_r,
                   rejected['residual_eta_mas'] / sy_r,
                   c='red', alpha=0.15, s=8)
    for sub, col in [(m_nonstars, 'orange'), (m_stars, 'blue')]:
        if len(sub) > 0:
            sx, sy = sub['sigma_xi_mas'].values, sub['sigma_eta_mas'].values
            ax.scatter(sub['residual_xi_mas'] / sx, sub['residual_eta_mas'] / sy,
                       c=col, alpha=0.5, s=15)
    ax.add_artist(plt.Circle((0, 0), 1, color='black', fill=False, ls='--', alpha=0.5))
    ax.add_artist(plt.Circle((0, 0), 5, color='red', fill=False, ls=':', alpha=0.5))
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
    ax.set_xlabel('Δξ / σ_ξ'); ax.set_ylabel('Δη / σ_η')
    ax.set_title('Normalized Residuals')

    # ── 3,0: |residual| vs Gaia mag (log y) ──────────────────────────────────
    ax = axes[3, 0]
    mag_min = all_df['gaia_gmag'].min(); mag_max = all_df['gaia_gmag'].max()
    mag_pad = (mag_max - mag_min) * 0.05
    if len(rejected) > 0:
        dr_r = np.sqrt(rejected['residual_xi_mas']**2 + rejected['residual_eta_mas']**2)
        ax.scatter(rejected['gaia_gmag'], dr_r, c='red', alpha=0.15, s=5, label='Rejected')
    for sub, col, lbl in [(m_nonstars, 'orange', 'Non-star'),
                          (m_stars, 'blue', 'Star')]:
        if len(sub) > 0:
            dr = np.sqrt(sub['residual_xi_mas']**2 + sub['residual_eta_mas']**2)
            ax.scatter(sub['gaia_gmag'], dr, c=col, alpha=0.5, s=10, label=lbl)
    ax.set_yscale('log')
    ax.set_xlabel('Gaia G (mag)'); ax.set_ylabel('|residual| (mas)')
    ax.set_title('Residual vs Gaia Mag'); ax.legend(fontsize=7)
    ax.set_xlim(mag_min - mag_pad, mag_max + mag_pad)

    # ── 3,1: Sigma histogram ─────────────────────────────────────────────────
    ax = axes[3, 1]
    bins = np.linspace(0, 10, 50)
    if len(m_nonstars) > 0:
        ax.hist(m_nonstars['residual_sigma'], bins=bins, color='orange', alpha=0.5, label='Non-star')
    if len(m_stars) > 0:
        ax.hist(m_stars['residual_sigma'], bins=bins, color='blue', alpha=0.6, label='Star')
    if len(rejected) > 0:
        rej_near = rejected[rejected['residual_sigma'] < 10.0]
        ax.hist(rej_near['residual_sigma'], bins=bins, color='red', alpha=0.3, label='Rejected (<10σ)')
    ax.axvline(5, color='red', ls='--')
    ax.set_yscale('log')
    ax.set_xlabel('σ (Mahalanobis)'); ax.set_ylabel('Count (log)')
    ax.set_title('Sigma Distribution'); ax.legend(fontsize=7)

    # ── 4,0: Sigma vs Gaia mag ───────────────────────────────────────────────
    ax = axes[4, 0]
    if len(rejected) > 0:
        rej_near = rejected[rejected['residual_sigma'] < 15.0]
        ax.scatter(rej_near['gaia_gmag'], rej_near['residual_sigma'],
                   c='red', alpha=0.15, s=5, label='Rejected (<15σ)')
    for sub, col, lbl in [(m_nonstars, 'orange', 'Non-star'),
                          (m_stars, 'blue', 'Star')]:
        if len(sub) > 0:
            ax.scatter(sub['gaia_gmag'], sub['residual_sigma'],
                       c=col, alpha=0.5, s=10, label=lbl)
    ax.axhline(5, color='red', ls='--', label='Threshold (5σ)')
    ax.set_xlabel('Gaia G (mag)'); ax.set_ylabel('σ (Mahalanobis)')
    ax.set_title('Sigma vs Gaia Mag'); ax.legend(fontsize=7)
    ax.set_xlim(mag_min - mag_pad, mag_max + mag_pad)

    # ── 4,1: Color-color (BP-RP vs G-instrumental) ───────────────────────────
    ax = axes[4, 1]
    if 'gaia_bprp' in all_df.columns:
        for sub, col, lbl in [(rejected, 'red', 'Rejected'),
                              (m_nonstars, 'orange', 'Non-star'),
                              (m_stars, 'blue', 'Star')]:
            if len(sub) > 0:
                color_inst = sub['gaia_gmag'] - sub['src_mag']
                ax.scatter(sub['gaia_bprp'], color_inst,
                           c=col, alpha=0.5 if col != 'red' else 0.15,
                           s=12 if col != 'red' else 5, label=lbl)
        ax.invert_yaxis()
        ax.set_xlabel('BP − RP (mag)'); ax.set_ylabel(f'G − {INST} (mag)')
        ax.set_title('Color-Color Diagram'); ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(det_dir / 'diagnostic_plots.png', dpi=150)
    plt.close(fig)


def _offset_histogram(result: dict, det_dir: Path) -> None:
    """Discovery voting histogram for the winning tier."""
    best = result.get('best') or {}
    hist = best.get('offset_hist')
    if hist is None:
        return
    xed, yed = best['offset_xed'], best['offset_yed']
    peaks = best.get('offset_peaks', [])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = hist.T.copy().astype(float)
    disp[disp == 0] = np.nan
    im = ax.imshow(disp, origin='lower', aspect='equal',
                   extent=[xed[0], xed[-1], yed[0], yed[-1]], cmap='viridis')
    plt.colorbar(im, ax=ax, label='weighted density')
    for dx, dy, _ in peaks:
        ax.axvline(dx, color='red', lw=0.8, ls='--', alpha=0.7)
        ax.axhline(dy, color='red', lw=0.8, ls='--', alpha=0.7)
    ax.set_xlabel('Δξ  (mas)')
    ax.set_ylabel('Δη  (mas)')
    ax.set_title(f"{result['meta'].image_id}  |  "
                 f"best {best.get('tier', '')}  [{best.get('gaia_tier', '')}]")
    fig.tight_layout()
    fig.savefig(det_dir / 'offset_histogram.png', dpi=120)
    plt.close(fig)


__all__ = ['make_xmatch_plots']
