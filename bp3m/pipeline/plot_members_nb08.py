"""Final pop-fit v2 membership drawn on the notebook-08 panels (user 2026-10-06).

Notebook 08 (member_seed_v2_selection.png) shows the v2 master catalogue -- HST-only
sources included -- on VPD / Gaia CMD / parallax / HST CMDs / colour-colour planes.
The pop-fit's own member_selection_panels.png works from the fit's star table and
reads as Gaia-only.  This module is a verbatim port of the notebook's catalogue
construction (cell 1), panel definitions (cell 2) and summary figure (cell 4); the
only change is the selection mask, which comes from <popfit_dir>/stellar_astrometry.csv
(is_member) instead of the drawn regions.

    python -m bp3m.pipeline.plot_members_nb08 FIELD [--popfit_dir BP3M_pop_fit_v2_results]
                                                   [--output_dir GaiaHub_results]
Writes <popfit_dir>/plots/member_selection_nb08style.png.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

NB_VERSION = '2026-10-01a'      # notebook 08 version this port follows
VPD_ZOOM = 3.0
MIN_PAIR = 50
GAIA_SOURCE = 'v2'
SIG_PM_MAX = 1.0
N_DET_MIN = 3

_WAVELENGTHS = {  # nm, blue->red ordering
    'F275W': 275, 'F336W': 336, 'F390W': 390, 'F435W': 435, 'F438W': 438,
    'F475W': 475, 'F555W': 555, 'F606W': 606, 'F625W': 625, 'F775W': 775,
    'F814W': 814, 'F850LP': 900, 'F110W': 1100, 'F125W': 1250, 'F160W': 1600,
}


def _wl(band):
    return _WAVELENGTHS.get(band.split('/')[0].upper(), 9999)


def build_master(field_dir: Path) -> pd.DataFrame:
    """Notebook-08 cell 1: master catalogue shown in the selection UI."""
    from bp3m.pipeline.run_pop_fit_v2 import _load_catalog
    mc = _load_catalog(field_dir, gaia_source=GAIA_SOURCE)
    _sa_path = field_dir / 'BP3M_v2_results' / 'stellar_astrometry.csv'
    if not _sa_path.exists():
        _sa_path = field_dir / 'BP3M_results' / 'stellar_astrometry.csv'
    if _sa_path.exists():
        _sa = pd.read_csv(_sa_path, dtype={'Gaia_id': np.int64},
                          usecols=lambda c: c in ('Gaia_id', 'gmag', 'rpmag', 'bp_rp', 'parallax_bp3m'))
        _sa = _sa.rename(columns={'Gaia_id': 'gaia_source_id'}).drop_duplicates('gaia_source_id')
        mc = mc.drop(columns=[c for c in ('gmag', 'rpmag', 'bp_rp', 'parallax_bp3m') if c in mc.columns])
        mc = mc.merge(_sa, on='gaia_source_id', how='left')
    else:
        mc['gmag'] = np.nan; mc['rpmag'] = np.nan; mc['bp_rp'] = np.nan; mc['parallax_bp3m'] = np.nan
    _bands_red_first = sorted((c for c in mc.columns if c.startswith('mag_wmean_')),
                              key=lambda c: -float(''.join(ch for ch in c.split('_')[-1] if ch.isdigit()) or 0))
    mc['G_any'] = mc['gmag']
    for _b in _bands_red_first:
        _ok = np.isfinite(mc['gmag']) & np.isfinite(mc[_b])
        if _ok.sum() < 10:
            continue
        _off = np.nanmedian(mc.loc[_ok, 'gmag'] - mc.loc[_ok, _b])
        _need = ~np.isfinite(mc['G_any']) & np.isfinite(mc[_b])
        mc.loc[_need, 'G_any'] = mc.loc[_need, _b] + _off
    mc['plx_any'] = np.where(np.isfinite(mc['parallax_bp3m']), mc['parallax_bp3m'],
                             pd.to_numeric(mc.get('parallax_xmatch'), errors='coerce'))
    sig_rms = np.sqrt((mc['sig_ra'] ** 2 + mc['sig_dec'] ** 2) / 2)
    n_det_fit = pd.to_numeric(mc.get('n_detect_fit'), errors='coerce').fillna(0)
    is_gaia = (mc['gaia_source_id'] != 0).to_numpy()
    _qual = (np.isfinite(sig_rms) & (sig_rms > 0) & (sig_rms < SIG_PM_MAX)
             & (n_det_fit >= N_DET_MIN)).to_numpy()
    show = np.isfinite(mc['pm_ra']).to_numpy() & np.isfinite(mc['pm_dec']).to_numpy() & (is_gaia | _qual)
    master = mc.loc[show].reset_index(drop=True)
    master['sid_str'] = master['source_index'].astype(str)
    return master


def build_panels(master: pd.DataFrame):
    """Notebook-08 cell 2: (title, x, y, xlabel, ylabel, invert_y) list in the notebook's order."""
    mag_cols = sorted((c for c in master.columns if c.startswith('mag_wmean_')),
                      key=lambda c: _wl(c.replace('mag_wmean_', '')))
    bands = [c.replace('mag_wmean_', '') for c in mag_cols]
    panels = []
    panels.append(('VPD (xmatch/v2)', master['pm_ra'], master['pm_dec'],
                   'pmra [mas/yr]', 'pmdec [mas/yr]', False))

    def _n_joint(*cols):
        m = np.ones(len(master), bool)
        for c in cols:
            m &= np.isfinite(master[c].to_numpy(float))
        return int(m.sum())

    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            b, r = bands[i], bands[j]
            if b.split('/')[0] == r.split('/')[0]:
                continue
            n = _n_joint(f'mag_wmean_{b}', f'mag_wmean_{r}')
            if n < MIN_PAIR:
                continue
            panels.append((f'{b} − {r} CMD  [n={n}]',
                           master[f'mag_wmean_{b}'] - master[f'mag_wmean_{r}'],
                           master[f'mag_wmean_{r}'], f'{b} − {r}', r, True))
    if len(bands) >= 3:
        for b1, b2, b3 in combinations(bands, 3):
            if len({b.split('/')[0] for b in (b1, b2, b3)}) < 3:
                continue
            n = _n_joint(f'mag_wmean_{b1}', f'mag_wmean_{b2}', f'mag_wmean_{b3}')
            if n < MIN_PAIR:
                continue
            panels.append((f'({b1}−{b2}) vs ({b2}−{b3})  [n={n}]',
                           master[f'mag_wmean_{b1}'] - master[f'mag_wmean_{b2}'],
                           master[f'mag_wmean_{b2}'] - master[f'mag_wmean_{b3}'],
                           f'{b1} − {b2}', f'{b2} − {b3}', False))
    _G_PIVOT = 640
    _by_filter = {}
    for _b in bands:
        _f = _b.split('/')[0]
        _n = _n_joint('gmag', f'mag_wmean_{_b}') if 'gmag' in master.columns else 0
        if _n > _by_filter.get(_f, (None, -1))[1]:
            _by_filter[_f] = (_b, _n)
    for _f, (_b, _n) in sorted(_by_filter.items(), key=lambda kv: _wl(kv[1][0])):
        if _n < MIN_PAIR:
            continue
        _m = master[f'mag_wmean_{_b}']
        if _wl(_b) < _G_PIVOT:
            _x, _xl = _m - master['gmag'], f'{_f} − G'
        else:
            _x, _xl = master['gmag'] - _m, f'G − {_f}'
        panels.append((f'G vs {_xl}  [n={_n}]', _x, master['gmag'], _xl, 'G', True))
    if {'gmag', 'bp_rp'}.issubset(master.columns) and np.isfinite(master['bp_rp']).sum() >= 10:
        panels.append(('Gaia CMD (Gaia-matched)', master['bp_rp'], master['gmag'], 'BP − RP', 'G', True))
    if {'G_any', 'plx_any'}.issubset(master.columns) and _n_joint('G_any', 'plx_any') >= MIN_PAIR:
        panels.append(('Parallax vs G (G_est for HST-only)', master['G_any'], master['plx_any'],
                       'G / G_est', 'parallax [mas]', False))
    if {'gmag', 'rpmag'}.issubset(master.columns):
        for _f, (_b, _n0) in sorted(_by_filter.items(), key=lambda kv: _wl(kv[1][0])):
            _n = _n_joint('gmag', 'rpmag', f'mag_wmean_{_b}')
            if _n < MIN_PAIR:
                continue
            _m = master[f'mag_wmean_{_b}']
            _y, _yl = ((_m - master['gmag'], f'{_f} − G') if _wl(_b) < _G_PIVOT
                       else (master['gmag'] - _m, f'G − {_f}'))
            panels.append((f'({_yl}) vs (G − RP)  [n={_n}]', master['gmag'] - master['rpmag'], _y,
                           'G − RP', _yl, False))

    def _panel_rank(p):
        t = p[0]
        if t.startswith('VPD'):                 return 0
        if t.startswith('Gaia CMD'):            return 1
        if t.startswith('Parallax'):            return 2
        if t.startswith('G vs '):               return 3
        if ' CMD' in t:                         return 4
        if 'vs (G − RP)' in t:                  return 5
        return 6
    return [p for _, p in sorted(enumerate(panels), key=lambda ip: (_panel_rank(ip[1]), ip[0]))]


def draw(master, panels, sel_mask, field_name, title_tag, out_png, stamp):
    """Notebook-08 cell 4 summary figure (Agg canvas, one layout pass)."""
    from matplotlib.figure import Figure as _Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as _Agg
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig = _Figure(figsize=(4.6 * ncol, 4.0 * nrow))
    _Agg(fig)
    axes = np.atleast_1d(fig.subplots(nrow, ncol)).ravel()
    for ax, (title, x, y, xl, yl, inv) in zip(axes, panels):
        xv = np.asarray(x, float); yv = np.asarray(y, float)
        fin = np.isfinite(xv) & np.isfinite(yv)
        ax.scatter(xv[fin & ~sel_mask], yv[fin & ~sel_mask], s=2, c='0.8', lw=0, label='not selected')
        ax.scatter(xv[fin & sel_mask], yv[fin & sel_mask], s=5, c='crimson', lw=0, label='selected member')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f'{title}  ({int((fin & sel_mask).sum())} sel)', fontsize=10)
        if inv:
            ax.invert_yaxis()
        xs = xv[fin & sel_mask]; ys = yv[fin & sel_mask]
        if len(xs) >= 2:
            def _lim(v):
                lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
                pad = 0.15 * max(hi - lo, 1e-3)
                return lo - pad, hi + pad
            ax.set_xlim(*_lim(xs))
            y0, y1 = _lim(ys)
            ax.set_ylim((y1, y0) if inv else (y0, y1))
        elif title.startswith('VPD'):
            ax.set_xlim(-VPD_ZOOM, VPD_ZOOM); ax.set_ylim(-VPD_ZOOM, VPD_ZOOM)
    for ax in axes[len(panels):]:
        fig.delaxes(ax)
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle(f'{field_name} — {title_tag}: {int(sel_mask.sum())} of {len(master)} shown sources', y=0.995)
    fig.text(0.995, 0.005, stamp, ha='right', va='bottom', fontsize=7, color='0.5')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor='white', transparent=False)
    return out_png


def plot_popfit_members(output_dir, field_name, popfit_dir='BP3M_pop_fit_v2_results'):
    field_dir = Path(output_dir).expanduser().resolve() / field_name
    pf = Path(popfit_dir) if Path(popfit_dir).is_absolute() else field_dir / popfit_dir
    sa = pd.read_csv(pf / 'stellar_astrometry.csv', usecols=lambda c: c in ('source_index', 'is_member'))
    members = set(sa.loc[sa['is_member'].astype(bool), 'source_index'].astype(int))
    master = build_master(field_dir)
    panels = build_panels(master)
    sel_mask = master['source_index'].astype(int).isin(members).to_numpy()
    out = pf / 'plots' / 'member_selection_nb08style.png'
    draw(master, panels, sel_mask, field_name, 'pop-fit v2 final membership', out,
         f'notebook-08 panels {NB_VERSION} · pop-fit members')
    print(f'Saved {out}  ({int(sel_mask.sum())} of {len(members)} members shown among {len(master)} sources, '
          f'{len(panels)} panels)')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('field'); ap.add_argument('--popfit_dir', default='BP3M_pop_fit_v2_results')
    ap.add_argument('--output_dir', default='.')
    a = ap.parse_args(argv)
    plot_popfit_members(a.output_dir, a.field, a.popfit_dir)


if __name__ == '__main__':
    main()
