"""Global per-image photometric zero points, one band per CAMERA/FILTER (user 2026-10-06).

The cross-match validator ties every image to one reference image per filter, and merges cameras
(ACS/WFC F606W and WFC3/UVIS F606W share one 'F606W' column).  In Sculptor and Draco that left
F606W in several never-tied groups, offset from each other by up to 0.15 mag, which shows up as
copies of the isochrone.  This module solves instead, for each band b = DETECTOR/FILTER separately
(no colour terms, so single-filter fields work identically):

    m_{s,i} = M_s + z_i + e_{s,i},     m = catalogue mag_st_gdc (header STMAG + GDC pixel-area term)

over ALL stars shared between ANY images of the band (alternating weighted estimates with
iterative sigma clipping, i.e. a robust least-squares network / "ubercal").  Images are grouped
into connected components of the shared-star graph; each component is pinned to the header
STMAG scale (weighted-median z = 0 over its images), so components are never silently tied
through a single star.  When a band splits into several components, they can optionally be
bridged by Gaia: the median of (G - M_s) at matched BP-RP is compared between components of the
SAME band (no cross-band colour term), and reported (and applied with --gaia_bridge).

Outputs in <field>/hst_xmatch/:
    photometric_zp.csv    band, image, zp, zp_err, n_stars, component, rms, old_cross_image_zp
    photometric_mags.csv  source_index, mag_<band>, magerr_<band>, n_<band>  (per camera/filter)

    python -m bp3m.pipeline.photometric_zp FIELD [--output_dir .] [--gaia_bridge] [--min_link 5]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MAG_ERR_MAX = 0.05      # detections used in the solve
N_ITER = 30
CLIP = 4.0
SYS_FLOOR = 0.005       # mag, added in quadrature to catalogue errors


def _band_of(flc: Path) -> tuple[str, str]:
    from astropy.io import fits
    h = fits.getheader(flc, 0)
    det = str(h.get('DETECTOR', '')).strip().upper()
    filt = next((str(h.get(k, '')).strip() for k in ('FILTER', 'FILTER1', 'FILTER2')
                 if str(h.get(k, '')).strip().upper().startswith('F')), '?')
    return f'{det}/{filt}', str(h.get('INSTRUME', '')).strip()


def load_detections(field_dir: Path) -> pd.DataFrame:
    """One row per (source_index, image, catalogue row) from master_combined_v2 hst_indices_*."""
    from astropy.io import fits
    mc = pd.read_csv(field_dir / 'hst_xmatch' / 'master_combined_v2.csv', low_memory=False,
                     usecols=lambda c: c.startswith('hst_indices_'))
    rows = []
    for c in mc.columns:
        for si, v in mc[c].items():
            if isinstance(v, str):
                for tok in v.split(','):
                    if ':' in tok:
                        ob, k = tok.rsplit(':', 1)
                        rows.append((int(si), ob[:9], int(k)))
    det = pd.DataFrame(rows, columns=['source_index', 'image', 'idx']).drop_duplicates(['image', 'idx'])
    hst = field_dir / 'HST' / 'mastDownload' / 'HST'
    parts, bands = [], {}
    for im, g in det.groupby('image'):
        cat = hst / im / f'{im}_flc_catalog.fits'
        flc = hst / im / f'{im}_flc.fits'
        if not cat.exists() or not flc.exists():
            continue
        bands[im] = _band_of(flc)
        t = fits.getdata(cat, 1)
        k = g.idx.to_numpy()
        ok = k < len(t)
        g = g[ok]; k = k[ok]
        cols = t.columns.names
        parts.append(pd.DataFrame({
            'source_index': g.source_index.to_numpy(), 'image': im, 'idx': k,
            'mag': np.asarray(t['mag_st_gdc'], float)[k],
            'magerr': np.asarray(t['mag_err_gdc'] if 'mag_err_gdc' in cols else t['mag_err'], float)[k],
            'n_sat': np.asarray(t['n_sat'], float)[k] if 'n_sat' in cols else 0.0,
            'chip': np.asarray(t['chip_ext'], int)[k] if 'chip_ext' in cols else 1,
        }))
    d = pd.concat(parts, ignore_index=True)
    d['band'] = d.image.map(lambda i: bands[i][0])
    return d


def _components(d: pd.DataFrame, min_link: int) -> dict:
    """Connected components of the image graph; an edge needs >= min_link shared stars."""
    by_star = d.groupby('source_index').image.apply(lambda s: sorted(set(s)))
    pair = defaultdict(int)
    for ims in by_star:
        for a in range(len(ims)):
            for b in range(a + 1, len(ims)):
                pair[(ims[a], ims[b])] += 1
    parent = {im: im for im in d.image.unique()}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for (a, b), n in pair.items():
        if n >= min_link:
            parent[find(a)] = find(b)
    roots = {im: find(im) for im in parent}
    order = pd.Series(roots).value_counts().index        # largest component = 0
    rank = {r: i for i, r in enumerate(order)}
    return {im: rank[r] for im, r in roots.items()}


def solve_band(d: pd.DataFrame, min_link: int = 5):
    """Robust network solve for one band.  Returns (zp table, per-star table)."""
    use = d[(d.magerr > 0) & (d.magerr < MAG_ERR_MAX) & (d.n_sat <= 0) & np.isfinite(d.mag)].copy()
    use = use[use.groupby('source_index').image.transform('nunique') >= 2]
    if not len(use):
        return None, None
    comp = _components(use, min_link)
    use['comp'] = use.image.map(comp)
    use['w'] = 1.0 / (use.magerr ** 2 + SYS_FLOOR ** 2)
    z = pd.Series(0.0, index=sorted(use.image.unique()))
    keep = np.ones(len(use), bool)
    for it in range(N_ITER):
        u = use[keep]
        r = u.mag - u.image.map(z)
        M = (r * u.w).groupby(u.source_index).sum() / u.w.groupby(u.source_index).sum()
        res_im = use.mag - use.source_index.map(M)
        zi = (res_im[keep] * use.w[keep]).groupby(use.image[keep]).sum() / use.w[keep].groupby(use.image[keep]).sum()
        # pin every component to the header scale: weighted median of its zps = 0
        cz = pd.DataFrame({'zp': zi, 'comp': zi.index.map(comp)})
        zi = zi - cz.groupby('comp').zp.transform('median')
        z_new = zi.reindex(z.index).fillna(0.0)
        resid = use.mag - use.source_index.map(M) - use.image.map(z_new)
        sig = np.sqrt(use.magerr ** 2 + SYS_FLOOR ** 2)
        mad = 1.4826 * np.nanmedian(np.abs(resid[keep] / sig[keep]))
        new_keep = (np.abs(resid / sig) < CLIP * max(mad, 1.0)).to_numpy() & np.isfinite(resid.to_numpy())
        dz = float(np.nanmax(np.abs(z_new - z))) if len(z) else 0.0
        z = z_new
        if dz < 1e-5 and (new_keep == keep).all():
            break
        keep = new_keep
    u = use[keep]; u = u.assign(res=u.mag - u.source_index.map(M) - u.image.map(z))
    g = u.groupby('image')
    zt = pd.DataFrame({'zp': z, 'n_stars': g.source_index.nunique(),
                       'rms': g.res.apply(lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))),
                       'component': pd.Series(comp)}).reindex(z.index)
    zt['zp_err'] = zt.rms / np.sqrt(zt.n_stars.clip(lower=1))
    # per-star calibrated magnitude from ALL detections (incl. faint) with the solved zps
    a = d[np.isfinite(d.mag) & (d.magerr > 0)].copy()
    a['mc'] = a.mag - a.image.map(z).fillna(0.0)
    a['w'] = 1.0 / (a.magerr ** 2 + SYS_FLOOR ** 2)
    sg = a.groupby('source_index')
    st = pd.DataFrame({'mag': (a.mc * a.w).groupby(a.source_index).sum() / sg.w.sum(),
                       'magerr': 1.0 / np.sqrt(sg.w.sum()), 'n': sg.size(),
                       'comp': a.image.map(comp).groupby(a.source_index).agg(lambda s: s.mode().iloc[0] if s.notna().any() else -1)})
    return zt.reset_index().rename(columns={'index': 'image'}), st


def gaia_bridge(field_dir: Path, band: str, st: pd.DataFrame) -> pd.DataFrame | None:
    """Offset of each component relative to component 0 from (G - M) at matched BP-RP (same band)."""
    mc = pd.read_csv(field_dir / 'hst_xmatch' / 'master_combined_v2.csv', low_memory=False, usecols=['gaia_source_id'])
    sa_p = field_dir / 'BP3M_v2_results' / 'stellar_astrometry.csv'
    if not sa_p.exists():
        sa_p = field_dir / 'BP3M_results' / 'stellar_astrometry.csv'
    if not sa_p.exists():
        return None
    sa = pd.read_csv(sa_p, usecols=['Gaia_id', 'gmag', 'bp_rp'], dtype={'Gaia_id': np.int64}).drop_duplicates('Gaia_id')
    gid = pd.to_numeric(mc.gaia_source_id, errors='coerce').fillna(0).astype(np.int64)
    s = st.join(pd.Series(gid.values, index=mc.index, name='Gaia_id'), how='left').merge(sa, on='Gaia_id', how='inner')
    s = s[np.isfinite(s.gmag) & np.isfinite(s.bp_rp) & s.bp_rp.between(0, 2.5) & (s.comp >= 0)]
    if s.comp.nunique() < 2 or (s.comp == 0).sum() < 10:
        return None
    ref = s[s.comp == 0]
    p = np.polyfit(ref.bp_rp, ref.gmag - ref.mag, 3)          # same-band relation, used only to compare components
    s = s.assign(res=s.gmag - s.mag - np.polyval(p, s.bp_rp))
    out = s.groupby('comp').res.agg(['size', 'median', lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))])
    out.columns = ['n_gaia', 'offset_vs_comp0', 'scatter']
    out['offset_err'] = out.scatter / np.sqrt(out.n_gaia.clip(lower=1))
    out['band'] = band
    return out.reset_index()


def run(output_dir, field, min_link=5, apply_gaia_bridge=False):
    field_dir = Path(output_dir).expanduser().resolve() / field
    d = load_detections(field_dir)
    old = {}
    zp_old = field_dir / 'magnitude_zp_offsets.csv'
    if zp_old.exists():
        o = pd.read_csv(zp_old); old = dict(zip(o.image, o.cross_image_zp))
    zts, mags, bridges = [], [], []
    for band, db in d.groupby('band'):
        zt, st = solve_band(db, min_link)
        if zt is None:
            continue
        zt['band'] = band
        if st.comp.nunique() > 1:
            br = gaia_bridge(field_dir, band, st)
            if br is not None:
                bridges.append(br)
                if apply_gaia_bridge:
                    off = dict(zip(br['comp'], br['offset_vs_comp0']))
                    zt['zp'] = zt.zp - zt.component.map(off).fillna(0.0)
                    st['mag'] = st.mag + st.comp.map(off).fillna(0.0)
        zt['old_cross_image_zp'] = zt.image.map(old)
        zts.append(zt)
        tag = band.replace('/', '_')
        mags.append(st[['mag', 'magerr', 'n']].rename(columns={'mag': f'mag_{tag}', 'magerr': f'magerr_{tag}', 'n': f'n_{tag}'}))
    zt = pd.concat(zts, ignore_index=True)
    out = field_dir / 'hst_xmatch'
    zt[['band', 'image', 'zp', 'zp_err', 'n_stars', 'rms', 'component', 'old_cross_image_zp']].to_csv(out / 'photometric_zp.csv', index=False)
    pm = pd.concat(mags, axis=1); pm.index.name = 'source_index'; pm.reset_index().to_csv(out / 'photometric_mags.csv', index=False)
    summ = zt.groupby('band').agg(images=('image', 'size'), components=('component', 'nunique'),
                                  zp_spread=('zp', lambda v: float(v.max() - v.min())), median_rms=('rms', 'median'))
    print(f'{field}: per-band photometric network'); print(summ.round(4).to_string())
    if bridges:
        br = pd.concat(bridges, ignore_index=True)
        br.to_csv(out / 'photometric_zp_gaia_bridge.csv', index=False)
        print('components bridged by Gaia (same band, offset relative to the largest component):')
        print(br.round(4).to_string(index=False))
    json.dump({'min_link': min_link, 'gaia_bridge_applied': bool(apply_gaia_bridge), 'mag_err_max': MAG_ERR_MAX,
               'clip': CLIP, 'sys_floor': SYS_FLOOR}, open(out / 'photometric_zp_config.json', 'w'), indent=1)
    return zt, pm


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('field'); ap.add_argument('--output_dir', default='.')
    ap.add_argument('--min_link', type=int, default=5)
    ap.add_argument('--gaia_bridge', action='store_true', help='apply the Gaia component bridge (same band)')
    a = ap.parse_args(argv)
    run(a.output_dir, a.field, a.min_link, a.gaia_bridge)


if __name__ == '__main__':
    main()
