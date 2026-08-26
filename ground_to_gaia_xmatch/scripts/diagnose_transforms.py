#!/usr/bin/env python
"""
Diagnose alignment transform solutions, optionally against a reference run.

Answers two kinds of question:

  1. Are the fitted transforms consistent with the ideal?
     rotation -> 0 deg, scale -> 1, on/off skew -> 0.

  2. Where a reference tree exists, do the new solutions agree with it, and
     does any disagreement correlate with an exposure-level property?
     The z-score (new - old)/sigma_new is the test: a well-behaved parameter
     has median z ~ 0 and essentially no |z| > 3.

The second question is why this script exists.  A discrepancy that shows up in
one parameter only (observed: D, the eta-scale, at ~5% |z|>3 while A/B/C sit at
0.0%) is not noise, and the way to find its cause is to regress it against every
exposure-level property available rather than guess one at a time.

Properties tested
-----------------
    ra0, dec0, cos(dec0)          pointing
    altitude, airmass             computed from ra0/dec0/mjd at the observatory
    mjd, decimal year             epoch
    iq_arcsec                     seeing
    maglim                        depth
    plate_scale_mas_px            plate scale
    rotation_deg                  instrument rotation
    n_stars, n_stars_alignment    star counts
    rms_xi_uas, rms_eta_uas       per-detector WCS residuals
    alpha_applied, n_alpha_ref    error-inflation diagnostics
    |delta_ra0_mas|, |delta_dec0_mas|   tangent-point excursion

Usage
-----
    # CFHT, against the pre-refactor bp3m_results tree
    python -m ground_to_gaia_xmatch.scripts.diagnose_transforms \
        --instrument cfht \
        --field-root /path/to/CFHT/UNIONS \
        --reference-root /path/to/CFHT/UNIONS/bp3m_results \
        --out diagnostics.csv

    # LSST, no reference: ideal-consistency and property regressions only
    python -m ground_to_gaia_xmatch.scripts.diagnose_transforms \
        --instrument lsst --field-root /path/to/Fornax_dSph

Re-run it as more exposures finish; it auto-discovers whatever is on disk.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .. import layout

warnings.filterwarnings('ignore')

PARAMS = ['A', 'B', 'C', 'D']
IDEALS = [('rotation_deg', 0.0), ('scale', 1.0),
          ('on_skew', 0.0), ('off_skew', 0.0)]

# Observatory sites for the airmass calculation.
SITES = {
    'cfht': dict(lat=19.82525, lon=-155.46886, height=4204.0),   # Mauna Kea
    'lsst': dict(lat=-30.24463, lon=-70.74942, height=2647.0),   # Cerro Pachon
}


# ── discovery ────────────────────────────────────────────────────────────────

def find_new(field_root: Path):
    """Every per-image alignment result under the canonical layout."""
    out = []
    root = layout.align_root(field_root)
    if not root.is_dir():
        return out
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith('joint_'):
            continue
        for det_dir in sorted(exp_dir.iterdir()):
            f = det_dir / layout.IMAGE_TRANSFORM_CSV
            if f.exists():
                out.append((exp_dir.name, det_dir.name, f))
    return out


def reference_path(ref_root: Path, exp_id: str, det_token: str) -> Path | None:
    """
    Locate the matching reference file.

    The pre-refactor CFHT tree used bare zero-padded detector numbers
    (bp3m_results/cfht_2159742/05/) rather than det_05, so both are tried.
    """
    if ref_root is None:
        return None
    num = det_token.replace('det_', '').split('_')[0]
    for cand in (ref_root / exp_id / det_token / 'image_transformations.csv',
                 ref_root / exp_id / num / 'image_transformations.csv'):
        if cand.exists():
            return cand
    return None


# ── assembly ─────────────────────────────────────────────────────────────────

def exposure_properties(field_root: Path, instrument: str) -> pd.DataFrame:
    """Per-(exposure, detector) metadata, where the instrument provides it."""
    if instrument == 'cfht':
        f = Path(field_root) / 'cfht_unions_detectors.csv'
        if f.exists():
            d = pd.read_csv(f)
            d['exp_id'] = 'cfht_' + d['expnum'].astype(str)
            d['det_num'] = d['ext'].astype(int)
            return d
    if instrument == 'lsst':
        try:
            from astropy.io import ascii as ascii_io
            hits = sorted(Path(field_root).glob('table_dp2.*-VisitDetector.tbl'))
            if hits:
                d = ascii_io.read(hits[0], format='ipac').to_pandas()
                d['exp_id'] = 'lsst_' + d['visitId'].astype(str)
                d['det_num'] = d['detector'].astype(int)
                d = d.rename(columns={'expMidptMJD': 'mjd', 'ra': 'ra0',
                                      'dec': 'dec0', 'pixelScale': 'plate_scale_arcsec'})
                d['plate_scale_mas_px'] = d['plate_scale_arcsec'] * 1000.0
                if 'seeing' in d.columns:
                    d['iq_arcsec'] = d['seeing']
                return d
        except Exception:
            pass
    return pd.DataFrame()


def add_airmass(t: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Altitude and airmass from pointing + epoch at the observatory."""
    site = SITES.get(instrument)
    if site is None or not {'ra0', 'dec0', 'mjd'} <= set(t.columns):
        t['alt_deg'] = np.nan
        t['airmass'] = np.nan
        return t
    try:
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord
        from astropy.time import Time
        import astropy.units as u
        loc = EarthLocation(lat=site['lat'] * u.deg, lon=site['lon'] * u.deg,
                            height=site['height'] * u.m)
        ok = np.isfinite(t['ra0']) & np.isfinite(t['dec0']) & np.isfinite(t['mjd'])
        alt = np.full(len(t), np.nan)
        if ok.any():
            sc = SkyCoord(ra=t.loc[ok, 'ra0'].values * u.deg,
                          dec=t.loc[ok, 'dec0'].values * u.deg)
            aa = sc.transform_to(AltAz(obstime=Time(t.loc[ok, 'mjd'].values, format='mjd'),
                                       location=loc))
            alt[ok.values] = aa.alt.deg
        t['alt_deg'] = alt
        # sec(z); NaN below the horizon rather than a spurious huge value
        t['airmass'] = np.where(alt > 3.0, 1.0 / np.sin(np.radians(alt)), np.nan)
    except Exception as e:
        print(f'  (airmass unavailable: {e})')
        t['alt_deg'] = np.nan
        t['airmass'] = np.nan
    return t


def build_table(field_root: Path, ref_root: Path | None, instrument: str,
                exposures=None) -> pd.DataFrame:
    """One row per detector: new solution, reference solution, z-scores, properties."""
    rows = []
    for exp_id, det_token, new_f in find_new(field_root):
        if exposures and not any(str(e) in exp_id for e in exposures):
            continue
        n = pd.read_csv(new_f).iloc[0]
        rec = {'exp_id': exp_id, 'det_token': det_token,
               'det_num': int(det_token.replace('det_', '').split('_')[0])}
        for k in ('ra0', 'dec0', 'mjd', 'n_stars_alignment', 'alpha_applied',
                  'alpha_raw', 'n_alpha_ref', 'saturated_alpha',
                  'delta_ra0_mas', 'delta_dec0_mas', 'pixel_scale_mas'):
            rec[k] = pd.to_numeric(n.get(k, np.nan), errors='coerce')
        for k, _ in IDEALS:
            rec[k + '_new'] = pd.to_numeric(n.get(k, np.nan), errors='coerce')
        for k in PARAMS:
            rec[k + '_new'] = pd.to_numeric(n.get(k, np.nan), errors='coerce')
            rec['sigma_' + k] = pd.to_numeric(n.get('sigma_' + k, np.nan), errors='coerce')

        ref_f = reference_path(ref_root, exp_id, det_token)
        if ref_f is not None:
            o = pd.read_csv(ref_f).iloc[0]
            for k, _ in IDEALS:
                rec[k + '_old'] = pd.to_numeric(o.get(k, np.nan), errors='coerce')
            for k in PARAMS:
                rec[k + '_old'] = pd.to_numeric(o.get(k, np.nan), errors='coerce')
                s = rec['sigma_' + k]
                rec['z_' + k] = ((rec[k + '_new'] - rec[k + '_old']) / s
                                 if s and np.isfinite(s) and s > 0 else np.nan)
            rec['n_stars_alignment_old'] = pd.to_numeric(
                o.get('n_stars_alignment', np.nan), errors='coerce')
        rows.append(rec)

    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t = t.replace([np.inf, -np.inf], np.nan)

    props = exposure_properties(field_root, instrument)
    if not props.empty:
        keep = [c for c in ('exp_id', 'det_num', 'iq_arcsec', 'maglim',
                            'plate_scale_mas_px', 'rotation_deg', 'n_stars',
                            'rms_xi_uas', 'rms_eta_uas', 'mjd', 'ra0', 'dec0')
                if c in props.columns]
        p = props[keep].rename(columns={c: c + '_meta' for c in keep
                                        if c not in ('exp_id', 'det_num')})
        t = t.merge(p, on=['exp_id', 'det_num'], how='left')
        for c in ('mjd', 'ra0', 'dec0'):
            if c + '_meta' in t.columns:
                t[c] = t[c].fillna(t[c + '_meta'])

    t['cos_dec0'] = np.cos(np.radians(t['dec0']))
    t['year'] = 2000.0 + (t['mjd'] - 51544.5) / 365.25
    t = add_airmass(t, instrument)
    return t


# ── reporting ────────────────────────────────────────────────────────────────

def _f(x, w=11, p=3):
    return 'nan'.rjust(w) if not np.isfinite(x) else f'{x:{w}.{p}e}'


def report_ideal(t: pd.DataFrame, has_ref: bool):
    print('\n' + '=' * 78)
    print('CONSISTENCY WITH THE IDEAL TRANSFORM  (rotation->0, scale->1, skews->0)')
    print('=' * 78)
    hdr = f"{'quantity':>18} {'median':>12} {'med|dev|':>12} {'p95|dev|':>12} {'stdev':>12}"
    print(hdr)
    for k, ref in IDEALS:
        for tag in (['_old', '_new'] if has_ref else ['_new']):
            col = k + tag
            if col not in t.columns:
                continue
            v = t[col].dropna()
            if v.empty:
                continue
            dev = np.abs(v - ref)
            print(f"{(tag[1:].upper()+' '+k):>18} {_f(np.median(v),12,4)} "
                  f"{_f(np.median(dev),12)} {_f(np.percentile(dev,95),12)} {_f(np.std(v),12)}")
        print()
    if has_ref:
        print('  ratio new/old of median |deviation from ideal|  (<1 = closer to ideal now)')
        for k, ref in IDEALS:
            if k + '_old' not in t.columns:
                continue
            do = np.median(np.abs(t[k + '_old'].dropna() - ref))
            dn = np.median(np.abs(t[k + '_new'].dropna() - ref))
            if do > 0:
                flag = '' if dn <= do else '   <-- worse'
                print(f'    {k:>14}: {dn/do:6.3f}{flag}')


def report_agreement(t: pd.DataFrame):
    print('\n' + '=' * 78)
    print('AGREEMENT WITH REFERENCE   z = (new - old) / sigma_new')
    print('=' * 78)
    print(f"{'param':>6} {'median z':>10} {'p95|z|':>8} {'frac|z|>3':>10} {'n':>6}")
    for k in PARAMS:
        z = t.get('z_' + k)
        if z is None:
            continue
        z = z.dropna()
        if z.empty:
            continue
        print(f'{k:>6} {np.median(z):+10.3f} {np.percentile(np.abs(z),95):8.2f} '
              f'{100*np.mean(np.abs(z)>3):9.1f}% {len(z):6d}')
    if 'n_stars_alignment_old' in t.columns:
        o = t['n_stars_alignment_old'].dropna()
        n = t['n_stars_alignment'].dropna()
        if len(o) and len(n):
            print(f'\n  stars per solve: {np.median(o):.0f} -> {np.median(n):.0f} '
                  f'({100*(np.median(n)/max(np.median(o),1)-1):+.0f}%)')


PROPERTIES = [
    ('cos_dec0', 'cos(dec0)'), ('dec0', 'dec0'), ('ra0', 'ra0'),
    ('airmass', 'airmass'), ('alt_deg', 'altitude'),
    ('year', 'epoch (yr)'), ('iq_arcsec_meta', 'seeing (IQ)'),
    ('maglim_meta', 'depth (maglim)'),
    ('plate_scale_mas_px_meta', 'plate scale'),
    ('rotation_deg_meta', 'instr rotation'),
    ('n_stars_meta', 'n_stars (WCS)'),
    ('n_stars_alignment', 'n_stars (align)'),
    ('rms_xi_uas_meta', 'WCS rms xi'), ('rms_eta_uas_meta', 'WCS rms eta'),
    ('alpha_applied', 'alpha_applied'), ('n_alpha_ref', 'n_alpha_ref'),
    ('delta_ra0_mas', '|delta_ra0|'), ('delta_dec0_mas', '|delta_dec0|'),
]


def report_regression(t: pd.DataFrame):
    """Rank every available property by how well it explains |z_D|."""
    print('\n' + '=' * 78)
    print('WHAT EXPLAINS THE DISAGREEMENT?  Spearman rho of |z| vs each property')
    print('=' * 78)
    avail = [(c, lbl) for c, lbl in PROPERTIES
             if c in t.columns and t[c].notna().sum() >= 10 and t[c].nunique() > 2]
    if not avail:
        print('  no properties with enough coverage')
        return

    print(f"{'property':>20} " + ''.join(f'{("rho|z_"+k+"|"):>12}' for k in PARAMS))
    ranked = []
    for c, lbl in avail:
        v = t[c].abs() if c.startswith('delta_') else t[c]
        cells, rho_d = [], np.nan
        for k in PARAMS:
            z = t.get('z_' + k)
            if z is None:
                cells.append(f'{"-":>12}')
                continue
            m = np.isfinite(z) & np.isfinite(v)
            if m.sum() < 10:
                cells.append(f'{"-":>12}')
                continue
            r = pd.Series(np.abs(z[m]).values).corr(pd.Series(v[m].values), method='spearman')
            cells.append(f'{r:+12.3f}')
            if k == 'D':
                rho_d = r
        print(f'{lbl:>20} ' + ''.join(cells))
        ranked.append((abs(rho_d) if np.isfinite(rho_d) else -1, lbl, rho_d))

    ranked.sort(reverse=True)
    print('\n  strongest |z_D| associations:')
    for a, lbl, r in ranked[:5]:
        if a >= 0:
            verdict = 'NEGLIGIBLE' if a < 0.15 else ('weak' if a < 0.3 else 'NOTABLE')
            print(f'    {lbl:>20}  rho = {r:+.3f}   {verdict}')


def report_per_exposure(t: pd.DataFrame):
    """Per-exposure view: the pattern so far is exposure-level, not detector-level."""
    if 'z_D' not in t.columns:
        return
    print('\n' + '=' * 78)
    print('PER-EXPOSURE BREAKDOWN  (sorted by D outlier fraction)')
    print('=' * 78)
    rows = []
    for exp, x in t.groupby('exp_id'):
        z = x['z_D'].dropna()
        if z.empty:
            continue
        rows.append(dict(
            exp=exp, n=len(z), fracD=100 * np.mean(np.abs(z) > 3),
            p95=np.percentile(np.abs(z), 95),
            ra0=x['ra0'].median(), dec0=x['dec0'].median(),
            airmass=x['airmass'].median(),
            iq=x.get('iq_arcsec_meta', pd.Series([np.nan])).median(),
            year=x['year'].median(),
            nstar=x['n_stars_alignment'].median()))
    if not rows:
        return
    d = pd.DataFrame(rows).sort_values('fracD', ascending=False)
    print(f"{'exposure':>16} {'n':>4} {'fracD>3':>9} {'p95|zD|':>8} {'ra0':>8} "
          f"{'dec0':>8} {'airmass':>8} {'IQ':>6} {'year':>8} {'nstar':>7}")
    for _, r in d.iterrows():
        print(f"{r.exp:>16} {int(r.n):4d} {r.fracD:8.1f}% {r.p95:8.2f} {r.ra0:8.2f} "
              f"{r.dec0:+8.2f} {r.airmass:8.3f} {r.iq:6.2f} {r.year:8.2f} {r.nstar:7.0f}")

    # Is the effect exposure-level or detector-level?  If exposure-level, the
    # between-exposure spread of fracD greatly exceeds binomial expectation.
    p = d.fracD.mean() / 100.0
    n = d.n.median()
    if 0 < p < 1 and n > 0:
        exp_sd = 100 * np.sqrt(p * (1 - p) / n)
        print(f'\n  fracD spread: observed sd {d.fracD.std():.1f}%  vs  '
              f'binomial {exp_sd:.1f}% if purely per-detector')
        if d.fracD.std() > 2 * exp_sd:
            print('  -> clustered BY EXPOSURE: cause is an exposure-level property')
        else:
            print('  -> consistent with per-detector scatter')


def report_calibration(field_root: Path, t: pd.DataFrame):
    """
    Are the source position uncertainties correctly scaled?

    Uses the per-star measurement-only chi2 written to stellar_astrometry.csv.
    A detection with n_det_chi2 == 1 gives a chi2 with 2 dof, so a correctly
    calibrated catalogue has median 2*ln(2) = 1.386 and p84/p50 = 2.644.

    CRITICAL: only detectors with alpha_applied == 1 are usable.  Where alpha
    fired it rescaled C_src precisely so the median chi2 matches expectation,
    making the test circular — a "perfect" 1.386 there measures the inflation
    machinery, not the data.  Detectors are split accordingly.
    """
    from scipy.stats import chi2 as chi2d
    med_th = 2.0 * np.log(2.0)
    ratio_th = chi2d.ppf(0.84, df=2) / chi2d.ppf(0.50, df=2)

    print('\n' + '=' * 78)
    print('SOURCE UNCERTAINTY CALIBRATION  (per-star measurement-only chi2)')
    print('=' * 78)
    print(f'  expected for correctly scaled errors: median {med_th:.3f}, '
          f'p84/p50 {ratio_th:.3f}')

    rows = []
    for _, r in t.iterrows():
        f = (layout.align_root(field_root) / r.exp_id / r.det_token
             / layout.STELLAR_CSV)
        if not f.exists():
            continue
        try:
            g = pd.read_csv(f, usecols=['chi2_xmatch', 'n_det_chi2', 'pmra'])
        except (ValueError, OSError):
            continue
        # Informative-prior stars only.  Diffuse-prior (Gaia 2p) stars have
        # their residual absorbed by their own free position parameters, giving
        # chi2 ~1e-7; where they are the majority they own the median and drive
        # it to zero.  Same failure that disabled the alpha estimator.
        informative = np.isfinite(pd.to_numeric(g['pmra'], errors='coerce'))
        one = g.loc[informative & (g['n_det_chi2'] == 1), 'chi2_xmatch'].dropna()
        one = one[one > 0]
        if len(one) < 30:
            continue
        q = np.percentile(one, [50, 84])
        rows.append(dict(exp_id=r.exp_id, det=r.det_token, n=len(one),
                         med=q[0], ratio=q[1] / q[0] if q[0] > 0 else np.nan,
                         alpha=r.get('alpha_applied', np.nan)))
    if not rows:
        print('  no per-star chi2 available (re-run the alignment to populate it)')
        return
    d = pd.DataFrame(rows)
    clean = d[d.alpha <= 1.001]
    forced = d[d.alpha > 1.001]

    print(f"\n  {'sample':>34} {'n_det':>7} {'med chi2':>10} {'p84/p50':>9}")
    for lbl, x in (('UNCONTAMINATED (alpha == 1)', clean),
                   ('alpha fired - CIRCULAR, ignore', forced)):
        if len(x):
            print(f'  {lbl:>34} {len(x):7d} {np.median(x.med):10.3f} '
                  f'{np.median(x.ratio):9.3f}')
    if len(clean) < 5:
        print('\n  WARNING: too few uncontaminated detectors to conclude anything.')
        return
    dev = np.median(clean.med) / med_th
    print(f'\n  uncontaminated median chi2 is {dev:.2f}x expectation')
    if dev < 0.8:
        print('  -> uncertainties OVER-estimated.  alpha cannot correct this: it is')
        print('     floored at 1.0 and only inflates, so the error stays uncorrected.')
    elif dev > 1.25:
        print('  -> uncertainties UNDER-estimated (alpha should be firing here).')
    else:
        print('  -> consistent with correctly scaled uncertainties.')
    print(f'  spread across detectors: {clean.med.min():.3f} to {clean.med.max():.3f}'
          f'  (a well-behaved catalogue is tight around {med_th:.3f})')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--instrument', required=True, choices=['lsst', 'cfht'])
    p.add_argument('--field-root', required=True, type=Path)
    p.add_argument('--reference-root', type=Path, default=None,
                   help='Tree of reference image_transformations.csv to compare against')
    p.add_argument('--exposure', nargs='+', default=None,
                   help='Restrict to these exposure ids')
    p.add_argument('--out', type=Path, default=None, help='Write the full table to CSV')
    args = p.parse_args(argv)

    t = build_table(args.field_root, args.reference_root, args.instrument, args.exposure)
    if t.empty:
        raise SystemExit('No alignment results found under '
                         f'{layout.align_root(args.field_root)}')

    has_ref = 'z_D' in t.columns and t['z_D'].notna().any()
    print(f'{len(t)} detectors over {t.exp_id.nunique()} exposures')
    if {'dec0', 'ra0'} <= set(t.columns):
        print(f'  ra0 {t.ra0.min():.1f} to {t.ra0.max():.1f}   '
              f'dec0 {t.dec0.min():+.1f} to {t.dec0.max():+.1f}   '
              f'cos(dec0) {t.cos_dec0.min():.3f} to {t.cos_dec0.max():.3f}')
    if t.airmass.notna().any():
        print(f'  airmass {t.airmass.min():.3f} to {t.airmass.max():.3f}')
    print(f'  reference comparison: {"YES" if has_ref else "NO (ideal-consistency only)"}')

    report_calibration(args.field_root, t)
    report_ideal(t, has_ref)
    if has_ref:
        report_agreement(t)
        report_per_exposure(t)
        report_regression(t)

    if args.out:
        t.to_csv(args.out, index=False)
        print(f'\nfull table -> {args.out}  ({len(t)} rows, {len(t.columns)} columns)')
    return t


if __name__ == '__main__':
    main()
