"""Lindegren et al. (2021) Gaia DR3 parallax zero point for BP3M's Gaia tables (2026-10-28; user: the zero point must be
applied at every level -- a 0.3 mpx position bias at the HST epochs is not negligible next to the GDC uncertainties).

BP3M applies it through a CORRECTED COPY of each field Gaia CSV, Gaia/corrected/<stem>_gaiacorr_l21.csv, used through the
Gaia CSV override (bp3m.gaia_model), so the indv / v1 fits, the v2 cross-match and loader, the pop fits, the GDC header-frame
labels and the QSO vetting all read the same corrected parallaxes.  The copy keeps every row and column of the original and
adds: parallax_orig (published), zp_l21 (mas, subtracted: parallax = parallax_orig - zp_l21), zp_l21_extrap (True where an
input was outside the calibrated range and was clamped to it), gaia_model.  A sidecar <copy>.json records the source file and
its md5, the package version, the clamping policy and the date.  2p sources (no parallax) are left untouched (zp NaN).

Calibrated range (L21; gaiadr3_zeropoint): G 6-21; 5p nu_eff 1.1-1.9 um^-1; 6p pseudocolour 1.24-1.72 um^-1.  Inputs outside
are CLAMPED to the edge and flagged (zp_l21_extrap); the flag lets a later test treat them separately.

Uses the official package gaiadr3_zeropoint (import name zero_point; a bp3m dependency in pyproject.toml / environment.yml,
LGPLv3+, used as a library, not vendored).  Also runs as a script:
    python bp3m/gaia_zp.py SRC_gaia.csv [SRC2_gaia.csv ...]      -> writes the copies next to the sources (Gaia/corrected/)
"""
from __future__ import annotations
import datetime, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

TAG = 'l21'
G_RANGE = (6.0, 21.0); NU_RANGE = (1.1, 1.9); PC_RANGE = (1.24, 1.72)


def copy_path(src) -> Path:
    src = Path(src)
    stem = src.name[:-len('_gaia.csv')] if src.name.endswith('_gaia.csv') else src.stem
    return src.parent / 'corrected' / f'{stem}_gaiacorr_{TAG}.csv'


def _md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def compute_zp(df: pd.DataFrame):
    """(zp mas, extrapolated flag) per row; NaN where the source has no parallax (2p) or no G."""
    from zero_point import zpt                      # official L21 implementation (gaiadr3_zeropoint)
    zpt.load_tables()
    n = len(df)
    aps = df['astrometric_params_solved'].to_numpy(float) if 'astrometric_params_solved' in df.columns else np.full(n, np.nan)
    if 'astrometric_params_solved' not in df.columns:                                   # fall back to the NaN pattern
        aps = np.where(np.isfinite(df['pseudocolour'].to_numpy(float)), 95,
                       np.where(np.isfinite(df['parallax'].to_numpy(float)), 31, 3)).astype(float)
    g = df['gmag'].to_numpy(float); nu = df['nu_eff_used_in_astrometry'].to_numpy(float)
    pc = df['pseudocolour'].to_numpy(float); bet = df['ecl_lat'].to_numpy(float)
    has = np.isfinite(df['parallax'].to_numpy(float)) & np.isfinite(g) & np.isin(aps, (31, 95)) & np.isfinite(bet)
    six = aps == 95
    gc = np.clip(g, *G_RANGE); nuc = np.clip(nu, *NU_RANGE); pcc = np.clip(pc, *PC_RANGE)
    extrap = has & ((gc != g) | (~six & (nuc != nu)) | (six & (pcc != pc)))
    zp = np.full(n, np.nan)
    if has.any():
        zp[has] = zpt.get_zpt(gc[has], np.where(six, np.nan, nuc)[has], np.where(six, pcc, np.nan)[has], bet[has],
                              aps[has].astype(int), _warnings=False)
    return zp, extrap


def build_copy(src, force: bool = False) -> Path:
    """Write (or reuse, when its sidecar matches the source md5) the L21-corrected copy of one Gaia CSV; return its path."""
    src = Path(src); out = copy_path(src); side = Path(str(out) + '.json'); m = _md5(src)
    if out.exists() and side.exists() and not force:
        meta = json.loads(side.read_text())
        if meta.get('source_md5') == m and meta.get('zp') == TAG:
            return out
    df = pd.read_csv(src, dtype={'source_id': np.int64})
    if 'zp_l21' in df.columns or 'parallax_orig' in df.columns:
        raise ValueError(f'{src}: already carries a corrected parallax (zp_l21 / parallax_orig); refusing to correct twice')
    zp, extrap = compute_zp(df)
    df['parallax_orig'] = df['parallax']
    df['zp_l21'] = zp
    df['zp_l21_extrap'] = extrap
    df['parallax'] = np.where(np.isfinite(zp), df['parallax'] - np.nan_to_num(zp), df['parallax'])
    df['gaia_model'] = f'zp={TAG}'
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_suffix('.csv.tmp'); df.to_csv(tmp, index=False); tmp.replace(out)
    try:
        from importlib.metadata import version
        pv = version('gaiadr3_zeropoint')
    except Exception:
        pv = 'unknown'
    side.write_text(json.dumps({'zp': TAG, 'source': str(src), 'source_md5': m, 'copy_md5': _md5(out),
                                'package': f'gaiadr3_zeropoint {pv}', 'clamp': {'G': G_RANGE, 'nu_eff': NU_RANGE, 'pseudocolour': PC_RANGE},
                                'n_rows': int(len(df)), 'n_corrected': int(np.isfinite(zp).sum()), 'n_extrapolated': int(extrap.sum()),
                                'zp_median_mas': float(np.nanmedian(zp)) if np.isfinite(zp).any() else None,
                                'created': datetime.datetime.now().isoformat(timespec='seconds')}, indent=1))
    return out


def is_corrected(path) -> bool:
    """True if a Gaia CSV carries the L21-corrected parallax: this module's copy (sidecar zp = l21), a model copy whose
    sidecar lists 'l21_zp' in its corrections (pop_fit v3real2p / corrvals, 2026-10-29), or the zp_l21 column."""
    side = Path(str(path) + '.json')
    if side.exists():
        meta = json.loads(side.read_text())
        if 'corrections' in meta:
            if ('l21_zp' in meta['corrections']) != ('l21_zp' in meta.get('corrections_qso_rows', meta['corrections'])):
                raise SystemExit(f'{path}: L21 applied to star rows but not QSO rows (or the reverse): {meta["corrections"]} / '
                                 f'{meta.get("corrections_qso_rows")}')
            return 'l21_zp' in meta['corrections']
        return meta.get('zp') == TAG
    cols = pd.read_csv(path, nrows=0).columns
    return 'zp_l21' in cols


if __name__ == '__main__':
    for s in sys.argv[1:]:
        p = build_copy(s)
        print(p, json.loads(Path(str(p) + '.json').read_text())['n_corrected'], 'corrected')
