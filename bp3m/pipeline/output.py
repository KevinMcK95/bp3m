"""
Output formatting and summary utilities for the GaiaHub Improved pipeline.

Provides human-readable summaries of pipeline outputs and helpers for
writing supplementary products (e.g. FITS tables, region files).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── Text summaries ────────────────────────────────────────────────────────────

def print_field_summary(output_dir: Path, field_name: str) -> None:
    """
    Print a concise summary of pipeline outputs for a field.
    """
    root = Path(output_dir) / field_name
    bp3m_dir = root / "BP3M_results"
    gaia_csv = root / "Gaia" / f"{field_name}_gaia.csv"

    print(f"\n{'='*55}")
    print(f"Field: {field_name}")
    print(f"{'='*55}")

    # Gaia
    if gaia_csv.exists():
        gaia = pd.read_csv(gaia_csv)
        print(f"\n  Gaia catalogue:   {len(gaia):,} stars")
        if 'gmag' in gaia.columns:
            print(f"    G range:        {gaia['gmag'].min():.1f} – {gaia['gmag'].max():.1f} mag")
    else:
        print("\n  Gaia catalogue:   not found")

    # BP3M results
    stars_csv = bp3m_dir / "stellar_astrometry.csv"
    imgs_csv  = bp3m_dir / "image_transformations.csv"
    if stars_csv.exists():
        stars = pd.read_csv(stars_csv)
        print(f"\n  BP3M results:")
        print(f"    Stars fitted:   {len(stars):,}")
        if 'n_hst_used' in stars.columns:
            med_hst = np.median(stars['n_hst_used'])
            print(f"    Median N_HST:   {med_hst:.0f}")
        if 'sigma_pmra_bp3m' in stars.columns and 'sigma_pmdec_bp3m' in stars.columns:
            sig_pm = 0.5 * (stars['sigma_pmra_bp3m'].median()
                            + stars['sigma_pmdec_bp3m'].median())
            print(f"    Median σ_PM:    {sig_pm:.3f} mas/yr")
    else:
        print("\n  BP3M results:     not found")

    if imgs_csv.exists():
        imgs = pd.read_csv(imgs_csv)
        print(f"    Images used:    {len(imgs)}")
    print()


def write_ds9_region_file(stars: pd.DataFrame, path: str | Path,
                          ra_col: str = 'ra', dec_col: str = 'dec',
                          color: str = 'green', radius_arcsec: float = 1.0) -> None:
    """
    Write a DS9 region file for the matched stars.

    Parameters
    ----------
    stars      : DataFrame with RA/Dec columns
    path       : output .reg file path
    color      : DS9 region colour (default 'green')
    radius_arcsec: circle radius in arcseconds
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write('# Region file format: DS9 version 4.1\n')
        fh.write(f'global color={color} dashlist=8 3 width=1 ')
        fh.write('font="helvetica 10 normal roman" select=1 highlite=1 '
                 'dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        fh.write('fk5\n')
        ok = np.isfinite(stars[ra_col].values) & np.isfinite(stars[dec_col].values)
        for _, row in stars[ok].iterrows():
            fh.write(f'circle({row[ra_col]:.7f},{row[dec_col]:.7f},'
                     f'{radius_arcsec:.2f}")\n')


def write_fits_catalog(stars: pd.DataFrame, path: str | Path,
                       field_name: str = '') -> None:
    """
    Write the stellar astrometry catalogue as a FITS binary table.

    Requires astropy.  Falls back to CSV if astropy is unavailable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from astropy.table import Table
        t = Table.from_pandas(stars)
        if field_name:
            t.meta['FIELD'] = field_name
        t.write(str(path), overwrite=True)
        print(f"  Written FITS catalogue: {path}")
    except ImportError:
        csv_path = path.with_suffix('.csv')
        stars.to_csv(csv_path, index=False)
        print(f"  astropy not available; written CSV: {csv_path}")


# ── Comparison tables ─────────────────────────────────────────────────────────

def pm_comparison_table(stars: pd.DataFrame,
                        gaia: pd.DataFrame) -> pd.DataFrame:
    """
    Build a merged table comparing Gaia and BP3M proper motions for each star.

    Joins on the Gaia source ID and returns a DataFrame with both sets of
    PMs and their uncertainties side by side.
    """
    gaia_id_col = 'source_id' if 'source_id' in gaia.columns else 'Gaia_id'
    stars_id_col = 'Gaia_id' if 'Gaia_id' in stars.columns else 'source_id'

    cols_gaia = [gaia_id_col, 'pmra', 'pmdec', 'pmra_error', 'pmdec_error']
    cols_gaia = [c for c in cols_gaia if c in gaia.columns]

    cols_stars = [stars_id_col, 'gmag', 'n_hst_used',
                  'pmra_bp3m', 'pmdec_bp3m',
                  'sigma_pmra_bp3m', 'sigma_pmdec_bp3m']
    cols_stars = [c for c in cols_stars if c in stars.columns]

    merged = stars[cols_stars].merge(
        gaia[cols_gaia].rename(columns={gaia_id_col: stars_id_col}),
        on=stars_id_col, how='left')

    return merged
