"""
LSST DP2 adapter.

Reads the IPAC tables exported from the Rubin Science Platform:

    <field_root>/table_dp2.<field>-data.tbl           source catalogue
    <field_root>/table_dp2.<field>-VisitDetector.tbl  per-detector WCS metadata
    <field_root>/Gaia/*_gaia.csv                      Gaia catalogue

Unit and column notes specific to this dataset
----------------------------------------------
* `raErr`/`decErr`/`ra_dec_Cov` are in **deg / deg / deg^2**, per row 14 of the
  IPAC header.  An earlier version divided them by 1e3 as if they were
  microarcsec, which underflowed every source onto a clip floor and destroyed
  all magnitude-dependent weighting.  Convert with DEG2MAS.
* `raErr` is already a great-circle error: median(raErr)/median(decErr) = 1.00
  across detectors, not the 1/cos(dec) = 1.21 a raw RA-angle error would give.
  No cos(dec) factor is applied.
* `extendedness_flag` arrives in three different guises and MUST be parsed
  defensively.  A portal IPAC export gives the lowercase strings
  'false'/'true'; a TAP download gives a real numpy bool; and a TAP result
  written back out through astropy's IPAC writer gives CAPITALISED 'False'/
  'True' (it renders the bool with str()).  A literal `== 'false'` test matches
  nothing in the last two cases, silently classifying every source as extended
  and destroying the star/galaxy separation.  Use _to_bool().
* Fluxes are nJy; magnitude uses the AB zero point of 31.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from astropy.io import ascii as ascii_io

from ..geometry import DEG2MAS, covariance_from_sigmas, gnomonic
from .base import ImageMeta, InstrumentConfig, SourceCatalog

NAME = 'lsst'

# nJy -> AB magnitude
AB_ZP = 31.4

# Guard rails on the reported position errors [mas].
ERR_MIN = 0.1        # against zero/underflow
ERR_MAX = 5000.0     # junk sources reach raErr ~ 0.013 deg = 47 arcsec
ERR_FALLBACK = 5000.0

# Instrument scale constants.  ~0.2"/px, so one pixel is ~200 mas.
CONFIG = InstrumentConfig(
    disc_max=3000.0,
    disc_bin=100.0,
    disc_seed_radii=(400.0, 1500.0),
    disc_floor=200.0,          # ~1 LSST pixel
    refine_radius=12000.0,
    pix_floor=0.5,             # deliberately negligible; resid_cov carries the
                               # systematics, as in CFHT (hst_pix_floor=0.01 px)
    max_mag_diff=3.0,
    min_matches=4,
    mag_step=1.0,
)


@dataclass
class LSSTInstrument:
    """Adapter for an LSST DP2 field export."""

    field_root: Path
    exposures: tuple[int, ...] | None = None   # restrict to these visits
    detectors: tuple[int, ...] | None = None   # restrict to these detectors
    name: str = NAME
    config: InstrumentConfig = None

    def __post_init__(self):
        self.field_root = Path(self.field_root)
        if self.config is None:
            self.config = CONFIG
        self._sources = None
        self._meta = None

    # ── table loading (cached) ───────────────────────────────────────────────

    def _find(self, suffix: str) -> Path:
        """Locate a table, preferring FITS (written for large fields)."""
        for ext in ('.fits', '.tbl'):
            hits = sorted(self.field_root.glob(f'table_dp2.*-{suffix}{ext}'))
            if hits:
                return hits[0]
        raise FileNotFoundError(
            f'No table_dp2.*-{suffix}.(fits|tbl) under {self.field_root}')

    @staticmethod
    def _to_bool(col) -> np.ndarray:
        """
        Interpret a flag column that may be bool, 'true'/'false', or
        'True'/'False' depending on how the table reached us.

        Anything unrecognised is treated as False rather than guessed at, so a
        new representation shows up as "no flags set" rather than as silently
        inverted classifications.
        """
        a = np.asarray(col)
        if a.dtype == bool:
            return a
        if np.issubdtype(a.dtype, np.number):
            return a.astype(bool)
        s = np.char.lower(np.char.strip(a.astype(str)))
        return np.isin(s, ('true', 't', '1', 'yes'))

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if path.suffix == '.fits':
            from astropy.table import Table
            df = Table.read(path).to_pandas()
            # FITS stores strings as bytes; decode so comparisons like
            # extendedness_flag == 'false' behave as they do for IPAC.
            for c in df.columns:
                if df[c].dtype == object and len(df) and isinstance(
                        df[c].iloc[0], (bytes, bytearray)):
                    df[c] = df[c].str.decode('utf-8')
            return df
        return ascii_io.read(path, format='ipac').to_pandas()

    @property
    def sources(self) -> pd.DataFrame:
        if self._sources is None:
            self._sources = self._read(self._find('data'))
        return self._sources

    @property
    def visit_detector(self) -> pd.DataFrame:
        if self._meta is None:
            self._meta = self._read(self._find('VisitDetector'))
        return self._meta

    def gaia_catalog(self, meta=None) -> pd.DataFrame:
        """
        Load the Gaia catalogue for this field.

        `meta` is accepted for interface symmetry with CFHT (one catalogue per
        exposure there) but ignored: LSST has a single catalogue per field.

        Returns the FULL catalogue.  No clean_label filter: clean_label derives
        from RUWE, which 2p sources lack, so filtering here would remove every
        one of them.  Quality enters downstream as a discovery seed mask only.
        """
        from ..layout import gaia_dir
        gdir = gaia_dir(self.field_root)
        hits = sorted(gdir.glob('*_gaia.csv')) if gdir.is_dir() else []
        if not hits:      # tolerate the pre-restructure location
            hits = sorted(self.field_root.glob('*_gaia.csv'))
        if not hits:
            raise FileNotFoundError(f'No *_gaia.csv in {gdir} or {self.field_root}')
        return pd.read_csv(hits[0])

    # ── Instrument protocol ──────────────────────────────────────────────────

    def iter_images(self) -> Iterator[ImageMeta]:
        vd = self.visit_detector
        combos = self.sources.groupby(['visit', 'detector', 'band']).size().index
        for visit, detector, band in combos:
            if self.exposures and int(visit) not in self.exposures:
                continue
            if self.detectors and int(detector) not in self.detectors:
                continue
            row = vd[(vd['visitId'] == visit) & (vd['detector'] == detector)]
            if len(row) == 0:
                continue      # no WCS metadata for this detector
            row = row.iloc[0]
            yield ImageMeta(
                instrument=NAME,
                exposure=int(visit),
                detector=int(detector),
                band=str(band),
                ra0=float(row['ra']),
                dec0=float(row['dec']),
                mjd=float(row['expMidptMJD']),
                pixel_scale=float(row['pixelScale']) * 1000.0,   # "/px -> mas/px
                det_width=3,
                key={'visit': int(visit), 'detector': int(detector), 'band': str(band)},
            )

    def load_catalog(self, meta: ImageMeta) -> SourceCatalog:
        s = self.sources
        sub = s[(s['visit'] == meta.exposure)
                & (s['detector'] == meta.detector)
                & (s['band'] == meta.band)].reset_index(drop=True)

        ra = sub['ra'].values.astype(float)
        dec = sub['dec'].values.astype(float)
        xi, eta = gnomonic(ra, dec, meta.ra0, meta.dec0)

        # deg -> mas; already great-circle, so no cos(dec) factor.
        sig_ra = np.abs(sub['raErr'].values.astype(float)) * DEG2MAS
        sig_dec = np.abs(sub['decErr'].values.astype(float)) * DEG2MAS
        cov = sub['ra_dec_Cov'].values.astype(float) * DEG2MAS**2

        sig_ra[~np.isfinite(sig_ra) | (sig_ra <= 0)] = ERR_FALLBACK
        sig_dec[~np.isfinite(sig_dec) | (sig_dec <= 0)] = ERR_FALLBACK
        sig_ra = np.clip(sig_ra, ERR_MIN, ERR_MAX)
        sig_dec = np.clip(sig_dec, ERR_MIN, ERR_MAX)
        C_src = covariance_from_sigmas(sig_ra, sig_dec, cov, floor=self.config.pix_floor)

        flux = sub['calibFlux'].values.astype(float)
        flux_err = sub['calibFluxErr'].values.astype(float)
        safe = np.maximum(flux, 1.0)
        mag = -2.5 * np.log10(safe) + AB_ZP
        magerr = np.clip(1.0857 * np.abs(flux_err) / safe, 0.01, 1.0)

        # extendedness_flag TRUE means extended, so a star is the negation.
        # Parsed via _to_bool because the representation varies by source; see
        # the module docstring.
        is_star = ~self._to_bool(sub['extendedness_flag'].values)

        cat = SourceCatalog(
            xi=xi, eta=eta, C_src=C_src,
            mag=mag, magerr=magerr, is_star=is_star,
            ra=ra, dec=dec,
            x=sub['x'].values.astype(float), y=sub['y'].values.astype(float),
            tier_keys={},          # DP2 has no qfit analogue
            extra={'sourceId': sub['sourceId'].values},
        )
        cat.validate()
        return cat


__all__ = ['LSSTInstrument', 'CONFIG', 'NAME']
