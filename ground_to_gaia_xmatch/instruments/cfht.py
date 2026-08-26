"""
CFHT/UNIONS MegaCam adapter.

Reads the SExtractor photcal catalogues and the per-detector WCS solutions:

    <field_root>/cat_files/<expnum>p.photcal.cat    one file per exposure, all 40 CCDs
    <field_root>/cfht_unions_detectors.csv          per (expnum, ext) WCS + distortion
    <field_root>/xmatch/cfht_<expnum>/Gaia/*_gaia.csv

Notes specific to this dataset
------------------------------
* Source positions are already calibrated ra/dec, so the tangent-plane
  projection is the same gnomonic used everywhere else.  The distortion
  polynomial is needed only to propagate the *pixel* centroid uncertainty into
  the tangent plane, via its Jacobian.
* `photflags` is a bit field: bit 0 set => non-stellar; bits 1 and 2 flag bad
  photometry.  `flags` must be 0.
* The discovery tier-walk cuts on magerr x mag (MAGERR_LIMITS), not on the
  qfit/chi2 axes used by the HST pipeline, and it always uses stars only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from ..geometry import covariance_from_sigmas, gnomonic
from .base import ImageMeta, InstrumentConfig, SourceCatalog

NAME = 'cfht'

# The detector table is ~848k rows (21,200 exposures x 40 CCDs) and takes
# seconds to parse.  Bulk processing builds one CFHTInstrument per exposure, so
# a per-instance cache would re-read it every time — hours of pure CSV parsing
# per worker.  Cache on the path instead, shared across instances.
_DET_TABLE_CACHE: dict = {}
_EXP_CACHE: dict = {}

POLY_ORDER = 3
CAT_COLUMNS = ['ra', 'dec', 'x', 'y', 'ext', 'mag', 'magerr',
               'flux_radius', 'flags', 'photflags']

# Source-quality axis for the discovery tier-walk.
MAGERR_LIMITS = [0.02, 0.05, 0.10, 0.20, np.inf]

# MegaCam is ~187 mas/px.
CONFIG = InstrumentConfig(
    disc_max=3000.0,
    disc_bin=100.0,
    disc_seed_radii=(400.0, 1500.0),
    disc_floor=187.0,          # ~1 MegaCam pixel
    refine_radius=12000.0,
    pix_floor=10.0,            # CFHT_PIX_FLOOR, unchanged from the original
    max_mag_diff=3.0,
    min_matches=4,
    mag_step=1.0,
)


def _n_terms(order: int) -> int:
    return (order + 1) * (order + 2) // 2


def _dpoly_dx(x, y, order):
    """d/dx of each polynomial basis function (normalised detector coords)."""
    cols = []
    for p in range(order + 1):
        for i in range(p + 1):
            j = p - i
            cols.append(np.zeros_like(x) if i == 0 else i * x**(i - 1) * y**j)
    return np.column_stack(cols)


def _dpoly_dy(x, y, order):
    """d/dy of each polynomial basis function (normalised detector coords)."""
    cols = []
    for p in range(order + 1):
        for i in range(p + 1):
            j = p - i
            cols.append(np.zeros_like(x) if j == 0 else x**i * j * y**(j - 1))
    return np.column_stack(cols)


def magerr_mag_tiers(cat: SourceCatalog, config: InstrumentConfig,
                     stars_only: bool):
    """
    CFHT discovery tier-walk: magerr x magnitude, stars only.

    Yields nothing for stars_only=False, so the shared discover() loop's
    all-sources pass is a no-op — the original pipeline seeded on stars only and
    this keeps that behaviour exactly.
    """
    if not stars_only:
        return
    star = cat.is_star & np.isfinite(cat.mag)
    if not star.any():
        return
    magerr = cat.tier_keys.get('magerr', np.zeros(len(cat)))
    mags = cat.mag[star]
    limits = np.arange(mags.min() + 1.0, mags.max() + 0.5, config.mag_step)
    if len(limits) == 0:
        limits = np.array([mags.max()])
    limits[-1] = mags.max()
    for qlim in MAGERR_LIMITS:
        for mlim in limits:
            yield (f'magerr<{qlim:g} m<{mlim:.1f}',
                   star & (magerr <= qlim) & (cat.mag <= mlim))


@dataclass
class CFHTInstrument:
    """Adapter for a CFHT/UNIONS MegaCam dataset."""

    field_root: Path
    exposures: tuple[int, ...] | None = None   # restrict to these expnums
    detectors: tuple[int, ...] | None = None   # restrict to these exts
    name: str = NAME
    config: InstrumentConfig = None

    def __post_init__(self):
        self.field_root = Path(self.field_root)
        if self.config is None:
            self.config = CONFIG
        self._det = None
        self._cat_cache: tuple[int, pd.DataFrame] | None = None
        self._gaia_cache: dict = {}

    @property
    def detector_table(self) -> pd.DataFrame:
        if self._det is None:
            key = str(self.field_root / 'cfht_unions_detectors.csv')
            if key not in _DET_TABLE_CACHE:
                _DET_TABLE_CACHE[key] = pd.read_csv(key)
            self._det = _DET_TABLE_CACHE[key]
        return self._det

    def _catalog_path(self, expnum: int) -> Path:
        return self.field_root / 'cat_files' / f'{expnum}p.photcal.cat'

    def _exposure_catalog(self, expnum: int) -> pd.DataFrame:
        """Whole-exposure catalogue (all CCDs), cached for consecutive detectors."""
        if self._cat_cache is not None and self._cat_cache[0] == expnum:
            return self._cat_cache[1]
        cat = pd.read_csv(self._catalog_path(expnum), comment='#', sep=r'\s+',
                          header=None, names=CAT_COLUMNS)
        self._cat_cache = (expnum, cat)
        return cat

    def available_exposures(self) -> list[int]:
        """Exposure numbers that have both a catalogue and detector rows."""
        key = str(self.field_root)
        if key in _EXP_CACHE:
            return _EXP_CACHE[key]
        have_det = set(self.detector_table['expnum'].unique().tolist())
        found = []
        for p in sorted((self.field_root / 'cat_files').glob('*p.photcal.cat')):
            try:
                e = int(p.name.split('p.')[0])
            except ValueError:
                continue
            if e in have_det:
                found.append(e)
        _EXP_CACHE[key] = found
        return found

    def gaia_catalog(self, meta=None) -> pd.DataFrame:
        """
        Gaia catalogue for one exposure.

        Returns the FULL catalogue — no clean_label filter, which would remove
        every 2p source.
        """
        from ..layout import exposure_id, xmatch_root
        expnum = getattr(meta, 'exposure', None) if meta is not None else None
        if expnum is not None and expnum in self._gaia_cache:
            return self._gaia_cache[expnum]
        if expnum is None:
            exps = self._selected_exposures()
            if not exps:
                raise FileNotFoundError('no exposures selected')
            expnum = exps[0]
        gdir = xmatch_root(self.field_root) / exposure_id(NAME, expnum) / 'Gaia'
        hits = sorted(gdir.glob('*_gaia.csv'))
        if not hits:   # tolerate the pre-restructure location
            legacy = self.field_root / 'Gaia_xmatch' / f'cfht_{expnum}' / 'Gaia'
            hits = sorted(legacy.glob('*_gaia.csv')) if legacy.is_dir() else []
        if not hits:
            raise FileNotFoundError(f'No *_gaia.csv for exposure {expnum} in {gdir}')
        # Exclude the qso/galaxy candidate side-files.
        main = [h for h in hits if 'candidates' not in h.name]
        df = pd.read_csv(main[0] if main else hits[0])
        self._gaia_cache[expnum] = df
        return df

    def _selected_exposures(self) -> list[int]:
        if self.exposures:
            return list(self.exposures)
        return self.available_exposures()

    # ── Instrument protocol ──────────────────────────────────────────────────

    def iter_images(self) -> Iterator[ImageMeta]:
        det = self.detector_table
        for expnum in self._selected_exposures():
            rows = det[det['expnum'] == expnum]
            for _, r in rows.iterrows():
                ext = int(r['ext'])
                if self.detectors and ext not in self.detectors:
                    continue
                mjd = float(r['mjd'])
                if not np.isfinite(mjd):
                    continue
                yield ImageMeta(
                    instrument=NAME,
                    exposure=int(expnum),
                    detector=ext,
                    band=None,
                    ra0=float(r['ra0']), dec0=float(r['dec0']),
                    mjd=mjd,
                    pixel_scale=float(r.get('plate_scale_mas_px', 187.0)),
                    det_width=2,
                    key={'expnum': int(expnum), 'ext': ext},
                )

    def load_catalog(self, meta: ImageMeta) -> SourceCatalog:
        cat = self._exposure_catalog(meta.exposure)
        sub = cat[cat['ext'] == meta.detector]

        # Quality filter: flags clean, photometry bits 1 and 2 clear.
        pf = sub['photflags'].values.astype(int)
        good = (sub['flags'].values == 0) & ((pf & 2) == 0) & ((pf & 4) == 0)
        sub = sub[good].reset_index(drop=True)
        pf = sub['photflags'].values.astype(int)

        ra = sub['ra'].values.astype(float)
        dec = sub['dec'].values.astype(float)
        xi, eta = gnomonic(ra, dec, meta.ra0, meta.dec0)

        # Pixel centroid uncertainty -> tangent plane, via the distortion Jacobian.
        det_row = self.detector_table[
            (self.detector_table['expnum'] == meta.exposure)
            & (self.detector_table['ext'] == meta.detector)].iloc[0]
        npar = _n_terms(POLY_ORDER)
        cx = np.array([float(det_row[f'cx{k}']) for k in range(npar)])
        cy = np.array([float(det_row[f'cy{k}']) for k in range(npar)])
        x0, y0 = float(det_row['x0']), float(det_row['y0'])
        xs, ys = float(det_row['xs']), float(det_row['ys'])

        xc = (sub['x'].values.astype(float) - x0) / xs
        yc = (sub['y'].values.astype(float) - y0) / ys
        sigma_px = np.clip(sub['flux_radius'].values.astype(float)
                           * sub['magerr'].values.astype(float) / 1.28,
                           0.002, np.inf)

        dPdxc, dPdyc = _dpoly_dx(xc, yc, POLY_ORDER), _dpoly_dy(xc, yc, POLY_ORDER)
        dxi_dx = (dPdxc @ cx) / xs * 1000      # arcsec/px -> mas/px
        dxi_dy = (dPdyc @ cx) / ys * 1000
        deta_dx = (dPdxc @ cy) / xs * 1000
        deta_dy = (dPdyc @ cy) / ys * 1000

        s_xi = np.sqrt((dxi_dx * sigma_px)**2 + (dxi_dy * sigma_px)**2)
        s_eta = np.sqrt((deta_dx * sigma_px)**2 + (deta_dy * sigma_px)**2)
        C_src = covariance_from_sigmas(s_xi, s_eta, None, floor=self.config.pix_floor)

        magerr = sub['magerr'].values.astype(float)
        cat_out = SourceCatalog(
            xi=xi, eta=eta, C_src=C_src,
            mag=sub['mag'].values.astype(float), magerr=magerr,
            is_star=((pf & 1) == 0),          # bit 0 set => non-stellar
            ra=ra, dec=dec,
            x=sub['x'].values.astype(float), y=sub['y'].values.astype(float),
            tier_keys={'magerr': magerr},
            extra={},
        )
        cat_out.validate()
        return cat_out


__all__ = ['CFHTInstrument', 'magerr_mag_tiers', 'CONFIG', 'NAME', 'MAGERR_LIMITS']
