"""
Instrument adapter contract for ground_to_gaia_xmatch.

The whole point of this module is the *normalisation boundary*: every adapter
returns its sources already projected onto the gnomonic tangent plane in
milliarcseconds, with a measurement covariance in mas^2.  Once an adapter has
done that, every downstream layer — discovery, affine refinement, the final
pass, the alignment solver, the plots — is unit-free and instrument-free.

This is the seam that was missing before.  The CFHT pipeline worked in MegaCam
pixels and the LSST pipeline worked in mas, so the shared algorithm had to be
copy-pasted and re-tuned per instrument, and bugs fixed in one never reached the
other.  Adapters absorb that difference here instead.

What an adapter is responsible for
----------------------------------
  * finding the images in a dataset               -> iter_images()
  * reading one image's source catalogue          -> load_catalog()
  * getting sources onto the tangent plane in mas -> SourceCatalog.xi/eta
  * building the measurement covariance in mas^2  -> SourceCatalog.C_src
  * classifying stars vs extended sources         -> SourceCatalog.is_star
  * supplying instrument scale constants          -> InstrumentConfig

What an adapter must NOT do
---------------------------
  * apply any Gaia-quality hard filter.  clean_label is derived from RUWE, which
    Gaia 2p sources do not have, so filtering on it silently removes every 2p
    star.  Gaia quality enters only as a *seed mask* for the discovery tiers.
  * fold the Gaia covariance into C_src.  C_src is the measurement error ALONE.
    The alignment solver applies the Gaia covariance separately as its prior, so
    anything Gaia-derived in C_src is double-counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np


# ── Normalised per-image source catalogue ────────────────────────────────────

@dataclass
class SourceCatalog:
    """
    One image's sources, normalised to the shared tangent-plane frame.

    All positional quantities are in milliarcseconds on the gnomonic tangent
    plane about the image's (ra0, dec0); all covariances are in mas^2.
    """

    xi: np.ndarray            # (n,) tangent-plane xi  [mas]
    eta: np.ndarray           # (n,) tangent-plane eta [mas]
    C_src: np.ndarray         # (n, 2, 2) measurement covariance [mas^2].
                              # Measurement error ALONE — never Gaia-derived.
    mag: np.ndarray           # (n,) instrumental/calibrated magnitude
    magerr: np.ndarray        # (n,) magnitude uncertainty
    is_star: np.ndarray       # (n,) bool — point-source classification

    ra: np.ndarray            # (n,) deg, for output tables
    dec: np.ndarray           # (n,) deg
    x: np.ndarray             # (n,) detector x [px], diagnostics only
    y: np.ndarray             # (n,) detector y [px], diagnostics only

    # Extra axes the discovery tier-walk may cut on, beyond magnitude.
    # CFHT supplies {'qfit': ..., 'chi2': ...}; LSST DP2 supplies {}.
    tier_keys: dict[str, np.ndarray] = field(default_factory=dict)

    # Passthrough columns copied verbatim into the match table.
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.xi)

    def validate(self) -> None:
        """Fail loudly on contract violations rather than silently biasing a fit."""
        n = len(self.xi)
        for name in ('eta', 'mag', 'magerr', 'is_star', 'ra', 'dec', 'x', 'y'):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f'SourceCatalog.{name} has length {len(arr)}, expected {n}')
        if self.C_src.shape != (n, 2, 2):
            raise ValueError(f'SourceCatalog.C_src has shape {self.C_src.shape}, expected {(n, 2, 2)}')
        if n == 0:
            return
        # Covariance must be positive semi-definite, or the Mahalanobis and
        # log-probability costs silently return garbage instead of erroring.
        det = self.C_src[:, 0, 0] * self.C_src[:, 1, 1] - self.C_src[:, 0, 1] * self.C_src[:, 1, 0]
        bad = (self.C_src[:, 0, 0] < 0) | (self.C_src[:, 1, 1] < 0) | (det < 0)
        if np.any(bad):
            raise ValueError(f'SourceCatalog.C_src is not positive semi-definite for '
                             f'{int(bad.sum())}/{n} sources')


# ── Per-image metadata ───────────────────────────────────────────────────────

@dataclass
class ImageMeta:
    """
    Identity and pointing for a single image.

    Directory names come from layout.py, never from string literals at the call
    site — that is what keeps the cross-match and alignment outputs in step.
    """

    instrument: str           # 'cfht', 'lsst', ...
    exposure: Any             # native exposure id (expnum / visit)
    detector: Any             # native detector id (ext / detector)
    ra0: float                # tangent point [deg]
    dec0: float               # tangent point [deg]
    mjd: float                # exposure mid-point
    pixel_scale: float        # [mas/px]

    band: str | None = None   # filter, where the instrument distinguishes one
    det_width: int = 0        # zero-padding for the detector number

    # Any further instrument-native identity, written verbatim into
    # transformation.csv so downstream tools can round-trip it.
    key: dict[str, Any] = field(default_factory=dict)

    @property
    def exposure_id(self) -> str:
        """Canonical exposure directory name, e.g. 'lsst_2025121400828'."""
        from ..layout import exposure_id
        return exposure_id(self.instrument, self.exposure)

    @property
    def det_token(self) -> str:
        """Canonical detector directory name, e.g. 'det_147_i'."""
        from ..layout import det_token
        return det_token(self.detector, self.band, self.det_width)

    @property
    def image_id(self) -> str:
        """Flat unique identifier, e.g. 'lsst_2025121400828_det_147_i'."""
        return f'{self.exposure_id}_{self.det_token}'

    def rel_dir(self) -> Path:
        """Output sub-directory for this image, relative to a stage root."""
        return Path(self.exposure_id) / self.det_token


# ── Instrument scale constants ───────────────────────────────────────────────

@dataclass
class InstrumentConfig:
    """
    Scale constants for one instrument, all in mas.

    These are the values that genuinely depend on plate scale and image quality.
    They are NOT algorithm choices — the algorithm lives in discovery.py and is
    identical for every instrument.
    """

    disc_max: float           # offset-histogram half-width
    disc_bin: float           # offset-histogram bin size
    disc_seed_radii: tuple[float, ...]   # tight-then-loose 4P seed match radii
    disc_floor: float         # discovery covariance floor (~1 pixel)
    refine_radius: float      # candidate radius in 6P refinement / final pass
    pix_floor: float          # systematic astrometric floor added to C_src

    max_mag_diff: float = 3.0
    min_matches: int = 4

    # Transform priors.  Left None, they fall back to the shared bp3m defaults
    # so every instrument uses the same priors unless it deliberately overrides.
    sigma_rot_deg: float | None = None
    sigma_scale: float | None = None
    sigma_skew: float | None = None

    # Magnitude tier-walk
    mag_step: float = 1.0

    def __post_init__(self):
        from bp3m.instrument_config import SIGMA_ROT_DEG, SIGMA_SCALE, SIGMA_SKEW
        if self.sigma_rot_deg is None:
            self.sigma_rot_deg = SIGMA_ROT_DEG
        if self.sigma_scale is None:
            self.sigma_scale = SIGMA_SCALE
        if self.sigma_skew is None:
            self.sigma_skew = SIGMA_SKEW


# ── Adapter protocol ─────────────────────────────────────────────────────────

class Instrument(Protocol):
    """
    What a dataset adapter must provide.

    Implementations live in instruments/cfht.py, instruments/lsst.py, ...
    """

    name: str
    config: InstrumentConfig

    def iter_images(self) -> Iterator[ImageMeta]:
        """Yield metadata for every image in the dataset."""
        ...

    def load_catalog(self, meta: ImageMeta) -> SourceCatalog:
        """Load and normalise one image's sources. Returns a validated catalog."""
        ...


__all__ = ['SourceCatalog', 'ImageMeta', 'InstrumentConfig', 'Instrument']
