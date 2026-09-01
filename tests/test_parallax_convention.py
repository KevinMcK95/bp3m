"""
Parallax-convention regression tests.

A frame-swap sign bug lived in bp3m.astro_utils.get_parallax_factors from the
initial commit (2026-06) until 2026-09: the classic Sun-geocentric almanac
formula's minus signs were kept after switching to Earth-BARYCENTRIC
coordinates.  These tests pin the convention to first-principles vector
geometry so it can never silently flip again, and guard against re-derived
inline copies of the physics.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from bp3m.astro_utils import (get_tele_position, get_parallax_factors,
                              propagate_gaia_positions)

RAD2MAS = 180.0 / np.pi * 3.6e6


def _apparent_shift_mas(ra_deg, dec_deg, plx_mas, tele_xyz):
    """Ground truth: apparent-minus-barycentric shift from pure vector geometry."""
    d_au = (1000.0 / plx_mas) * 206264.806          # distance in AU
    rr, dd = np.radians(ra_deg), np.radians(dec_deg)
    u = np.array([np.cos(dd) * np.cos(rr), np.cos(dd) * np.sin(rr), np.sin(dd)])
    v = d_au * u - np.asarray(tele_xyz)
    v = v / np.linalg.norm(v)
    e_a = np.array([-np.sin(rr), np.cos(rr), 0.0])
    e_d = np.array([-np.cos(rr) * np.sin(dd), -np.sin(rr) * np.sin(dd), np.cos(dd)])
    return np.dot(v - u, e_a) * RAD2MAS, np.dot(v - u, e_d) * RAD2MAS


@pytest.mark.parametrize("mjd", [58849.0, 58940.0, 59032.0, 59124.0])  # 4 seasons
@pytest.mark.parametrize("ra,dec", [(152.1, 12.3),   # Leo I (on-ecliptic)
                                    (325.1, -23.2),  # M30
                                    (10.0, 75.0)])   # high dec
def test_factors_match_vector_geometry(mjd, ra, dec):
    from astropy.time import Time
    xyz = get_tele_position(Time(mjd, format='mjd'), curr_id='earth')
    plx = 100.0
    true_ra, true_dec = _apparent_shift_mas(ra, dec, plx, xyz)
    f_ra, f_dec = get_parallax_factors(ra, dec, xyz)
    assert np.isclose(plx * float(f_ra),  true_ra,  atol=1e-3)
    assert np.isclose(plx * float(f_dec), true_dec, atol=1e-3)


def test_propagate_gaia_positions_matches_manual():
    from astropy.time import Time
    xyz = get_tele_position(Time(59000.0, format='mjd'), curr_id='earth')
    ra, dec = np.array([152.1, 200.0]), np.array([12.3, -40.0])
    pmra, pmdec = np.array([5.0, -3.0]), np.array([-2.0, 7.0])
    plx, dt = np.array([10.0, 0.5]), -6.0
    ra_p, dec_p = propagate_gaia_positions(ra, dec, pmra, pmdec, plx, dt, xyz)
    f_ra, f_dec = get_parallax_factors(ra, dec, xyz)
    exp_ra  = ra  + (pmra * dt + plx * f_ra) / 3.6e6 / np.cos(np.radians(dec))
    exp_dec = dec + (pmdec * dt + plx * f_dec) / 3.6e6
    assert np.allclose(ra_p, exp_ra, atol=1e-12)
    assert np.allclose(dec_p, exp_dec, atol=1e-12)


def test_propagate_treats_nonfinite_as_zero():
    xyz = np.array([0.9, 0.3, 0.1])
    ra_p, dec_p = propagate_gaia_positions(150.0, 10.0, np.nan, np.nan, np.nan,
                                           5.0, xyz)
    assert np.isclose(float(ra_p), 150.0) and np.isclose(float(dec_p), 10.0)


def test_no_inline_parallax_factor_copies():
    """The parallax-factor formula must exist ONLY in bp3m/astro_utils.py."""
    root = Path(__file__).resolve().parents[1]
    # the p_dec fingerprint: cos(ra)*sin(dec) paired with sin(ra)*sin(dec)
    pat = re.compile(r"np\.sin\(\s*\w+\s*\)\s*\*\s*np\.sin\(\s*dec", re.I)
    pat2 = re.compile(r"np\.cos\(\w*ra\w*\)\s*\*\s*np\.sin\(\w*dec\w*\)")
    offenders = []
    for sub in ("bp3m", "gaia_cross_match", "ground_to_gaia_xmatch", "pypass"):
        for f in (root / sub).rglob("*.py"):
            if f.name == "astro_utils.py":
                continue
            src = f.read_text(errors="ignore")
            if pat2.search(src) and ("parallax" in src.lower()):
                # only flag when it looks like the parallax-factor formula
                for m in pat2.finditer(src):
                    ctx = src[max(0, m.start()-200):m.end()+200]
                    if "plx" in ctx or "parallax" in ctx:
                        offenders.append(str(f.relative_to(root)))
                        break
    assert not offenders, f"inline parallax-factor copies found: {offenders}"
