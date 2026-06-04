import numpy as np


def mag_from_flux(flux, flux_err, zero_point=0.0):
    """Return (mag, mag_err) in the AB/Vega system offset by zero_point."""
    if flux <= 0:
        return np.inf, np.inf
    mag = zero_point - 2.5 * np.log10(flux)
    if np.isfinite(flux_err) and flux_err >= 0:
        mag_err = (2.5 / np.log(10.0)) * flux_err / flux
    else:
        mag_err = np.inf
    return float(mag), float(mag_err)
