"""Filter bookkeeping shared by the plotting / membership code (2026-10-07, jwst-integration).

HST UV/optical filter names encode the wavelength in nm (F814W = 814 nm), while HST WFC3/IR and every
JWST NIRCam / NIRISS / MIRI filter name encodes it in units of 10 nm (F160W = 1.6 um, F277W = 2.77 um,
F1000W = 10 um).  Before this helper the CMD / colour-colour panels (hst_catalog_crossmatch, the member-
selection notebooks, the pop-fit diagnostics) parsed the digits as nm and put F277W bluewards of F606W.
"""
from __future__ import annotations

import re

# HST UV/optical/IR effective wavelengths (nm) -- same table as the v2 CMD plots
_WL_NM = {
    'F225W': 237, 'F275W': 271, 'F300X': 282, 'F336W': 336, 'F390W': 392, 'F435W': 432, 'F438W': 438,
    'F475W': 476, 'F555W': 531, 'F606W': 592, 'F600LP': 747, 'F625W': 626, 'F775W': 764, 'F814W': 803,
    'F850LP': 918, 'F098M': 986, 'F105W': 1055, 'F110W': 1153, 'F125W': 1249, 'F140W': 1392, 'F160W': 1537,
    'G': 673, 'BP': 511, 'RP': 777,                     # Gaia
    'DELVEG': 473, 'DELVER': 642, 'DELVEI': 784, 'DELVEZ': 926,
}

# names whose digits are in units of 10 nm: JWST NIRCam / NIRISS / MIRI and HST WFC3/IR
_X10 = {
    'F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F150W2', 'F162M', 'F164N', 'F182M', 'F187N', 'F200W',
    'F210M', 'F212N', 'F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M', 'F356W', 'F360M', 'F380M',
    'F405N', 'F410M', 'F430M', 'F444W', 'F460M', 'F466N', 'F470N', 'F480M',
    'F560W', 'F770W', 'F1000W', 'F1065C', 'F1130W', 'F1140C', 'F1280W', 'F1500W', 'F1550C', 'F1800W',
    'F2100W', 'F2300C', 'F2550W',
    'F098M', 'F105W', 'F110W', 'F125W', 'F126N', 'F127M', 'F128N', 'F130N', 'F132N', 'F139M', 'F140W',
    'F153M', 'F160W', 'F167N',
}


def filter_wavelength_nm(band: str, default: float = 9999.0) -> float:
    """Approximate effective wavelength (nm) of a band label such as 'F814W', 'F277W', 'F606W/ACS_WFC',
    'DELVEg' or 'G'.  Unknown labels without digits return *default* (sorted last)."""
    key = str(band).split('/')[0].strip().upper()
    if key in _WL_NM:
        return float(_WL_NM[key])
    m = re.search(r'(\d{3,4})', key)
    if not m:
        return float(default)
    return float(m.group(1)) * (10.0 if key in _X10 else 1.0)


def sort_bands(bands) -> list:
    """Band labels sorted blue -> red."""
    return sorted(bands, key=filter_wavelength_nm)
