"""bp3m-setup: Download HST PSF/GDC library files and QSO reference catalogs."""

import argparse
import re
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError

BASE_URL = "https://www.stsci.edu/~jayander/HST1PASS/LIB"

# ── QSO reference catalog URLs ───────────────────────────────────────────────
# Quaia: Storey-Fisher et al. 2024 (Gaia DR3 + unWISE photometric QSOs).
# Zenodo DOI 10.5281/zenodo.10403370 — stable version-locked URL.
# Key columns: source_id (Gaia DR3 int64), ra, dec, redshift_quaia,
#              phot_g_mean_mag, mag_w1_vg, mag_w2_vg.
_QUAIA_URL = (
    "https://zenodo.org/records/10403370/files/quaia_G20.5.fits?download=1"
)
_QUAIA_FILENAME = "quaia_G20.5.fits"

# MILLIQUAS v8: Flesch 2023 — Final Edition (~907 k spectroscopic + ~66 k
# radio/X-ray candidates).  No Gaia source_id; positions for ~61% of sources
# use Gaia EDR3 astrometry (flagged by 'G' in the Comment column), giving
# sub-arcsec accuracy.  Cross-match against Gaia must be positional.
# Key columns: RA/RAdeg, Dec/DEdeg, Name, Type, z/Redshift, Comment.
_MILLIQUAS_URL = "https://quasars.org/milliquas.fits.zip"
_MILLIQUAS_FILENAME = "milliquas.fits"

def _bp3m_home() -> Path:
    """Base directory for bp3m config and default lib. Override with BP3M_HOME."""
    import os
    return Path(os.environ["BP3M_HOME"]) if "BP3M_HOME" in os.environ else Path.home() / ".bp3m"

CONFIG_FILE = _bp3m_home() / "config.toml"
DEFAULT_LIB_DIR = _bp3m_home() / "lib"

PSF_INSTRUMENTS = ["ACSWFC", "ACSHRC", "WFC3UV"]
GDC_INSTRUMENTS = ["ACSWFC", "ACSHRC", "WFC3UV"]
# WFC3IR has PSFs on the server but no GDCs; not yet supported by pypass.
# Users can request it explicitly with --instruments WFC3IR.
_OPTIONAL_PSF_ONLY = {"WFC3IR"}


def _list_fits(url: str) -> list:
    """Return list of full .fits file URLs by scraping the STScI directory listing."""
    try:
        with urlopen(url, timeout=30) as r:
            html = r.read().decode("utf-8", errors="replace")
        names = re.findall(r'href="([^"]+\.fits)"', html, re.IGNORECASE)
        base = url.rstrip("/")
        return [f"{base}/{n}" for n in names]
    except URLError as e:
        print(f"  WARNING: could not list {url}: {e}")
        return []


def _download(url: str, dest: Path) -> bool:
    """Download url to dest. Returns True on success."""
    tmp = dest.with_suffix(".tmp")
    try:
        urlretrieve(url, str(tmp))
        tmp.rename(dest)
        return True
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


def _download_large(url: str, dest: Path, label: str = '') -> bool:
    """Download a large file with MB-progress display. Returns True on success."""
    import urllib.request
    tmp = dest.with_suffix('.tmp')
    try:
        shown_mb = [0]
        def _hook(block, block_size, total):
            mb = block * block_size / 1e6
            if mb - shown_mb[0] >= 20:
                shown_mb[0] = int(mb / 20) * 20
                tot = f'/{total/1e6:.0f} MB' if total > 0 else ''
                print(f'    {label}: {mb:.0f}{tot} MB...', end='\r', flush=True)
        # Use a browser User-Agent — some servers (e.g. quasars.org) return 406
        # when the default Python UA string is detected.
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; bp3m-setup/1.0)'},
        )
        with urllib.request.urlopen(req) as resp:
            total_size = int(resp.headers.get('Content-Length', 0))
            block_size = 65536
            block_num  = 0
            with open(tmp, 'wb') as f:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    block_num += 1
                    _hook(block_num, block_size, total_size)
        print()
        tmp.rename(dest)
        return True
    except Exception as e:
        print(f'\n  ERROR downloading {label}: {e}')
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


def _download_and_extract_zip(url: str, dest_fits: Path, label: str = '') -> bool:
    """Download a zip, extract the first .fits inside, delete the zip."""
    zip_tmp = dest_fits.with_suffix('.zip.tmp')
    if not _download_large(url, zip_tmp, label):
        return False
    try:
        with zipfile.ZipFile(zip_tmp) as zf:
            fits_names = [n for n in zf.namelist() if n.lower().endswith('.fits')]
            if not fits_names:
                print(f'  ERROR: no .fits file found inside {zip_tmp.name}')
                zip_tmp.unlink(missing_ok=True)
                return False
            name = fits_names[0]
            print(f'    Extracting {name}...')
            tmp = dest_fits.with_suffix('.extract.tmp')
            with zf.open(name) as src, open(tmp, 'wb') as dst:
                dst.write(src.read())
            tmp.rename(dest_fits)
        zip_tmp.unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f'  ERROR extracting {label}: {e}')
        zip_tmp.unlink(missing_ok=True)
        return False


def main():
    p = argparse.ArgumentParser(
        description=(
            "Download HST PSF and geometric distortion correction (GDC) library "
            "files for bp3m from STScI (https://www.stsci.edu/~jayander/HST1PASS/LIB). "
            "Saves the lib_dir path to config.toml so --lib_dir is optional "
            "when running bp3m. Config location defaults to ~/.bp3m/ but can be "
            "overridden by setting the BP3M_HOME environment variable."
        )
    )
    p.add_argument(
        "--lib-dir",
        default=None,
        help=f"Directory to store PSF/GDC files (default: {DEFAULT_LIB_DIR})",
    )
    p.add_argument(
        "--no-config",
        action="store_true",
        help="Skip writing lib_dir to config.toml",
    )
    p.add_argument(
        "--instruments",
        nargs="+",
        default=None,
        metavar="INST",
        help=(
            "Instruments to download PSFs/GDCs for (default: all). "
            "PSF choices: ACSWFC ACSHRC WFC3UV WFC3IR. "
            "GDC choices: ACSWFC ACSHRC WFC3UV."
        ),
    )
    p.add_argument(
        "--no-gdcs",
        action="store_true",
        help="Skip downloading GDC files",
    )
    p.add_argument(
        "--no-psfs",
        action="store_true",
        help="Skip downloading PSF files",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that already exist locally",
    )
    p.add_argument(
        "--no-qso-catalogs",
        action="store_true",
        help="Skip downloading MILLIQUAS v8 and Quaia QSO reference catalogs "
             "(saved to lib_dir/qso_catalogs/; used to vet Gaia qso_candidates "
             "before assigning QSO anchor status)",
    )
    args = p.parse_args()

    lib_dir = Path(args.lib_dir) if args.lib_dir else DEFAULT_LIB_DIR

    if args.instruments:
        requested = {i.upper() for i in args.instruments}
        all_psf = PSF_INSTRUMENTS + list(_OPTIONAL_PSF_ONLY)
        psf_insts = [i for i in all_psf if i in requested]
        gdc_insts = [i for i in GDC_INSTRUMENTS if i in requested]
    else:
        psf_insts = PSF_INSTRUMENTS
        gdc_insts = GDC_INSTRUMENTS

    print("bp3m library setup")
    print(f"  lib_dir  : {lib_dir}")
    print(f"  PSF insts: {', '.join(psf_insts)}")
    print(f"  GDC insts: {', '.join(gdc_insts)}")
    print()

    n_ok = n_skip = n_err = 0

    # ── PSF files ─────────────────────────────────────────────────────────────
    if not args.no_psfs:
        print("Downloading PSF files...")
        for inst in psf_insts:
            url = f"{BASE_URL}/PSFs/STDPSFs/{inst}"
            files = _list_fits(url)
            if not files:
                print(f"  {inst}: no .fits files found at {url}")
                continue
            dest_dir = lib_dir / "STDPSFs" / inst
            dest_dir.mkdir(parents=True, exist_ok=True)
            for file_url in files:
                fname = file_url.rsplit("/", 1)[-1]
                dest = dest_dir / fname
                if dest.exists() and not args.force:
                    n_skip += 1
                    continue
                print(f"  {inst}/{fname}")
                if _download(file_url, dest):
                    n_ok += 1
                else:
                    n_err += 1
        print()

    # ── GDC files ─────────────────────────────────────────────────────────────
    if not args.no_gdcs:
        print("Downloading GDC files...")
        for inst in gdc_insts:
            # ACSWFC GDCs live in a VINTAGE_2005 subdirectory
            if inst == "ACSWFC":
                url = f"{BASE_URL}/GDCs/STDGDCs/{inst}/VINTAGE_2005"
            else:
                url = f"{BASE_URL}/GDCs/STDGDCs/{inst}"
            files = _list_fits(url)
            if not files:
                print(f"  {inst}: no .fits files found at {url}")
                continue
            dest_dir = lib_dir / "STDGDCs" / inst
            dest_dir.mkdir(parents=True, exist_ok=True)
            for file_url in files:
                fname = file_url.rsplit("/", 1)[-1]
                dest = dest_dir / fname
                if dest.exists() and not args.force:
                    n_skip += 1
                    continue
                print(f"  {inst}/{fname}")
                if _download(file_url, dest):
                    n_ok += 1
                else:
                    n_err += 1
        print()

    # ── QSO reference catalogs ────────────────────────────────────────────────
    if not args.no_qso_catalogs:
        print("Downloading QSO reference catalogs...")
        qso_dir = lib_dir / "qso_catalogs"
        qso_dir.mkdir(parents=True, exist_ok=True)

        # Quaia — Gaia DR3 + unWISE photometric QSO catalog (~171 MB FITS)
        quaia_dest = qso_dir / _QUAIA_FILENAME
        if quaia_dest.exists() and not args.force:
            sz = quaia_dest.stat().st_size / 1e6
            print(f"  Quaia: already present ({sz:.0f} MB)")
            n_skip += 1
        else:
            print(f"  Quaia G<20.5 (Storey-Fisher et al. 2024, ~171 MB FITS):")
            if _download_large(_QUAIA_URL, quaia_dest, 'Quaia'):
                sz = quaia_dest.stat().st_size / 1e6
                print(f"  Saved: {quaia_dest} ({sz:.0f} MB)")
                n_ok += 1
            else:
                n_err += 1

        # MILLIQUAS v8 — spectroscopic + photometric QSOs (~40 MB zip → FITS)
        milliquas_dest = qso_dir / _MILLIQUAS_FILENAME
        if milliquas_dest.exists() and not args.force:
            sz = milliquas_dest.stat().st_size / 1e6
            print(f"  MILLIQUAS: already present ({sz:.0f} MB)")
            n_skip += 1
        else:
            print(f"  MILLIQUAS v8 (Flesch 2023, ~40 MB zip → FITS):")
            if _download_and_extract_zip(_MILLIQUAS_URL, milliquas_dest, 'MILLIQUAS'):
                sz = milliquas_dest.stat().st_size / 1e6
                print(f"  Saved: {milliquas_dest} ({sz:.0f} MB)")
                n_ok += 1
            else:
                n_err += 1
        print()

    print(f"Done: {n_ok} downloaded, {n_skip} already present, {n_err} errors.")

    # ── Write config ──────────────────────────────────────────────────────────
    if not args.no_config:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(f'lib_dir = "{lib_dir}"\n')
        print(f"Config written to {CONFIG_FILE}")
        print(f"bp3m will use lib_dir={lib_dir} by default (override with --lib_dir).")

    if n_err > 0:
        sys.exit(1)
