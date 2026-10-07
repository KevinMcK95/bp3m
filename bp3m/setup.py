"""bp3m-setup: Download HST and JWST PSF/GDC library files and QSO reference catalogs.

JWST support (``--telescope JWST|both``) is adapted from Liwen Chen's bp3m fork
(2026-06 to 2026-09: ``bp3m/setup.py`` there), which first scraped Jay Anderson's
JWST1PASS library, normalised the inconsistent NIRCam filename token order and
rejected non-FITS download bodies.  The on-disk layout differs from her fork: JWST
files are flattened into the same per-camera directories bp3m uses for HST
(``STDPSFs/NIRCAM/STDPSF_NRCA1_F200W.fits`` next to ``STDPSFs/ACSWFC/...``), so one
``find_psf``/``find_gdc`` lookup serves both telescopes.
"""

import argparse
import re
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError

BASE_URL = "https://www.stsci.edu/~jayander/HST1PASS/LIB"
JWST_BASE_URL = "https://www.stsci.edu/~jayander/JWST1PASS/LIB"

# ── JWST library ─────────────────────────────────────────────────────────────
# Server layout (2026-10):  PSFs/STDPSFs/{NIRCam/{SWC/<filter>/,LWC/},NIRISS/,MIRI/}
# and the same under GDCs/STDGDCs/.  NIRCam SWC files sit in one subdirectory per
# filter, LWC files are flat; NIRISS and MIRI are flat and carry a PLEASE_README.txt
# with the citation (Libralato et al. 2023, NIRISS ApJ / MIRI PASP) that is saved
# alongside the tables.  Locally everything is flattened into one directory per
# instrument, matching the HST layout (STDPSFs/ACSWFC/, STDGDCs/WFC3UV/, ...):
#   STDPSFs/NIRCAM/STDPSF_<NRCA1..NRCB4|NRCAL|NRCBL>_<filter>.fits
#   STDPSFs/NIRISS/STDPSF_NIRISS_<filter>.fits,  STDPSFs/MIRI/STDPSF_MIRI_<filter>.fits
# GDC availability (2026-10): NIRCam SWC all 9 filters x 8 detectors, LWC F277W only,
# NIRISS 12 filters, MIRI F560W/F770W/F1000W.  NIRCam GDC tables are ~76 MB each.
JWST_INSTRUMENTS = ["NIRCAM", "NIRISS", "MIRI"]
_JWST_SERVER_DIR = {"NIRCAM": "NIRCam", "NIRISS": "NIRISS", "MIRI": "MIRI"}
_NIRCAM_CHANNELS = ["SWC", "LWC"]

# STScI is not consistent about the filename token order for NIRCam tables: most
# are STD{X}_{detector}_{filter}.fits but some filters (observed: F210M and F070W
# GDCs) are published as STD{X}_{filter}_{detector}.fits.  Files are always SAVED
# detector-first, the form the library lookup constructs.  (Liwen Chen's fix; a
# template-based downloader had saved HTML 404 pages as .fits for those filters.)
_NRC_DET_RE = re.compile(r"(NRC[AB](?:L|[1-4]))", re.IGNORECASE)
_FILT_RE = re.compile(r"(F\d{2,4}[A-Z]{1,2})", re.IGNORECASE)


def _canonical_nircam_name(kind: str, basename: str) -> str:
    """Reorder a scraped NIRCam filename into STD{kind}_{detector}_{filter}.fits
    (kind 'PSF' or 'GDC'); unchanged if both tokens cannot be parsed."""
    det_m = _NRC_DET_RE.search(basename)
    filt_m = _FILT_RE.search(basename)
    if not det_m or not filt_m:
        return basename
    return f"STD{kind}_{det_m.group(1).upper()}_{filt_m.group(1).upper()}.fits"


def _valid_fits(path: Path) -> bool:
    """True if *path* exists and starts with the FITS 'SIMPLE' card (an HTML error page
    saved as .fits, or a truncated transfer, does not)."""
    try:
        if path.stat().st_size < 2880:
            return False
        with open(path, "rb") as f:
            return f.read(6) == b"SIMPLE"
    except OSError:
        return False

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


def _list_dirs(url: str) -> list:
    """Return subdirectory names linked from an STScI directory listing (skips the
    parent-directory link and the column-sort query links)."""
    try:
        with urlopen(url, timeout=30) as r:
            html = r.read().decode("utf-8", errors="replace")
        return re.findall(r'href="([^"?/][^"]*)/"', html)
    except URLError as e:
        print(f"  WARNING: could not list {url}: {e}")
        return []


def _list_text(url: str) -> list:
    """Return full URLs of .txt files (READMEs) in an STScI directory listing."""
    try:
        with urlopen(url, timeout=30) as r:
            html = r.read().decode("utf-8", errors="replace")
        names = re.findall(r'href="([^"]+\.txt)"', html, re.IGNORECASE)
        base = url.rstrip("/")
        return [f"{base}/{n}" for n in names]
    except URLError:
        return []


def _download(url: str, dest: Path) -> bool:
    """Download url to dest. Returns True on success.

    A ``.fits`` destination must start with the FITS 'SIMPLE' card; anything else
    (an error page served with status 200, a truncated body) is discarded instead
    of being saved under a .fits name.
    """
    tmp = dest.with_suffix(".tmp")
    try:
        urlretrieve(url, str(tmp))
        if dest.suffix.lower() == ".fits" and not _valid_fits(tmp):
            with open(tmp, "rb") as f:
                head = f.read(6)
            raise ValueError(f"downloaded content is not a FITS file (starts with {head!r})")
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


def _download_jwst_group(lib_dir: Path, kind: str, instruments: list, force: bool) -> tuple:
    """Download one JWST library (*kind* 'PSF' or 'GDC') for *instruments*.

    Walks the server layout (NIRCam SWC per-filter subdirectories, LWC flat, NIRISS
    and MIRI flat) and flattens everything into ``lib_dir/STD{kind}s/<INSTRUMENT>/``
    under canonical detector-first names.  Existing files are kept unless *force*
    or they fail the FITS check (a previously saved error page is re-fetched).
    Returns (n_ok, n_skip, n_err).
    """
    label = "STDPSFs" if kind == "PSF" else "STDGDCs"
    top = "PSFs" if kind == "PSF" else "GDCs"
    n_ok = n_skip = n_err = 0
    for inst in instruments:
        server_dir = _JWST_SERVER_DIR[inst]
        inst_url = f"{JWST_BASE_URL}/{top}/{label}/{server_dir}"
        if inst == "NIRCAM":
            listings = []
            for channel in _NIRCAM_CHANNELS:
                chan_url = f"{inst_url}/{channel}"
                filters = _list_dirs(chan_url)
                listings += [f"{chan_url}/{f}" for f in filters] if filters else [chan_url]
        else:
            listings = [inst_url]
        dest_dir = lib_dir / label / inst
        dest_dir.mkdir(parents=True, exist_ok=True)
        n_found = 0
        for list_url in listings:
            for file_url in _list_fits(list_url):
                n_found += 1
                raw = file_url.rsplit("/", 1)[-1]
                fname = _canonical_nircam_name(kind, raw) if inst == "NIRCAM" else raw
                dest = dest_dir / fname
                if dest.exists() and not force and _valid_fits(dest):
                    n_skip += 1
                    continue
                print(f"  {inst}/{fname}" + (f"  (server name {raw})" if raw != fname else ""))
                if _download(file_url, dest):
                    n_ok += 1
                else:
                    n_err += 1
            for txt_url in _list_text(list_url):
                # citation / release notes (Libralato et al. 2023 for NIRISS and MIRI)
                tdest = dest_dir / f"README_{label}_{txt_url.rsplit('/', 1)[-1]}"
                if not tdest.exists() or force:
                    _download(txt_url, tdest)
        if n_found == 0:
            print(f"  {inst}: no .fits files found under {inst_url}")
    return n_ok, n_skip, n_err


def _write_config(lib_dir: Path) -> None:
    """Set lib_dir in config.toml, keeping any other keys (e.g. pos_corr_model)."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if CONFIG_FILE.exists():
        lines = [l for l in CONFIG_FILE.read_text().splitlines() if not l.strip().startswith("lib_dir")]
    CONFIG_FILE.write_text("\n".join([f'lib_dir = "{lib_dir}"'] + lines).rstrip("\n") + "\n")


def main():
    p = argparse.ArgumentParser(
        description=(
            "Download HST (and optionally JWST) PSF and geometric distortion "
            "correction (GDC) library files for bp3m from STScI "
            "(https://www.stsci.edu/~jayander/HST1PASS/LIB and .../JWST1PASS/LIB). "
            "Saves the lib_dir path to config.toml so --lib_dir is optional "
            "when running bp3m. Config location defaults to ~/.bp3m/ but can be "
            "overridden by setting the BP3M_HOME environment variable."
        )
    )
    p.add_argument(
        "--telescope",
        choices=["HST", "JWST", "both"],
        default="HST",
        help=(
            "Which telescope's library to download (default: HST). JWST NIRCam GDC "
            "tables are ~76 MB each (~6 GB for all filters) -- opt in with "
            "--telescope JWST or both."
        ),
    )
    p.add_argument(
        "--jwst-instruments",
        nargs="+",
        default=None,
        metavar="INST",
        help="JWST instruments to download (default: all). Choices: NIRCAM NIRISS MIRI.",
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
    do_hst = args.telescope in ("HST", "both")
    do_jwst = args.telescope in ("JWST", "both")

    if args.instruments:
        requested = {i.upper() for i in args.instruments}
        all_psf = PSF_INSTRUMENTS + list(_OPTIONAL_PSF_ONLY)
        psf_insts = [i for i in all_psf if i in requested]
        gdc_insts = [i for i in GDC_INSTRUMENTS if i in requested]
    else:
        psf_insts = PSF_INSTRUMENTS
        gdc_insts = GDC_INSTRUMENTS
    if not do_hst:
        psf_insts = gdc_insts = []
    jwst_insts = []
    if do_jwst:
        req = {i.upper() for i in (args.jwst_instruments or JWST_INSTRUMENTS)}
        jwst_insts = [i for i in JWST_INSTRUMENTS if i in req]
        unknown = req - set(JWST_INSTRUMENTS)
        if unknown:
            print(f"  WARNING: unknown JWST instrument(s) ignored: {', '.join(sorted(unknown))}")

    print("bp3m library setup")
    print(f"  lib_dir  : {lib_dir}")
    print(f"  telescope: {args.telescope}")
    if do_hst:
        print(f"  HST PSF insts: {', '.join(psf_insts)}")
        print(f"  HST GDC insts: {', '.join(gdc_insts)}")
    if do_jwst:
        print(f"  JWST insts   : {', '.join(jwst_insts)}")
    print()

    n_ok = n_skip = n_err = 0

    # ── PSF files ─────────────────────────────────────────────────────────────
    if not args.no_psfs and do_hst:
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
    if not args.no_gdcs and do_hst:
        print("Downloading GDC files...")
        for inst in gdc_insts:
            url = f"{BASE_URL}/GDCs/STDGDCs/{inst}"
            files = _list_fits(url)
            if not files:
                print(f"  {inst}: no .fits files found at {url}")
                continue
            dest_dir = lib_dir / "STDGDCs" / inst
            dest_dir.mkdir(parents=True, exist_ok=True)
            # Only the top-level listing: the STDGDC_OFFICIAL_JFRAME_* tables (which
            # pypass.io.find_gdc prefers) plus the plain STDGDC_<det>_<filt> files for
            # filters that have no OFFICIAL table (ACS/WFC F775W).  Never a VINTAGE_*
            # subdirectory: ACS/WFC ran on VINTAGE_2005 by mistake until 2026-09-24.
            names = {u.rsplit("/", 1)[-1]: u for u in files}
            keep = {}
            for n, u in names.items():
                if "STDGDC_OFFICIAL_JFRAME_" in n:
                    keep[n] = u
                elif n.startswith("STDGDC_") and "OFFICIAL" not in n:
                    if n.replace("STDGDC_", "STDGDC_OFFICIAL_JFRAME_", 1) not in names:
                        keep[n] = u
            files = sorted(keep.items())
            for fname, file_url in files:
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

    # ── JWST PSF / GDC files ──────────────────────────────────────────────────
    if do_jwst and jwst_insts:
        for kind, skip in (("PSF", args.no_psfs), ("GDC", args.no_gdcs)):
            if skip:
                continue
            print(f"Downloading JWST {kind} files...")
            ok, sk, er = _download_jwst_group(lib_dir, kind, jwst_insts, args.force)
            n_ok += ok; n_skip += sk; n_err += er
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
        _write_config(lib_dir)
        print(f"Config written to {CONFIG_FILE}")
        print(f"bp3m will use lib_dir={lib_dir} by default (override with --lib_dir).")

    if n_err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
