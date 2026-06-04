"""bp3m-setup: Download HST PSF and GDC library files from STScI."""

import argparse
import re
import sys
from pathlib import Path
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError

BASE_URL = "https://www.stsci.edu/~jayander/HST1PASS/LIB"
CONFIG_FILE = Path.home() / ".bp3m" / "config.toml"
DEFAULT_LIB_DIR = Path.home() / ".bp3m" / "lib"

# WFC3IR has PSFs but no GDCs
PSF_INSTRUMENTS = ["ACSWFC", "ACSHRC", "WFC3UV", "WFC3IR"]
GDC_INSTRUMENTS = ["ACSWFC", "ACSHRC", "WFC3UV"]


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


def main():
    p = argparse.ArgumentParser(
        description=(
            "Download HST PSF and geometric distortion correction (GDC) library "
            "files for bp3m from STScI (https://www.stsci.edu/~jayander/HST1PASS/LIB). "
            "Saves the lib_dir path to ~/.bp3m/config.toml so --lib_dir is optional "
            "when running bp3m."
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
        help="Skip writing lib_dir to ~/.bp3m/config.toml",
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
    args = p.parse_args()

    lib_dir = Path(args.lib_dir) if args.lib_dir else DEFAULT_LIB_DIR

    if args.instruments:
        requested = {i.upper() for i in args.instruments}
        psf_insts = [i for i in PSF_INSTRUMENTS if i in requested]
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

    print(f"Done: {n_ok} downloaded, {n_skip} already present, {n_err} errors.")

    # ── Write config ──────────────────────────────────────────────────────────
    if not args.no_config:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(f'lib_dir = "{lib_dir}"\n')
        print(f"Config written to {CONFIG_FILE}")
        print(f"bp3m will use lib_dir={lib_dir} by default (override with --lib_dir).")

    if n_err > 0:
        sys.exit(1)
