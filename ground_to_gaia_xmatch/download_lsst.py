"""
Download LSST DP2 source and visit-detector tables for a cone on the sky.

Mirrors bp3m.pipeline.download_gaia's role: given (ra, dec, radius), fetch what
the pipeline needs and write it where the LSST adapter expects it, so a new
field becomes one command instead of a portal export plus manual file shuffling.

Authentication
--------------
The Rubin TAP service requires a bearer token — /api/tap/tables returns 401
without one.  Generate one in the RSP (your name -> Security tokens -> new
token, scope `read:tap`) and either:

    export RSP_TOKEN=gt-...
    # or
    echo 'gt-...' > ~/.rsp_token && chmod 600 ~/.rsp_token

The token is read from RSP_TOKEN, then --token-file, then ~/.rsp_token.  It is
never logged or written into the output.

Output
------
Written as IPAC tables under `field_root`, with the names the adapter globs:

    table_dp2.<field>-data.tbl            dp2.Source rows in the cone
    table_dp2.<field>-VisitDetector.tbl   per-(visit, detector) metadata
    table_dp2.<field>-query.json          the ADQL actually issued, for provenance

Queries are async (TAP job), so results larger than the sync row cap come back
in full — the 50,000-row exports from the portal are a UI limit, not the true
source count.  That cuts both ways: a deep-drilling field is genuinely huge.
COSMOS in a 0.25 deg cone returns 4.7 MILLION sources over 717 visits.

Format
------
IPAC is fixed-width text, so astropy must scan every row to size the columns;
at millions of rows the write takes longer than the query and produces a
multi-GB file.  Above FITS_THRESHOLD rows the tables are written as FITS binary
tables instead, which the LSST adapter reads transparently.  FITS rather than
Parquet because astropy writes it natively with no extra dependency.  Smaller
fields stay IPAC so hand-exported portal files remain interchangeable.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

TAP_URL = 'https://data.lsst.cloud/api/tap'

SOURCE_TABLE = 'dp2.Source'
VISIT_DETECTOR_TABLE = 'dp2.VisitDetector'

# The visit-detector cone must be widened relative to the source cone, because
# VisitDetector rows are indexed by detector CENTRE: a detector centred well
# outside the source cone still contributes sources inside it, and an image with
# no metadata row is silently skipped by the adapter.
#
# The margin is ADDITIVE, not multiplicative.  An LSST detector is ~0.23 deg
# across, so its centre can sit up to half a diagonal (~0.16 deg) from any
# source it contributes.  A multiplicative factor fails badly for small cones:
# at r=0.05 a 2x rule gives 0.1 deg and misses 77% of contributing detectors
# (measured on COSMOS), while at r=0.25 the same rule happens to be wide enough.
DETECTOR_HALF_DIAG_DEG = 0.17
VD_RADIUS_FACTOR = 2.0          # retained as a floor for large cones

# Above this many rows, write a FITS binary table rather than IPAC.  IPAC's
# fixed-width text writer must scan every row to size the columns and balloons
# on wide tables; COSMOS (4.7M rows x 40 cols) never finished writing.
FITS_THRESHOLD = 200_000

SOURCE_COLUMNS = [
    'x', 'y', 'xErr', 'yErr', 'ra', 'dec', 'raErr', 'decErr', 'ra_dec_Cov',
    'calibFlux', 'calibFluxErr', 'ixx', 'iyy', 'ixy',
    'ixxPSF', 'iyyPSF', 'ixyPSF',
    'ixxDebiasedPSF', 'iyyDebiasedPSF', 'ixyDebiasedPSF',
    'gaussianFlux', 'gaussianFluxErr',
    'extendedness', 'sizeExtendedness', 'blendedness_abs',
    'blendedness_flag', 'blendedness_flag_noCentroid', 'blendedness_flag_noShape',
    'extendedness_flag', 'sizeExtendedness_flag',
    'footprintArea_value', 'invalidPsfFlag', 'centroid_flag_badError',
    'calib_astrometry_used', 'sky_source',
    'visit', 'detector', 'band', 'physical_filter', 'sourceId',
]

VISIT_DETECTOR_COLUMNS = [
    'band', 'ra', 'dec', 'pixelScale', 'zenithDistance', 'expTime', 'zeroPoint',
    'psfSigma', 'astromOffsetMean', 'astromOffsetStd', 'visitId', 'detector',
    'seeing', 'skyRotation', 'expMidptMJD', 'obsStartMJD',
    'xSize', 'ySize', 'magLim',
]


# ── auth ─────────────────────────────────────────────────────────────────────

def read_token(token_file: str | Path | None = None) -> str:
    """RSP bearer token, from the environment or a file."""
    tok = os.environ.get('RSP_TOKEN', '').strip()
    if tok:
        return tok
    for cand in ([token_file] if token_file else []) + ['~/.rsp_token']:
        p = Path(cand).expanduser()
        if p.is_file():
            tok = p.read_text().strip()
            if tok:
                return tok
    raise RuntimeError(
        'No RSP token found.  Set RSP_TOKEN, pass --token-file, or write the '
        'token to ~/.rsp_token.  Generate one in the RSP under your name -> '
        'Security tokens, with scope read:tap.')


def tap_service(token: str | None = None, url: str = TAP_URL):
    """Authenticated pyvo TAP service."""
    import pyvo
    import requests
    session = requests.Session()
    session.headers['Authorization'] = f'Bearer {token or read_token()}'
    return pyvo.dal.TAPService(url, session=session)


# ── queries ──────────────────────────────────────────────────────────────────

def source_adql(ra: float, dec: float, radius: float,
                columns=None, table: str = SOURCE_TABLE,
                bands=None, max_rows: int | None = None) -> str:
    cols = ', '.join(columns or SOURCE_COLUMNS)
    top = f'TOP {int(max_rows)} ' if max_rows else ''
    where = [f"CONTAINS(POINT('ICRS', ra, dec), "
             f"CIRCLE('ICRS', {ra}, {dec}, {radius}))=1"]
    if bands:
        band_list = ', '.join(f"'{b}'" for b in bands)
        where.append(f'band IN ({band_list})')
    return f"SELECT {top}{cols}\nFROM {table}\nWHERE " + '\n  AND '.join(where)


def visit_detector_adql(ra: float, dec: float, radius: float,
                        columns=None, table: str = VISIT_DETECTOR_TABLE,
                        mjd_range=None) -> str:
    """
    Per-(visit, detector) metadata over a cone.

    Queried by position rather than by `visitId IN (...)`: the visit list is not
    known until the sources come back, and a positional query with a margin also
    picks up detectors whose centre sits just outside the source cone but which
    still contribute sources to it.
    """
    cols = ', '.join(columns or VISIT_DETECTOR_COLUMNS)
    where = [f"CONTAINS(POINT('ICRS', ra, dec), "
             f"CIRCLE('ICRS', {ra}, {dec}, {radius}))=1"]
    if mjd_range:
        lo, hi = mjd_range
        where.append(f'(expMidptMJD >= {lo} AND expMidptMJD <= {hi})')
    return f"SELECT {cols}\nFROM {table}\nWHERE " + '\n  AND '.join(where)


def run_async(service, adql: str, poll: float = 10.0, timeout: float = 3600.0,
              quiet: bool = False):
    """
    Submit an async TAP job and wait.

    Async rather than sync so results are not silently truncated at the sync row
    cap — the reason portal exports come back as exactly 50,000 rows.
    """
    job = service.submit_job(adql)
    job.run()
    t0 = time.time()
    while job.phase in ('QUEUED', 'EXECUTING', 'PENDING', 'UNKNOWN'):
        if time.time() - t0 > timeout:
            job.abort()
            raise TimeoutError(f'TAP job exceeded {timeout:.0f}s (phase {job.phase})')
        if not quiet:
            print(f'    ... {job.phase} ({time.time()-t0:.0f}s)', flush=True)
        time.sleep(poll)
    if job.phase != 'COMPLETED':
        raise RuntimeError(f'TAP job ended in phase {job.phase}: '
                           f'{getattr(job, "errorsummary", None)}')
    return job.fetch_result().to_table()


def _write_table(t, path: Path, quiet: bool = False) -> Path:
    """
    Write an astropy table, choosing the format by size.

    Returns the path actually written, which may have a .parquet suffix.
    """
    if len(t) > FITS_THRESHOLD:
        fp = path.with_suffix('.fits')
        if not quiet:
            print(f'  {len(t)} rows > {FITS_THRESHOLD} — writing FITS '
                  f'({fp.name}); IPAC would take longer than the query',
                  flush=True)
        # Object/str columns need an explicit width for FITS; astropy handles
        # this, but bytes-vs-str round-tripping is the usual failure, so the
        # adapter decodes on read.
        t.write(fp, format='fits', overwrite=True)
        return fp
    t.write(path, format='ipac', overwrite=True)
    return path


# ── driver ───────────────────────────────────────────────────────────────────

def download_lsst(ra: float, dec: float, radius: float,
                  field_root: str | Path, field_name: str,
                  bands=None, max_rows=None, token_file=None,
                  source_table: str = SOURCE_TABLE,
                  vd_table: str = VISIT_DETECTOR_TABLE,
                  vd_radius_factor: float = VD_RADIUS_FACTOR,
                  mjd_range=None,
                  force: bool = False, quiet: bool = False):
    """
    Fetch DP2 sources in a cone plus the matching visit-detector metadata.

    Returns (source_path, visit_detector_path).
    """
    field_root = Path(field_root)
    field_root.mkdir(parents=True, exist_ok=True)
    src_path = field_root / f'table_dp2.{field_name}-data.tbl'
    vd_path = field_root / f'table_dp2.{field_name}-VisitDetector.tbl'

    if src_path.exists() and vd_path.exists() and not force:
        if not quiet:
            print(f'  already present (use force=True to redownload):\n'
                  f'    {src_path.name}\n    {vd_path.name}')
        return src_path, vd_path

    svc = tap_service(read_token(token_file))

    q_src = source_adql(ra, dec, radius, table=source_table,
                        bands=bands, max_rows=max_rows)
    if not quiet:
        print(f'  querying {source_table}: cone ({ra}, {dec}) r={radius} deg'
              + (f', bands {bands}' if bands else ''), flush=True)
    src = run_async(svc, q_src, quiet=quiet)
    if len(src) == 0:
        raise RuntimeError('no sources returned — check ra/dec/radius and that '
                           'the field is inside the DP2 footprint')
    if not quiet:
        print(f'  -> {len(src)} sources, {len(set(src["visit"]))} visits, '
              f'{len(set(src["detector"]))} detectors, '
              f'bands {sorted(set(str(b) for b in src["band"]))}', flush=True)

    # Whichever is larger: the additive detector margin (correct for small
    # cones) or the legacy multiplicative floor (already generous for large).
    vd_radius = max(radius + DETECTOR_HALF_DIAG_DEG, radius * vd_radius_factor)
    q_vd = visit_detector_adql(ra, dec, vd_radius, table=vd_table,
                               mjd_range=mjd_range)
    if not quiet:
        print(f'  querying {vd_table}: cone r={vd_radius:.3f} deg '
              f'(source cone + {DETECTOR_HALF_DIAG_DEG} deg detector margin)',
              flush=True)
    vd = run_async(svc, q_vd, quiet=quiet)
    if not quiet:
        print(f'  -> {len(vd)} visit-detector rows', flush=True)

    # Every (visit, detector) that produced sources must have a metadata row,
    # or the adapter silently skips that image.  Check rather than assume.
    want = {(int(v), int(d)) for v, d in zip(src['visit'], src['detector'])}
    have = {(int(v), int(d)) for v, d in zip(vd['visitId'], vd['detector'])}
    missing = want - have
    if missing:
        print(f'  WARNING: {len(missing)} of {len(want)} (visit, detector) pairs '
              f'with sources have NO metadata row and will be skipped by the '
              f'adapter. Increase DETECTOR_HALF_DIAG_DEG (currently '
              f'{DETECTOR_HALF_DIAG_DEG}) — the metadata cone was '
              f'{vd_radius:.3f} deg.')
        for v, d in sorted(missing)[:10]:
            print(f'    visit {v} detector {d}')
    elif not quiet:
        print(f'  all {len(want)} (visit, detector) pairs have metadata')

    src_path = _write_table(src, src_path, quiet=quiet)
    vd_path = _write_table(vd, vd_path, quiet=quiet)
    (field_root / f'table_dp2.{field_name}-query.json').write_text(json.dumps(
        {'ra': ra, 'dec': dec, 'radius_deg': radius, 'bands': bands,
         'source_table': source_table, 'visit_detector_table': vd_table,
         'vd_radius_deg': radius * vd_radius_factor, 'mjd_range': mjd_range,
         'n_sources': len(src), 'n_visit_detector': len(vd),
         'source_adql': q_src, 'visit_detector_adql': q_vd}, indent=2))
    if not quiet:
        print(f'  wrote {src_path.name} and {vd_path.name}')
    return src_path, vd_path


__all__ = ['download_lsst', 'source_adql', 'visit_detector_adql', 'tap_service',
           'read_token', 'run_async', 'TAP_URL', 'SOURCE_TABLE',
           'VISIT_DETECTOR_TABLE', 'VD_RADIUS_FACTOR', 'DETECTOR_HALF_DIAG_DEG',
           'SOURCE_COLUMNS', 'VISIT_DETECTOR_COLUMNS']
