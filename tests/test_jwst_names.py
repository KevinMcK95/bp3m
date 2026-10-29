"""bp3m.jwst_links.parse_jwst_name: activity ids end in base 36 (2026-10-29: '0210b' frames were never linked)."""
from bp3m.jwst_links import parse_jwst_name


def test_alphanumeric_activity():
    d = parse_jwst_name('jw01072001001_0210b_00001_nrcb1')
    assert d is not None and d['visit_id'] == 'jw01072001001' and d['exposure'] == 'jw01072001001_0210b_00001' and d['module'] == 'NRCB'
    assert parse_jwst_name('JW01072001001_0210B_00002_NRCB1')['exposure'] == 'jw01072001001_0210b_00002'


def test_numeric_and_non_jwst():
    assert parse_jwst_name('jw01227002001_02103_00001_nrcalong')['module'] == 'NRCA'
    assert parse_jwst_name('idan10qxq') is None
