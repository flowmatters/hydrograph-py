from hydrograph.minify import increment_identifier,minify_name
import pytest

INCREMENT_TESTS=[
    ('' ,'a'),
    ('a', 'b'),
    ('j', 'k'),
    ('z', 'aa'),
    ('aa','ab'),
    ('az','ba'),
    ('zz','aaa')
]

@pytest.mark.parametrize('test',INCREMENT_TESTS)
def test_increment(test):
    orig, incr = test

    assert increment_identifier(orig)==incr


def test_minify():
    existing = {'a','b','f'}

    assert minify_name('apples',existing)=='c'
    assert minify_name('candy',existing)=='c'
    assert minify_name('fruit',existing)=='c'
