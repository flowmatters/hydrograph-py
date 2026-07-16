import numpy as np
import pandas as pd
import pytest

from hydrograph.server import HydrographServerDataset, open_server_dataset

DS_URL = 'https://staging.hydrograph.io/api/datasets/joel/test-ds'

TAGS_RESPONSE = [
    {'name': 'location', 'valueType': 'any'},
    {'name': 'variable', 'valueType': 'any'},
]

TAG_VALUES = {
    'location': ['Hamilton', 'Kerang', None],
    'variable': ['Rain', 'Temperature'],
}

# Two hours of hourly data starting 2025-07-10 00:00 UTC
DATES_MS = [1752105600000, 1752109200000, 1752112800000]

def timeseries_response(params):
    if params and params.get('variable') == 'nothing':
        return {'indexes': [], 'timeseries': []}
    return {
        'indexes': [DATES_MS],
        'timeseries': [{
            'tags': {'location': 'Hamilton', 'variable': 'Rain'},
            'attributes': {'source': 'test'},
            'index': 0,
            'values': [0, None, 1.4],
            'title': 'Rain',
            'units': 'mm',
            'noData': -9999,
        }],
    }

def tables_response(params):
    tables = [
        {
            'tags': {'variable': 'soil-moisture'},
            'index': ['Site A', 'Site B'],
            'data': {'current-soil-moisture': [100, None]},
            'indexCol': 'site',
        },
        {
            'tags': {'variable': 'rainfall-summary'},
            'index': [0, 1],
            'data': {'total': [10.0, 12.5]},
            'indexCol': None,
        },
    ]
    if params:
        tables = [t for t in tables
                  if all(t['tags'].get(k) == v for k, v in params.items())]
    return {'tables': tables}

class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload

class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, auth=None):
        self.calls.append((url, params))
        path = url[len(DS_URL) + 1:]
        if path == 'tags':
            return FakeResponse(TAGS_RESPONSE)
        if path.startswith('tags/'):
            tag = path[len('tags/'):]
            if tag not in TAG_VALUES:
                return FakeResponse(None, 404)
            return FakeResponse(TAG_VALUES[tag])
        if path == 'timeseries':
            return FakeResponse(timeseries_response(params))
        if path == 'tables':
            return FakeResponse(tables_response(params))
        return FakeResponse(None, 404)

@pytest.fixture
def ds():
    return HydrographServerDataset(DS_URL, session=FakeSession())

def test_url_parsing(ds):
    assert ds.owner == 'joel'
    assert ds.dataset == 'test-ds'
    assert ds.url == DS_URL

def test_tags(ds):
    assert ds.tags() == {'location', 'variable'}
    # cached - only one request
    ds.tags()
    assert len(ds._session.calls) == 1

def test_tag_values(ds):
    assert ds.tag_values('variable') == {'Rain', 'Temperature'}
    ds.tag_values('variable')
    assert len(ds._session.calls) == 1

def test_get_timeseries(ds):
    result = ds.get_timeseries(location='Hamilton', variable='Rain')
    assert len(result) == 1
    ts = result[0]
    assert list(ts.columns) == ['Rain']
    assert ts.index[0] == pd.Timestamp('2025-07-10 00:00:00')
    assert ts['Rain'].iloc[0] == 0
    assert np.isnan(ts['Rain'].iloc[1])  # null value
    assert ts['Rain'].iloc[2] == 1.4
    assert ts.attrs['units'] == 'mm'
    assert ts.attrs['tags'] == {'location': 'Hamilton', 'variable': 'Rain'}

def test_get_timeseries_no_data_value():
    session = FakeSession()
    ds = HydrographServerDataset(DS_URL, session=session)
    # inject noData value into the response
    original = timeseries_response(None)
    original['timeseries'][0]['values'] = [0, -9999, 1.4]
    session.get = lambda url, params=None, auth=None: FakeResponse(
        original if url.endswith('timeseries') else TAGS_RESPONSE)
    ts = ds.get_timeseries()[0]
    assert np.isnan(ts['Rain'].iloc[1])

def test_get_timeseries_empty(ds):
    assert ds.get_timeseries(variable='nothing') == []

def test_get_tables(ds):
    tables = ds.get_tables()
    assert len(tables) == 2
    tbl = tables[0]
    assert list(tbl.index) == ['Site A', 'Site B']
    assert tbl.index.name == 'site'
    assert tbl['current-soil-moisture'].iloc[0] == 100

def test_get_table_single(ds):
    tbl = ds.get_table(variable='soil-moisture')
    assert list(tbl.columns) == ['current-soil-moisture']

def test_get_table_multiple_raises(ds):
    with pytest.raises(Exception):
        ds.get_table()

def test_get_table_none_raises(ds):
    with pytest.raises(Exception):
        ds.get_table(variable='no-such-thing')

def test_match_list_expansion(ds):
    ds.match('tables', variable=['soil-moisture', 'rainfall-summary'])
    table_calls = [c for c in ds._session.calls if c[0].endswith('tables')]
    assert len(table_calls) == 2
    assert table_calls[0][1] == {'variable': 'soil-moisture'}
    assert table_calls[1][1] == {'variable': 'rainfall-summary'}

def test_match_table(ds):
    result = ds.match_table('tables')
    assert len(result) == 2
    assert set(result['variable']) == {'soil-moisture', 'rainfall-summary'}

def test_unique_tag_groups(ds):
    groups = ds.unique_tag_groups('tables')
    assert ('variable',) in groups

def test_read_only(ds):
    with pytest.raises(Exception, match='read-only'):
        ds.add_table(pd.DataFrame({'a': [1]}))
    with pytest.raises(Exception, match='read-only'):
        ds.add_metadata('key', 'value')

def test_open_server_dataset():
    ds = open_server_dataset(DS_URL, session=FakeSession())
    assert isinstance(ds, HydrographServerDataset)
    assert ds.auth is None
