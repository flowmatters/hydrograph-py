import pandas as pd

import hydrograph as hg

def make_dataset(tmp_path):
    ds = hg.open_dataset(str(tmp_path / 'ds'), 'w')
    ds.add_table(pd.DataFrame({'a': [1]}), variable='Temperature', location='X')
    ds.add_table(pd.DataFrame({'a': [2]}), variable='Humidity', location='X')
    ds.add_table(pd.DataFrame({'a': [3]}), variable='Rain', location='Y')
    return ds

def test_match_single_tag_value(tmp_path):
    ds = make_dataset(tmp_path)
    m = ds.match('tables', variable='Temperature')
    assert len(m) == 1
    assert m[0]['tags']['location'] == 'X'

def test_match_list_of_tag_values(tmp_path):
    # A list of values for a tag matches any of the values
    ds = make_dataset(tmp_path)
    m = ds.match('tables', variable=['Temperature', 'Humidity'])
    assert len(m) == 2
    assert {r['tags']['variable'] for r in m} == {'Temperature', 'Humidity'}

def test_get_tables_list_of_tag_values(tmp_path):
    ds = make_dataset(tmp_path)
    tables = ds.get_tables(variable=['Temperature', 'Rain'])
    assert len(tables) == 2

def test_match_list_combined_with_scalar(tmp_path):
    ds = make_dataset(tmp_path)
    m = ds.match('tables', variable=['Temperature', 'Rain'], location='X')
    assert len(m) == 1
    assert m[0]['tags']['variable'] == 'Temperature'
