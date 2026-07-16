'''
Read-only client for Hydrograph server-side datasets.

Server-side datasets share the tag/tag-value concepts of static (index.json)
datasets, but are queried through the Hydrograph REST API:

  GET <url>/tags                 -> [{"name":..., "valueType":...}, ...]
  GET <url>/tags/<tag>           -> [value, ...]
  GET <url>/tables?tag=value     -> {"tables":[{tags,index,data,indexCol},...]}
  GET <url>/timeseries?tag=value -> {"indexes":[[ms,...],...],
                                     "timeseries":[{tags,attributes,index,
                                                    values,title,units,noData},...]}

The starting point is a dataset URL such as
https://staging.hydrograph.io/api/datasets/joel/agvic-mait-service
'''
import logging
from itertools import product

import numpy as np
import pandas as pd
import requests

from .general import HydrographDataset, DEFAULT_OPTIONS, MODE_READ, API_URL

logger = logging.getLogger('hydrograph')

class HydrographServerDataset(HydrographDataset):
  '''
  Read-only access to a Hydrograph server-side dataset.

  Mirrors the query API of HydrographDataset (tags, tag_values, match,
  get_table, get_tables, get_timeseries, get_coverages) but backed by the
  Hydrograph REST API rather than an index.json manifest.

  Notes:
  * Time series indexes are millisecond epoch timestamps; they are converted
    to naive UTC datetimes.
  * Tag values of None cannot be used as query filters (they are dropped from
    the request query string).
  * Tag and tag-value lookups are cached; call clear_cache() to refresh.
  '''
  def __init__(self,url,auth=None,session=None,options=DEFAULT_OPTIONS,**kwargs):
    self.url = url.rstrip('/')
    self.path = self.url
    parts = self.url.split('/')
    self.dataset = parts[-1]
    self.owner = parts[-2]
    self.mode = MODE_READ
    self.is_remote = True
    self.options = DEFAULT_OPTIONS.copy()
    self.options.update(options)
    self.options.update(kwargs)
    if auth is None:
      self.auth = None
    else:
      self.auth = requests.auth.HTTPBasicAuth(auth[0], auth[1])
    self._session = session if session is not None else requests
    self._rewrite = False
    self.hosting = False
    self.port = None
    self.host_process = None
    self._tags_cache = None
    self._tag_values_cache = {}

  def _get_json(self,path,params=None):
    url = '%s/%s'%(self.url,path)
    r = self._session.get(url,params=params,auth=self.auth)
    if r.status_code != 200:
      raise Exception('Request failed (%d): %s'%(r.status_code,url))
    return r.json()

  def clear_cache(self):
    self._tags_cache = None
    self._tag_values_cache = {}

  def require_writable(self):
    raise Exception('Hydrograph server datasets are read-only')

  def tags(self,datatype=None,**tags):
    '''
    Tag names used in this dataset.

    With no arguments, uses the server's tags endpoint (covers the whole
    dataset). With a datatype and/or filter tags, falls back to querying
    matching records like the static-dataset API.
    '''
    if datatype is None and not tags:
      if self._tags_cache is None:
        self._tags_cache = self._get_json('tags')
      return set(t['name'] for t in self._tags_cache)
    return super().tags(datatype or 'tables',**tags)

  def tag_values(self,tag,datatype=None,**tags):
    '''
    Values of a tag across the dataset (server tags endpoint), or across
    matching records when a datatype and/or filter tags are given.
    '''
    if datatype is None and not tags:
      if tag not in self._tag_values_cache:
        self._tag_values_cache[tag] = self._get_json('tags/%s'%tag)
      return set(self._tag_values_cache[tag])
    return super().tag_values(tag,datatype,**tags)

  def match(self,datatype='tables',**tags):
    '''
    Records matching the given tags, as returned by the server.

    Table records have keys: tags, index, data, indexCol.
    Time series records have keys: tags, attributes, values, title, units,
    noData and _dates (the resolved millisecond timestamp index).
    Coverages are table records with a geometry column.

    A list of values for a tag matches any of the values (one request per
    combination).
    '''
    result = []
    for query in self._expand_tag_queries(tags):
      result += self._match_single(datatype,query)
    return result

  def _expand_tag_queries(self,tags):
    keys = list(tags.keys())
    values = [v if isinstance(v,list) else [v] for v in tags.values()]
    return [dict(zip(keys,combo)) for combo in product(*values)]

  def _match_single(self,datatype,tags):
    if datatype == 'tables':
      return self._get_json('tables',params=tags).get('tables',[])
    if datatype == 'timeseries':
      response = self._get_json('timeseries',params=tags)
      indexes = response.get('indexes',[])
      records = []
      for ts in response.get('timeseries',[]):
        record = dict(ts)
        record['_dates'] = indexes[record.pop('index')]
        records.append(record)
      return records
    if datatype == 'coverages':
      tables = self._get_json('tables',params=tags).get('tables',[])
      return [t for t in tables if 'geometry' in (t.get('data') or {})]
    return []

  def match_table(self,datatype='tables',**tags):
    result = self.match(datatype,**tags)
    return pd.DataFrame([r['tags'] for r in result])

  def _table_to_dataframe(self,record):
    df = pd.DataFrame(record.get('data') or {})
    index = record.get('index')
    index_col = record.get('indexCol')
    if index is not None and len(index) == len(df):
      df.index = pd.Index(index,name=index_col or None)
    elif index_col and index_col in df.columns:
      df = df.set_index(index_col)
    df.attrs['tags'] = record.get('tags',{})
    return df

  def _timeseries_to_dataframe(self,record):
    idx = pd.to_datetime(record.get('_dates',[]),unit='ms')
    values = pd.to_numeric(pd.Series(record.get('values',[]),dtype=object),errors='coerce')
    no_data = record.get('noData')
    if no_data is not None:
      values = values.where(values != no_data)
    name = record.get('title') or 'value'
    df = pd.DataFrame({name:values.values},index=idx)
    df.attrs['tags'] = record.get('tags',{})
    df.attrs['attributes'] = record.get('attributes',{})
    df.attrs['units'] = record.get('units')
    df.attrs['title'] = record.get('title')
    return df

  def _coverage_to_geodataframe(self,record):
    import json
    import geopandas as gpd
    from shapely.geometry import shape
    df = self._table_to_dataframe(record)
    geometry = [shape(json.loads(g)) if isinstance(g,str) else shape(g)
                for g in df['geometry']]
    gdf = gpd.GeoDataFrame(df.drop(columns=['geometry']),geometry=geometry)
    gdf.attrs['tags'] = record.get('tags',{})
    return gdf

  def get_tables(self,**tags):
    result = []
    for record in self.match('tables',**tags):
      if 'geometry' in (record.get('data') or {}):
        try:
          result.append(self._coverage_to_geodataframe(record))
          continue
        except ImportError:
          logger.warning('geopandas not available, returning coverage as plain DataFrame')
      result.append(self._table_to_dataframe(record))
    return result

  def get_timeseries(self,**tags):
    return [self._timeseries_to_dataframe(r) for r in self.match('timeseries',**tags)]

  def get_coverages(self,**tags):
    return [self._coverage_to_geodataframe(r) for r in self.match('coverages',**tags)]

class APIDataSet(HydrographServerDataset):
  '''
  Deprecated: use open_server_dataset / HydrographServerDataset instead.
  '''
  def __init__(self,name,url_base=API_URL,owner='joel'):
    super().__init__(url_base + owner + '/' + name)

def open_server_dataset(url,auth=None,**kwargs) -> HydrographServerDataset:
  '''
  Open a Hydrograph server-side dataset from its URL, eg
  https://staging.hydrograph.io/api/datasets/joel/agvic-mait-service

  auth is an optional (user,password) tuple.
  '''
  assert isinstance(url,str)
  return HydrographServerDataset(url,auth=auth,**kwargs)
