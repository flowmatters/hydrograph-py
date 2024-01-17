import atexit
import os
from io import StringIO
import shutil
from collections import OrderedDict
import logging
import json
import subprocess
import pandas as pd
import numpy as np
from glob import glob
import tempfile
#from json import encoder
from .minify import minify_geojson
from time import sleep
import requests

logger = logging.getLogger('hydrograph')

INDEX_FN='index.json'
FILE_PREFIX={
  'table':'tb',
  'timeseries':'ts',
  'timeseriesCollection':'tc',
  'content':'cn',
  'coverage':'cv',
  'index':'idx'
}

COLLECTION_TYPES=[
  'tables',
  'timeseries',
  'timeseriesCollections',
  'content',
  'coverages'
]

METADATA_KEY='metadata'

OPT_UGLIFY_TAGS='uglify_tags'
OPT_UGLIFY_COVERAGE='uglify_coverage_attributes'
OPT_COMMON_TIMESERIES_INDEX='extract_timeseries_index'
DEFAULT_OPTIONS={
  OPT_UGLIFY_TAGS:False,
  OPT_UGLIFY_COVERAGE:False,
  OPT_COMMON_TIMESERIES_INDEX:False
}

DEFAULT_HOST_PORT=8000
GIVE_UP_HOST_PORT=8100

def _open(fn,mode='r', raw=False, **kwargs):
  if '://' in fn:
    r = requests.get(fn, **kwargs)
    assert mode == 'r'
    if r.status_code != 200:
      raise Exception('Failed to open remote file: %s'%fn)
    if raw:
      # return r.text
      # print(r.json())
      return r.json()
    #   return r.raw.read().decode('utf-8')
    return StringIO(r.text)
  elif raw:
    return fn
  return open(fn,mode)

def _exists(fn, **kwargs):
  if '://' in fn:
    r = requests.head(fn, **kwargs)
    return r.status_code == 200
  else:
    return os.path.exists(fn)

class HydrographDataset(object):
  def __init__(self,path,mode,options=DEFAULT_OPTIONS, auth = None, **kwargs):
    self.mode = mode
    self.options = DEFAULT_OPTIONS.copy()
    self.options.update(options)
    self.options.update(kwargs)
  
    if "://" in path:
      # assume a remote dataset
      if mode in ['w','rw']:
        raise Exception('Cannot open remote dataset in write mode')
      self.is_remote = True
    else:
      self.is_remote = False

    if auth == None:
      auth = ("", "")
      if self.is_remote:
        logging.warning("No authentication provided for remote dataset")
    self.auth = requests.auth.HTTPBasicAuth(auth[0], auth[1]) # User, Password
    self.path = path
    if mode=='rw':
      self.ensure_directory()

    if mode=='w':
      self.clear()

    try:
      self.load_index() # TD
    except:
      if mode in ['r','ro']:
        raise Exception('Could not load index file')
      self.index = self.init_index() # TD
    
    self._rewrite = True

    self.hosting = False
    self.port = None
    self.host_process = None

  def require_writable(self):
    if 'w' not in self.mode:
      raise Exception('Dataset is not in read/write mode')

  def rewrite(self,val,compressed=False):
    '''
    Enable/disable the writing of the index file after each change to the dataset

    Disable (val=False) to prevent writing and hence speed up bulk writes
    '''
    self.require_writable()
    self._rewrite = val
    if val:
      self.write_index(compressed) # TD

  def expand_path(self,fn): # TD
    # if self.is_remote:
    #   return os.path.join(self.path,fn, )
    return os.path.join(self.path,fn)

  def create_fn(self,prefix,ftype,contents): # TD
    ident = None
    try:
      from hashlib import md5
      ident = md5(contents.encode('utf-8')).hexdigest()[:8]
    except:
      logger.error('Could not generate MD5')
      ident = None

    if ident is None:
      if prefix in self.index:
        existing = self.index[prefix]
      else:
        existing = self.index[prefix+'s']
      i = len(existing)
      ident = i
      while True:
        valid = True
        test_fn = '%s_%s.%s'%(FILE_PREFIX[prefix],str(ident),ftype)
        for e in existing:
          if e['filename']==test_fn:
            valid=False
            break
        if valid:
          return test_fn
        i += 1
        ident = i
    return '%s_%s.%s'%(FILE_PREFIX[prefix],str(ident),ftype)

  def load_index(self): # TD
    index_fn = self.expand_path(INDEX_FN)
    if self.is_remote:
      try:
        r = requests.get(index_fn,auth=self.auth)
        self.index = r.json()
      except:
        raise Exception('Could not load index file from {}'.format(index_fn))
    elif _exists(index_fn, auth=self.auth):
      self.index = json.load(_open(index_fn,'r', auth=self.auth),
                       object_pairs_hook=OrderedDict)
    elif self.mode in ['r','ro']:
      raise Exception('Index file does not exist')
    else:
      self.index = self.init_index()

  def init_index(self):
    result = OrderedDict()
    result[METADATA_KEY] = OrderedDict()
    for collection in COLLECTION_TYPES:
      result[collection] = []
    if self.options[OPT_UGLIFY_TAGS]:
      result['tag_lookup'] = {}
    # if self.options[OPT_COMMON_TIMESERIES_INDEX]:
    #   result['timeseries_index'] = []
    return result

  def write_index(self,compressed=False):
    if not self._rewrite:
      return
    self.require_writable()

    index_fn = self.expand_path(INDEX_FN)
    if compressed:
      json.dump(self.index,_open(index_fn,'w', auth=self.auth))
    else:
      json.dump(self.index,_open(index_fn,'w', auth=self.auth),indent=2)

  def ensure_directory(self):
    self.require_writable()
    if not _exists(self.path, auth=self.auth):
      os.makedirs(self.path)

  def clear(self):
    self.require_writable()
    if _exists(self.path, auth=self.auth):
      shutil.rmtree(self.path)
    self.ensure_directory()
    self.load_index()

  def find_unreferenced_files(self):
    entries = sum([self.index[coll] for coll in COLLECTION_TYPES],[])
    data_filenames = [e['filename'] for e in entries]
    index_filenames = [e['index'] for e in entries if 'index' in e]
    filenames = set(data_filenames + index_filenames + ['index.json'])
    all_filenames = set([os.path.basename(fn) for fn in glob(self.expand_path('*'))])
    return all_filenames - filenames

  def remove_unreferenced_files(self):
    self.require_writable()
    files_to_remove = self.find_unreferenced_files()
    for fn in files_to_remove:
      os.remove(self.expand_path(fn))
    return files_to_remove

  def ensure_all_files_exist(self):
    missing = []
    for collection in COLLECTION_TYPES:
      missing += self.ensure_files_exist(collection)
    if len(missing)>0:
      unique_missing = list(set([fn for fn,_ in missing]))
      raise Exception('Missing %d files: %s'%(len(unique_missing),','.join(unique_missing)))

  def ensure_files_exist(self,datatype,**tags):
    missing = []
    matching = self.match(datatype,**tags)
    for d in matching:
      if not _exists(self.expand_path(d['filename']), auth=self.auth):
        missing.append((d['filename'],d))
      elif 'index' in d and not _exists(self.expand_path(d['index']), auth=self.auth):
        missing.append((d['index'],d))
    return missing

  def tags(self,datatype='tables',**tags):
    matching = self.match(datatype,**tags)
    all_tags = [set(d['tags'].keys()) for d in matching]
    result = set.union(set(),*all_tags)
    return result

  def tag_values(self,tag,datatype='tables',**tags):
    if datatype is None:
      all_sets = [self.tag_values(tag,c,**tags) for c in COLLECTION_TYPES]
      return set.union(*all_sets)
    matching = self.match(datatype,**tags)
    all_tags = [set([d['tags'][tag]]) for d in matching if tag in d['tags']]
    result = set.union(set(),*all_tags)
    return result

  def match(self,datatype='tables',**tags):
    collection = self.index[datatype]
    return self.records_matching(collection,**tags)

  def records_matching(self,records,**tags):
    return [r for r in records if self.matches(r,**tags)]

  def match_table(self,datatype='tables',**tags):
    result = self.match(datatype,**tags)
    filenames = [r['filename'] for r in result]
    tag_sets = [r['tags'] for r in result]
    return pd.DataFrame(tag_sets,index=filenames)

  def unique_tag_groups(self,datatype='tables',**tags):
    result = self.match(datatype,**tags)
    tag_groupings = set()
    for match in result:
      tag_names = tuple(sorted(match['tags'].keys()))
      if not tag_names in tag_groupings:
        tag_groupings.add(tag_names)
    return tag_groupings

  def matches(self,entry,**tags):
    for k,v in tags.items():
      if not k in entry['tags']:
        return False
      if not isinstance(v, list):
        v = [v]

      if not entry['tags'][k] in v:
        return False
      # if entry['tags'][k] == v:
      #   return False
    return True

  def _sanitize_value(self,v):
    if isinstance(v,float):
      if np.isnan(v):
        return None
    return v

  def _sanitize_tags(self,tags):
    return {k:self._sanitize_value(v) for k,v in tags.items()}

  def _add_data_record(self,collection,fn,attributes={},**tags):
    tags = self._sanitize_tags(tags)
    existing = self.match(collection,**tags)
    if len(existing):
      logger.debug('Updating existing record')
      record = existing[0]
      if fn != record['filename']:
        existing_fn = self.expand_path(record['filename'])
        if _exists(existing_fn, auth=self.auth):
          logger.debug('Removing existing file: %s'%record['filename'])
          os.unlink(existing_fn)
    else:
      record = OrderedDict()
      self.index[collection].append(record)

    record['filename'] = fn
    record['tags'] = OrderedDict(**tags)
    record.update(attributes)
    self.write_index()
    return record

  def add_partitioned(self,table,partition_by,csv_options={},**tags):
    self.require_writable()
    if not len(partition_by):
      self.add_table(table,csv_options=csv_options,**tags)
      return

    first = partition_by[0]
    rest = partition_by[1:]
    for val in set(table[first]):
      subset = table[table[first]==val]
      tags[first] = val
      self.add_partitioned(subset,rest,csv_options,**tags)

  def _write_csv(self,data,prefix,csv_options={},fn=None):
    sio = StringIO()
    data.to_csv(sio,**csv_options)
    txt = sio.getvalue()
    auto_fn = fn is None
    if auto_fn:
      fn = self.create_fn(prefix,'csv',txt)

    full_fn = self.expand_path(fn)
    if auto_fn and _exists(full_fn, auth=self.auth):
      return fn

    f = _open(full_fn,'w', auth=self.auth)
    try:
      f.write(txt)
    finally:
      f.close()

    assert _exists(full_fn, auth=self.auth)
    return fn

  def _add_tabular(self,data,prefix,collection,csv_options={},fn=None,attributes={},**tags):
    fn = self._write_csv(data,prefix,csv_options,fn)
    self._add_data_record(collection,fn,attributes,**tags)
    return fn

  def add_table(self,table,csv_options={},fn=None,**tags):
    self.require_writable()
    if hasattr(table, 'geometry'):
      raise Warning('Use add_coverage to add GeoDataFrames')
    return self._add_tabular(table,'table','tables',csv_options,fn,**tags)

  def add_table_existing(self,fn,**tags):
    self.require_writable()
    self._add_data_record('tables',fn,**tags)

  def add_coverage(self,coverage,name_attr=None,decimal_places=None,fn=None,**tags):
    self.require_writable()
    if hasattr(coverage,'to_json'):
      content = coverage.to_json(na='null')
    elif hasattr(coverage,'keys'):
      # Dictionary
      content = json.dumps(coverage)
    elif _exists(coverage, auth=self.auth):
      # Assume string - is it a filename?
      content = _open(coverage,'r', auth=self.auth).read()
    else:
      # Assume loaded JSON text
      content = coverage

    if self.options[OPT_UGLIFY_COVERAGE]:
      content = minify_geojson(content)

    if name_attr is not None:
      parsed = json.loads(content)
      for f in parsed['features']:
        f['properties']['name'] = f['properties'][name_attr]
      content = json.dumps(parsed)

    if fn is None:
      fn = self.create_fn('coverage','json',content)

    self._add_data_record('coverages',fn,**tags)

    full_fn = self.expand_path(fn)

    def write_to(fn):
      f = _open(fn,'w', auth=self.auth)
      try:
        f.write(content)
      finally:
        f.close()

    if decimal_places is None:
      write_to(full_fn)
    else:
      import math
      simplify= math.pow(10,-decimal_places)
      tmp_fn = tempfile.mktemp() + '.json'
      logger.info(f'Writing coverage to {tmp_fn} ahead of simplification to {decimal_places} decimal places')
      write_to(tmp_fn)
      assert os.system('ogr2ogr -f GeoJSON -simplify %f -lco COORDINATE_PRECISION=%s %s %s'%(simplify,decimal_places,full_fn,tmp_fn))==0
      logger.info(f'ogr2ogr successful')
      os.remove(tmp_fn)

  def add_timeseries(self,series,csv_options={},fn=None,**tags):
    self.require_writable()
    attributes={}
    options = csv_options.copy()
    options['header']=True

    if self.options[OPT_COMMON_TIMESERIES_INDEX] and not fn:
      idx = pd.Series(series.index)
      idx_fn = self._write_csv(idx,'index',dict(index=False,header=False))
      options['index']=False
      attributes['index']=idx_fn

    return self._add_tabular(series,'timeseries','timeseries',options,fn,attributes,**tags)

  def add_multiple_time_series(self,dataframe,column_tag,csv_options={},**tags):
    self.require_writable()
    if not self.options[OPT_COMMON_TIMESERIES_INDEX]:
      for col in dataframe.columns:
        ctag = {
          column_tag:col
        }
        self.add_timeseries(dataframe[col],**ctag,csv_options=csv_options,**tags)
      return

    attributes={}
    options = csv_options.copy()
    options['header']=True

    idx = pd.Series(dataframe.index)
    idx_fn = self._write_csv(idx,'index',dict(index=False,header=False))
    options['index']=False
    attributes['index']=idx_fn

    for col in dataframe.columns:
      series = dataframe[col].rename('COL')
      ctag = {
        column_tag:col
      }
      self._add_tabular(series,'timeseries','timeseries',options,attributes=attributes,**ctag,**tags)

  def add_timeseries_collection(self,series,column_tag,csv_options={},fn=None,**tags):
    self.require_writable()
    if fn is None:
      fn = self.create_fn('timeseriesCollection','csv')

    record = self._add_data_record('timeseriesCollections',fn,**tags)
    record['columnTag'] = column_tag
    record['columns'] = series.columns[1:]
    full_fn = self.expand_path(fn)
    series.to_csv(full_fn,header=True,**csv_options)

  def add_content(self,content,fn=None,**tags):
    raise Exception('Not implemented')

  def remove_record(self,record,collection='tables'):
    self.require_writable()
    idx = self.index[collection]
    idx = [rec for rec in idx if rec != record]
    self.index[collection] = idx

  def get_tables(self,**tags):
    coverages = self.match('coverages',**tags)
    if len(coverages):
      import geopandas as gpd
      # return [_open(self.expand_path(c['filename']), raw=True, auth=self.auth) for c in coverages]
      coverages = [gpd.GeoDataFrame.from_features(_open(self.expand_path(c['filename']), raw=True, auth=self.auth)["features"]) for c in coverages]

    tables = self.match('tables',**tags)
    tables = [pd.read_csv(_open(self.expand_path(t['filename']), auth=self.auth),index_col=0,parse_dates=True) for t in tables]

    return coverages + tables

  def get_table(self,**tags):
    tables = self.get_tables(**tags)
    if len(tables)==0:
      raise Exception('No matching tables for tags: %s'%str(tags))
    if len(tables)>1:
      raise Exception('Multiple tables matching tags: %s'%str(tags))
    return tables[0]

  def get_timeseries(self,**tags):
    series = self.match('timeseries',**tags)
    return [self.load_time_series(ts) for ts in series]

  def get_coverages(self,**tags):
    coverages = self.match('coverages',**tags)
    return [self.load_coverage(c) for c in coverages]

  def load_coverage(self,record):
    import geopandas as gpd
    with _open(self.expand_path(record['filename']), auth=self.auth) as fp:
      raw = json.load(fp)
      gdf = gpd.GeoDataFrame.from_features(raw['features'])
      if '_names' in raw:
        gdf = gdf.rename(columns=raw['_names'])
      return gdf

  def load_time_series(self,record):
    try:
      if 'index' in record:
        idx = pd.read_csv(_open(self.expand_path(record['index']), auth=self.auth),index_col=0,parse_dates=True,header=None)
        data = pd.read_csv(_open(self.expand_path(record['filename']), auth=self.auth),parse_dates=True)
        return data.set_index(idx.index)
      return pd.read_csv(_open(self.expand_path(record['filename']), auth=self.auth),parse_dates=True,index_col=0)
    except Exception as e:
      logger.error('Failed to load time series: %s'%str(record))
      raise e

  def copy(self,source,**tags):
    try:
      self.rewrite(False)
      for data_type in ['tables','timeseries','coverages']:
        self._copy_data(data_type,source,**tags)
    finally:
      self.rewrite(True)

  def copy_tables(self,source,**tags):
    self._copy_data('tables',source,**tags)

  def copy_timeseries(self,source,**tags):
    self._copy_data('timeseries',source,**tags)

  def _copy_data(self,datatype,source,**tags):
    matching = source.match(datatype,**tags)
    for entry in matching:
      copy_if_not_exist(source.expand_path(entry['filename']),
                      self.expand_path(entry['filename']))
      rec = self._add_data_record(datatype,entry['filename'],**entry['tags'])

      if 'index' in entry:
        copy_if_not_exist(source.expand_path(entry['index']),
                        self.expand_path(entry['index']))
        rec['index'] = entry['index']

  def add_metadata(self,key,value):
    self.require_writable()
    self.index[METADATA_KEY][key] = value
    self.write_index()

  def get_metadata(self,key):
    return self.index[METADATA_KEY][key]

  def host(self,port=None):
    assert not self.hosting

    if port is None:
      port = DEFAULT_HOST_PORT
      while port <= GIVE_UP_HOST_PORT:
        try:
          self.host(port)
          return
        except Exception as e:
          logger.info('Failed to host on port %d: %s'%(port,e))
          port += 1
      raise Exception('Failed to find open port to host on')

    self.host_process = subprocess.Popen(['python','-m','hydrograph._host',str(port)],cwd=self.path)
    sleep(0.5)
    if self.host_process.poll() is not None:
      self.host_process = None
      raise Exception('Failed to start host process')
    self.port = port
    self.hosting = True

    def close_on_exit():
      self.stop_hosting()

    atexit.register(close_on_exit)

  def stop_hosting(self):
    if not self.hosting:
      return
    self.hosting = False
    self.host_process.kill()
    self.port = None

def open_dataset(path,mode='rw',options=DEFAULT_OPTIONS,**kwargs) -> HydrographDataset:
  assert isinstance(mode,str)
  return HydrographDataset(path,mode,options, auth=None, **kwargs)

def open_remote(url, auth = None, options=DEFAULT_OPTIONS, **kwargs) -> HydrographDataset:
  assert isinstance(url,str)
  return HydrographDataset(url, mode="r", options=options, auth = auth, **kwargs)

def make_reference_dashboard(owner,name,prefix='',content={},**kwargs):
  full_content = dict()

  for k,v in list(content.items())+list(kwargs.items()):
    full_content[f'{prefix}{k.replace(" ","-")}'] = v

  dashboard = OrderedDict(
    owner = owner,
    name = name,
    title = name,
    includes = [],
    reference = full_content
  )

  return dashboard

def copy_if_not_exist(source,dest):
  if not os.path.exists(dest):
    shutil.copy(source,dest) # TD: this will be a problem for remote sources

