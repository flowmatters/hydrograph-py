import os
from io import StringIO
import shutil
from collections import OrderedDict
import logging
import json
import pandas as pd
import numpy as np
from glob import glob
import tempfile
#from json import encoder

logger = logging.getLogger('hydrograph')

INDEX_FN='index.json'
FILE_PREFIX={
  'table':'tb',
  'timeseries':'ts',
  'timeseriesCollection':'tc',
  'content':'cn',
  'coverage':'cv'
}

def open_dataset(path):
  return HydrographDataset(path)

class HydrographDataset(object):
  def __init__(self,path):
    self.path = path
    self.ensure()
    try:
      self.load_index()
    except:
      pass
    self._rewrite = True

  def rewrite(self,val,compressed=False):
    '''
    Enable/disable the writing of the index file after each change to the dataset

    Disable (val=False) to prevent writing and hence speed up bulk writes
    '''
    self._rewrite = val
    if val:
      self.write_index(compressed)

  def expand_path(self,fn):
    return os.path.join(self.path,fn)

  def create_fn(self,prefix,ftype,contents):
    ident = None
    try:
      from hashlib import md5
      ident = md5(contents.encode('utf-8')).hexdigest()[:8]
    except:
      print('Could not generate MD5')
      ident = None

    if prefix in self.index:
      existing = self.index[prefix]
    else:
      existing = self.index[prefix+'s']
    i = len(existing)
    if ident is None:
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

  def load_index(self):
    index_fn = self.expand_path(INDEX_FN)
    if os.path.exists(index_fn):
      self.index = json.load(open(index_fn,'r'),
                       object_pairs_hook=OrderedDict)
    else:
      self.index = self.init_index()

  def init_index(self):
    result = OrderedDict()
    result['tables'] = []
    result['timeseries'] = []
    result['timeseriesCollections'] = []
    result['content'] = []
    result['coverages'] = []
    return result

  def write_index(self,compressed=False):
    if not self._rewrite:
      return

    index_fn = self.expand_path(INDEX_FN)
    if compressed:
      json.dump(self.index,open(index_fn,'w'))
    else:
      json.dump(self.index,open(index_fn,'w'),indent=2)

  def ensure(self):
    if not os.path.exists(self.path):
      os.makedirs(self.path)

  def clear(self):
    if os.path.exists(self.path):
      shutil.rmtree(self.path)
    self.ensure()
    self.load_index()

  def find_unreferenced_files(self):
    entries = sum([v for _,v in self.index.items()],[])
    filenames = set([e['filename'] for e in entries] + ['index.json'])
    all_filenames = set([os.path.basename(fn) for fn in glob(self.expand_path('*'))])
    return all_filenames - filenames

  def remove_unreferenced_files(self):
    files_to_remove = self.find_unreferenced_files()
    for fn in files_to_remove:
      os.remove(self.expand_path(fn))
    return files_to_remove

  def tags(self,datatype='tables',**tags):
    matching = self.match(datatype,**tags)
    all_tags = [set(d['tags'].keys()) for d in matching]
    result = set()
    for tags in all_tags: result = result.union(tags)
    return result

  def tag_values(self,tag,datatype='tables',**tags):
    matching = self.match(datatype,**tags)
    all_tags = [set([d['tags'][tag]]) for d in matching if tag in d['tags']]
    result = set()
    for tags in all_tags: result = result.union(tags)
    return result

  def match(self,datatype='tables',**tags):
    collection = self.index[datatype]
    return [e for e in collection if self.matches(e,**tags)]

  def match_table(self,datatype='tables',**tags):
    result = self.match(datatype,**tags)
    filenames = [r['filename'] for r in result]
    tag_sets = [r['tags'] for r in result]
    return pd.DataFrame(tag_sets,index=filenames)

  def matches(self,entry,**tags):
    for k,v in tags.items():
      if not k in entry['tags']:
        return False
      if entry['tags'][k] != v:
        return False
    return True

  def _sanitize_value(self,v):
    if isinstance(v,float):
      if np.isnan(v):
        return None
    return v

  def _sanitize_tags(self,tags):
    return {k:self._sanitize_value(v) for k,v in tags.items()}

  def _add_data_record(self,collection,fn,**tags):
    tags = self._sanitize_tags(tags)
    existing = self.match(collection,**tags)
    if len(existing):
      logger.info('Updating existing record')
      record = existing[0]
      existing_fn = self.expand_path(record['filename'])
      if os.path.exists(existing_fn):
        logger.info('Removing existing file: %s'%record['filename'])
        os.unlink(existing_fn)
    else:
      record = OrderedDict()
      self.index[collection].append(record)

    record['filename'] = fn
    record['tags'] = OrderedDict(**tags)

    self.write_index()
    return record

  def add_partitioned(self,table,partition_by,csv_options={},**tags):
    if not len(partition_by):
      self.add_table(table,csv_options=csv_options,**tags)
      return

    first = partition_by[0]
    rest = partition_by[1:]
    for val in set(table[first]):
      subset = table[table[first]==val]
      tags[first] = val
      self.add_partitioned(subset,rest,csv_options,**tags)

  def _add_tabular(self,data,prefix,collection,csv_options={},fn=None,**tags):
    sio = StringIO()
    data.to_csv(sio,**csv_options)
    txt = sio.getvalue()
    if fn is None:
      fn = self.create_fn(prefix,'csv',txt)

    self._add_data_record(collection,fn,**tags)

    full_fn = self.expand_path(fn)
    f = open(full_fn,'w')
    try:
      f.write(txt)
    finally:
      f.close()
    return fn

  def add_table(self,table,csv_options={},fn=None,**tags):
    return self._add_tabular(table,'table','tables',csv_options,fn,**tags)
  
  def add_table_existing(self,fn,**tags):
    self._add_data_record('tables',fn,**tags)

  def add_coverage(self,coverage,name_attr=None,decimal_places=None,fn=None,**tags):
    if hasattr(coverage,'to_json'):
      content = coverage.to_json(na='null')
    elif hasattr(coverage,'keys'):
      # Dictionary
      content = json.dumps(coverage)
    elif os.path.exists(coverage):
      # Assume string - is it a filename?
      content = open(coverage,'r').read()
    else:
      # Assume loaded JSON text
      content = coverage

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
      f = open(fn,'w')
      try:
        f.write(content)
      finally:
        f.close()

    if decimal_places is None:
      write_to(fn)
    else:
      import math
      simplify= math.pow(10,-decimal_places)
      tmp_fn = tempfile.mktemp() + '.json'
      write_to(tmp_fn)
      assert os.system('ogr2ogr -f GeoJSON -simplify %f -lco COORDINATE_PRECISION=%s %s %s'%(simplify,decimal_places,full_fn,tmp_fn))==0
      shutil.copyfile(tmp_fn,full_fn)
      os.remove(tmp_fn)

  def add_timeseries(self,series,csv_options={},fn=None,**tags):
    options = csv_options.copy()
    options['header']=True
    return self._add_tabular(series,'timeseries','timeseries',options,fn,**tags)
  
  def add_timeseries_collection(self,series,column_tag,csv_options={},fn=None,**tags):
    if fn is None:
      fn = self.create_fn('timeseriesCollection','csv')

    record = self._add_data_record('timeseriesCollections',fn,**tags)
    record['columnTag'] = column_tag
    record['columns'] = series.columns[1:]
    full_fn = self.expand_path(fn)
    series.to_csv(full_fn,header=True,**csv_options)

  def add_content(self,content,fn=None,**tags):
    raise Exception('Not implemented')

  def get_tables(self,**tags):
    coverages = self.match('coverages',**tags)
    if len(coverages):
      import geopandas as gpd
      coverages = [gpd.read_file(self.expand_path(c['filename'])) for c in coverages]

    tables = self.match('tables',**tags)
    tables = [pd.read_csv(self.expand_path(t['filename']),index_col=0,parse_dates=True) for t in tables]

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
    return [pd.read_csv(self.expand_path(ts['filename']),
                        index_col=0,parse_dates=True) for ts in series]

  def copy_tables(self,source,**tags):
    self._copy_data('tables',source,**tags)

  def copy_timeseries(self,source,**tags):
    self._copy_data('timeseries',source,**tags)

  def _copy_data(self,datatype,source,**tags):
    matching = source.match(datatype,**tags)
    for entry in matching:
      shutil.copyfile(source.expand_path(entry['filename']),
                      self.expand_path(entry['filename']))
      self._add_data_record(datatype,entry['filename'],**entry['tags'])




