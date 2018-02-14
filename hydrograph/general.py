import os
import shutil
from collections import OrderedDict
import logging
import json

logger = logging.getLogger('hydrograph')

INDEX_FN='index.json'

def open_dataset(path):
  return HydrographDataset(path)

class HydrographDataset(object):
  def __init__(self,path):
    self.path = path
    self.ensure()
    self.load_index()

  def expand_path(self,fn):
    return os.path.join(self.path,fn)

  def create_fn(self,prefix,ftype):
    if prefix in self.index:
      existing = self.index[prefix]
    else:
      existing = self.index[prefix+'s']
    i = 1
    while True:
      valid = True
      test_fn = '%s_%d.%s'%(prefix,i,ftype)
      for e in existing:
        if e['filename']==test_fn:
          valid=False
          break
      if valid:
        return test_fn
      i += 1

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

  def write_index(self):
    index_fn = self.expand_path(INDEX_FN)
    json.dump(self.index,open(index_fn,'w'),indent=2)

  def ensure(self):
    if not os.path.exists(self.path):
      os.makedirs(self.path)

  def clear(self):
    if os.path.exists(self.path):
      shutil.rmtree(self.path)
    self.ensure()
    self.load_index()

  def match(self,datatype='tables',**tags):
    collection = self.index[datatype]
    return [e for e in collection if self.matches(e,**tags)]

  def matches(self,entry,**tags):
    for k,v in tags.items():
      if not k in entry['tags']:
        return False
      if entry['tags'][k] != v:
        return False
    return True

  def _add_data_record(self,collection,fn,**tags):
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

  def add_table(self,table,csv_options={},fn=None,**tags):
    if fn is None:
      fn = self.create_fn('table','csv')

    self._add_data_record('tables',fn,**tags)

    full_fn = self.expand_path(fn)
    table.to_csv(full_fn,**csv_options)

  def add_coverage(self,coverage,fn=None,**tags):
    if fn is None:
      fn = self.create_fn('coverage','json')

    self._add_data_record('coverages',fn,**tags)

    full_fn = self.expand_path(fn)
    f = open(full_fn,'w')
    try:
      f.write(coverage.to_json())
    finally:
      f.close()

  def add_timeseries(self,series,csv_options={},fn=None,**tags):
    if fn is None:
      fn = self.create_fn('timeseries','csv')

    self._add_data_record('timeseries',fn,**tags)

    full_fn = self.expand_path(fn)
    series.to_csv(full_fn,header=True,**csv_options)
  
  def add_timeseries_collection(self,series,column_tag,fn=None,**tags):
    raise Exception('Not implemented')

  def add_content(self,content,fn=None,**tags):
    raise Exception('Not implemented')



