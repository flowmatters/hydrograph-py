# hydrograph-py

Build and query datasets for publication on [Hydrograph](https://hydrograph.io).

Supports:

* Creating and updating indexed datasets (`index.json` plus tagged tables, time series and coverages), locally or for publication
* Read-only access to remotely hosted indexed datasets
* Read-only access to Hydrograph server-side datasets via the REST API

## Installation

Install directly from GitHub with pip:

```bash
pip install https://github.com/flowmatters/hydrograph-py/archive/refs/heads/master.zip
```

To upgrade an existing installation, add `--upgrade`:

```bash
pip install --upgrade https://github.com/flowmatters/hydrograph-py/archive/refs/heads/master.zip
```

### Optional dependencies

Core dependencies (`numpy`, `pandas`, `requests`) are installed automatically.
For spatial data (coverages / GeoDataFrames), also install the optional
dependencies:

```bash
pip install geopandas shapely
```

Writing coverages with reduced coordinate precision additionally requires the
`ogr2ogr` command line tool (GDAL).

### Development install

```bash
git clone https://github.com/flowmatters/hydrograph-py.git
cd hydrograph-py
pip install -e '.[optional]'
pytest
```

## Quick start

```python
import hydrograph as hg

# Create a local dataset
ds = hg.open_dataset('path/to/dataset', 'rw')
ds.add_timeseries(series, location='Hamilton', variable='Rain')

# Query it by tags
rain = ds.get_timeseries(location='Hamilton', variable='Rain')

# Open a remote dataset (static index.json or server-side REST API)
remote = hg.open_remote('https://staging.hydrograph.io/api/datasets/owner/dataset-name')
remote.tags()
remote.tag_values('variable')
tables = remote.get_tables(variable='soil-moisture')
```
