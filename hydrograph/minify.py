import string
import json

WORKING_SET=string.ascii_lowercase

def increment_identifier(name:string)->str:
    if not len(name):
        return WORKING_SET[0]
    new_idx = WORKING_SET.index(name[-1]) + 1
    if new_idx == len(WORKING_SET):
        return increment_identifier(name[:-1]) + WORKING_SET[0]

    return name[:-1] + WORKING_SET[new_idx]

def minify_name(name,existing):
  if name[0].lower() not in existing:
    return name[0].lower()

  new_name = WORKING_SET[0]
  while new_name in existing:
      new_name = increment_identifier(new_name)
  return new_name

def minify_geojson(json_content:str):
  minified_names = set()
  attribute_lookup = {}
  coverage = json.loads(json_content)
  for f in coverage['features']:
    property_names = list(f['properties'].keys())
    minified_properties={}

    for old_name in property_names:
      if old_name not in attribute_lookup:
        minified_name = minify_name(old_name,minified_names)
        minified_names = minified_names.union({minified_name})
        attribute_lookup[old_name]=minified_name

      new_name = attribute_lookup[old_name]
      minified_properties[new_name] = f['properties'][old_name]
    f['properties'] = minified_properties
  inverted_lookup = {v:k for k,v in attribute_lookup.items()}
  coverage['_names'] = inverted_lookup
  return json.dumps(coverage)
