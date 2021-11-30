import os
import json

DATA_PATH = '/media/anthony/Storage_1/aviation_data/dataset-big'

days = 0
fires = []
for fname in os.listdir(DATA_PATH):
    id = fname.split('-')[1]
    fires.append(id)
    days += 1
fires = set(fires)

d = {'days': days,
     'fires': len(fires)}

with open(os.path.join(DATA_PATH, 'metadata.json'), 'w') as fout:
    json.dump(d, fout)