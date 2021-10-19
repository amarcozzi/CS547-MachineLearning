import os
import sys
import numpy as np
import xarray as xr
import rioxarray as rioxr
from utils import *

sys.setrecursionlimit(10000)

if sys.platform == "linux":
    directory = "Aviation/fire_spread_analysis/burned_area_files/"
else:
    directory = None

accum = 0
year = 2005
start_month = 5
end_month = 11

# Load in the first raster for the selected year
filename, jd = build_filename(year, start_month)
dxr = rioxr.open_rasterio(directory + filename)
accumulator = xr.where(dxr > 0, dxr, 0.)

d1 = xr.open_rasterio(directory + filename)
d = d1.squeeze().values
accum = np.zeros(d.shape, dtype=np.int16)
accum[d>0] = d[d>0]

for month in range(start_month + 1, end_month):
    filename, jd = build_filename(year,month)
    path = directory + filename

    # rio xarray approach
    temp_raster = rioxr.open_rasterio(path)
    accumulator = xr.where(temp_raster > 0, temp_raster, accumulator)

    # Numpy approach
    d1 = xr.open_rasterio(path)
    d = d1.squeeze().values
    accum[d>0] = d[d>0]

if np.all(accumulator.values.squeeze() == accum):
    print('!worked!')

label_array = label_array_func(accumulator.values.squeeze())

for fid in range(2, label_array.max()):
    days = accumulator.values.squeeze()[label_array == fid]
    start = int(days.min())
    stop = int(days.max())

    # Let's get rid of the tiny fires
    if days.size > 1:
        print("Processing fire %i of %i." % (fid, label_array.max()))

        row_min, row_max = int(1e8), -1
        col_min, col_max = int(1e8), -1
        lat_min, lat_max = int(1e8), -1
        lon_min, lon_max = int(1e8), -1000
        for d in range(start, stop + 1):

            condition = np.logical_and(accumulator == d, label_array == fid)
            row, col = np.where(condition.values.squeeze())
            latitude = accumulator.y.values[row]
            longitude = accumulator.x.values[col]

            # Keep track of bounding box around fire
            if row.size > 0:
                if row.min() < row_min:
                    row_min = row.min()
                if row.max() > row_max:
                    row_max = row.max()
                if col.min() < col_min:
                    col_min = col.min()
                if col.max() > col_max:
                    col_max = col.max()
            if row.size > 0:
                if latitude.min() < lat_min:
                    lat_min = latitude.min()
                if latitude.max() > lat_max:
                    lat_max = latitude.max()
                if longitude.min() < lon_min:
                    lon_min = longitude.min()
                if longitude.max() > lon_max:
                    lon_max = longitude.max()

        print(f'Got bounding box around fire {fid}')
        accum_subset = accumulator.rio.slice_xy(lon_min, lat_min, lon_max, lat_max)
        print('test')