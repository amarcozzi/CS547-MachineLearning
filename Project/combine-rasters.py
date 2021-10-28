import os
import sys
import numpy as np
import xarray as xr
import rioxarray as rioxr
from utils import *
from rioxarray.merge import merge_arrays
import multiprocessing
import threading

from dask.distributed import Client, LocalCluster, Lock
from dask.utils import SerializableLock

# xr.set_options(file_cache_maxsize=2)
sys.setrecursionlimit(10000)

if sys.platform == "linux":
    directory = "Aviation/fire_spread_analysis/burned_area_files/"
else:
    directory = "Aviation/fire_spread_analysis/burned_area_files/"

year = 2005
start_month = 5
end_month = 11

# Load in the accumulated raster for the selected year
# test = xr.open_rasterio(f'Data/accumulated_fires/{year}-fires.tif')

# accumulator = rioxr.open_rasterio(f'Data/accumulated_fires/{year}-fires.tif')

# Test load in FWI
# test = xr.open_dataarray('/Users/anthony/CompSci/UMontana/2021_Fall/CS547-ML/Project/Aviation/fire_spread_analysis/fwi_files/ECMWF_FWI_FWI_20010501_1200_hr_v3.1_con.nc', decode_coords="all")
# fwi_raster = rioxr.open_rasterio('/Users/anthony/CompSci/UMontana/2021_Fall/CS547-ML/Project/Aviation/fire_spread_analysis/fwi_files/ECMWF_FWI_FWI_20010501_1200_hr_v3.1_con.nc')

# Test load in the dem
dem_raster = rioxr.open_rasterio('Data/LC16_Elev_200.tif', chunks=True, cache=False)

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

            # Process the first day of the fire data:
            if d == start:
                condition = np.logical_and(accumulator == d, label_array == fid)
                row, col = np.where(condition.values.squeeze())
                latitude = accumulator.y.values[row]
                longitude = accumulator.x.values[col]

            else:
                row_slice = slice(min(row_min-500, 0), max(row_max+500, accumulator.shape[0]))
                col_slice = slice(min(col_min - 500, 0), max(col_max + 500, accumulator.shape[1]))
                condition = np.logical_and(accumulator[0, row_slice, col_slice] == d,
                                           label_array[row_slice, col_slice] == fid)
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

        # Reproject accumulator
        # reprojected_accumulator = accum_subset.rio.reproject_match(dem_raster)
        reprojected_accumulator = accum_subset.rio.reproject(dem_raster.rio.crs)
        print(f'Before: {accum_subset.rio.crs}, After: {reprojected_accumulator.rio.crs}')
        print('test')