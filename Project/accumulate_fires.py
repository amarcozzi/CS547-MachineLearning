import os
import sys
import numpy as np
import xarray as xr
import rioxarray as rioxr
from utils import *

in_dir = "Aviation/fire_spread_analysis/burned_area_files/"
years = range(2001, 2020)
start_month = 5
end_month = 11

for year in years:
    print(f'Accumulating year: {year}')

    # Load in the first raster for the selected year
    filename, jd = build_filename(year, start_month)
    dxr = rioxr.open_rasterio(in_dir + filename)
    accumulator = dxr.where(dxr > 0, 0., drop=False)

    for month in range(start_month+1, end_month):
        print(f'Accumulating month {month}')

        # Load in the month's raster
        filename, jd = build_filename(year, month)
        path = in_dir + filename
        temp_raster = rioxr.open_rasterio(path)

        # merge the temp raster with the accumulator
        accumulator = temp_raster.where(temp_raster > 0, accumulator, drop=False)

    # Reproject the raster to NAD83
    accumulator = accumulator.rio.reproject("EPSG:5070")
    accumulator.rio.to_raster(f"Data/accumulated_fires/{year}-fires.tif")
    accumulator = None
    temp_raster = None