import os
import sys
import datetime
import numpy as np
import rasterio as rio
import xarray as xr
import rioxarray as rioxr
from utils import *
from rioxarray.merge import merge_arrays

sys.setrecursionlimit(10000)
store_path = '/media/anthony/Storage_1/aviation_data'

# Load in the dem
dem_raster = rioxr.open_rasterio(os.path.join(store_path, 'LC16_Elev_200.tif'))

# Load in sb40
sb40_raster = rioxr.open_rasterio(os.path.join(store_path, 'LC20_F40_200.tif'))

in_dir = os.path.join(store_path, 'burned_area_files')
years = range(2001, 2020)
start_month = 5
end_month = 11

for year in years:
    print(f'Accumulating year: {year}')

    # Load in the first raster for the selected year
    print(f'Accumulating month {start_month}')
    filename, jd = build_filename(year, start_month)
    dxr = rioxr.open_rasterio(os.path.join(in_dir, filename))
    accumulator = dxr.where(dxr > 0, 0., drop=False)

    for month in range(start_month + 1, end_month):
        print(f'Accumulating month {month}')

        # Load in the month's raster
        filename, jd = build_filename(year, month)
        path = os.path.join(in_dir, filename)
        temp_raster = rioxr.open_rasterio(path)

        # merge the temp raster with the accumulator
        accumulator = temp_raster.where(temp_raster > 0, accumulator, drop=False)
    res = accumulator.rio.resolution()[0]

    # Label accumulated fires
    print(f'Labeling fires in year {year}')
    label_array = label_array_func(accumulator.values.squeeze())

    # Loop over all fires in the year
    for fid in range(2, label_array.max()):
        days = accumulator.values.squeeze()[label_array == fid]
        start = int(days.min())
        stop = int(days.max())
        pixels_burned_time_series = [days[days == i].size for i in range(start, stop + 1)]
        pruned_ts, ts_ind = prune_ts(pixels_burned_time_series)
        num_days_in_ts = len(pruned_ts)

        # Let's get rid of the tiny fires
        if num_days_in_ts < 5:
            continue



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
                row_slice = slice(min(row_min - 500, 0), max(row_max + 500, accumulator.shape[0]))
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
                if latitude.min() < lat_min:
                    lat_min = latitude.min()
                if latitude.max() > lat_max:
                    lat_max = latitude.max()
                if longitude.min() < lon_min:
                    lon_min = longitude.min()
                if longitude.max() > lon_max:
                    lon_max = longitude.max()

        # Draw a box around the lat/long of the fire and reproject to EPSG 5070
        n = 4
        accum_subset = accumulator.rio.clip_box(minx=lon_min - n * res, miny=lat_min - n * res, maxx=lon_max + n * res,
                                                maxy=lat_max + n * res, auto_expand=True)
        accum_subset = accum_subset.rio.reproject('EPSG:5070', nodata=0.)

        # Get the new EPSG 5070 bounds of the data
        bounds = accum_subset.rio.bounds()

        # Sample the DEM and SB40 raster
        dem_subset = dem_raster.rio.clip_box(*bounds)
        sb40_subset = sb40_raster.rio.clip_box(*bounds)

        # Skip over entries outside the U.S.
        invalid_subset_flag = (dem_subset.max() == -9999 and dem_subset.min() < -1000) or (sb40_subset.max() == -9999
                                                                                           and sb40_subset < -1000)
        if invalid_subset_flag:
            continue

        print("Processing fire %i of %i." % (fid, label_array.max()))
        # Match all of the rasters to the DEM
        sb40_match = sb40_subset.rio.reproject_match(dem_subset)

        # Get wind and FWI data for each day
        for i in range(stop - start):
            jday = i + start
            # Convert the julian day to a year-month-day format
            julian_day = f'{str(year)[-2:]}{jday}'
            date = datetime.datetime.strptime(julian_day, '%y%j').date()
            day = f'{date.day:02d}'
            month = f'{date.month:02d}'

            # Load in the day's FWI and wind files
            try:
                fwi_raster = rioxr.open_rasterio(os.path.join(store_path, f'FWI/fwi-{year}-{month}-{day}.tif'))
                day, month = date.day, date.month
                gust_raster = rioxr.open_rasterio(os.path.join(store_path, 'winds', f'wind-gust-{year}-{month}-{day}.tif'), mask_and_scale=True)
                u_raster = rioxr.open_rasterio(os.path.join(store_path, 'winds', f'wind-u-{year}-{month}-{day}.tif'), mask_and_scale=True)
                v_raster = rioxr.open_rasterio(os.path.join(store_path, 'winds', f'wind-v-{year}-{month}-{day}.tif'), mask_and_scale=True)
            except FileNotFoundError:
                print('Could not find daily raster')
                continue

            # Match the daily rasters to the DEM
            fwi_match = fwi_raster.rio.reproject_match(dem_subset, resampling=3)
            gust_match = gust_raster.rio.reproject_match(dem_subset, resampling=3)
            u_match = u_raster.rio.reproject_match(dem_subset, resampling=3)
            v_match = v_raster.rio.reproject_match(dem_subset, resampling=3)

            # Assign all of the pixels already burned to 0
            accum_today = accum_subset.where(accum_subset.values.squeeze() <= jday, 0.)
            accum_today.values.squeeze()[tuple(accum_today > 0)] = 1.
            accum_tomorrow = accum_subset.where(accum_subset.values.squeeze() <= jday+1, 0.)
            accum_tomorrow.values.squeeze()[tuple(accum_tomorrow > 0)] = 1.

            # Match the accumulated subset to the DEM
            accum_today = accum_today.rio.reproject_match(dem_subset)
            accum_tomorrow = accum_tomorrow.rio.reproject_match(dem_subset)

            # Write all of the rasters to a multi-banded tif
            all_together = xr.concat(
                [accum_today, dem_subset, sb40_match, fwi_match, u_match, v_match, gust_match, accum_tomorrow],
                dim='band')
            fname = f'{year}-{fid}-{month}-{jday}.tif'
            all_together.rio.to_raster(os.path.join(store_path, 'dataset', fname))
            print(f'Wrote raster for day {jday} - fire {fid}')
