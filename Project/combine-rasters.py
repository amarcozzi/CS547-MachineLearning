import os
import sys
import datetime
import numpy as np
import rasterio as rio
import xarray as xr
import rioxarray as rioxr
from utils import *
from rioxarray.exceptions import NoDataInBounds
from rioxarray.merge import merge_arrays

sys.setrecursionlimit(10000)
prune_data_flag = True
store_path = '/media/anthony/Storage_1/aviation_data'
out_folder = 'dataset-pruned-padded-40-max'

# Load in the dem
dem_raster = rioxr.open_rasterio(os.path.join(store_path, 'LC16_Elev_200.tif'))

# Load in sb40
sb40_raster = rioxr.open_rasterio(os.path.join(store_path, 'LC20_F40_200.tif'))

in_dir = os.path.join(store_path, 'burned_area_files')
years = range(2001, 2020)
start_month = 5
end_month = 11

for year in years:

    if year != 2019:
        continue

    accumulator = accumulate_year(year, start_month, end_month, in_dir)
    res = accumulator.rio.resolution()[0]

    # Label accumulated fires
    print(f'Labeling fires in year {year}')
    label_array = label_array_func(accumulator.values.squeeze())

    print('Finding maximum raster shape')
    # if year == 2019:
    #     max_shape = [57, 101]
    # else:
    # max_shape = get_max_shape(accumulator, label_array, prune_data_flag, res, 4)
    max_shape = (40, 40)

    # Loop over all fires in the year
    for fid in range(2, label_array.max()):

        days = get_fire_days(accumulator, label_array, fid, prune_data_flag)
        if not days:
            continue
        else:
            start, stop = days

        vals = find_fire_size(start, stop, accumulator, label_array, fid)
        row_min, row_max, col_min, col_max, lat_min, lat_max, lon_min, lon_max = vals

        # Match the shape of the subset to the maximum extracted shape
        accum_test = accumulator[..., row_min:row_max, col_min:col_max]
        test_shape = accum_test.values.shape[-2:]
        if np.any(np.array(test_shape) > np.array(max_shape)):
            print('Fire too big')
            continue

        row_diff = row_max - row_min
        col_diff = col_max - col_min
        if row_diff < max_shape[0]:
            max_diff = max_shape[0] - row_diff
            row_min -= max_diff // 2
            row_max += max_diff // 2
            if max_diff % 2 != 0:
                row_max += 1
        row_diff = row_max - row_min
        if row_diff != max_shape[0]:
            print(f'Shape error for fire {fid}')
            quit()

        if col_diff < max_shape[1]:
            max_diff = max_shape[1] - col_diff
            col_min -= max_diff // 2
            col_max += max_diff // 2
            if max_diff % 2 != 0:
                col_max += 1
        col_diff = col_max - col_min
        if col_diff != max_shape[1]:
            print(f'Shape error for fire {fid}')
            quit()

        # Draw a box around the lat/long of the fire and reproject to EPSG 5070
        # n = 4
        # accum_subset = accumulator.rio.clip_box(minx=lon_min - n * res, miny=lat_min - n * res, maxx=lon_max + n * res,
        #                                         maxy=lat_max + n * res, auto_expand=True)
        # accum_subset = accum_subset.rio.reproject('EPSG:5070', nodata=0.)

        # Draw a box around the lat/long of the fire and reproject to EPSG 5070
        accum_subset = accumulator[..., row_min:row_max, col_min:col_max]
        try:
            accum_subset = accum_subset.rio.reproject('EPSG:5070', nodata=0.)
        except NoDataInBounds:
            print('No data in bounds')
            continue

        # Get the new EPSG 5070 bounds of the data
        bounds = accum_subset.rio.bounds()

        # Sample the DEM and SB40 raster
        try:
            dem_subset = dem_raster.rio.clip_box(*bounds)
        except NoDataInBounds:
            print('No data in bounds - DEM')
            continue
        try:
            sb40_subset = sb40_raster.rio.clip_box(*bounds)
        except NoDataInBounds:
            print('No data in bounds - SB40')
            continue

        # Skip over entries outside the U.S.
        invalid_DEM_flag = dem_subset.max() == -9999 or dem_subset.min() == -9999
        if invalid_DEM_flag:
            print('no-data values present in DEM')
            continue

        invalid_SB40_flag = sb40_subset.max() == -9999 or sb40_subset.min() == -9999
        if invalid_SB40_flag:
            print('no-data values present in SB40')
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
                gust_raster = rioxr.open_rasterio(
                    os.path.join(store_path, 'winds', f'wind-gust-{year}-{month}-{day}.tif'), mask_and_scale=True)
                u_raster = rioxr.open_rasterio(os.path.join(store_path, 'winds', f'wind-u-{year}-{month}-{day}.tif'),
                                               mask_and_scale=True)
                v_raster = rioxr.open_rasterio(os.path.join(store_path, 'winds', f'wind-v-{year}-{month}-{day}.tif'),
                                               mask_and_scale=True)
            except FileNotFoundError:
                print('Could not find daily raster')
                continue

            # Match the daily rasters to the DEM
            fwi_match = fwi_raster.rio.reproject_match(dem_subset, resampling=3)
            gust_match = gust_raster.rio.reproject_match(dem_subset, resampling=3)
            u_match = u_raster.rio.reproject_match(dem_subset, resampling=3)
            v_match = v_raster.rio.reproject_match(dem_subset, resampling=3)

            # Assign all of the pixels already burned to 0
            accum_today = accum_subset.where(accum_subset.values.squeeze() <= jday, 0)
            accum_today.values.squeeze()[tuple(accum_today > 0)] = 1
            accum_tomorrow = accum_subset.where(accum_subset.values.squeeze() <= jday + 1, 0)
            accum_tomorrow.values.squeeze()[tuple(accum_tomorrow > 0)] = 1
            accum_tomorrow.values.squeeze()[tuple(accum_subset == jday + 1)] = 2

            # Match the accumulated subset to the DEM
            accum_today = accum_today.rio.reproject_match(dem_subset)
            accum_tomorrow = accum_tomorrow.rio.reproject_match(dem_subset)

            # Write all of the rasters to a multi-banded tif
            all_together = xr.concat(
                [accum_today, dem_subset, sb40_match, fwi_match, u_match, v_match, gust_match, accum_tomorrow],
                dim='band')
            fname = f'{year}-{fid}-{month}-{jday}.tif'
            if not os.path.isdir(os.path.join(store_path, out_folder)):
                os.mkdir(os.path.join(store_path, out_folder))

            all_together.rio.to_raster(os.path.join(store_path, out_folder, fname), dtype=np.float32)
