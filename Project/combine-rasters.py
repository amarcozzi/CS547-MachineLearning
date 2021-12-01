import os
import gc
import sys
import json
import datetime
import numpy as np
import rasterio as rio
import xarray as xr
import rioxarray as rioxr
from utils import *
from rasterio import windows
from rioxarray.exceptions import NoDataInBounds
from rioxarray.merge import merge_arrays

sys.setrecursionlimit(10000)

if len(sys.argv) > 1:
    arg_year = int(sys.argv[1])
else:
    arg_year = None

prune_data_flag = True
data_path = '/media/anthony/Storage_1/aviation_data'
out_folder = 'dataset-big'

# Load in the dem
print('loading DEM')
dem_raster = rioxr.open_rasterio(os.path.join(data_path, 'LC16_Elev_200.tif'))

# Load in sb40
print('loading SB40')
sb40_raster = rioxr.open_rasterio(os.path.join(data_path, 'LC20_F40_200.tif'))

max_shape = [64, 64]
start_year = 2001
end_year = 2019
start_month = 5
end_month = 11

num_fires = 0
num_days = 0
for year in range(start_year, end_year+1):

    if arg_year:
        if year != arg_year:
            continue

    print('accumulating fires')
    accumulator = accumulator_loader(year, start_month, end_month, data_path)
    res = accumulator.rio.resolution()[0]

    # Label accumulated fires
    print(f'Labeling fires in year {year}')
    label_array = label_loader(accumulator.values.squeeze(), year, data_path)


    # Loop over all fires in the year
    max_fid = label_array.max()
    for fid in range(2, max_fid):
        days = accumulator.values.squeeze()[label_array == fid]
        start = int(days.min())
        stop = int(days.max())
        pixels_burned_time_series = [days[days == i].size for i in range(start, stop + 1)]
        pruned_ts, ts_ind = prune_ts(pixels_burned_time_series)
        num_days_in_ts = len(pruned_ts)

        # Let's get rid of the tiny fires
        if num_days_in_ts < 3:
            continue

        vals = find_fire_size(start, stop, accumulator, label_array, fid)
        row_min, row_max, col_min, col_max, lat_min, lat_max, lon_min, lon_max = vals

        if prune_data_flag:
            old_start = start
            old_stop = stop
            stop = min(old_stop, start + ts_ind[-1] + 1)
            start += ts_ind[0]

        # Match the shape of the subset to the maximum extracted shape
        accum_test = accumulator[..., row_min:row_max, col_min:col_max]
        test_shape = accum_test.values.shape[-2:]
        if np.any(np.array(test_shape) > np.array(max_shape)):
            print(f'Fire too big - {fid} of {max_fid}')
            continue

        vals = round_idx_to_max(row_min, row_max, col_min, col_max, max_shape, fid)
        row_min, row_max, col_min, col_max = vals

        # Draw a box around the lat/long of the fire
        # n = 4
        # accum_subset = accumulator.rio.clip_box(minx=lon_min - n * res, miny=lat_min - n * res, maxx=lon_max + n * res,
        #                                         maxy=lat_max + n * res, auto_expand=True)

        # Draw a box around the lat/long of the fire
        # accum_subset = accumulator.where(label_array == fid, 0)[..., row_min:row_max, col_min:col_max]
        accum_subset = accumulator[..., row_min:row_max, col_min:col_max]
        if accum_subset.shape != (1, *max_shape):
            print(f'raster is the wrong shape - {fid} of {max_fid}')
            continue
        bounds = accum_subset.rio.bounds()

        # Sample the DEM and SB40 raster
        try:
            dem_subset = dem_raster.rio.clip_box(*bounds)
            dem_subset = dem_subset.rio.reproject_match(accum_subset, resampling=3)
        except NoDataInBounds:
            print(f'No data in bounds - DEM - {fid} of {max_fid}')
            continue
        try:
            sb40_subset = sb40_raster.rio.clip_box(*bounds)
            sb40_subset = sb40_subset.rio.reproject_match(accum_subset)
        except NoDataInBounds:
            print(f'No data in bounds - SB40 - {fid} of {max_fid}')
            continue

        # Skip over entries outside the U.S.
        invalid_DEM_flag = dem_subset.max() == -9999 or dem_subset.min() == -9999
        if invalid_DEM_flag:
            print(f'no-data values present in DEM - {fid} of {max_fid}')
            continue

        invalid_SB40_flag = sb40_subset.max() == -9999 or sb40_subset.min() == -9999
        if invalid_SB40_flag:
            print(f'no-data values present in SB40 - {fid} of {max_fid}')
            continue

        accumulator_fid = accumulator.where(label_array == fid, 0)
        accum_subset = accumulator_fid[..., row_min:row_max, col_min:col_max]
        del accumulator_fid
        gc.collect()

        print("Processing fire %i of %i." % (fid, max_fid))
        num_fires += 1
        # Get wind and FWI data for each day
        for i in tqdm(range(stop - start)):
            jday = i + start
            # Convert the julian day to a year-month-day format
            julian_day = f'{str(year)[-2:]}{jday}'
            date = datetime.datetime.strptime(julian_day, '%y%j').date()
            day = f'{date.day:02d}'
            month = f'{date.month:02d}'

            # Load in the day's FWI and wind files
            try:
                fwi_raster = rioxr.open_rasterio(os.path.join(data_path, f'FWI/fwi-{year}-{month}-{day}.tif'))
                day, month = date.day, date.month
                gust_raster = rioxr.open_rasterio(
                    os.path.join(data_path, 'winds', f'wind-gust-{year}-{month}-{day}.tif'), mask_and_scale=True)
                u_raster = rioxr.open_rasterio(os.path.join(data_path, 'winds', f'wind-u-{year}-{month}-{day}.tif'),
                                               mask_and_scale=True)
                v_raster = rioxr.open_rasterio(os.path.join(data_path, 'winds', f'wind-v-{year}-{month}-{day}.tif'),
                                               mask_and_scale=True)
            except FileNotFoundError:
                print('Could not find daily raster')
                continue

            # Match the daily rasters to the DEM
            # fwi_match = fwi_raster.rio.reproject_match(dem_subset, resampling=3)
            # gust_match = gust_raster.rio.reproject_match(dem_subset, resampling=3)
            # u_match = u_raster.rio.reproject_match(dem_subset, resampling=3)
            # v_match = v_raster.rio.reproject_match(dem_subset, resampling=3)

            # Match the daily rasters to the accumualted subset
            fwi_match = fwi_raster.rio.reproject_match(accum_subset, resampling=3)
            gust_match = gust_raster.rio.reproject_match(accum_subset, resampling=3)
            u_match = u_raster.rio.reproject_match(accum_subset, resampling=3)
            v_match = v_raster.rio.reproject_match(accum_subset, resampling=3)

            # Assign all of the pixels already burned to 0
            accum_today = accum_subset.where(accum_subset.values.squeeze() <= jday, 0)

            # Assign labels of 1 and 2 for previously and newly burnt pixels
            accum_today.values.squeeze()[tuple(accum_today > 0)] = 1
            accum_tomorrow = accum_subset.where(accum_subset.values.squeeze() <= jday + 1, 0)
            accum_tomorrow.values.squeeze()[tuple(accum_tomorrow > 0)] = 0
            accum_tomorrow.values.squeeze()[tuple(accum_subset == jday + 1)] = 1

            # Match the accumulated subset to the accumulated subset
            accum_today = accum_today.rio.reproject_match(accum_subset)
            accum_tomorrow = accum_tomorrow.rio.reproject_match(accum_subset)

            if accum_today.values.min() != 0:
                print('Error - no min values')

            if accum_tomorrow.values.min() != 0:
                print('Error - no min values')

            raster_list = [accum_today, dem_subset, sb40_subset, fwi_match, u_match, v_match, gust_match,
                           accum_tomorrow]
            if not all(raster.shape == (1, *max_shape) for raster in raster_list):
                print('Not all rasters have the same size!')
                continue

            # Write all of the rasters to a multi-banded tif
            all_together = xr.concat(raster_list, dim='band')
            fname = f'{year}-{fid}-{month}-{jday}.tif'
            if not os.path.isdir(os.path.join(data_path, out_folder)):
                os.mkdir(os.path.join(data_path, out_folder))

            all_together.rio.to_raster(os.path.join(data_path, out_folder, fname), dtype=np.float32)
            num_days += 1

# data = {'shape': max_shape,
#         'num-fires': num_fires,
#         'num-days': num_days,
#         'mean-days-per-fire': num_days/num_fires}
# with open(os.path.join(data_path, out_folder, 'metadata.json')) as fout:
#     json.dump(data, fout)