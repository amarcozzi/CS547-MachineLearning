import os
import glob
import rioxarray as rioxr
import numpy as np
import xarray as xr
from calendar import monthrange

if __name__ == '__main__':

    # Load in the data
    input_path = '/media/anthony/Storage_1/aviation_data/winds.nc'
    wind_nc = xr.open_dataset(input_path)
    times = wind_nc.time

    index = 0
    for year in np.arange(2001, 2021):
        for month in np.arange(5, 11):
            week_day, days_in_month = monthrange(year, month)
            for day in range(1, days_in_month + 1):
                print(f'Processing wind data for {year}-{month}-{day}')
                # Get the correct time from the wind data
                wind_ds = wind_nc.sel(time=times[index])

                # Pull the u, v, and gust components from the dataset
                wind_u = wind_ds.u10
                wind_v = wind_ds.v10
                wind_gust = wind_ds.fg10

                # Add a crs to each varaible and reproject to NAD83
                wind_u = wind_u.rio.write_crs('EPSG:4326')
                wind_u = wind_u.rio.reproject('NAD83')
                wind_u.rio.to_raster(f'/media/anthony/Storage_1/aviation_data/winds/wind-u-{year}-{month}-{day}.tif')

                wind_v = wind_v.rio.write_crs('EPSG:4326')
                wind_v = wind_v.rio.reproject('NAD83')
                wind_v.rio.to_raster(f'/media/anthony/Storage_1/aviation_data/winds/wind-v-{year}-{month}-{day}.tif')

                wind_gust = wind_gust.rio.write_crs('EPSG:4326')
                wind_gust = wind_gust.rio.reproject('EPSG:5070')
                wind_gust.rio.to_raster(f'/media/anthony/Storage_1/aviation_data/winds/wind-gust-{year}-{month}-{day}'
                                        f'.tif')

                index += 1

    print('All done!')
