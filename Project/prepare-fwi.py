import os
import glob
import rioxarray as rioxr
import xarray as xr


def parse_year_month_day(s):
    underscore_split = s.split('_')
    dot_split = underscore_split[-1].split('.')
    year = dot_split[0]
    month = dot_split[1]
    day = dot_split[2]
    return year, month, day


if __name__ == '__main__':

    input_path = 'Aviation/fire_spread_analysis/fwi_files/'
    directories = sorted(os.listdir('Aviation/fire_spread_analysis/fwi_files/'))

    # Grab the .nc file inside each year-month-day's directory
    for dir_name in directories:
        print(f'processing {dir_name}')

        year, month, day = parse_year_month_day(dir_name)

        # Find the name of the file inside of the directory
        dir_path = os.path.join(input_path, dir_name)
        f_name = os.listdir(dir_path)[0]
        f_path = os.path.join(dir_path, f_name)

        # Load in the raster and reproject it to NAD83
        fwi_data_xr = xr.open_dataset(f_path)
        fwi_data_xr = fwi_data_xr.rio.write_crs('EPSG:4326')
        fwi_data_xr = fwi_data_xr.rio.reproject("EPSG:5070")
        fwi_data_xr.fwi.rio.to_raster(f'/media/anthony/Storage_1/aviation_data/FWI/fwi-{year}-{month}-{day}.tif')

    print('All done!')
