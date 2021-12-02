import os
import rioxarray as rioxr

if __name__ == '__main__':

    input_path = '/media/anthony/Storage_1/aviation_data/burned_area_files'
    output_path = '/media/anthony/Storage_1/aviation_data/burned_area_files_reproj'

    # Grab the .nc file inside each year-month-day's directory
    files = os.listdir(input_path)
    for fname in files:
        print(f'processing {fname}')

        # Load in the raster and reproject it to NAD83
        f_path_in = os.path.join(input_path, fname)
        fwi_data_xr = rioxr.open_rasterio(f_path_in)
        fwi_data_xr = fwi_data_xr.rio.write_crs('EPSG:4326')
        fwi_data_xr = fwi_data_xr.rio.reproject("EPSG:5070", nodata=0.)

        f_path_out = os.path.join(output_path, fname)
        fwi_data_xr.rio.to_raster(f_path_out)

    print('All done!')
