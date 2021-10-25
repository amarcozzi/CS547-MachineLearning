import rioxarray as rioxr
import xarray as xr

# fwi_data_rioxr = rioxr.open_rasterio()
fwi_data_xr = xr.open_dataset('Aviation/fire_spread_analysis/fwi_files/fire_weather_index_2001.05.01.nc/ECMWF_FWI_FWI_20010501_1200_hr_v3.1_con.nc')
fwi_data_xr = fwi_data_xr.rio.write_crs('EPSG:4326')
fwi_data_xr = fwi_data_xr.rio.reproject("EPSG:5070")
fwi_data_xr.fwi.rio.to_raster('Data/fwi-test.tif')
print()