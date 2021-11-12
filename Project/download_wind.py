import cdsapi
import numpy as np
from calendar import monthrange

c = cdsapi.Client()

# Loop to plot over:
for year in np.arange(2001, 2021):
    for month in np.arange(5, 11):
        week_day, days_in_month = monthrange(year, month)
        for day in range(1, days_in_month + 1):
            print("****************************************************************")
            print("**** Acquiring wind data for %i %i %i ****" % (year, month, day))
            print("****************************************************************")

            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': [
                        '10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_gust_since_previous_post_processing',
                    ],
                    'month': str(month),
                    'day': str(day),
                    'year': str(year),
                    'area': [
                        49.34, -124.78, -24.74,
                        -66.95,
                    ],
                },
                f'/media/anthony/Storage_1/aviation_data/wind_{year}_{month}_{day}.nc')
