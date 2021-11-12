from downloaders import *
import numpy as np
from calendar import monthrange

# Loop to plot over:
for year in np.arange(2001,2021):
    for month in np.arange(5,11):
        print("****************************************************************")
        print("****Acquiring burned area data for %i %i ****"%(year,month))
        print("****************************************************************")

        f   = download_burned_area(month,year,download_dir = 'burned_area_files')
        week_day,days_in_month = monthrange(year,month)
        for day in range(1,days_in_month+1):
            print("****************************************************************")
            print("**** Acquiring fire weather data for %i %i %i ****"%(year,month,day))
            print("****************************************************************")
            fwi = download_fire_historical(month,year,day,var='fire_weather_index',download_dir='fwi_files')
