import cdsapi
from urllib.request import urlopen
import xarray as xr
from io import BytesIO
from zipfile import ZipFile
import urllib3
import pysftp
import datetime
import os
import time


def download_fire_historical(month, year, day, var='fire_weather_index', download_dir=None):
    """
    This returns an xarray containing the data. This is
    not going to fill the disk with data, but that's got
    its own hazards.
    """

    dataset = 'cems-fire-historical'
    if month < 10:
        month_str = '0' + str(month)
    else:
        month_str = str(month)

    if day < 10:
        day_str = '0' + str(day)
    else:
        day_str = str(day)

    file_name = var + "_" + str(year) + '.' + month_str + '.' + day_str + ".nc.gz"

    params = {
        'product_type': 'reanalysis',
        'variable': var,
        'version': '3.1',
        'dataset': 'Consolidated dataset',
        'year': str(year),
        'month': month_str,
        'day': day_str,
        'format': 'zip',
    }
    c = cdsapi.Client(quiet=True)

    if download_dir:
        # Create the download directory if it doesn't exist
        if not os.path.exists(download_dir):
            os.mkdir(download_dir)
        try:
            fl = c.retrieve(dataset, params, download_dir + "/" + file_name)
        except:
            print("Something went wrong downloading fire weather for %s" % (month_str + " " + str(year)))
            print("Will try again in 3 seconds.")
            time.sleep(3)
        return download_dir + "/" + file_name

    else:
        # Get file object, some error handling added
        print("Query posed to Copernicus (ERA5)")
        fl = None
        while not fl:
            try:
                fl = c.retrieve(dataset, params)
            except:
                print("Something went wrong downloading fire weather for %s" % (month_str + " " + str(year)))
                print("Will try again in 3 seconds.")
                time.sleep(3)

        print("Retrieved a locator from Copernicus, getting the data:")
        # load into memory as xarray dataset
        # some trickery as the unzipping is also done in memory
        with urlopen(fl.location) as f:
            zipdownload = None
            while not zipdownload:
                try:
                    zipdownload = ZipFile(BytesIO(f.read()))
                    nc_file = BytesIO(zipdownload.open(zipdownload.namelist()[0]).read())
                except:
                    print("Something went wrong downloading fire weather for %s" % (month_str + " " + str(year)))
                    print("Will try again in 3 seconds.")
                    time.sleep(3)

            print("Copernicus data aquired, returning the x array.")
            return xr.open_dataset(nc_file)


def download_burned_area(month, year, download_dir='burned_area_files'):
    # Create the download directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    fmt = '%Y.%m.%d'

    if month < 10:
        month_str = '0' + str(month)
    else:
        month_str = str(month)

    # The next 4 lines get a Julian Day, or days since 1 Jan.
    # Always at the beginning of the month because the files
    # are monthly
    date_str = str(year) + '.' + month_str + '.01'
    dt = datetime.datetime.strptime(date_str, fmt)
    tt = dt.timetuple()
    yd = tt.tm_yday

    # The file format specifies the date as follows:
    date_str = 'A' + str(year) + str(yd)

    # Spatial window, see https://modis-fire.umd.edu/files/MODIS_C6_BA_User_Guide_1.3.pdf
    # page 15 for details, Win03 is CONUS, Win01 AK, Win24 Hawaii, Win02 Canada, etc.
    window = 'Win03'

    # Version number, 006 is current
    version = '006'

    # There are two layers 'burndate' (what we want) and 'ba_qa' (quality assurance?)
    layer = 'burndate'

    filename = "MCD64monthly." + date_str + '.' + window + '.' + version + '.' + layer + '.tif'

    directory = 'data/MODIS/C6/MCD64A1/TIFF/' + window + '/' + str(year) + '/'

    path_file = download_dir + '/' + filename

    # Don't download again if already done.
    if os.path.exists(path_file):
        print("File %s found locally, no download performed." % path_file)
        return path_file

    cwd = os.getcwd()

    os.chdir(download_dir)

    with pysftp.Connection('fuoco.geog.umd.edu', username='fire', password='burnt') as sftp:

        if sftp.isfile(directory + filename):
            with sftp.cd(directory):  # chdir to directory
                sftp.get(filename)  # get a remote file
            sftp.close()
            print("Downloaded %s" % filename)
        else:
            print("File %s not found." % (directory + filename))

    os.chdir(cwd)
    return download_dir + '/' + filename
