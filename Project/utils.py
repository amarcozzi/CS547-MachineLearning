import os
import datetime
import numpy as np
import rioxarray as rioxr
from tqdm import tqdm
from rioxarray.exceptions import NoDataInBounds


def build_filename(year, month):
    if month < 10:
        month_str = '0' + str(month)
    else:
        month_str = str(month)
    fmt = '%Y.%m.%d'
    date_str = str(year) + '.' + month_str + '.01'
    dt = datetime.datetime.strptime(date_str, fmt)
    tt = dt.timetuple()
    # The point of the above, the Julian day
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

    return "MCD64monthly." + date_str + '.' + window + '.' + version + '.' + layer + '.tif', yd


def label_array_func(a):
    labeled = np.array(a > 0, dtype=np.int16)
    ind = np.argwhere(labeled > 0)
    curr_label = 2
    for pos in ind:
        pos = tuple(pos)
        if labeled[pos] == 1:
            label_cluster(pos, labeled, curr_label)
            curr_label += 1

    print("Identified and labeled %i unique clusters." % (curr_label - 1))
    return labeled


def get_nn(pos, a):
    """
    returns a list of nearest neigbors, nn
    ignores neighbors that would be beyond
    the edges of the matrix a
    """

    n, m = a.shape
    i, j = pos[0], pos[1]
    pos = np.array(pos)
    nn = []
    if i < (n - 1):
        nn.append(pos + np.array([1, 0]))
    if j < (m - 1):
        nn.append(pos + np.array([0, 1]))
    if i > 0:
        nn.append(pos + np.array([-1, 0]))
    if j > 0:
        nn.append(pos + np.array([0, -1]))
    return nn


def label_cluster(pos, a, l):
    """
    Given an array a that contains 1 where there
    is a fire, and non-zero, non-one for labeled
    clusters. This method recursivly assigns
    l to all members in the cluster identified at
    position pos.

    Method is 'flood fill'.
    """
    a[pos] = l
    nn = get_nn(pos, a)

    for n_pos in nn:
        if a[tuple(n_pos)] == 1:
            label_cluster(tuple(n_pos), a, l)


def test_raster_bounds(a, b):
    if a[0] >= b[0] or a[1] >= b[1] or a[2] <= b[2] or a[3] <= b[3]:
        return False
    return True


def prune_ts(l):
    mv = max(l)
    mv_ind = l.index(mv)
    nl = []
    v = -1
    i = mv_ind
    while v != 0 and i != -1:
        v = l[i]
        nl.insert(0, v)
        i -= 1
    min_ind = i
    v = -1
    i = mv_ind + 1
    while v != 0 and i != len(l):
        v = l[i]
        nl.append(v)
        i += 1
    max_ind = i
    return nl, range(min_ind + 1, max_ind)


def get_pad_span_and_shape(dem, accumulator, label_array, prune_data_flag, res, n):
    max_shape = [0, 0]
    max_bounds = (0., 0., 0., 0.)
    max_size = 0

    # Loop over all fires in the year
    max_fid = label_array.max()
    for fid in tqdm(range(2, max_fid)):
        days = accumulator.values.squeeze()[label_array == fid]
        start = int(days.min())
        stop = int(days.max())
        pixels_burned_time_series = [days[days == i].size for i in range(start, stop + 1)]
        pruned_ts, ts_ind = prune_ts(pixels_burned_time_series)
        num_days_in_ts = len(pruned_ts)

        # Let's get rid of the tiny fires
        if num_days_in_ts < 5:
            continue

        vals = find_fire_size(start, stop, accumulator, label_array, fid)
        row_min, row_max, col_min, col_max, lat_min, lat_max, lon_min, lon_max = vals

        accum_subset = accumulator.rio.clip_box(minx=lon_min - n * res, miny=lat_min - n * res, maxx=lon_max + n * res,
                                                maxy=lat_max + n * res, auto_expand=True)
        bounds = accum_subset.rio.bounds()

        # Sample the DEM and SB40 raster
        try:
            dem_subset = dem.rio.clip_box(*bounds)
        except NoDataInBounds:
            continue

        size = dem_subset.values.size
        if size > max_size:
            max_shape = accum_subset.values.squeeze().shape
            max_bounds = accum_subset.rio.bounds()

    return max_shape, max_bounds


def get_bounds_span(accumulator, label_array, prune_data_flag, res, n):
    return


def find_fire_size(start, stop, accumulator, label_array, fid):
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

    return row_min, row_max, col_min, col_max, lat_min, lat_max, lon_min, lon_max


def get_fire_days(accumulator, label_array, fid, prune_data_flag):
    days = accumulator.values.squeeze()[label_array == fid]
    start = int(days.min())
    stop = int(days.max())
    pixels_burned_time_series = [days[days == i].size for i in range(start, stop + 1)]
    pruned_ts, ts_ind = prune_ts(pixels_burned_time_series)
    num_days_in_ts = len(pruned_ts)

    # Let's get rid of the tiny fires
    if num_days_in_ts < 5:
        return None

    if prune_data_flag:
        old_start = start
        old_stop = stop
        stop = min(old_stop, start + ts_ind[-1] + 1)
        start += ts_ind[0]

    return start, stop


def accumulate_year(year, start_month, end_month, in_dir):
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

    return accumulator


def accumulator_loader(year, start_month, end_month, data_path):
    in_path = os.path.join(data_path, 'burned_area_files')
    accum_path = os.path.join(data_path, 'accumulated_fires')
    accum_name = f'{year}-fires.tif'

    # Load in the accumulated fires if the file has already been created
    fpath = os.path.join(accum_path, accum_name)
    if os.path.exists(fpath):
        return rioxr.open_rasterio(fpath)

    # Run the accumulator
    else:
        accumulator = accumulator = accumulate_year(year, start_month, end_month, in_path)
        accumulator = accumulator.rio.reproject('EPSG:5070', nodata=0.)
        accumulator.rio.to_raster(fpath)
        return accumulator


def round_idx_to_max(row_min, row_max, col_min, col_max, max_shape, fid):
    # Modify the max/min rows and columns to match the default size
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

    return row_min, row_max, col_min, col_max