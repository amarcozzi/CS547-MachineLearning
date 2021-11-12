import datetime
import numpy as np


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
        nl.insert(0,v)
        i-=1
    min_ind=i
    v = -1
    i = mv_ind+1
    while v != 0 and i != len(l):
        v = l[i]
        nl.append(v)
        i+=1
    max_ind = i
    return nl,range(min_ind,max_ind-1)