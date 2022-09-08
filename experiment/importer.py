import csv
import os
from datetime import datetime
import numpy as np
from vg import angle
import pandas as pd
from geopy.distance import great_circle
from pyproj import Proj
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import math
import utils

import warnings

warnings.filterwarnings('ignore')


def handle_ride_file(file):
    with open(file, 'r') as f:
        ride_data = []
        split_found = False
        for line in f.readlines():
            if "#" in line:
                version = line.split('#')[0]
            if "======" in line:
                split_found = True
                continue
            if split_found:
                ride_data.append(line)
        df = ride_file_to_df(ride_data, version)
        df['date'] = utils.interpolate_dates(df, 'date_raw')
        return df


def ride_file_to_df(ride_data, version):
    data = csv.DictReader(ride_data[1:], delimiter=",")

    raw_coords = []
    accuracies = []
    timestamps = []

    ride_df = []

    j = 0
    for i, row in enumerate(data):
        if row["lat"]:
            raw_coords.append([float(row["lon"]), float(row["lat"])])
            try:
                if row["acc"]:
                    # if float(row["acc"]) > 100.0:  # ride goes to trash
                    #    print("Ride is filtered due to accuracies > 100m")
                    #    return
                    accuracies.append(float(row["acc"]))
            except KeyError:
                return
            ts = datetime.utcfromtimestamp(int(row["timeStamp"]) / 1000)  # timeStamp is in Java TS Format
            timestamps.append(ts)
            coord_index = j
            j += 1
        try:
            if row["X"]:
                if row['lon']:
                    ts = datetime.utcfromtimestamp(int(row["timeStamp"]) / 1000)
                else:
                    ts = np.datetime64('NaT')
            line = ()
            line += (
                i, coord_index, raw_coords[-1][0], raw_coords[-1][1], accuracies[-1], ts, float(row["X"]),
                float(row["Y"]),
                float(row["Z"]),)
            if version >= '72':
                line += (float(row["XL"]), float(row["YL"]), float(row["ZL"]), float(row["RX"]), float(row["RY"]),
                         float(row["RZ"]), float(row["RC"]),)
            else:
                line += (
                    float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'),)

            ride_df.append(line)
        except TypeError as e:
            raise (e)
            return

    if len(raw_coords) == 0:
        print("Ride is filtered due to len(coords) == 0")
        return

    if is_teleportation(timestamps):
        print("Ride is filtered due to teleportation")
        return

    return pd.DataFrame(ride_df,
                        columns=['id', 'coord_index', 'lon', 'lat', 'acc', 'date_raw', 'X', 'Y', 'Z', 'XL', 'YL', 'ZL',
                                 'RX', 'RY', 'RZ', 'RC'])


def is_teleportation(timestamps):
    for i, t in enumerate(timestamps):
        if i + 1 < len(timestamps):
            if (timestamps[i + 1] - timestamps[i]).seconds > 20:
                return True
    return False


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def low_pass(data):
    b, a = butter(10, 0.05, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def calc_dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def rolling_window(a, step):
    shape = a.shape[:-1] + (a.shape[-1] - step + 1, step)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


## From https://scikit-surgerycore.readthedocs.io/en/latest/_modules/sksurgerycore/algorithms/averagequaternions.html#average_quaternions
def average_quaternions(quaternions):

    # Number of quaternions to average
    samples = quaternions.shape[0]
    mat_a = np.zeros(shape=(4, 4), dtype=np.float64)

    for i in range(0, samples):
        quat = quaternions[i, :]
        # multiply quat with its transposed version quat' and add mat_a
        mat_a = np.outer(quat, quat) + mat_a

    # scale
    mat_a = (1.0/ samples)*mat_a
    # compute eigenvalues and -vectors
    eigen_values, eigen_vectors = np.linalg.eig(mat_a)
    # Sort by largest eigenvalue
    eigen_vectors = eigen_vectors[:, eigen_values.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(np.ravel(eigen_vectors[:, 0]))


def avg_rotations(rot_arr):
    rs = [R.from_matrix(rot) for rot in rot_arr]
    qs = [tuple(r.as_quat()) for r in rs]
    r = R.from_quat(average_quaternions(np.array(qs)))
    return R.as_matrix(r)


def import_files(path):
    files = []
    for r, d, f in os.walk(path, followlinks=True):
        for file in f:
            if '.' not in file:
                files.append(os.path.join(r, file))

    dfs = []
    i = 0
    for file in tqdm(files):
        if "Profiles" in file:
            continue
        dfs.append(handle_ride_file(file))
        print(file, i)
        i += 1
    return dfs



### Vars with _bike are specified in the reference system of the cyclists (x is direction of travel).
### Vars with _world are specified in the reference system of the world (x is east, y is north).
def preprocess_basics(df, simplify):
    g = 9.81

    df['G_raw'] = df.apply(lambda x: np.array([x['X'], x['Y'], x['Z']]), axis=1)

    df['date'] = utils.interpolate_dates(df, 'date')

    # cf = DataFrame with only Coord Entries
    cf = df[~df['date_raw'].isnull()]
    cf_shift = cf.shift(1)

    cf['l_lon'] = cf_shift['lon']
    cf['l_lat'] = cf_shift['lat']
    cf = cf[~cf['l_lon'].isnull()]
    cf['duration'] = (cf['date_raw'] - cf_shift['date_raw']).dt.total_seconds()
    cf['dist'] = cf.apply(lambda x: great_circle([x['l_lat'], x['l_lon']], [x['lat'], x['lon']]).meters, axis=1)
    cf['velo'] = cf['dist'] / cf['duration']

    cf_shift = cf.shift(1)

    cf['a'] = (cf['velo'] - cf_shift['velo']) / cf['duration']

    proj = Proj('epsg:5243')

    proj_coords = cf.apply(lambda x: proj(x['lon'], x['lat']), axis=1)
    cf.loc0[:, ['x', 'y']] = list(map(list, proj_coords))

    cf['dir_x'] = cf['x'] - cf.shift(1)['x']
    cf['dir_y'] = cf['y'] - cf.shift(1)['y']
    cf['velo_vec'] = cf.apply(lambda x: utils.resize_vector_to_one(np.array([x['dir_x'], x['dir_y']])) * x['velo'], axis=1)
    cf['a_wrld'] = cf.apply(
        lambda x: np.nan_to_num(np.hstack([utils.resize_vector_to_one(np.array([x['dir_x'], x['dir_y']])) * x['a'], g])),
        axis=1)

    cf['a_bike'] = cf.apply(lambda x: np.nan_to_num(np.array([x['a'], 0, g])), axis=1)

    df['duration'] = cf['duration']
    df['velo'] = cf['velo']
    df['velo_vec'] = cf['velo_vec']
    df['velo_vec'] = df['velo_vec'].fillna(method='bfill')
    df['a_bike'] = cf['a_bike']
    df['a'] = cf['a']
    df['a_wrld'] = cf['a_wrld']
    df['x'] = cf['x']
    df['y'] = cf['y']
    df['x'] = df['x'].interpolate()
    df['y'] = df['y'].interpolate()
    df['dist'] = cf['dist']

    df['G_size'] = df.apply(lambda x: np.linalg.norm(x['G_raw']), axis=1)

    first_velo_entry = df[~df['velo'].isnull()].iloc[0].id
    last_velo_entry = df[~df['velo'].isnull()].iloc[-1].id
    df_cut = df[first_velo_entry:last_velo_entry + 1]

    df_cut['a_bike'] = df_cut['a_bike'].fillna(method='backfill')
    df_cut['a_wrld'] = df_cut['a_wrld'].fillna(method='backfill')

    df_cut['a_size'] = df_cut.apply(lambda x: np.linalg.norm(x['a_bike']), axis=1)

    df_cut['spike'] = utils.find_velocity_spikes(df_cut)

    df_cut['section'] = df_cut['spike'].cumsum()
    df_cut['section'] = df_cut.apply(lambda x: int(x['section']) if x['spike'] is False else -1, axis=1)

    if simplify:
        return simplify_for_gps_analysis(df_cut)
    else:
        return df_cut


### Hereby, the findings from thesis 3.2.2 were derived.
### The R_b are Rotation matrices for rotating the acceleration data into the reference system of the bike
### The R_2 are Rotation matrices for rotating the acceleration data into the reference system of the world
### It was tried to average over the Rotation matrices to obtain more resilient results. Thus, the R_mean_X vars were calculated.
def preprocess_advanced(df_cut):

    df_cut = preprocess_basics(df_cut, False)

    rw_size = 10

    #df_cut['R_b'] = df_cut.apply(lambda x: rotation_matrix_from_vectors(x['G_raw'], x['a_bike']), axis=1)
    df_cut['R_w'] = df_cut.apply(lambda x: rotation_matrix_from_vectors(x['G_raw'], x['a_wrld']), axis=1)

    groups = df_cut.groupby(['coord_index'])

    df_new = pd.DataFrame()
    df_new['coord_index'] = groups.apply(lambda x: int(np.mean(x['coord_index'])))

    # mean of 1 coord index (~3sek). Calulation takes times. Therefore, one (bike or world) is commented to safe time.
    #df_new['R_mean'] = groups.apply(lambda x: avg_rotations(x['R_b'].to_numpy()))
    #mean_dict = df_new.set_index('coord_index').T.to_dict('list')
    #df_cut['R_mean_b'] = df_cut['coord_index'].map(mean_dict)
    #df_cut['R_mean_b'] = df_cut['R_mean_b'].apply(lambda x: x[0])

    # mean of 1 coord index (~3sek)
    df_new['R_mean'] = groups.apply(lambda x: avg_rotations(x['R_w'].to_numpy()))
    mean_dict = df_new.set_index('coord_index').T.to_dict('list')
    df_cut['R_mean_w'] = df_cut['coord_index'].map(mean_dict)
    df_cut['R_mean_w'] = df_cut['R_mean_w'].apply(lambda x: x[0])


    # Moving average approach. Window size can be adjusted via $rw_size. Calulation takes times. Therefore, one (bike or world) is commented to safe time.
    # rws = rolling_window(df_cut['R_b'].to_numpy(), rw_size)
    # buffer = [(np.zeros(9) + np.nan).reshape(3, 3) for i in range(len(df_cut.index) - len(rws))]
    # R_mas = [avg_rotations(w) for w in rws]
    # R_mas = np.vstack([buffer, R_mas])
    # df_cut['R_ma_b'] = R_mas.tolist()

    rws = rolling_window(df_cut['R_w'].to_numpy(), rw_size)
    buffer = [(np.zeros(9) + np.nan).reshape(3, 3) for i in range(len(df_cut.index) - len(rws))]
    R_mas = [avg_rotations(w) for w in rws]
    R_mas = np.vstack([buffer, R_mas])
    df_cut['R_ma_w'] = R_mas.tolist()

    df_cut['G_clean'] = df_cut.apply(lambda x: np.dot(x['R_ma_w'], x['G_raw']), axis=1)
    #df_cut['G_clean'] = df_cut.apply(lambda x: np.dot(x['R_ma_b'], x['G_raw']), axis=1)


    df_cut['X_'] = df_cut.apply(lambda x: x['G_clean'][0], axis=1)
    df_cut['Y_'] = df_cut.apply(lambda x: x['G_clean'][1], axis=1)
    df_cut['Z_'] = df_cut.apply(lambda x: x['G_clean'][2], axis=1)

    df_cut = df_cut[~df_cut['X_'].isnull()]

    df_cut['angle_G_Gc'] = df_cut.apply(lambda x: angle(x['G_clean'], x['G_raw']), axis=1)
    df_cut['angle_G_1'] = df_cut.apply(lambda x: angle(x['G_raw'], np.array([1, 0, 0])), axis=1)
    df_cut['angle_Gc_1'] = df_cut.apply(lambda x: angle(x['G_clean'], np.array([1, 0, 0])), axis=1)
    df_cut['angle_G_aw'] = df_cut.apply(lambda x: angle(x['G_raw'], x['a_wrld']), axis=1)
    df_cut['angle_Gc_aw'] = df_cut.apply(lambda x: angle(x['G_clean'], x['a_wrld']), axis=1)

    ### Here the acceleration is intigrated. If the rotations would procude sufficient results, the $check would equal the $velo var.
    ### As they do not, the rotations are no adequate preprocessing measure for the SimRa data.
    df_cut['check'] = (df_cut['X_'] * (df_cut.shift(-1)['date'] - df_cut['date']).dt.total_seconds()).cumsum() + \
                 df_cut[~df_cut['velo'].isnull()].iloc[0]['velo']

    return df_cut


def simplify_for_gps_analysis(df):
    df = df[~df['date_raw'].isnull()]
    df = df.drop(['X', 'Y', 'Z', 'XL', 'YL', 'ZL', 'RX', 'RY', 'RZ', 'RC', 't_int', 'date_raw'], axis=1)
    return df
