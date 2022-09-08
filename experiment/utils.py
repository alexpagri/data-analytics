import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import math


def resize_vector_to_one(vec):
    length = np.linalg.norm(vec)
    if length == 0.0:
        return np.array([0., 0.])
    else:
        return vec / length


def interpolate_dates(df, date_column):
    t0 = df[date_column].min()
    m = df[date_column].notnull()
    df.loc0[m, 't_int'] = (df.loc0[m, date_column] - t0).dt.total_seconds()

    return t0 + pd.to_timedelta(df.t_int.interpolate(), unit='s')


def low_pass(data, order, fv):
    b, a = butter(order, fv, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def calc_dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_velocity_spikes_(df):
    threshold = df['a'].std() * 0.8
    print(df['a'].std(), threshold)

    return ((df['a'] > threshold) & (df.shift(-1)['a'] < -threshold)) | (df['a'] > 1.3)


def find_velocity_spikes(df):
    threshold = df['a'].std() * 0.5
    print(df['a'].std(), threshold)

    return ((df[~df.velo.isnull()]['a'] > threshold) & (df[~df.velo.isnull()].shift(-1)['a'] < -threshold)) | (df['a'] > 2)

def find_closest_index(df, dt_obj):
    return np.argmin(np.abs(df['timestamp'] - dt_obj))
