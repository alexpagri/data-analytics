from settings import *
from pyproj import Proj
import numpy as np
from scipy.signal import butter, filtfilt

SILENT = True

def apply_smoothing_filters(ride_df):
    ride_df = apply_gauss_kernel_location(ride_df)
    ride_df = remove_speed_artifacts(ride_df)
    ride_df = apply_low_pass_velocity(ride_df)
    return ride_df

def remove_speed_artifacts(ride_df):
    if len(ride_df[ride_df.velo > 20]) > 5:
        raise Exception("Too many speed artifacts > 20")
    ride_df.loc[ride_df.velo > 20, 'velo'] = 20
    return ride_df

def apply_gauss_kernel_location(ride_df):
    win_type = 'gaussian'
    window_size = 22
    std = 2.5
    max_bad_acc = 100
    normalized_acc = np.power(ride_df.accuracy, 2.0/5) / np.power(max_bad_acc, 2.0/5)
    ride_df['x_k'] = ride_df.x.rolling(window=window_size, win_type=win_type, center=True, min_periods=1).mean(std=std) * (normalized_acc) + ride_df.x * (1 - normalized_acc)
    ride_df['y_k'] = ride_df.y.rolling(window=window_size, win_type=win_type, center=True, min_periods=1).mean(std=std) * (normalized_acc) + ride_df.y * (1 - normalized_acc)

    proj = Proj('epsg:5243')
    proj_coords = ride_df.apply(lambda x: proj(x['x_k'], x['y_k'], inverse=True), axis=1)
    ride_df.loc[:, ['lon_k', 'lat_k']] = list(map(list, proj_coords))

    return ride_df


# not in use
def apply_gauss_kernel_velocity(ride_df):
    tmp = ride_df[ride_df['section'] >= 0]
    threshold = 0.5
    win_type = 'gaussian'
    window_size = 17
    std = 2.0
    tmp['velo_k0'] = np.where(tmp.velo < threshold, 0,
                              tmp.velo.rolling(window=window_size, win_type=win_type, center=True, min_periods=1).mean(
                                  std=std))
    ride_df['velo_k0'] = tmp['velo_k0']
    return ride_df


def low_pass(data, order, fv):
    b, a = butter(order, fv, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def apply_low_pass_velocity(ride_df):
    threshold = 0.45
    tmp = ride_df[ride_df['section'] >= 0].iloc[1:]
    lp_order = 1
    lp_filter_value = 0.12
    tmp['velo_lp0'] = np.where(tmp.velo < threshold, 0, low_pass(tmp.velo, lp_order, lp_filter_value))
    ride_df['velo_lp0'] = tmp['velo_lp0']
    ride_df['velo_lp0'] = ride_df.iloc[1:]['velo_lp0'].interpolate() # compute AVG for NaN values
    return ride_df


def apply_removal_filters(ride_df):
    ride_distance = calc_spatial_dist(ride_df[['x', 'y']].to_numpy())
    ride_duration = (ride_df.iloc[-1].timestamp - ride_df.iloc[0].timestamp).seconds
    return apply_short_distance_filter(ride_distance) | \
           apply_short_duration_filter(ride_duration) | \
           apply_high_avg_speed_filter(ride_distance, ride_duration) | \
           apply_user_forgot_to_stop_filter(ride_df)


def apply_short_distance_filter(dist):
    if dist < MIN_RIDE_DISTANCE:
        if not SILENT: print("Ride filtered due to short distance ({}m).".format(dist))
        return True
    else:
        return False


def apply_short_duration_filter(duration):
    if duration < MIN_RIDE_DURATION:
        if not SILENT: print("Ride filtered due to short duration ({}sec).".format(duration))
        return True
    else:
        return False


def apply_high_avg_speed_filter(distance, duration):
    if duration <= 0:
        return True
    avg_speed = (distance / duration) * 3.6
    if avg_speed > MAX_RIDE_AVG_SPEED:
        if not SILENT: print("Ride filtered due to high average speed ({}km/h).".format(duration))
        return True
    else:
        return False


#   heuristic approach
#
#   ride will be classified as 'forgot to stop' when User does not
#   exceed $MIN_DISTANCE_TO_COVER_IN_5_MIN in 5min (300sec) (300*6000millis)
#
#   5min in 3sec steps = 100steps
#
def apply_user_forgot_to_stop_filter(ride_df):
    for i in range(len(ride_df)):
        if i + 100 < len(ride_df):
            dist = calc_spatial_dist(ride_df[['x', 'y']].to_numpy()[i:i + 100])
        else:
            break
        if dist < MIN_DISTANCE_TO_COVER_IN_5_MIN:
            if not SILENT: print("Ride filtered due to user forgot to stop recording (user only covered {}m in ~5min).".format(dist))
            return True
    return False


def calc_spatial_dist(coords):
    x = coords[:, 0]
    y = coords[:, 1]
    dist_array = (x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2
    return np.sum(np.sqrt(dist_array))
