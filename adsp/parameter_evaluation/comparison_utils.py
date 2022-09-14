import imp
from re import I
import sys
sys.path.extend(['..', '../..'])

from typing import Tuple, Dict, List
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import rc
from scipy.spatial.distance import jensenshannon
from shapely.geometry import box, Point
import contextily as cx

from db_utils import get_rect_to_rect_data

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# use this style for better visibility in histograms
plt.style.use('seaborn-white')
plt_kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, ec="k")

# latex style plot font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 16})
rc('text', usetex=True)

COLORS = ['blue', 'orange', 'green', 'grey', 'brown', 'white', 'black']
IMAGE_DIR = './images/'


def analyze_and_plot_ride_data(ride_data: Dict[str, float], scenario_name: str):
    print("Ride paths:")
    plot_ride_paths(ride_data, scenario_name)

    print("Velocity histograms (normalized):")
    plot_velocity_histograms(ride_data, scenario_name)    

    print("Ride duration histograms (normalized):")
    durations = calculate_and_plot_ride_durations(ride_data, scenario_name)

    return duration_js_divergence(durations)


def get_hist_bins(data: List[List[float]], binwidth: float):
    data_flattened = [item for sublist in data for item in sublist]    
    return np.arange(min(data_flattened), max(data_flattened) + binwidth, binwidth)


def duration_js_divergence(durations: Dict[str, List[float]]):
    reference_data = durations['SimRa']
    hist_bins = get_hist_bins(durations.values(), binwidth=10)
    
    reference_data_hist, _ = np.histogram(reference_data, bins=hist_bins, density=True)

    divergences = {}
    for data_name in [k for k in durations.keys() if k.startswith('SUMO_')]:
        test_data_hist, _ = np.histogram(durations[data_name], bins=hist_bins, density=True)
        divergences[data_name] = jensenshannon(reference_data_hist, test_data_hist)
    
    return divergences


def calculate_and_plot_ride_durations(ride_data: Dict[str, pd.DataFrame], scenario_name: str):
    simra_path_durations = [td.total_seconds() for td in ride_data['SimRa'].groupby('ride_id').first().duration]
    meta_path_durations = {'SimRa': simra_path_durations}

    for ride_data_name in [k for k in ride_data.keys() if k.startswith('SUMO_')]:
        meta_path_durations[ride_data_name] = list(ride_data[ride_data_name].groupby('ride_id').ts.agg(np.ptp))
    
    _, ax = plt.subplots()
    hist_bins = get_hist_bins(list(meta_path_durations.values()), binwidth=2)
    for idx, (ride_data_name, path_durations) in enumerate(meta_path_durations.items()):
        ax.hist(path_durations, color=COLORS[idx], bins=hist_bins, label=ride_data_name, **plt_kwargs)   
    
    ax.set_xlabel("duration (in s)")
    ax.set_ylabel("histogram normalization")
    plt.grid()
    plt.legend()
    plt.title(f"{scenario_name}")
    
    plt.savefig(IMAGE_DIR + 'simra-vs-sumo_ride-durations_' + scenario_name + '.png', transparent=True)
    plt.show()

    return meta_path_durations


def plot_velocity_histograms(ride_data: Dict[str, pd.DataFrame], scenario_name: str):
    # for data_name, ride_data_ in ride_data.items():
    #     print(f"{data_name} - min: {min(ride_data_.velo)} | max: {max(ride_data_.velo)}")
    #     display(ride_data_)
    fig, ax = plt.subplots(figsize=(10, 20))

    hist_bins = get_hist_bins([list(d.velo) for d in ride_data.values()], binwidth=0.2)
    # hist_bins = 10
    for idx, (ride_data_name, ride_data_) in enumerate(ride_data.items()):
        ax = ride_data_.velo.hist(color=COLORS[idx], bins=hist_bins, label=ride_data_name, **plt_kwargs)
    ax.set_xlabel("velocity (in m/s)")
    ax.set_ylabel("histogram normalization")

    plt.title(f"{scenario_name}")
    plt.legend()

    plt.savefig(IMAGE_DIR + 'simra-vs-sumo_velocities_' + scenario_name + '.png', transparent=True)
    plt.show()


def plot_ride_paths(ride_data: Dict[str, pd.DataFrame], scenario_name: str):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # plot rides for each dataframe
    for data_idx, (data_name, df) in enumerate(ride_data.items()):
        df_grouped = df.groupby('ride_id')
        for idx, ride_group_name in enumerate(df_grouped.groups):
            df_ride_group = df_grouped.get_group(ride_group_name)
            
            # only add label for first ride
            label = None
            if idx == 0:
                label = data_name

            ax.plot(df_ride_group.lon, df_ride_group.lat, color=COLORS[data_idx], label=label, linewidth=1)

    # ax.xaxis.set_major_locator(ticker.LinearLocator(4))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    # ax.yaxis.set_major_locator(ticker.LinearLocator(4))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.set_xlabel('Longitude in decimal degrees')
    ax.set_ylabel('Latitude in decimal degrees')

    cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenStreetMap.BZH)
    ax.set_aspect(1.7)

    plt.title(f"{scenario_name}")
    plt.legend()

    plt.savefig(IMAGE_DIR + 'simra-vs-sumo_ride-paths_' + scenario_name + '.png', transparent=True, bbox='tight')
    plt.show()


def get_ride_data(sumo_sim_data_folder: str, sumo_sim_files: List[str], scenario_name: str,
    start_rect_coords: Tuple[float, float], end_rect_coords: Tuple[float, float]):

    sim_data_folder_path = Path(sumo_sim_data_folder)
    
    print(f"----- {scenario_name} -----")
    print("SimRa\n---")
    print(f"Color: {COLORS[0]}")
    df_simra = remove_simra_distance_outliers(get_rect_to_rect_data(start_rect_coords, end_rect_coords))
    df_simra_paths = df_simra[['filename', 'ts', 'lon', 'lat', 'velo', 'time_diff']]
    df_simra_paths.rename({'filename': 'ride_id', 'time_diff': 'duration'}, axis='columns', inplace=True)
    # filter out nans for velo (probably someone started ride exactly in start rect)
    df_simra_paths = df_simra_paths[df_simra_paths.velo.notnull()]
    print("-----")
    
    ride_data = {'SimRa': df_simra_paths}

    for idx, sumo_sim_file in enumerate(sumo_sim_files):
        param_type = '-'.join(sumo_sim_file[:-len('.csv')].split('_')[1:])
        ride_data_name = f"SUMO_{param_type}"
       
        print(f"SUMO [{sumo_sim_file}]\n---")
        print(f"color: {COLORS[idx + 1]}")
        df_sumo = pd.read_csv(sim_data_folder_path / sumo_sim_file, delimiter=';')
        df_sumo_paths = df_sumo[['vehicle_id', 'timestep_time', 'vehicle_x', 'vehicle_y', 'vehicle_speed']]
        df_sumo_paths.rename({'vehicle_id': 'ride_id', 'timestep_time': 'ts', 'vehicle_x': 'lon', 'vehicle_y': 'lat', 'vehicle_speed': 'velo'}, axis='columns', inplace=True)
        # filter out nans (timesteps after flow endend in simulation)
        df_sumo_paths = df_sumo_paths[df_sumo_paths.ride_id.notnull()]
    
        df_sumo_paths_grouped = df_sumo_paths.groupby('ride_id')
        print(f"Number of rides: {len(df_sumo_paths_grouped)}")
        for _, ride_group in df_sumo_paths_grouped:
            df_sumo_paths.drop(index=get_indices_to_delete(ride_group, start_rect_coords, end_rect_coords), inplace=True)
             
        ride_data[ride_data_name] = df_sumo_paths
        print("-----")

    print("-----")
    return ride_data

def remove_simra_distance_outliers(df_simra: pd.DataFrame, iqr_factor: int = 6, verbose: bool = False):
    # filter out riders who drive in circles
    df_simra_grouped = df_simra.groupby('filename').agg({'dist': 'sum'})

    # Computing IQR
    Q1 = df_simra_grouped['dist'].quantile(0.25)
    Q3 = df_simra_grouped['dist'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Filter by multiple of IQR
    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR
    outlier_filenames = list(df_simra_grouped.query('dist < @lower_bound | dist > @upper_bound').index)
    df_filtered = df_simra[~df_simra.filename.isin(outlier_filenames)]

    if verbose:
        df_simra_grouped.hist()
        print(f"lowerbound: {lower_bound}, upper_bound: {upper_bound}")
        print(f"n_rides unfiltered: {df_simra_grouped.size}, n_rides filtered: {len(df_filtered.groupby('filename').groups)}")
  
    return df_filtered


def get_indices_to_delete(ride_group, start_rect_coords: Tuple[float, float], end_rect_coords: Tuple[float, float]):
    start_rect = box(*start_rect_coords)
    end_rect = box(*end_rect_coords)
    
    mask_first = ride_group.apply(lambda coord: start_rect.contains(Point(coord['lon'], coord['lat'])), axis=1)
    mask_end = ride_group.apply(lambda coord: end_rect.contains(Point(coord['lon'], coord['lat'])), axis=1)    
    try:
        start_idx = mask_first[mask_first==True].index[0]
        end_idx = mask_end[mask_end==True].index[-1]
        return [idx for idx in ride_group.index if idx < start_idx or idx > end_idx]
    except: 
        # probably vehicle (ride) does not arrive in end box because simulation ended beforehand
        return list(ride_group.index)





