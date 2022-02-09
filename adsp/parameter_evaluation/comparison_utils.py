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
from shapely.geometry import box, Point
import contextily as cx

from db_utils import get_rect_to_rect_data

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# use this style for better visibility in histograms
plt.style.use('seaborn-white')
plt_kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, ec="k")

COLORS = ['blue', 'orange', 'green']

def analyze_and_plot_ride_data(ride_data: Dict[str, float]):
    print("Ride paths:")
    plot_ride_paths(ride_data)

    print("Velocity histograms (normalized):")
    plot_velocity_histograms(ride_data)    

    print("Ride duration histograms (normalized):")
    calculate_and_plot_ride_durations(ride_data)

def get_hist_bins(data: List[List[float]], binwidth: float):
    data_flattened =  [item for sublist in data for item in sublist]
    return np.arange(min(data_flattened), max(data_flattened) + binwidth, binwidth)

def plot_velocity_histograms(ride_data: Dict[str, pd.DataFrame]):
    hist_bins = get_hist_bins([list(d.velo) for d in ride_data.values()], binwidth=0.5)
    for idx, (ride_data_name, ride_data) in enumerate(ride_data.items()):
        ax = ride_data.velo.hist(color=COLORS[idx], bins=hist_bins, label=ride_data_name, **plt_kwargs)
    ax.set_xlabel("velocity (in m/s)")
    ax.set_ylabel("histogram normalization")
    plt.legend()
    plt.show()

def plot_ride_paths(ride_data: Dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect(1.5)
    
    # plot rides for each dataframe
    for data_idx, (data_name, df) in enumerate(ride_data.items()):
        df_grouped = df.groupby('ride_id')
        for ride_group_name in df_grouped.groups:
            df_ride_group = df_grouped.get_group(ride_group_name)
            ax.plot(df_ride_group.lon, df_ride_group.lat, color=COLORS[data_idx], label=data_name, linewidth=1)
            # add labels to legend

    ax.xaxis.set_major_locator(ticker.LinearLocator(4))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_locator(ticker.LinearLocator(4))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.set_xlabel('Longitude in decimal degrees')
    ax.set_ylabel('Latitude in decimal degrees')

    cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.Stamen.Toner)

    # plt.savefig('simra_vs_sumo_ride_path.png', transparent=True)
    # plt.legend()
    plt.show()

def calculate_and_plot_ride_durations(ride_data: Dict[str, pd.DataFrame]):
    simra_path_durations = [td.total_seconds() for td in ride_data['SimRa'].groupby('ride_id').first().duration]
    meta_path_durations = {'SimRa': simra_path_durations}

    for ride_data_name in [k for k in ride_data.keys() if k.startswith('SUMO_')]:
        meta_path_durations[ride_data_name] = list(ride_data[ride_data_name].groupby('ride_id').ts.agg(np.ptp))
    
    _, ax = plt.subplots()
    hist_bins = get_hist_bins(list(meta_path_durations.values()), binwidth=5)
    for idx, (ride_data_name, path_durations) in enumerate(meta_path_durations.items()):
        ax.hist(path_durations, color=COLORS[idx], bins=hist_bins, label=ride_data_name, **plt_kwargs)   
    ax.set_xlabel("duration (in s)")
    ax.set_ylabel("histogram normalization")
    plt.legend()
    plt.grid()
    plt.show()


def get_ride_data(sumo_sim_data_folder: str, scenario_name: str, sumo_sim_files: List[str], 
        start_rect_coords: Tuple[float, float], end_rect_coords: Tuple[float, float], **kwargs):

    sim_data_folder_path = Path(sumo_sim_data_folder)
    
    print("SimRa:")
    print(f"color: {COLORS[0]}")
    df_simra = remove_simra_distance_outliers(get_rect_to_rect_data(start_rect_coords, end_rect_coords))
    df_simra_paths = df_simra[['filename', 'ts', 'lon', 'lat', 'velo', 'time_diff']]
    df_simra_paths.rename({'filename': 'ride_id', 'time_diff': 'duration'}, axis='columns', inplace=True)

    ride_data = {'SimRa': df_simra_paths}

    for idx, sumo_sim_file in enumerate(sumo_sim_files):
        print(f"SUMO [{sumo_sim_file}]:")
        print(f"color: {COLORS[idx + 1]}")
        df_sumo = pd.read_csv(sim_data_folder_path / sumo_sim_file, delimiter=';')
        df_sumo_paths = df_sumo[['vehicle_id', 'timestep_time', 'vehicle_x', 'vehicle_y', 'vehicle_speed']]
        df_sumo_paths.rename({'vehicle_id': 'ride_id', 'timestep_time': 'ts', 'vehicle_x': 'lon', 'vehicle_y': 'lat', 'vehicle_speed': 'velo'}, axis='columns', inplace=True)
    
        df_sumo_paths_grouped = df_sumo_paths.groupby('ride_id')
        print(f"Number of rides: {len(df_sumo_paths_grouped)}")
        for _, ride_group in df_sumo_paths_grouped:
            df_sumo_paths.drop(index=get_indices_to_delete(ride_group, start_rect_coords, end_rect_coords), inplace=True)
        
        ride_data_name = f"SUMO_{sumo_sim_file[:-len('.csv')]}"
        ride_data[ride_data_name] = df_sumo_paths

    return ride_data

def remove_simra_distance_outliers(df_simra: pd.DataFrame, iqr_factor: int = 6, verbose: bool = False):
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





