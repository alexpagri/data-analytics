import imp
import sys
sys.path.extend(['..', '../..'])

from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from shapely.geometry import box, Point

from db_utils import get_rect_to_rect_data

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

def analyze_and_plot_ride_data(ride_data: Dict[str, float]):
    print("Ride paths:")
    plot_ride_paths(ride_data)

    print("Velocity histograms (normalized):")
    plot_velocity_histograms(ride_data)    

    print("Ride duration histograms (normalized):")
    calculate_and_plot_ride_durations(ride_data)

def plot_velocity_histograms(ride_data: Dict[str, pd.DataFrame]):
    ax_simra = ride_data['SimRa'].velo.hist(density=True, color='blue', alpha=0.5)
    ax_sumo = ride_data['SUMO'].velo.hist(density=True, color='orange', alpha=0.5)
    for ax in [ax_simra, ax_sumo]:
        ax.set_ylabel("velocity")
    plt.show()

def plot_ride_paths(ride_data: Dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect(1.5)
    
    colors = ['blue', 'orange']

    # plot rides for each dataframe
    for data_idx, (data_name, df) in enumerate(ride_data.items()):
        df_grouped = df.groupby('ride_id')
        for ride_group_name in df_grouped.groups:
            df_ride_group = df_grouped.get_group(ride_group_name)
            ax.plot(df_ride_group.lon, df_ride_group.lat, color=colors[data_idx], label=data_name, linewidth=1)
            # add labels to legend

    ax.xaxis.set_major_locator(ticker.LinearLocator(4))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_locator(ticker.LinearLocator(4))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.set_xlabel('Longitude in decimal degrees')
    ax.set_ylabel('Latitude in decimal degrees')

    # plt.savefig('simra_vs_sumo_ride_path.png', transparent=True)
    # plt.legend()
    plt.show()


def calculate_and_plot_ride_durations(ride_data: Dict[str, pd.DataFrame]):
    sumo_path_durations = list(ride_data['SUMO'].groupby('ride_id').ts.agg(np.ptp))
    simra_path_durations = [td.total_seconds() for td in ride_data['SimRa'].groupby('ride_id').first().duration]

    _, ax = plt.subplots()
    ax.hist(sumo_path_durations, color='orange', alpha=0.5, density=True, bins=10)
    ax.hist(simra_path_durations, color='blue', alpha=0.5, density=True, bins=60)
    ax.set_xlabel("duration in s")
    ax.set_ylabel("normalized histogram")
    plt.xlim(0, 100)
    plt.show()



def get_ride_data(sumo_sim_file: str, start_rect_coords: Tuple[float, float], end_rect_coords: Tuple[float, float]):
    print("SimRa:")
    df_simra = get_rect_to_rect_data(start_rect_coords, end_rect_coords)
    df_simra_paths = df_simra[['filename', 'ts', 'lon', 'lat', 'velo', 'time_diff']]
    df_simra_paths.rename({'filename': 'ride_id', 'time_diff': 'duration'}, axis='columns', inplace=True)

    print("SUMO:")
    df_sumo = pd.read_csv(sumo_sim_file, delimiter=';')
    df_sumo_paths = df_sumo[['vehicle_id', 'timestep_time', 'vehicle_x', 'vehicle_y', 'vehicle_speed']]
    df_sumo_paths.rename({'vehicle_id': 'ride_id', 'timestep_time': 'ts', 'vehicle_x': 'lon', 'vehicle_y': 'lat', 'vehicle_speed': 'velo'}, axis='columns', inplace=True)
    
    df_sumo_paths_grouped = df_sumo_paths.groupby('ride_id')
    print(f"Number of rides: {len(df_sumo_paths_grouped)}")
    for _, ride_group in df_sumo_paths_grouped:
        df_sumo_paths.drop(index=get_indices_to_delete(ride_group, start_rect_coords, end_rect_coords), inplace=True)

    return {'SimRa': df_simra_paths, 'SUMO': df_sumo_paths}

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





