import sys
# add parent directory and its parent to sys.path so that python finds the modules
sys.path.append('..')

from typing import Tuple, List, Dict
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from db_utils import get_rect_to_rect_data

plt.rcParams.update({
    "figure.facecolor":  'white', 
    "axes.facecolor":    'white', 
    "savefig.facecolor": 'white', 
})

def get_path_rotated(df_simra: pd.DataFrame) -> Tuple[np.float, np.float]:
    # rotate vector by 90 degrees clockwise
    #    (x, y)     -> (y, -x)
    # =  (lon, lat) -> (lat, -lon

    lat_start = df_simra.iloc[0].lat
    lat_end = df_simra.iloc[-1].lat

    delta_lat = lat_end - lat_start

    lon_start = df_simra.iloc[0].lon
    lon_end = df_simra.iloc[-1].lon

    delta_lon = lon_end - lon_start

    path_rotated_lon_unnormalized = delta_lat
    path_rotated_lat_unnormalized = (-1) *  delta_lon

    path_rotated_lon = path_rotated_lon_unnormalized / np.sqrt(path_rotated_lon_unnormalized**2+ path_rotated_lat_unnormalized**2)
    path_rotated_lat = path_rotated_lat_unnormalized / np.sqrt(path_rotated_lon_unnormalized**2+ path_rotated_lat_unnormalized**2)

    # return path_rotated_lon, path_rotated_lat
    return path_rotated_lat, path_rotated_lon

def get_max_projections(df_simra_grouped: pd.DataFrame, df_simra: pd.DataFrame, path_rotated: Tuple[np.float, np.float]) -> np.ndarray:
    max_projections = []
    df_simra_grouped_ = df_simra_grouped.reset_index()
    for filename in df_simra_grouped_['filename']:
        projections = []
        x, y = path_rotated
        x_norm = x / np.sqrt(x**2 + y**2)
        y_norm = y / np.sqrt(x**2 + y**2)
        for lat, lon in zip(df_simra[df_simra['filename'] == filename]['lat'], df_simra[df_simra['filename'] == filename]['lon']):
            projection = (lat * x_norm) + (lon * y_norm)
            projections.append(projection)
        max_projections.append(np.max(projections))

    # df_simra_grouped['max_projection'] = max_projections
    return np.array(max_projections)


def min_max_scale_features(features: Dict[str, np.ndarray]):
    Scaler = MinMaxScaler()
    return {feature_name: Scaler.fit_transform(feature_values.reshape(-1, 1))
            for feature_name, feature_values in features.items()}


def cluster_with_kmeans(features: Dict[str, np.ndarray], n_cluster: int = 2, plot: bool = True, **kwargs) -> np.ndarray:
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)

    feature_names = list(features.keys())
    features = list(features.values())
    features_combined = np.hstack(features)
    cluster_labels = kmeans.fit_predict(features_combined)
    cluster_centers = kmeans.cluster_centers_

    # does only work with n_cluster = 2
    colors = ['blue' if label == 0 else 'orange' for label in cluster_labels]
    
    if 'direction' in kwargs:
        plt.title('k-means clustering\n' + kwargs['direction'])
            
    if plot and len(features) == 2:
        plt.scatter(features_combined[:,0], features_combined[:,1], c=colors)
        plt.scatter(cluster_centers[:,0], cluster_centers[:,1], color='red', s=100)
        plt.xlabel(f'{feature_names[0]} (min-max-scaled)')
        plt.ylabel(f'{feature_names[1]} (min-max-scaled)')
    if plot and len(features) == 1:
        sns.stripplot(x=[x_ for x_, label in zip(features[0], cluster_labels) if label == 0], color='blue')
        sns.stripplot(x=[x_ for x_, label in zip(features[0], cluster_labels) if label == 1], color='orange')
        sns.stripplot(x=cluster_centers, color='red', size=10, jitter=False)
        plt.xlabel(f'{feature_names[0]} (min-max-scaled)')

    return cluster_labels



def plot_ride_paths(df_simra: pd.DataFrame, cluster_labels: np.ndarray, **kwargs):

    if 'figsize_paths' in kwargs:
        figsize_paths = kwargs['figsize_paths']
    else:
        figsize_paths = (12, 12)
    fig, ax = plt.subplots(figsize=figsize_paths)
    ax.set_aspect(1.5)

    colors = ['blue', 'orange']

    df_simra_grouped = df_simra.groupby('filename')
    for i, ride_group_name in enumerate(df_simra_grouped.groups):
        df_ride_group = df_simra_grouped.get_group(ride_group_name)
        ax.plot(df_ride_group.lon, df_ride_group.lat, color=colors[cluster_labels[i]], linewidth=1, label = ride_group_name)

        # df_ride_group_vec = df_simra_grouped_vec[df_simra_grouped_vec.filename == ride_group_name]
        # vec_rot_lon = [lon_start, lon_start + path_rotated_lon]
        # vec_rot_lat = [lat_start, lat_start + path_rotated_lat]
        # ax.plot(vec_rot_lon, vec_rot_lat, color='green')

        # projection_point_lon = np.float(path_rotated_lon) * df_ride_group_vec['max_projection']
        # projection_point_lat = np.float(path_rotated_lat) * df_ride_group_vec['max_projection']

        # print(projection_point_lon, projection_point_lat)
        # ax.scatter([projection_point_lon], [projection_point_lat], color='red', s=50)

    # ax.set_xlim(min(df_simra.lon), max(df_simra.lon))
    # ax.set_ylim(min(df_simra.lat), max(df_simra.lat))

    ax.xaxis.set_major_locator(ticker.LinearLocator(4))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_locator(ticker.LinearLocator(4))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.set_xlabel('Longitude in decimal degrees')
    ax.set_ylabel('Latitude in decimal degrees')

    if 'direction' in kwargs:
        plt.title('Clustered ride paths\n' + kwargs['direction'])

    plt.savefig('clustered_ride_path.png', transparent=True)
    plt.show()


def cluster_by_max_projection_and_distance(df_simra: pd.DataFrame, **kwargs):
    df_simra_grouped = df_simra.groupby('filename').agg({'dist': 'sum'})
    distances = np.array(df_simra_grouped.dist)

    path_rotated = get_path_rotated(df_simra)
    max_projections = get_max_projections(df_simra_grouped, df_simra, path_rotated)

    features = {'max_projections': max_projections, 'distances': distances}
    features_scaled = min_max_scale_features(features)

    cluster_labels = cluster_with_kmeans(features_scaled, **kwargs)
    plot_ride_paths(df_simra, cluster_labels, **kwargs)

    print(f"Percentage of orange turns: {round(cluster_labels.sum() / len(cluster_labels),2)}")
    print(f"Percentage of blue turns: {round((len(cluster_labels) - cluster_labels.sum()) / len(cluster_labels),2)}")


def analyse_df_for_faulty_entries(df_simra, show_faulty_entries = False):
    
    # Some entries contain nans, or no speed, even though a distance is given. Inspect further. Option for filtering or preprocessing.

    faulty_entries = df_simra[((df_simra.velo == 0) | (df_simra.velo.isna())) & (df_simra.dist != 0.0)]

    n_entries = len(df_simra)
    n_faulty_entries = len(faulty_entries) 
    percentage_faulty = n_faulty_entries / n_entries * 100

    print(f'Number of faulty rows (velocity is nan or zero even though distance is given): {n_faulty_entries}')
    print(f'Total rows: {n_entries}')
    print(f'Share of faulty rows: {round(percentage_faulty,2)}%.')

    if show_faulty_entries: display(faulty_entries)

def cluster_and_plot_for_intersection(start_end_coords, end_date_str = '2099-01-01 00:00:00', files_to_exclude = None, **kwargs):
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S')
    for direction, (start_coord, end_coord) in start_end_coords.items():
        print('######## ' + direction + ' ########')
        print('Start:', start_coord)
        print('End:', end_coord)
        df_simra = get_rect_to_rect_data(start_coord, end_coord, end_date=end_date, files_to_exclude=files_to_exclude)
        if df_simra is None: continue
        for key, value in kwargs.items():
            if key == 'analyse_for_faulty_entries':
                analyse_df_for_faulty_entries(df_simra)
        cluster_by_max_projection_and_distance(df_simra, direction = direction, **kwargs)
        print('\n')








