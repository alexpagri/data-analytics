from array import array
import math
import sys
# add parent directory and its parent to sys.path so that python finds the modules
sys.path.append('..')

from typing import Tuple, List, Dict
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from db_utils import get_rect_to_rect_data
import contextily as cx
from geopy.distance import great_circle
from pyproj import Proj

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

def get_off_center(df_simra_grouped: pd.DataFrame, df_simra: pd.DataFrame, path_rotated: Tuple[np.float, np.float]) -> np.ndarray:
    off_center = []
    df_simra_grouped_ = df_simra_grouped.reset_index()
    for filename in df_simra_grouped_['filename']:
        projections = []
        x, y = path_rotated
        x_norm = x / np.sqrt(x**2 + y**2)
        y_norm = y / np.sqrt(x**2 + y**2)
        for lat, lon in zip(df_simra[df_simra['filename'] == filename]['lat'], df_simra[df_simra['filename'] == filename]['lon']):
            projection = (lat * x_norm) + (lon * y_norm)
            projections.append(projection)
        off_center.append(np.max(projections))

    return np.array(off_center)


def min_max_scale_features(features: Dict[str, np.ndarray]):
    Scaler = MinMaxScaler()
    return {feature_name: Scaler.fit_transform(feature_values.reshape(-1, 1))
            for feature_name, feature_values in features.items()}


def cluster_with_kmeans(features: Dict[str, np.ndarray], turn_series: pd.Series, n_cluster: int = 2,  plot: bool = True, **kwargs) -> np.ndarray:

    intersection_number = turn_series['intersection number']
    name = turn_series['name']
    direction = turn_series['direction']
    
    
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)

    feature_names = list(features.keys())
    features = list(features.values())
    features_combined = np.hstack(features)
    cluster_labels = kmeans.fit_predict(features_combined)
    cluster_centers = kmeans.cluster_centers_

    # does only work with n_cluster = 2
    colors = ['blue' if label == 0 else 'orange' for label in cluster_labels]
            
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
    
    
    plt.title(f'intersection {intersection_number}: \n{name} \ndirection: {direction}')
    plt.savefig(f'images/k-means_{intersection_number}_{direction}.png', transparent=True, bbox_inches='tight')
    
    plt.show()

    return cluster_labels


def plot_ride_paths(df_simra: pd.DataFrame, cluster_labels: np.ndarray, turn_series: pd.Series, rides: int, fraction_cluster_1: np.ndarray, **kwargs):
    
    intersection_number = turn_series['intersection number']
    name = turn_series['name']
    direction = turn_series['direction']

    if 'figsize_rides' in kwargs:
        figsize_rides = kwargs['figsize_rides']
    else:
        figsize_rides = (12, 12)
    fig, ax = plt.subplots(figsize=figsize_rides)

    if 'group_name' in kwargs:
        group_name = f"{kwargs['group_name']}-"
    else:
        group_name = ""

    colors = ['blue', 'orange']

    if cluster_labels is None:
        cluster_labels = [0]

    df_simra_grouped = df_simra.groupby('filename', sort=False)
    for i, ride_group_name in enumerate(df_simra_grouped.groups):
        # if i > 0:
        #     break
        df_ride_group = df_simra_grouped.get_group(ride_group_name)
        ax.plot(df_ride_group.lon, df_ride_group.lat, color=colors[cluster_labels[i]], linewidth=1)

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

    cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenStreetMap.BZH)

    fraction_cluster_1_percentage = round(100*fraction_cluster_1,2)
    lines = [Line2D([0],[0], color = colors[0]),
                Line2D([0],[0], color = colors[1])]
    labels = [f'{group_name}cluster 1: '+ str(fraction_cluster_1_percentage)+'\%',
                f'{group_name}cluster 2: '+ str(round(100-fraction_cluster_1_percentage,2))+'\%']
    plt.legend(lines, labels)

    ax.set_aspect(1.7)

    plt.title(f'{group_name}intersection {intersection_number}:\n{name} \ndirection: {direction}')
    plt.savefig(f'images/{group_name}clustered_rides_{intersection_number}_{direction}.png', transparent=True)    
    plt.show()


def cluster(df_simra: pd.DataFrame, turn_series, **kwargs):
    df_simra_grouped = df_simra.groupby('filename').agg({'dist': 'sum'})
    distances = np.array(df_simra_grouped.dist)

    path_rotated = get_path_rotated(df_simra)
    off_center = get_off_center(df_simra_grouped, df_simra, path_rotated)

    features = {'off center': off_center, 'distances': distances}
    features_scaled = min_max_scale_features(features)

    cluster_labels = cluster_with_kmeans(features_scaled, turn_series)
    
    fraction_cluster_1 = 1 - cluster_labels.sum() / len(cluster_labels)

    rides = df_simra_grouped.shape[0]

    plot_ride_paths(df_simra, cluster_labels, turn_series, rides, fraction_cluster_1 , **kwargs)

    return fraction_cluster_1, rides

def cluster_return(df_simra: pd.DataFrame, turn_series):
    df_simra_grouped = df_simra.groupby('filename', sort=False).agg({'dist': 'sum'})
    distances = np.array(df_simra_grouped.dist)

    path_rotated = get_path_rotated(df_simra)
    off_center = get_off_center(df_simra_grouped, df_simra, path_rotated)

    features = {'off center': off_center, 'distances': distances}
    features_scaled = min_max_scale_features(features)

    cluster_labels = cluster_with_kmeans(features_scaled, turn_series)
    
    fraction_cluster_1 = cluster_labels.sum()

    rides = df_simra_grouped.shape[0]

    return fraction_cluster_1, rides, cluster_labels

def testf(df_simra: pd.DataFrame, cluster_labels):
    
    proj = Proj('epsg:5243')

    proj_coords = df_simra.apply(lambda x: proj(x['lon'], x['lat']), axis=1)
    df_simra.loc[:, ['x', 'y']] = list(map(list, proj_coords))

    df_simra_s = df_simra.shift(1)
    df_simra['l_x'] = df_simra_s['x']
    df_simra['l_y'] = df_simra_s['y']
    df_simra['l_lon'] = df_simra_s['lon']
    df_simra['l_lat'] = df_simra_s['lat']
    df_simra = df_simra[~df_simra['l_lon'].isnull()]
    #df_simra['dist'] = df_simra.apply(lambda x: great_circle([x['l_lat'], x['l_lon']], [x['lat'], x['lon']]).meters, axis=1)

    df_simra['ang'] = df_simra.apply(lambda x: math.degrees(math.atan2(x['y'] - x['l_y'], x['x'] - x['l_x'])), axis=1)

    df_simra['ang_w'] = df_simra.apply(lambda x: {'ang': x['ang'], 'dist': x['dist']}, axis=1)

    df_g = df_simra[['filename', 'ang_w']].groupby('filename').aggregate(lambda x: np.average(pd.DataFrame(list(x))[['ang']], weights=pd.DataFrame(list(x))[['dist']])).rename(columns={'ang_w': 'ang_avg'})
    df_simra = df_simra.merge(df_g, how='inner', on='filename')

    df_simra['ang'] = df_simra.apply(lambda x: x['ang'] - x['ang_avg'], axis=1).apply(lambda x: x if x < 180.0 else (x - 360.0)).apply(lambda x: x if x > -180.0 else (x + 360.0))

    colors = ['blue', 'orange']
    df_simra_grouped = df_simra.groupby('filename', sort=False)
    for i, ride_group_name in enumerate(df_simra_grouped.groups):
        # if i > 0:
        #     break
        df_ride_group = df_simra_grouped.get_group(ride_group_name)
        plt.xlim(-180, 180)
        plt.hist(df_ride_group['ang'], 45, density=True, weights=df_ride_group['dist'], color=colors[cluster_labels[i]])
        plt.show()


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


def return_cluster_results_and_plot_path(turn_series, end_date_str = '2099-01-01 00:00:00', **kwargs):
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S')
    df_simra = get_rect_to_rect_data(turn_series['start_rect_coords'], turn_series['end_rect_coords'],
        exclude_coords = turn_series['exclude_coords'])
    if (df_simra is not None) and ('group' in kwargs):
        df_simra = df_simra.query(f"group == {kwargs['group']}")
    if (df_simra is None) or (len(df_simra) == 0):
        print('No rides')
        return None, None
    # if only 1 ride, not possible to cluster
    if len(set(df_simra['filename'])) == 1:
        fraction_cluster_1 = 1
        plot_ride_paths(df_simra, [0], fraction_cluster_1=fraction_cluster_1, rides=1, turn_series = turn_series, **kwargs)
        return 1, 1
    if 'analyse_for_faulty_entries' in kwargs: analyse_df_for_faulty_entries(df_simra)
    share_cluster_1, rides = cluster(df_simra, turn_series, **kwargs)
    return share_cluster_1, rides

def cluster_one(turn_series, end_date_str = '2099-01-01 00:00:00', **kwargs):
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S')
    df_simra = get_rect_to_rect_data(turn_series['start_rect_coords'], turn_series['end_rect_coords'],
        exclude_coords = turn_series['exclude_coords'])
    if (df_simra is not None) and ('group' in kwargs):
        df_simra = df_simra.query(f"group == {kwargs['group']}")
    if (df_simra is None) or (len(df_simra) == 0):
        print('No rides')
        return None, None, 0, 0
    # if only 1 ride, not possible to cluster
    if len(set(df_simra['filename'])) == 1:
        fraction_cluster_1 = 1
        #plot_ride_paths(df_simra, [0], fraction_cluster_1=fraction_cluster_1, rides=1, turn_series = turn_series, **kwargs)
        return df_simra, [0], 1, 1
    if 'analyse_for_faulty_entries' in kwargs: analyse_df_for_faulty_entries(df_simra)
    share_cluster_1, rides, cluster_labels = cluster_return(df_simra, turn_series)
    #testf(df_simra, cluster_labels)
    return df_simra, cluster_labels, share_cluster_1, rides

def return_cluster_results_and_plot_path_grouped(turn_series_grp, end_date_str = '2099-01-01 00:00:00', **kwargs):
    turn_series_g = turn_series_grp.iloc[0]
    turn_series_g['direction'] = 'all'
    df_simra_g = cluster_labels_g = share_cluster_g = rides_g = None
    for turn_series in turn_series_grp.iloc:
        df_simra, cluster_labels, share_cluster, rides = cluster_one(turn_series, **kwargs)

        if df_simra_g is None:
            df_simra_g = df_simra
        else:
            df_simra_g = pd.concat([df_simra_g, df_simra])
        
        if cluster_labels_g is None:
            cluster_labels_g = cluster_labels
        elif cluster_labels is not None:
            cluster_labels_g = np.append(cluster_labels_g, cluster_labels)

        if share_cluster_g is None:
            share_cluster_g = share_cluster
        else:
            share_cluster_g = share_cluster_g + share_cluster

        if rides_g is None:
            rides_g = rides
        else:
            rides_g += rides
    
    if df_simra_g is not None:
        plot_ride_paths(df_simra_g.reset_index(), cluster_labels_g, turn_series_g, rides_g, (rides_g - share_cluster_g) / rides_g, **kwargs)



