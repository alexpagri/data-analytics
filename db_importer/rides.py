import csv
from postgis import LineString, Point
from datetime import datetime
from db_connection import DatabaseConnection
import filters
import numpy as np
import pandas as pd
import preprocess_service
import acceleration_service
from adsp.visualization.view_rides import add_path, show_path
from pyproj import Proj

def show_path2():
    show_path()

def handle_ride_file(file, cur):
    with open(file, 'r') as f:
        ride_data = []
        split_found = False
        bike_head = 0
        for line in f.readlines():
            if split_found == False:
                ride_data.append(line)
            if "======" in line and split_found == False:
                split_found = True
                bike_head = handle_ride_head(ride_data)
                ride_data = []
                continue
            if split_found:
                ride_data.append(line)

        handle_ride(ride_data, bike_head, file, cur)

def handle_ride_head(data):
    data = csv.DictReader(data[1:3], delimiter=",")
    for i, row in enumerate(data):
        if row["bike"]:
            return row["bike"]
    return 0

def handle_ride(data, bike_head, filename, cur):
    data = csv.DictReader(data[1:], delimiter=",")

    raw_coords = []
    accuracies = []
    timestamps = []

    for i, row in enumerate(data):
        if row["lat"]:
            raw_coords.append([float(row["lon"]), float(row["lat"])])
            try:
                if row["acc"]:
                    if float(row["acc"]) > 100.0:  # ride goes to trash
                        print("Ride is filtered due to accuracies > 100")
                        return
                    accuracies.append(float(row["acc"]))
            except KeyError:
                return
            ts = datetime.utcfromtimestamp(int(row["timeStamp"]) / 1000)  # timeStamp is in Java TS Format
            timestamps.append(ts)

    ride_df = pd.DataFrame(
        {'lon': np.array(raw_coords)[:,0],
         'lat': np.array(raw_coords)[:,1],
         'accuracy': accuracies,
         'timestamp': timestamps
         })

    indices = ride_df.index[ride_df.accuracy < 35.0]

    ride_df = ride_df[indices[0]:indices[-1]] # cut constant low accuracy (> 15) head and tail

    if ride_df.accuracy.mean() > 35.0:
        return

    if len(ride_df) == 0:
        print("Ride is filtered due to len(ride_df) == 0")
        return

    if is_teleportation(ride_df.timestamp):
        print("Ride is filtered due to teleportation")
        return

    #print("RIDE OK")
    #return

    ride_df = preprocess_service.preprocess_basics(ride_df)

    if filters.apply_removal_filters(ride_df):
        return
    ride_df = filters.apply_smoothing_filters(ride_df)

    #add_path(ride_df.rename(columns={'accuracy': 'rad'}))
    #add_path(ride_df[['lat_k', 'lon_k', 'accuracy', 'timestamp']].rename(columns={'lat_k': 'lat', 'lon_k': 'lon', 'accuracy': 'rad'}), 'red', 1)
    #return

    filename = filename.split("/")[-1]
    acceleration_service.process_acceleration_segments(ride_df, filename, cur)

    coords = ride_df[['lon', 'lat']].values
    coords_k = ride_df[['lon_k', 'lat_k']].values

    ls = LineString([tuple(x) for x in coords], srid=4326)
    ls_k = LineString([tuple(x) for x in coords_k], srid=4326)

    velos = list(ride_df.velo.values)
    velos_lp0 = list(ride_df.velo_lp0.values)
    durations = list(ride_df.duration.values)
    distances = list(ride_df.dist.values)
    accuracy = list(ride_df.accuracy.values)

    start = Point(tuple(coords_k[0]), srid=4326)
    end = Point(tuple(coords_k[-1]), srid=4326)

    try:
        cur.execute("""
            INSERT INTO public."ride" (geom_raw, geom, timestamps, filename, velos_raw, velos, durations, distances, "start", "end", accuracy, bike) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, [ls, ls_k, timestamps, filename, velos, velos_lp0, durations, distances, start, end, accuracy, bike_head])
    except Exception as e:
        print("Values to insert:")
        for value in [ls, ls_k, timestamps, filename, velos, velos_lp0, durations, distances, start, end, accuracy, bike_head]:
            print(f"-----\n{value}")
        
        print(f"Original exception: {e}")
        print(f"Problem parsing {filename}")
        raise Exception("Can not parse ride!")


def is_teleportation(timestamps):
    for i, t in enumerate(timestamps):
        if i + 1 < len(timestamps):
            if (timestamps[i + 1] - timestamps[i]).seconds > 20:
                return True
    return False


if __name__ == '__main__':
    filepath = "/Users/max/Documents/Projects/Uni/ADSP/ma/datasets/Berlin/Rides/2021/11/VM2_-17358072"
    with DatabaseConnection() as cur:
        handle_ride_file(filepath, cur)

