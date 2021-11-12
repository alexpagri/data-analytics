import csv
from postgis import LineString, Point
from datetime import datetime
from db_connection import DatabaseConnection
import filters
import numpy as np
import pandas as pd
import preprocess_service
import acceleration_service


def handle_ride_file(file, cur):
    with open(file, 'r') as f:
        ride_data = []
        split_found = False
        for line in f.readlines():
            if "======" in line:
                split_found = True
                continue
            if split_found:
                ride_data.append(line)

        handle_ride(ride_data, file, cur)


def handle_ride(data, filename, cur):
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

    if len(ride_df) == 0:
        print("Ride is filtered due to len(ride_df) == 0")
        return

    if is_teleportation(ride_df.timestamp):
        print("Ride is filtered due to teleportation")
        return

    ride_df = preprocess_service.preprocess_basics(ride_df)

    ride_df = filters.apply_smoothing_filters(ride_df)
    if filters.apply_removal_filters(ride_df):
        return

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

    start = Point(tuple(coords_k[0]), srid=4326)
    end = Point(tuple(coords_k[-1]), srid=4326)


    try:
        cur.execute("""
            INSERT INTO public."ride" (geom_raw, geom, timestamps, filename, velos_raw, velos, durations, distances, "start", "end") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, [ls, ls_k, timestamps, filename, velos, velos_lp0, durations, distances, start, end])
    except Exception as e:
        print("Values to insert:")
        for value in [ls, ls_k, timestamps, filename, velos, velos_lp0, durations, distances, start, end]:
            print(f"-----\n{value}")
        
        print(f"Original exception: {e}")
        print(f"Problem parsing {filename}")
        raise Exception("Can not parse ride!")


if __name__ == '__main__':
    filepath = "../csvdata/Berlin/Rides/VM2_-351907452"
    with DatabaseConnection() as cur:
        handle_ride_file(filepath, cur)


def is_teleportation(timestamps):
    for i, t in enumerate(timestamps):
        if i + 1 < len(timestamps):
            if (timestamps[i + 1] - timestamps[i]).seconds > 20:
                return True
    return False
