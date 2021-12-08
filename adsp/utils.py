import sys
# add parent directory to sys.path so that python finds the modules
sys.path.append('..')

from typing import List, Tuple
from datetime import datetime
from ast import literal_eval

import psycopg2
from postgis.psycopg import register
from shapely.geometry import box, Point
import pandas as pd

from db_importer.settings import *


"""Every geometric shape has a spatial reference system associated with it, and each such reference system has a Spatial Reference System ID (SRID). The SRID is used to tell which spatial reference system will be used to interpret each spatial object.
A common SRID in use is 4326, which represents spatial data using longitude and latitude coordinates on the Earth's surface as defined in the WGS84 standard, which is also used for the Global Positioning System (GPS) [https://www.cockroachlabs.com/docs/stable/srid-4326.html]"""
SRID = 4326 

RECT_CORNER_OFFSETS = {
    'MEDIUM': 7.899999999949614e-05
}


class DatabaseConnection(object):
    def __enter__(self):
        self.conn = psycopg2.connect(f"dbname='{DB_NAME}' user='{DB_USER}' password='{DB_PASSWORD}' host='{DB_HOST}' port='{DB_PORT}'")
        self.conn.autocommit = True

        register(self.conn)
        self.cur = self.conn.cursor()
 
        return self.cur

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is None:
            self.conn.commit()
            self.cur.close()
            self.conn.close()
        else:
            self.conn.rollback()
            self.cur.close()
            self.conn.close()

def get_rect_to_rect_data(start_rect_coords: Tuple[float], end_rect_coords: Tuple[float],
    start_date: datetime = None, end_date: datetime = None, files_to_exclude: List[str] = None) -> pd.DataFrame:

    res = build_and_execute_query(start_rect_coords, end_rect_coords, start_date, end_date)
        
    df = pd.DataFrame(res, columns=['filename', 'coords', 'timestamps', 'velos', 'distances'])
    df = df[df.coords.notnull()]
    
    if files_to_exclude: 
        df = df[~df['filename'].isin(files_to_exclude)]
    
    # convert string representation of coordinate list into actual list of floats
    # would not be necessary if stored as floats in the first place! 
    df.coords = df.coords.apply(lambda coord: literal_eval(coord))
    
    df_grouped = df.groupby('filename', as_index=False)
    print(f"Number of rides BEFORE intersection checking and cropping: {len(df_grouped.groups)}")

    df_cropped = df_grouped.apply(lambda group: crop_intersection_SimRa(group, start_rect_coords, end_rect_coords))
    print(f"Number of rides AFTER intersection checking and cropping: {len(df_cropped.groupby('filename').groups)}")
    return df_cropped


def build_and_execute_query(start_rect_coords: Tuple[float], end_rect_coords: Tuple[float], 
    start_date: datetime = None, end_date: datetime = None):
    
    with DatabaseConnection() as cur:
        query = f"""        
            SELECT filename,
                json_array_elements_text(st_asgeojson(geom_raw) :: json -> 'coordinates') AS coordinates,
                unnest(timestamps) timestamps,
                unnest(velos) velos,
                unnest(distances) distances
            FROM ride
            where st_intersects(geom, st_setsrid(
                    st_makebox2d(st_makepoint({start_rect_coords[0]}, {start_rect_coords[1]}), 
                        st_makepoint({start_rect_coords[2]}, {start_rect_coords[3]})), {SRID}))
            AND st_intersects(geom, st_setsrid(
                    st_makebox2d(st_makepoint({end_rect_coords[0]}, {end_rect_coords[1]}), 
                        st_makepoint({end_rect_coords[2]}, {end_rect_coords[3]})), {SRID}))
        """

        if start_date:
            query += f"AND timestamps[1] > '{start_date.isoformat()}'"
        if end_date:
            query += f"AND timestamps[1] < '{end_date.isoformat()}'"

        cur.execute(query)
        res = cur.fetchall()
    
    return res


def crop_intersection_SimRa(group, start_rect_coords: Tuple[float], end_rect_coords: Tuple[float]):
    start_rect = box(*start_rect_coords)
    end_rect = box(*end_rect_coords)

    mask_first = group.coords.apply(lambda coords: start_rect.contains(Point(coords)))
    mask_end = group.coords.apply(lambda coords: end_rect.contains(Point(coords)))
    
    if any(mask_first) and any(mask_end):
        first = mask_first[mask_first==True].index[0]
        last = mask_end[mask_end==True].index[-1]
        return group.loc[first:last]
    else:
        # print(f"No path intersections found for filename: '{group.filename.values[0]}'")
        return None


def get_rect_coords_from_center_point(point_coords: Tuple[float], rect_size: str = 'MEDIUM') -> Tuple[float]:
    try:
        assert len(point_coords) == 2
    except AssertionError:
        print(f"Rectangle needs 4 coordinates, {len(point_coords)} given.")
    
    try:
        offset = RECT_CORNER_OFFSETS[rect_size]
    except KeyError:
        print(f"'rect_size' must be one of {list(RECT_CORNER_OFFSETS.keys())}")

    if point_coords[0] < point_coords[1]:
        print(f"Point {point_coords} interpreted as (longitude, latitude)")
    elif point_coords[0] > point_coords[1]:
        print(f"Point {point_coords} interpreted as (latitude, longitude)")
        point_coords = (point_coords[1], point_coords[0])
    
    return (point_coords[0] - offset, point_coords[1] - offset, point_coords[0] + offset, point_coords[1] + offset)


def get_corner_offset_from_rect_coords(rect_coords: Tuple[float]) -> float:
    try:
        assert len(rect_coords) == 4
    except AssertionError:
        print(f"Rectangle needs 4 coordinates, {len(rect_coords)} given.")
    
    return box(*rect_coords).length / 8

