import sys
# add parent directory to sys.path so that python finds the modules
sys.path.append('..')

from typing import List, Tuple, Union
from datetime import datetime
from ast import literal_eval
import functools
import numpy as np

import psycopg2
from postgis.psycopg import register
from shapely.geometry import box, Point
import pandas as pd

from db_importer.settings import *


"""Every geometric shape has a spatial reference system associated with it, and each such reference system has a Spatial Reference System ID (SRID). The SRID is used to tell which spatial reference system will be used to interpret each spatial object.
A common SRID in use is 4326, which represents spatial data using longitude and latitude coordinates on the Earth's surface as defined in the WGS84 standard, which is also used for the Global Positioning System (GPS) [https://www.cockroachlabs.com/docs/stable/srid-4326.html]"""
SRID = 4326 

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

# @functools.lru_cache(maxsize=200)
def get_rect_data(rect_coords: Tuple[float], files_to_exclude: List[str] = None, limit_rows: int = 10000) -> pd.DataFrame:

    res = build_and_execute_query(rect_coords, limit_rows)
    df = pd.DataFrame(res, columns=['filename', 'coords', 'velo', 'dur', 'dist', 'ts'])
    # df = pd.DataFrame(res, columns=['filename', 'dp', 'velo', 'dur', 'dist', 'ts'])
    
    df = df[df.coords.notnull()]

    if not df.shape[0]:
        print("Query has no results!")
        return None

    df['lon'], df['lat'] = zip(*df.coords.values)

    if files_to_exclude: 
        df = df[~df['filename'].isin(files_to_exclude)]
    
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of rides: {len(df.groupby('filename').groups)}")
    return df


def build_and_execute_query(rect_coords: Tuple[float], limit_rows: int):
    with DatabaseConnection() as cur:
        query = f"""

                SELECT filename, to_json((ST_DumpPoints(geom::geometry)).geom :: json -> 'coordinates') AS coords,
                    unnest(velos) velo, unnest(durations) dur, unnest(distances) dist, unnest(timestamps) ts
                FROM (
                    SELECT *
                    FROM ride
                    WHERE st_intersects(geom,
                    st_setsrid( st_makebox2d( st_makepoint(13.4112,52.5031), st_makepoint(13.4117,52.5039)), 4326))
                AND st_intersects(geom,
                    st_setsrid( st_makebox2d( st_makepoint(13.426,52.4991), st_makepoint(13.4264,52.4998)), 4326))
                ) tmp
                """

        cur.execute(query)
        res = cur.fetchall()
    
    return res

