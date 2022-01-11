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
    df = pd.DataFrame(res, columns=['filename', 'coords', 'velo', 'dur', 'dist', 'ts', 'min_ts', 'max_ts', 'time_diff'])
    
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


def build_and_execute_query(start_rect_coords: Tuple[float], end_rect_coords: Tuple[float], 
    start_date: datetime = None, end_date: datetime = None):
    
    with DatabaseConnection() as cur:
        query = f"""
        SELECT *
            FROM
            (
                SELECT ride.filename, json_array_elements(st_asgeojson(geom_raw) :: json -> 'coordinates') AS coords,
                        unnest(velos) velo, unnest(durations) dur, unnest(distances) dist, unnest(timestamps) ts, 
                        min_ts, max_ts, (max_ts - min_ts) as time_diff 
                FROM ride
                JOIN
                (
                    SELECT a.filename, min_ts, max_ts
                    FROM
                    (
                        SELECT filename, min(tmp.ts) as min_ts
                        FROM (
                            SELECT *, ST_DumpPoints(geom::geometry) as dp, unnest(timestamps) ts
                            FROM ride
                        ) tmp
                        WHERE st_intersects((tmp.dp).geom, ST_MakeEnvelope ({start_rect_coords[0]}, {start_rect_coords[1]}, 
                                                                            {start_rect_coords[2]}, {start_rect_coords[3]},
                                                                            {SRID}
                                                            )
                        )
                        GROUP BY filename
                    ) a
                    INNER JOIN
                    (
                        SELECT filename, max(tmp.ts) as max_ts
                        FROM (
                            SELECT *, ST_DumpPoints(geom::geometry) as dp, unnest(timestamps) ts
                            FROM ride
                        ) tmp
                        WHERE st_intersects((tmp.dp).geom, ST_MakeEnvelope ({end_rect_coords[0]}, {end_rect_coords[1]}, 
                                                                            {end_rect_coords[2]}, {end_rect_coords[3]}, 
                                                                            {SRID}
                                                            )
                        )
                        GROUP BY filename
                    ) b
                    ON a.filename = b.filename
                    WHERE min_ts < max_ts
                ) c
                ON ride.filename = c.filename
            ) tmp
            WHERE ts >= min_ts AND ts <= max_ts AND extract(epoch FROM time_diff) < 300
        """

        if start_date:
            query += f"AND min_ts > '{start_date.isoformat()}'"
        if end_date:
            query += f"AND max_ts < '{end_date.isoformat()}'"

        cur.execute(query)
        res = cur.fetchall()
    
    return res

