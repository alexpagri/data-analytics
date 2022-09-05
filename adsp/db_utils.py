import sys
# add parent directory to sys.path so that python finds the modules
sys.path.extend(['..', '../..'])

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

@functools.lru_cache(maxsize=200)
def get_rect_to_rect_data(start_rect_coords: Tuple[float], end_rect_coords: Tuple[float],
    start_date: datetime = None, end_date: datetime = None, files_to_exclude: List[str] = None,
    exclude_coords: Union[Tuple[float],np.float64] = np.nan) -> pd.DataFrame:
    if exclude_coords is np.nan:
        exclude_coords = (0,0,0,0)
    res = build_and_execute_query(start_rect_coords, end_rect_coords, exclude_coords, start_date, end_date)
    df = pd.DataFrame(res, columns=['filename', 'coords', 'velo', 'dur', 'dist', 'ts', 'min_ts', 'max_ts', 'time_diff', 'avg_v', 'group'])
    
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
    exclude_coords: Tuple[float], start_date: datetime = None, end_date: datetime = None):
    
    with DatabaseConnection() as cur:
        avg_filter = lambda name: f"{name} > 0.2 AND {name} != 'NaN' AND {name} < 15"
        query = f"""
        SELECT *, CASE WHEN tmp.avg_v < 4.3638 THEN 0 ELSE CASE WHEN tmp.avg_v < 5.6694 THEN 1 ELSE 2 END END as group
            FROM
            (
                SELECT ride.filename as abc, json_array_elements(st_asgeojson(geom_raw) :: json -> 'coordinates') AS coords,
                        unnest(velos) velo, unnest(durations) dur, unnest(distances) dist, unnest(timestamps) ts, 
                        min_ts, max_ts, (max_ts - min_ts) as time_diff, (SELECT AVG(avg_v) FROM unnest(velos) avg_v WHERE {avg_filter("avg_v")}) as avg_v
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
                    WHERE (min_ts < max_ts) and
                        (a.filename NOT IN
                            (
                                SELECT filename
                                FROM (
                                    SELECT *, ST_DumpPoints(geom::geometry) as dp, unnest(timestamps) ts
                                    FROM ride
                                    ) tmp
                                WHERE st_intersects((tmp.dp).geom, ST_MakeEnvelope ({exclude_coords[0]}, {exclude_coords[1]}, 
                                                                                    {exclude_coords[2]}, {exclude_coords[3]}, 
                                                                                    {SRID}
                                                                                    )
                                )
                                GROUP BY filename
                            )
                        )
                ) c
                ON ride.filename = c.filename
            ) tmp
            WHERE ts >= min_ts AND ts <= max_ts AND extract(epoch FROM time_diff) < 3000
        """

        if start_date:
            query += f"AND min_ts > '{start_date.isoformat()}'"
        if end_date:
            query += f"AND max_ts < '{end_date.isoformat()}'"

        cur.execute(query)
        res = cur.fetchall()
    
    return res

