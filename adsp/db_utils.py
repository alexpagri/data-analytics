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

#@functools.lru_cache(maxsize=200)
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


def build_and_execute_query(start_rect_coords, end_rect_coords, 
    exclude_coords = (0, 0, 0, 0), start_date: datetime = None, end_date: datetime = None):
    
    with DatabaseConnection() as cur:
        velo_filter = lambda name: f"{name} > 0.2 AND {name} != 'NaN' AND {name} < 15"
        group_q = lambda perc: f"""SELECT percentile_cont({perc}) WITHIN GROUP (ORDER BY one.avg_v) FROM (SELECT AVG(velo) as avg_v FROM accels WHERE {velo_filter("velo")} GROUP BY filename) as one"""
        query = f"""
        SELECT *, CASE WHEN tmp.avg_v < ({group_q("0.25")}) THEN 0 ELSE CASE WHEN tmp.avg_v < ({group_q("0.75")}) THEN 1 ELSE 2 END END as group
            FROM
            (
                SELECT ride.filename as abc, json_array_elements(st_asgeojson(geom) :: json -> 'coordinates') AS coords,
                        unnest(velos) velo, unnest(durations) dur, unnest(distances) dist, unnest(timestamps) ts, 
                        min_ts, max_ts, (max_ts - min_ts) as time_diff,
                        --(SELECT SUM(velo2 * duration)/SUM(duration) FROM unnest(velos) velo2, unnest(durations) duration WHERE {velo_filter("velo2")}) as avg_v --check why slow, correctness ok
                        (SELECT AVG(velo2) FROM unnest(velos) velo2 WHERE {velo_filter("velo2")}) as avg_v
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

def getRideGeneralStatisticsAccel(accels_where = "True", ride_where = "True"): # Checked
    with DatabaseConnection() as cur:
        velo_filter = lambda name: f"{name} > 0.2 AND {name} != 'NaN' AND {name} < 15"
        group_q = lambda perc: f"""SELECT percentile_cont({perc}) WITHIN GROUP (ORDER BY one.avg_v) FROM (SELECT AVG(velo) as avg_v FROM accels WHERE {velo_filter("velo")} GROUP BY filename) as one"""
        cur.execute(f"""
                SELECT two.filename, two.avg_v, two.max_v, one.accel, one.decel, bike, CASE WHEN two.avg_v < ({group_q(0.25)}) THEN 0 ELSE CASE WHEN two.avg_v < ({group_q(0.75)}) THEN 1 ELSE 2 END END as group FROM (
                    SELECT filename, MAX(accel) as accel, MIN(accel) as decel FROM accels GROUP BY filename
                ) as one JOIN (
                    SELECT filename, AVG(velo) as avg_v, MAX(velo) as max_v FROM accels WHERE {velo_filter("velo")} AND {accels_where} GROUP BY filename
                ) as two ON (one.filename = two.filename) JOIN (
                    SELECT filename, bike FROM ride WHERE True AND {ride_where}
                ) as three ON (two.filename = three.filename)
                """)
        return cur.fetchall()
        #return pd.DataFrame(res, columns=['filename', 'max_v', 'max_accel', 'max_decel', 'group'])

def getUncutRideLocationStatisticsAccel(start_rect_coords, end_rect_coords, exclude_coords = (0, 0, 0, 0), accels_where = "True", ride_where = "True", time_diff = "5000"): # Checked
    with DatabaseConnection() as cur:
        velo_filter = lambda name: f"{name} > 0.2 AND {name} != 'NaN' AND {name} < 15"
        group_q = lambda perc: f"""SELECT percentile_cont({perc}) WITHIN GROUP (ORDER BY one.avg_v) FROM (SELECT AVG(velo) as avg_v FROM accels WHERE {velo_filter("velo")} GROUP BY filename) as one"""
        cur.execute(f"""
                SELECT two.filename, two.avg_v, two.max_v, one.accel, one.decel, bike, CASE WHEN two.avg_v < ({group_q(0.25)}) THEN 0 ELSE CASE WHEN two.avg_v < ({group_q(0.75)}) THEN 1 ELSE 2 END END as group FROM (
                    SELECT filename, MAX(accel) as accel, MIN(accel) as decel FROM accels WHERE True AND {accels_where} GROUP BY filename
                ) as one JOIN (
                    SELECT filename, AVG(velo) as avg_v, MAX(velo) as max_v FROM accels WHERE {velo_filter("velo")} AND {accels_where} GROUP BY filename
                ) as two ON (one.filename = two.filename) JOIN (
                    SELECT filename, bike FROM ride WHERE True AND {ride_where}
                ) as three ON (two.filename = three.filename) JOIN (

                    SELECT a.filename, min_ts, max_ts
                        FROM
                        (
                            SELECT filename, min(timestamp) as min_ts
                            FROM unnested_ride
                            WHERE st_intersects((coords).geom, ST_MakeEnvelope ({start_rect_coords[0]}, {start_rect_coords[1]}, 
                                                                                {start_rect_coords[2]}, {start_rect_coords[3]},
                                                                                {SRID}
                                                                )
                            )
                            GROUP BY filename
                        ) a
                        INNER JOIN
                        (
                            SELECT filename, max(timestamp) as max_ts
                            FROM unnested_ride
                            WHERE st_intersects((coords).geom, ST_MakeEnvelope ({end_rect_coords[0]}, {end_rect_coords[1]}, 
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
                                    FROM unnested_ride
                                    WHERE st_intersects((coords).geom, ST_MakeEnvelope ({exclude_coords[0]}, {exclude_coords[1]}, 
                                                                                        {exclude_coords[2]}, {exclude_coords[3]}, 
                                                                                        {SRID}
                                                                                        )
                                    )
                                    GROUP BY filename
                                )
                            )
                ) as four ON (four.filename = two.filename) AND extract(epoch FROM (max_ts - min_ts)) < {time_diff}

                """)
        return cur.fetchall()
        #return pd.DataFrame(res, columns=['filename', 'max_v', 'max_accel', 'max_decel', 'group'])

def getCutRideLocationStatisticsAccel(start_rect_coords, end_rect_coords, exclude_coords = (0, 0, 0, 0), accels_where = "True", ride_where = "True", time_diff = "5000"): # Checked - req extra memory
    with DatabaseConnection() as cur:
        velo_filter = lambda name: f"{name} > 0.2 AND {name} != 'NaN' AND {name} < 15"
        group_q = lambda perc: f"""SELECT percentile_cont({perc}) WITHIN GROUP (ORDER BY one.avg_v) FROM (SELECT AVG(velo) as avg_v FROM accels WHERE {velo_filter("velo")} GROUP BY filename) as one"""
        cur.execute(f"""
                WITH ride_location_cut_accels as (
                    SELECT r.filename, r.duration, r.timestamp, a.accel, r.velo FROM unnested_ride r JOIN accels a ON (a.filename = r.filename AND a.timestamp = r.timestamp) JOIN ( --only data present in accels, accels included in unnested_ride/ride
                        SELECT a.filename, min_ts, max_ts
                        FROM
                        (
                            SELECT filename, min(timestamp) as min_ts
                            FROM unnested_ride
                            WHERE st_intersects((coords).geom, ST_MakeEnvelope ({start_rect_coords[0]}, {start_rect_coords[1]}, 
                                                                                {start_rect_coords[2]}, {start_rect_coords[3]},
                                                                                {SRID}
                                                                )
                            )
                            GROUP BY filename
                        ) a
                        INNER JOIN
                        (
                            SELECT filename, max(timestamp) as max_ts
                            FROM unnested_ride
                            WHERE st_intersects((coords).geom, ST_MakeEnvelope ({end_rect_coords[0]}, {end_rect_coords[1]}, 
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
                                    FROM unnested_ride
                                    WHERE st_intersects((coords).geom, ST_MakeEnvelope ({exclude_coords[0]}, {exclude_coords[1]}, 
                                                                                        {exclude_coords[2]}, {exclude_coords[3]}, 
                                                                                        {SRID}
                                                                                        )
                                    )
                                    GROUP BY filename
                                )
                            )
                    ) loc_filter ON (r.filename = loc_filter.filename)
                    WHERE r.timestamp > loc_filter.min_ts AND r.timestamp < loc_filter.max_ts AND extract(epoch FROM (max_ts - min_ts)) < {time_diff} AND {accels_where}
                )
                SELECT two.filename, two.avg_v, two.max_v, one.accel, one.decel, bike, CASE WHEN two.avg_v < ({group_q(0.25)}) THEN 0 ELSE CASE WHEN two.avg_v < ({group_q(0.75)}) THEN 1 ELSE 2 END END as group FROM (
                    SELECT filename, MAX(accel) as accel, MIN(accel) as decel FROM ride_location_cut_accels GROUP BY filename
                ) as one JOIN (
                    SELECT filename, AVG(velo) as avg_v, MAX(velo) as max_v FROM ride_location_cut_accels WHERE {velo_filter("velo")} GROUP BY filename
                ) as two ON (one.filename = two.filename) JOIN (
                    SELECT filename, bike FROM ride WHERE True AND {ride_where}
                ) as three ON (two.filename = three.filename)
                """)
        return cur.fetchall()
        #return pd.DataFrame(res, columns=['filename', 'max_v', 'max_accel', 'max_decel', 'group'])

def getRideLocationStatisticsAccel(start_rect_coords, end_rect_coords, exclude_coords = (0, 0, 0, 0), accels_where = "True", ride_where = "True", time_diff = "5000"): # Checked - req extra memory
    with DatabaseConnection() as cur:
        velo_filter = lambda name: f"{name} > 0.2 AND {name} != 'NaN' AND {name} < 15"
        group_q = lambda perc: f"""SELECT percentile_cont({perc}) WITHIN GROUP (ORDER BY one.avg_v) FROM (SELECT AVG(velo) as avg_v FROM accels WHERE {velo_filter("velo")} GROUP BY filename) as one"""
        cur.execute(f"""
                WITH ride_location_cut_accels as (
                    SELECT r.filename, r.duration, r.timestamp, a.accel, r.velo FROM unnested_ride r JOIN accels a ON (a.filename = r.filename AND a.timestamp = r.timestamp) JOIN ( --only data present in accels, accels included in unnested_ride/ride
                        SELECT a.filename, min_ts, max_ts
                        FROM
                        (
                            SELECT filename, min(timestamp) as min_ts
                            FROM unnested_ride
                            WHERE st_intersects((coords).geom, ST_MakeEnvelope ({start_rect_coords[0]}, {start_rect_coords[1]}, 
                                                                                {start_rect_coords[2]}, {start_rect_coords[3]},
                                                                                {SRID}
                                                                )
                            )
                            GROUP BY filename
                        ) a
                        INNER JOIN
                        (
                            SELECT filename, max(timestamp) as max_ts
                            FROM unnested_ride
                            WHERE st_intersects((coords).geom, ST_MakeEnvelope ({end_rect_coords[0]}, {end_rect_coords[1]}, 
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
                                    FROM unnested_ride
                                    WHERE st_intersects((coords).geom, ST_MakeEnvelope ({exclude_coords[0]}, {exclude_coords[1]}, 
                                                                                        {exclude_coords[2]}, {exclude_coords[3]}, 
                                                                                        {SRID}
                                                                                        )
                                    )
                                    GROUP BY filename
                                )
                            )
                    ) loc_filter ON (r.filename = loc_filter.filename)
                    WHERE r.timestamp > loc_filter.min_ts AND r.timestamp < loc_filter.max_ts AND extract(epoch FROM (max_ts - min_ts)) < {time_diff} AND {accels_where}
                )
                SELECT two.filename, four.avg_v, two.max_v, one.accel, one.decel, bike, CASE WHEN four.avg_v < ({group_q(0.25)}) THEN 0 ELSE CASE WHEN four.avg_v < ({group_q(0.75)}) THEN 1 ELSE 2 END END as group FROM (
                    SELECT filename, MAX(accel) as accel, MIN(accel) as decel FROM ride_location_cut_accels GROUP BY filename
                ) as one JOIN (
                    SELECT filename, MAX(velo) as max_v FROM ride_location_cut_accels WHERE {velo_filter("velo")} GROUP BY filename
                ) as two ON (one.filename = two.filename) JOIN (
                    SELECT filename, bike FROM ride WHERE True AND {ride_where}
                ) as three ON (two.filename = three.filename) JOIN (
                    SELECT filename, AVG(velo) as avg_v FROM accels WHERE {velo_filter("velo")} AND {accels_where} GROUP BY filename
                ) as four ON (two.filename = four.filename)
                """)
        return cur.fetchall()
        #return pd.DataFrame(res, columns=['filename', 'max_v', 'max_accel', 'max_decel', 'group'])

def getRideGeneralStatistics(unnested_ride_where = "True", accels_where = "True", ride_where = "True"): # Checked
    with DatabaseConnection() as cur:
        velo_filter = lambda name: f"{name} > 0.2 AND {name} != 'NaN' AND {name} < 15"
        group_q_ride = lambda perc: f"""SELECT percentile_cont({perc}) WITHIN GROUP (ORDER BY one.avg_v) FROM (SELECT AVG(velo) as avg_v FROM unnested_ride WHERE {velo_filter("velo")} GROUP BY filename) as one"""
        cur.execute(f"""
                SELECT two.filename, two.avg_v, two.max_v, one.accel, one.decel, bike, CASE WHEN two.avg_v < ({group_q_ride(0.25)}) THEN 0 ELSE CASE WHEN two.avg_v < ({group_q_ride(0.75)}) THEN 1 ELSE 2 END END as group FROM (
                    SELECT filename, MAX(accel) as accel, MIN(accel) as decel FROM accels WHERE True AND {accels_where} GROUP BY filename
                ) as one JOIN (
                    SELECT filename, AVG(velo) as avg_v, MAX(velo) as max_v FROM unnested_ride WHERE {velo_filter("velo")} AND {unnested_ride_where} GROUP BY filename
                ) as two ON (one.filename = two.filename) JOIN (
                    SELECT filename, bike FROM ride WHERE True AND {ride_where}
                ) as three ON (two.filename = three.filename)
                """)
        return cur.fetchall()
        #return pd.DataFrame(res, columns=['filename', 'max_v', 'max_accel', 'max_decel', 'group'])

def getUncutRideLocationStatistics(start_rect_coords, end_rect_coords, exclude_coords = (0, 0, 0, 0), unnested_ride_where = "True", unnested_ride_lc_where = "True", accels_where = "True", ride_where = "True", time_diff = "5000"):
    with DatabaseConnection() as cur:
        # JOIN accels a ON (a.filename = r.filename AND a.timestamp = r.timestamp) --only data present in accels, accels included in unnested_ride/ride
        velo_filter = lambda name: f"{name} > 0.2 AND {name} != 'NaN' AND {name} < 15"
        group_q_ride = lambda perc: f"""SELECT percentile_cont({perc}) WITHIN GROUP (ORDER BY one.avg_v) FROM (SELECT AVG(velo) as avg_v FROM unnested_ride WHERE {velo_filter("velo")} GROUP BY filename) as one"""
        cur.execute(f"""
                WITH ride_location_cut as (
                    SELECT r.filename, r.duration, r.timestamp, r.velo FROM unnested_ride r JOIN (
                        SELECT a.filename, min_ts, max_ts
                        FROM
                        (
                            SELECT filename, min(timestamp) as min_ts
                            FROM unnested_ride
                            WHERE st_intersects((coords).geom, ST_MakeEnvelope ({start_rect_coords[0]}, {start_rect_coords[1]}, 
                                                                                {start_rect_coords[2]}, {start_rect_coords[3]},
                                                                                {SRID}
                                                                )
                            )
                            GROUP BY filename
                        ) a
                        INNER JOIN
                        (
                            SELECT filename, max(timestamp) as max_ts
                            FROM unnested_ride
                            WHERE st_intersects((coords).geom, ST_MakeEnvelope ({end_rect_coords[0]}, {end_rect_coords[1]}, 
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
                                    FROM unnested_ride
                                    WHERE st_intersects((coords).geom, ST_MakeEnvelope ({exclude_coords[0]}, {exclude_coords[1]}, 
                                                                                        {exclude_coords[2]}, {exclude_coords[3]}, 
                                                                                        {SRID}
                                                                                        )
                                    )
                                    GROUP BY filename
                                )
                            )
                    ) loc_filter ON (r.filename = loc_filter.filename)
                    WHERE r.timestamp > loc_filter.min_ts AND r.timestamp < loc_filter.max_ts AND extract(epoch FROM (max_ts - min_ts)) < {time_diff}
                )
                SELECT two.filename, four.avg_v, two.max_v, one.accel, one.decel, bike, CASE WHEN four.avg_v < ({group_q_ride(0.25)}) THEN 0 ELSE CASE WHEN four.avg_v < ({group_q_ride(0.75)}) THEN 1 ELSE 2 END END as group FROM (
                    SELECT filename, MAX(a.accel) as accel, MIN(a.accel) as decel FROM accels a JOIN ride_location_cut lc ON (lc.filename = a.filename AND lc.timestamp = a.timestamp) WHERE True AND {accels_where} GROUP BY filename
                ) as one JOIN (
                    SELECT filename, MAX(velo) as max_v FROM ride_location_cut WHERE {velo_filter("velo")} AND {unnested_ride_lc_where} GROUP BY filename
                ) as two ON (one.filename = two.filename) JOIN (
                    SELECT filename, bike FROM ride WHERE True AND {ride_where}
                ) as three ON (two.filename = three.filename) JOIN (
                    SELECT filename, AVG(velo) as avg_v FROM unnested_ride WHERE {velo_filter("velo")} AND {unnested_ride_where} GROUP BY filename
                ) as four ON (two.filename = four.filename)
                """)
        return cur.fetchall()
        #return pd.DataFrame(res, columns=['filename', 'max_v', 'max_accel', 'max_decel', 'group'])

def getRideLocation(start_rect_coords, end_rect_coords, exclude_coords = (0, 0, 0, 0), ride_where = "True", time_diff = "5000"): # Checked
    with DatabaseConnection() as cur:
        # JOIN accels a ON (a.filename = r.filename AND a.timestamp = r.timestamp) --only data present in accels, accels included in unnested_ride/ride
        #velo_filter = lambda name: f"{name} > 0.2 AND {name} != 'NaN' AND {name} < 15"
        #group_q_ride = lambda perc: f"""SELECT percentile_cont({perc}) WITHIN GROUP (ORDER BY one.avg_v) FROM (SELECT AVG(velo) as avg_v FROM unnested_ride WHERE {velo_filter("velo")} GROUP BY filename) as one"""
        cur.execute(f"""
                WITH ride_location_cut as (
                    SELECT r.filename, r.duration, r.timestamp, r.velo, (r.coords).geom::json -> 'coordinates' as coords FROM unnested_ride r JOIN (
                        SELECT a.filename, min_ts, max_ts
                        FROM
                        (
                            SELECT filename, min(timestamp) as min_ts
                            FROM unnested_ride
                            WHERE st_intersects((coords).geom, ST_MakeEnvelope ({start_rect_coords[0]}, {start_rect_coords[1]}, 
                                                                                {start_rect_coords[2]}, {start_rect_coords[3]},
                                                                                {SRID}
                                                                )
                            )
                            GROUP BY filename
                        ) a
                        INNER JOIN
                        (
                            SELECT filename, max(timestamp) as max_ts
                            FROM unnested_ride
                            WHERE st_intersects((coords).geom, ST_MakeEnvelope ({end_rect_coords[0]}, {end_rect_coords[1]}, 
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
                                    FROM unnested_ride
                                    WHERE st_intersects((coords).geom, ST_MakeEnvelope ({exclude_coords[0]}, {exclude_coords[1]}, 
                                                                                        {exclude_coords[2]}, {exclude_coords[3]}, 
                                                                                        {SRID}
                                                                                        )
                                    )
                                    GROUP BY filename
                                )
                            )
                    ) loc_filter ON (r.filename = loc_filter.filename)
                    WHERE r.timestamp > loc_filter.min_ts AND r.timestamp < loc_filter.max_ts AND extract(epoch FROM (max_ts - min_ts)) < {time_diff}
                )
                SELECT lc.filename, lc.duration, lc.timestamp, lc.velo, lc.coords, bike
                FROM ride_location_cut lc JOIN (
                    SELECT filename, bike FROM ride WHERE True AND {ride_where}
                ) as three ON (lc.filename = three.filename) ORDER BY lc.timestamp ASC
                """)
        return cur.fetchall()
        #return pd.DataFrame(res, columns=['filename', 'max_v', 'max_accel', 'max_decel', 'group'])

def getUncutRideLocationWGrouping(start_rect_coords, end_rect_coords, exclude_coords = (0, 0, 0, 0), ride_where = "True", accels_where = "True", time_diff = "5000"): # Checked
    with DatabaseConnection() as cur:
        # JOIN accels a ON (a.filename = r.filename AND a.timestamp = r.timestamp) --only data present in accels, accels included in unnested_ride/ride
        velo_filter = lambda name: f"{name} > 0.2 AND {name} != 'NaN' AND {name} < 15"
        group_q = lambda perc: f"""SELECT percentile_cont({perc}) WITHIN GROUP (ORDER BY one.avg_v) FROM (SELECT AVG(velo) as avg_v FROM accels WHERE {velo_filter("velo")} GROUP BY filename) as one"""
        cur.execute(f"""
                WITH ride_location_cut as (
                    SELECT r.filename, r.duration, r.timestamp, r.velo, (r.coords).geom::json -> 'coordinates' as coords FROM unnested_ride r JOIN (
                        SELECT a.filename, min_ts, max_ts
                        FROM
                        (
                            SELECT filename, min(timestamp) as min_ts
                            FROM unnested_ride
                            WHERE st_intersects((coords).geom, ST_MakeEnvelope ({start_rect_coords[0]}, {start_rect_coords[1]}, 
                                                                                {start_rect_coords[2]}, {start_rect_coords[3]},
                                                                                {SRID}
                                                                )
                            )
                            GROUP BY filename
                        ) a
                        INNER JOIN
                        (
                            SELECT filename, max(timestamp) as max_ts
                            FROM unnested_ride
                            WHERE st_intersects((coords).geom, ST_MakeEnvelope ({end_rect_coords[0]}, {end_rect_coords[1]}, 
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
                                    FROM unnested_ride
                                    WHERE st_intersects((coords).geom, ST_MakeEnvelope ({exclude_coords[0]}, {exclude_coords[1]}, 
                                                                                        {exclude_coords[2]}, {exclude_coords[3]}, 
                                                                                        {SRID}
                                                                                        )
                                    )
                                    GROUP BY filename
                                )
                            )
                    ) loc_filter ON (r.filename = loc_filter.filename)
                    WHERE r.timestamp > loc_filter.min_ts AND r.timestamp < loc_filter.max_ts AND extract(epoch FROM (max_ts - min_ts)) < {time_diff}
                )
                SELECT lc.filename, lc.duration, lc.timestamp, lc.velo, lc.coords, bike, CASE WHEN two.avg_v < ({group_q(0.25)}) THEN 0 ELSE CASE WHEN two.avg_v < ({group_q(0.75)}) THEN 1 ELSE 2 END END as group
                FROM ride_location_cut lc JOIN (
                    SELECT filename, AVG(velo) as avg_v FROM accels WHERE {velo_filter("velo")} AND {accels_where} GROUP BY filename
                ) as two ON (lc.filename = two.filename) JOIN (
                    SELECT filename, bike FROM ride WHERE True AND {ride_where}
                ) as three ON (lc.filename = three.filename) ORDER BY lc.timestamp ASC
                """)
        return cur.fetchall()
        #return pd.DataFrame(res, columns=['filename', 'max_v', 'max_accel', 'max_decel', 'group'])

#bug: decide on weighted avg or simple avg
#bug: check why some velos are NaN (importing, only rides)
#velo_filter means no stops in rides/calculations
#check: usually avg_v has to use velo_filter (filtered/calculated)
#check: grouping avg_v must check with avg_v of query (what<>what) + try sticking to accels or ride globally for the group query (group_q) + group query cannot be cut, rides must not be location-cut
#interest: avg_v|how is cut+(filtered/calculated), max_v|how is cut+(filtered/calculated), a/d|how is cut+(filtered/calculated), groups|from what<>what
#how is cut: [ride-uncut][accels-cut][location-cut][(accels)location-cut], note a/d cannot be ride-uncut
#filtered/calculated: [unfiltered][{some_filter}]
#dataset props (rows, especially focused on coordinates):
    #statistic (aggregated, grouped by ride/filename):
        #accel: (needs filter {velo_filter} for statistics)
            #general+ (getRideGeneralStatisticsAccel - avg_v is accels-cut+({velo_filter}), max_v accels-cut+({velo_filter}), a/d accels-cut+(unfiltered), groups from avg_v<>group_q)
            #location:
                #N/A?
                #uncut?
                #acut+ (getUncutRideLocationStatisticsAccel - avg_v accels-cut+({velo_filter}), max_v accels-cut+({velo_filter}), a/d accels-cut+(unfiltered), groups from avg_v<>group_q)
                #acut+ (getRideLocationStatisticsAccel - avg_v accels-cut+({velo_filter}), max_v location-cut+({velo_filter}), a/d location-cut+(unfiltered), groups from avg_v<>group_q)
                #lcut+ (getCutRideLocationStatisticsAccel - avg_v location-cut+({velo_filter}), max_v location-cut+({velo_filter}), a/d location-cut+(unfiltered), groups from avg_v<>group_q) !! avg_v does not check with group_q
        #ride: (needs filter {velo_filter} for statistics)
            #general+ (getRideGeneralStatistics - avg_v is ride-uncut+({velo_filter}), max_v is ride-uncut+({velo_filter}), a/d accels-cut+(unfiltered), groups from avg_v<>group_q_ride)
            #location:
                #N/A?
                #uncut+ (getUncutRideLocationStatistics - avg_v is ride-uncut+({velo_filter}), max_v is location-cut+({velo_filter}), a/d (accels)location-cut+(unfiltered), groups from avg_v<>group_q_ride)
                #acut?
                #lcut?

    #non_statistic (pure, non-aggregated, usually for maps):
        #accel: (needs join to ride)
            #general?x
            #location:
                #N/A?x
                #uncut?x
                #acut?x
                #lcut?x
        #ride:
            #general?
            #location:
                #N/A+ (getRideLocation - - N/A, - N/A, -/- N/A, - N/A)
                #uncut+ (build_and_execute_query - avg_v is ride-uncut+({velo_filter}), -(max_v) N/A, -/-(a/d) N/A, groups from avg_v<>group_q) !! avg_v does not check with group_q)
                #acut+ (getUncutRideLocationWGrouping - avg_v accels-cut+({velo_filter}), - N/A, -/- N/A, groups from avg_v<>group_q)
                #lcut?
#group query:
    #statistic only
        #accel:
            #general only+ ALL:(group_q) - avg_v is accels-cut+({velo_filter})
        #ride:
            #general only+ (group_q_ride) - avg_v is ride-uncut+({velo_filter})

#group_new generation -> uncut rides, filter out stops - unnested_ride: avg_v is ride-uncut+({velo_filter}), max_v is ride-uncut+({velo_filter}), a/d accels-cut+(unfiltered), groups from avg_v<>group_q_ride
#straight evaluation -> unnested_ride: avg_v is ride-uncut+({velo_filter}), max_v is location-cut+({velo_filter}), a/d (accels)location-cut+(unfiltered), groups from avg_v<>group_q_ride
#display query 1 -> unnested_ride: - N/A, - N/A, -/- N/A, - N/A