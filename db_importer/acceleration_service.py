import pandas as pd
import numpy as np
import utils
import sqlalchemy
from settings import *

engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

seg_id = 0

def process_acceleration_segments(df, filename, cur):
    cur.execute(""" SELECT seg_id FROM accels ORDER BY seg_id DESC LIMIT 1""")
    result = cur.fetchone()
    if result is not None:
        seg_id = result[0]
    else:
        print('seg id = 0 ')
        seg_id = 0

    df = df[df['section'] >= 0]
    tmp = df.copy()


    df['velo_lp0'] = tmp['velo_lp0']
    df['a_lp0'] = (df['velo_lp0'] - df.shift(1)['velo_lp0']) / df['duration']

    tmp = insert_empty_rows(df).copy()
    tmp_ak = tmp[['timestamp', 'a_lp0']].copy()

    ## es werden empty rows reingebuffert, damit die Auflösung von a erhöht werden kann (über interpolate).
    ## So lassen sich values nahe 0 finden. Bei 1/3Hz Auflösung gelänge das bei zu wenigen.
    t0 = tmp_ak['timestamp'].min()
    m = tmp_ak['timestamp'].notnull()
    tmp_ak.loc[m, 't_int'] = (tmp_ak.loc[m, 'timestamp'] - t0).dt.total_seconds()
    tmp_ak['timestamp'] = t0 + pd.to_timedelta(tmp_ak.t_int.interpolate(), unit='s')

    tmp_ak = tmp_ak[~tmp_ak.timestamp.isnull()]
    tmp_ak['a_lp0'] = tmp_ak['a_lp0'].astype('float64').interpolate()

    th = 0.005
    extrema = tmp_ak[tmp_ak['a_lp0'].between(-th, th)].timestamp.values

    # df = df.set_index('timestamp')
    segments = []
    for i in range(len(extrema)):
        if i + 1 == len(extrema):
            break
        e1 = extrema[i]
        e2 = extrema[i + 1]
        segments.append(df[utils.find_closest_index(df, e1):utils.find_closest_index(df, e2)].drop(
            ['lon', 'lat', 'accuracy', 'l_lon', 'l_lat', 'x', 'y', 'spike', 'section', 'x_k', 'y_k',
             'lon_k', 'lat_k'], axis=1))

    for s in segments:
        if len(s) < 1:
            continue
        if (s.dist.sum() < 20) | (s.dist.sum() > 350):
            continue
        if (s.duration.sum() < 5) | (s.duration.sum() > 60):
            continue

        initial_speed = s['velo_lp0'].iloc[0]
        final_speed = s['velo_lp0'].iloc[-1]
        if final_speed == initial_speed:
            continue
        t = abs(initial_speed - final_speed) / max(initial_speed, final_speed)
        if t < 0.5:
            continue

        if final_speed > initial_speed:
            ## hier < x, da es vorkommen kann, dass an den Grenzen des Segments marginal negative values vorkommen.
            if (s['a_lp0'] < -0.1).any():
                continue
            s['type'] = 'a'
        else:
            if (s['a_lp0'] > 0.1).any():
                continue
            s['type'] = 'd'

        s['seg_id'] = seg_id
        seg_id += 1

        s['filename'] = filename

        s.columns = ['timestamp', 'duration', 'dist', 'velo_raw', 'accel_raw', 'velo', 'accel', 'type', 'seg_id', 'filename']
        s.to_sql('accels', engine, index=False, if_exists='append')


def insert_empty_rows(df):
    data = df.values
    for _ in range(4):
        nans = np.where(np.empty_like(data), np.nan, np.nan)
        data = np.hstack([nans, data])
    return pd.DataFrame(data.reshape(-1, df.shape[1]), columns=df.columns)
