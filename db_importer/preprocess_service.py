from pyproj import Proj
import utils
from geopy.distance import great_circle


def preprocess_basics(df):
    df_shift = df.shift(1)

    df['l_lon'] = df_shift['lon']
    df['l_lat'] = df_shift['lat']
    df = df[~df['l_lon'].isnull()]
    df['duration'] = (df['timestamp'] - df_shift['timestamp']).dt.total_seconds()
    df['dist'] = df.apply(lambda x: great_circle([x['l_lat'], x['l_lon']], [x['lat'], x['lon']]).meters, axis=1)
    df['velo'] = df['dist'] / df['duration']

    df_shift = df.shift(1)

    df['a'] = (df['velo'] - df_shift['velo']) / df['duration']

    proj = Proj('epsg:5243')

    proj_coords = df.apply(lambda x: proj(x['lon'], x['lat']), axis=1)
    df.loc0[:, ['x', 'y']] = list(map(list, proj_coords))

    df['spike'] = utils.find_velocity_spikes(df)

    df['section'] = df['spike'].cumsum()
    df['section'] = df.apply(lambda x: int(x['section']) if x['spike'] is False else -1, axis=1)

    return df
