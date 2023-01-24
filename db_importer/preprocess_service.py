from pyproj import Proj
import utils
from geopy.distance import great_circle


def preprocess_basics(df):
    df_shift = df.shift(1)

    df['l_lon'] = df_shift['lon']
    df['l_lat'] = df_shift['lat']
    #df = df[~df['l_lon'].isnull()] # kind of a filter for the first element
    df_valid = df.iloc[1:]
    df['duration'] = (df_valid['timestamp'] - df_shift['timestamp']).dt.total_seconds() # duration in relation to the previous element
    df['dist'] = df_valid.apply(lambda x: great_circle([x['l_lat'], x['l_lon']], [x['lat'], x['lon']]).meters, axis=1) # distance in relation to the previous element
    df_valid = df.iloc[1:]
    df['velo'] = df_valid['dist'] / df_valid['duration'] # velocity in relation to the previous element

    filtered = df[((df.velo < 25) & (df.dist < 90) & (df.duration > 1.5) & (df.duration < 6.5)) | (df.index == df.index[0])] # filter out invalid but keep current first
    if len(df) - len(filtered) > 4:
        raise Exception("Too many artifacts in file")
    
    df = filtered.copy()

    df_shift = df.shift(1) # copy again

    df['a'] = (df['velo'] - df_shift['velo']) / df['duration']

    proj = Proj('epsg:5243')

    proj_coords = df.apply(lambda x: proj(x['lon'], x['lat']), axis=1)
    df.loc[:, ['x', 'y']] = list(map(list, proj_coords))

    df['spike'] = utils.find_velocity_spikes(df)

    df['section'] = df['spike'].cumsum()
    df['section'] = df.apply(lambda x: int(x['section']) if x['spike'] is False else -1, axis=1)
    #section = 0, 0, -1, 1, 1, 1, -1, 2, -1, 3, 3...

    return df
