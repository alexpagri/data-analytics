import matplotlib.pyplot as plt
import numpy as np
#import cudf as cf
import pandas as pd
import contextily as cx
#import geopandas as gp
import folium
import geopy.distance
from scipy.spatial.transform import Rotation

m = folium.Map()

tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(m)

def ang_to_vec(lat, lon):
    return Rotation.from_euler('xyz', [-lat, lon, 0], degrees=True).apply([0, 0, 1])

def rot_g(center, dest, radius_m=5, deg=90):
    dist = geopy.distance.geodesic((center['lat'], center['lon']), (dest['lat'], dest['lon'])).meters
    rotation_axis = ang_to_vec(center['lat'], center['lon'])
    vector_to_rotate_to = ang_to_vec(dest['lat'], dest['lon'])
    cross = np.cross(rotation_axis, vector_to_rotate_to)
    cross_len = np.linalg.norm(cross)
    if cross_len == 0:
        result_vector = rotation_axis
    else:
        cross_norm = cross / cross_len
        angle_between = np.arcsin(cross_len) # reverse AxBxSin
        small_vector = Rotation.from_rotvec(cross_norm * angle_between * radius_m / dist).apply(rotation_axis)
        result_vector = Rotation.from_rotvec(rotation_axis * deg, degrees=True).apply(small_vector)
    lat = np.arcsin(result_vector[1]) * 180 / np.pi #y
    lon = np.arctan2(result_vector[0], result_vector[2]) * 180 / np.pi #y'/x' = x/z
    return [lat, lon]

def rot_90(vec, dec=10):
    #return np.round(Rotation.from_euler('xyz', [0, 0, 90], degrees=True).apply(vec + [0])[:2], dec)
    return np.array(-vec[1], vec[0])

def build_trapezoid(a, b, ar, br, color):
    s1a = rot_g(a, b, ar)
    s1b = rot_g(b, a, br, -90)
    s2a = rot_g(a, b, ar, -90)
    s2b = rot_g(b, a, br)
    folium.PolyLine([s1a, s1b], color=color).add_to(m)
    folium.PolyLine([s2a, s2b], color=color).add_to(m)
    #a = np.array([1, 1])
    #b = np.array([2, 2])
    #c = np.
    #r = cf.DataFrame([np.arange(1, 100000), np.arange(1, 100000)]).transpose().rename(columns={0:'0', 1:'1'})
    #r = pd.DataFrame({'0':np.arange(1, 100000000), '1':np.arange(1, 100000000)})
    #r = cf.DataFrame({'a':[1,2,3],'b':[4,5,6]})
    #r['c'] = r.apply(lambda x: x['0'] + x['1'], axis=1)
    #print(r)

def add_path(path, color='blue', lod=3):
    ll, llr, r, ts = path[['lat', 'lon']], path[['lat', 'lon', 'rad']], path['rad'], path['timestamp']
    if lod < 3:
        folium.PolyLine(ll, color=color).add_to(m)
    if lod > 1:
        for i in np.arange(0, llr.shape[0] - 1):
            a_ll, a_llr, a_r, a_ts = ll.iloc[i], llr.iloc[i], r.iloc[i], ts.iloc[i]
            b_ll, b_llr, b_r = ll.iloc[i+1], llr.iloc[i+1], r.iloc[i+1]
            folium.Circle(a_ll, radius=a_r, color=color, tooltip=str(a_r)+", "+str(i)+", "+str(a_ts)).add_to(m)
            if lod > 2:
                build_trapezoid(a_ll, b_ll, a_r, b_r, color)
        folium.Circle(ll.iloc[-1], radius=r.iloc[-1], color='green', tooltip=str(r.iloc[-1])+", "+str(r.shape[0])+", "+str(ts.iloc[-1])).add_to(m) 

def show_path():
    m.show_in_browser()

#a0 = osmnx.graph_from_point([47.124, 23.873], dist=750, dist_type="bbox")

#osmnx.plot_graph_folium(a0).show_in_browser()

#r0 = pd.DataFrame({'lon': 23.873 + np.array(np.arange(1, 300))/100, 'lat': 47.124 + np.array(np.arange(1, 300))/100, 'rad': np.arange(100.0, 100.0 + 299 * 30, 30)})

#add_path(r0)

#show_path()

#add_path(pd.DataFrame({'lon': 23.873 + np.array(np.arange(1, 1000))/1000, 'lat': 47.124 + np.array(np.arange(1, 1000))/1000, 'rad': np.arange(1, 1000)}))

#show_path()

##fig, ax = plt.subplots()

#x = np.arange(23.873, 23.88701, 0.0001)
#y = np.arange(47.124, 47.13801, 0.0001)

#m = folium.Map()

#m.fit_bounds([[y.min(), x.max()], [y.max(), x.min()]])

#folium.PolyLine(np.column_stack([y, x])).add_to(m)

#m.show_in_browser()

#df_2 = pd.DataFrame(np.column_stack([x, y]), columns=['lat', 'lon'])

#df = gp.GeoDataFrame(df_2, geometry=gp.points_from_xy(df_2['lat'], df_2['lon']), crs='EPSG:4326').to_crs(epsg=3857)['geometry']

#ax = df.plot()

#cx.add_basemap(ax, crs=df.crs, source=cx.providers.OpenStreetMap.DE)

#plt.show(block=True)