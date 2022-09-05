from adsp.work.db_con import DatabaseConnection
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import contextily as cx

with DatabaseConnection() as cur:
    cur.execute("""

        SELECT two.fn,
            CASE WHEN two.ve = 'NaN' THEN 1 ELSE two.ve END,
            two.pts,
            CASE WHEN two.ve > 0.2 AND two.ve != 'NaN' AND two.ve < 15 THEN 'tab:blue' ELSE 'tab:orange' END
            FROM (
        SELECT one.fn as fn, unnest(one.v) as ve, json_array_elements(one.pts) as pts FROM (
            SELECT filename as fn, velos as v, (((ST_Points(geom_raw::geometry))::json) -> 'coordinates') as pts from ride LIMIT 1
        ) as one
        ) as two

    """)
    objs = cur.fetchall()

pdf = pd.DataFrame(objs, columns=['file', 'speed', 'c', 'color'])

pdf['lon'], pdf['lat'] = zip(*pdf.c.values)

fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(pdf.lon, pdf.lat, 3 * pdf['speed'], pdf['color'])
moving_point, = ax.plot([], [], 'o')

cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenStreetMap.BZH)

ax.set_aspect(1.7)


#axtime = plt.axes([0.25, 0.1, 0.65, 0.03])

#timesl = Slider(axtime, "time", 0, pdf['lat'].size - 1, valstep=1)

def anim(i):
    moving_point.set_data(pdf['lon'][i], pdf['lat'][i])
    return moving_point

#timesl.on_changed(anim)

an = FuncAnimation(fig, anim, frames=pdf['lat'].size, interval=200)

plt.show()