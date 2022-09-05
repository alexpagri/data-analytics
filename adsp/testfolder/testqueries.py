from adsp.work.db_con import DatabaseConnection



with DatabaseConnection() as cur:
    cur.execute("""

        SELECT one.fn, unnest(one.v), json_array_elements(one.pts) FROM (
            SELECT filename as fn, velos as v, (((ST_Points(geom::geometry))::json) -> 'coordinates') as pts from ride LIMIT 1
        ) as one

    """)
    objs = cur.fetchall()

    print(objs)