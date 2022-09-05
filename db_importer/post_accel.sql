UPDATE ride
SET accels_raw = three.accels_raw FROM (
SELECT filename as fn, array_agg(a ORDER BY t) as accels_raw FROM (
SELECT filename, v, v2, t, CASE v WHEN 0 THEN 'NaN' ELSE CASE v2 WHEN 0 THEN 'NaN' ELSE CASE d WHEN 0 THEN 'NaN' ELSE (v2 - v) / d END END END as a FROM (
SELECT filename, unnest(velos_raw) as v, unnest(velos_raw[2:]) as v2, unnest(timestamps) as t, unnest(durations) as d from ride
) one WHERE v is not null AND v2 is not null
) two GROUP BY filename
) three
WHERE filename = three.fn;

UPDATE ride
SET accels = three.accels FROM (
SELECT filename as fn, array_agg(a ORDER BY t) as accels FROM (
SELECT filename, v, v2, t, CASE v WHEN 0 THEN 'NaN' ELSE CASE v2 WHEN 0 THEN 'NaN' ELSE CASE d WHEN 0 THEN 'NaN' ELSE (v2 - v) / d END END END as a FROM (
SELECT filename, unnest(velos) as v, unnest(velos[2:]) as v2, unnest(timestamps) as t, unnest(durations) as d from ride
) one WHERE v is not null AND v2 is not null
) two GROUP BY filename
) three
WHERE filename = three.fn;

--compute AVG for NaN values (filters.py#L53)