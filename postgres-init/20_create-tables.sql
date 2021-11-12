CREATE TABLE parsedfiles (
    filename VARCHAR(255),
    region VARCHAR(255),
    importtimestamp TIMESTAMP
);

CREATE TABLE accels (
      seg_id INT,
      timestamp TIMESTAMP,
      duration FLOAT, 
      dist FLOAT, 
      velo_raw FLOAT,  
      accel_raw FLOAT, 
      velo FLOAT, 
      accel FLOAT, 
      type VARCHAR(255),
      filename VARCHAR(255)
);

CREATE TABLE ride (
    geom_raw geography(LINESTRING), 
    geom geography(LINESTRING), 
    timestamps TIMESTAMP[], 
    filename VARCHAR(255), 
    velos_raw FLOAT[], 
    velos FLOAT[], 
    durations FLOAT[], 
    distances FLOAT[], 
    "start" geography(POINT), 
    "end" geography(POINT)
);