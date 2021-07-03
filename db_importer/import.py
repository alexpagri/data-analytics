from datetime import datetime
from settings import *
import os
from db_connection import DatabaseConnection
import tqdm
from pandas.core.common import SettingWithCopyWarning

import profile, rides

import warnings

if __name__ == '__main__':

    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    files = []
    for r, d, f in os.walk(IMPORT_DIRECTORY, followlinks=True):
        for file in f:
            if '.' not in file:
                files.append(os.path.join(r, file))
    for file in tqdm.tqdm(files):
        if "Profiles" in file:
            continue
        filename = file.split("/")[-1]
        region = file.split("/")[-3]
        with DatabaseConnection() as cur:
            cur.execute("""
                SELECT * FROM public."parsedfiles" WHERE filename LIKE %s
            """, (f'%{filename}%', ))
            if cur.fetchone() is not None:
                continue
        try:
            with DatabaseConnection() as cur:   # new database connection for the whole transaction
                if "Profiles" in file:
                    continue
                else:
                    print(file)
                    rides.handle_ride_file(file, cur)

                cur.execute("""
                    INSERT INTO public."parsedfiles" ("filename", "region", "importtimestamp") VALUES (%s, %s, %s)
                """, [filename, region, datetime.utcnow()])
        except Exception as e:
            #raise e
            print(f"Skipped ride {filename} due to exception {e}")
