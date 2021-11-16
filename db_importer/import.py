from datetime import datetime
from settings import *
import os
from db_connection import DatabaseConnection
import tqdm
from pandas.core.common import SettingWithCopyWarning
from multiprocessing.dummy import Pool as ThreadPool
import profile, rides
import warnings
from datetime import datetime

def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

def get_file_paths(IMPORT_DIRECTORY):
    files = []
    for r, d, f in os.walk(IMPORT_DIRECTORY, followlinks=True):
        for file in f:
            if '.' not in file:
                files.append(os.path.join(r, file))
    return files   

def import_file(file):
    if "Profiles" in file:
        return
    filename = file.split("/")[-1]
    region = file.split("/")[-3]
    with DatabaseConnection() as cur:
        cur.execute("""
            SELECT * FROM public."parsedfiles" WHERE filename LIKE %s
        """, (f'%{filename}%', ))
        if cur.fetchone() is not None:
            return
    try:
        with DatabaseConnection() as cur:   # new database connection for the whole transaction
            if "Profiles" in file:
                return
            else:
                print(file)
                rides.handle_ride_file(file, cur)

            cur.execute("""
                INSERT INTO public."parsedfiles" ("filename", "region", "importtimestamp") VALUES (%s, %s, %s)
            """, [filename, region, datetime.utcnow()])
    except Exception as e:
        #raise e
        print(f"Skipped ride {filename} due to exception {e}")

if __name__ == '__main__':

    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    
    files = get_file_paths(IMPORT_DIRECTORY)

    print(f"Number of path entries in directory: {len(files)}")

    print_time()
    # Make the Pool of workers
    pool = ThreadPool(8)

    # and return the results
    pool.map(import_file, files)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()
    print_time()