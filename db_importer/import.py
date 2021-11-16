import os
import warnings
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool

from pandas.core.common import SettingWithCopyWarning
from tqdm import tqdm

from db_connection import DatabaseConnection
from settings import *
import rides


def get_file_paths(IMPORT_DIRECTORY):
    files = []
    for r, d, f in os.walk(IMPORT_DIRECTORY, followlinks=True):
        for file in f:
            # filter out profile folders and  hidden files (e.g.: '.DS_Store')
            if '.' not in file and 'Profiles' not in r:
                files.append(os.path.join(r, file))
    return files


def import_file(file):
    filename = Path(file).name
    region = Path(file).parents[2].name

    with DatabaseConnection() as cur:
        cur.execute("""
            SELECT * FROM public."parsedfiles" WHERE filename LIKE %s
        """, (f'%{filename}%', ))
        if cur.fetchone() is not None:
            return
    try:
        with DatabaseConnection() as cur:   # new database connection for the whole transaction
            print(file)
            rides.handle_ride_file(file, cur)

            cur.execute("""
                INSERT INTO public."parsedfiles" ("filename", "region", "importtimestamp") VALUES (%s, %s, %s)
            """, [filename, region, datetime.utcnow()])
    except Exception as e:
        print(f"Skipped ride {filename} due to exception {e}")

if __name__ == '__main__':

    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    
    files = get_file_paths(IMPORT_DIRECTORY)

    print(f"Number of path entries in directory: {len(files)}")

    num_files = len(files)

    with Pool() as p:
        with tqdm(total=num_files) as pbar:
            for i in p.imap_unordered(import_file, files):
                pbar.update()

