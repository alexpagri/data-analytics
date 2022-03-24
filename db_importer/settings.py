from dotenv import dotenv_values

# take environment variables from .env and load into dict
dotenv_values_dict = dotenv_values()  

DB_HOST = dotenv_values_dict['DB_HOST']
DB_NAME = dotenv_values_dict['DB_NAME']
DB_USER = dotenv_values_dict['DB_USER']
DB_PASSWORD = dotenv_values_dict['DB_PASSWORD']
DB_PORT = dotenv_values_dict['DB_PORT']

IMPORT_DIRECTORY = "../simra_data_2022-02-14/"

MIN_RIDE_DISTANCE = 200  # in meters
MIN_RIDE_DURATION = 3 * 60  # in seconds
MAX_RIDE_AVG_SPEED = 40  # in km/h
MIN_DISTANCE_TO_COVER_IN_5_MIN = 100  # in meters

