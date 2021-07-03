import pandas as pd
from datetime import datetime, timezone
from pyproj import Proj, proj


def read_ubx(filename):
    res = []

    # you may also want to remove whitespace characters like `\n` at the end of each line
    with open(filename) as f:

        coord_index = 0
        info = gps_information()
        for message in f:
            # cut off checksum and split at commas
            message = message[:message.index('*')]
            message = message.split(',')
            if message[0] == '$GPRMC':
                info = gps_information()
                info.time, info.valid, info.lat, info.lat_direction, info.lon, info.lon_direction = message[1:7]
            if message[0] == '$GPVTG':
                info.velo = message[7]
            if message[0] == '$GPGSA':
                info.pdop, info.hdop, info.vdop = message[15:]
            if message[0] == '$GPGLL':
                info.lat, info.lat_direction, info.lon, info.lon_direction, info.time, info.valid = message[1:7]
            if message[0] == '$GPGST':
                info.time, info.rms = message[1:3]
                info.lat_std, info.lon_std = message[6:8]
            if message[0] == '$GPZDA':
                info.time, info.day, info.month, info.year = message[1:5]
                info.coord_index = coord_index
                coord_index += 1
                try:
                    res.append(individual_raw_line_to_values(info).to_array())
                except AttributeError:
                    print('Attribute error. Coordinate is ignored.') 
    return pd.DataFrame(res,
                        columns=['coord_index', 'date', 'valid', 'lat', 'lon', 'lat_dir', 'lon_dir', 'x', 'y', 'velo',
                                 'rms', 'lat_std',
                                 'lon_std'])


def change_gps_format_to_degree(ublox_format):
    degree = int(ublox_format / 100)
    minutes = ublox_format - 100 * degree
    return round(degree + minutes / 60, 7)


def change_gps_format_to_degree_new(decimal_format):
    return decimal_degrees(*dm(decimal_format))


def dm(x):
    degrees = int(x) // 100
    minutes = x - 100 * degrees
    return degrees, minutes


def decimal_degrees(degrees, minutes):
    return degrees + minutes / 60


def change_utc_to_unix(time, day, month, year):
    date_str = '%s-%s-%s-%s' % (year, month, day, time)
    dt = datetime.strptime(date_str, '%Y-%m-%d-%H%M%S.%f')
    return dt


def individual_raw_line_to_values(info):
    if info.valid == 'A':
        info.date = change_utc_to_unix(info.time, info.day, info.month, info.year)
        info.lat = change_gps_format_to_degree_new(float(info.lat))
        info.lon = change_gps_format_to_degree_new(float(info.lon))
        info.velo = float(info.velo) / 3.6
        info.rms = float(info.rms)
        info.lat_std = float(info.lat_std)
        info.lon_std = float(info.lon_std)
        info.x, info.y = info.proj(info.lon, info.lat)
        return info
    return None


class gps_information:

    def __init__(self):
        self.coord_index = ''
        self.date = ''
        self.time = ''
        self.day = ''
        self.month = ''
        self.year = ''
        self.valid = ''
        self.lat = ''
        self.lon = ''
        self.lat_direction = ''
        self.lon_direction = ''
        self.x = ''
        self.y = ''
        self.velo = ''
        self.rms = ''  # RMS value of the standard deviation of the range inputs to the navigation process.
        self.lat_std = ''
        self.lon_std = ''
        self.proj = Proj('epsg:5243')

    def to_array(self):
        return [self.coord_index, self.date, self.valid, self.lat, self.lon,
                self.lat_direction, self.lon_direction, self.x, self.y, self.velo, self.rms,
                self.lat_std, self.lon_std]

    def __str__(self):
        attrs = vars(self)
        return ', \n'.join("%s: %s" % item for item in attrs.items())
