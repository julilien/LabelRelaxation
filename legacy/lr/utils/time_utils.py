import time
from datetime import datetime


def get_time_str():
    """
    Function returning a string representation of the current time in ms.

    :return: Returns the current time ms as string
    """
    return str(int(round(time.time() * 1000)))


def get_curr_date_str():
    """
    Function returning the date of today in the format YYYYmmdd.

    :return: Returns the current data as string
    """
    return datetime.today().strftime('%Y%m%d')


def get_full_timestamp_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")