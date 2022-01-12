import logging
import pandas as pd
import config
import requests
import pickle
from config import LOG_CONFIG


def get_data():
    # SOURCE
    # https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30120-1/fulltext
    logger.info("%s", "------   Pulling data...   ------")

    confirmed_df = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/\
master/csse_covid_19_data/csse_covid_19_time_series/\
time_series_covid19_confirmed_global.csv')
    logger.debug("%s", "Pulling mobility data...")
    apple_mobility = pd.read_csv(get_current_apple_url())
    logger.info("%s", "Pulling lockdown data...")
    ch_lockdown_data = pd.read_csv(
        "https://raw.githubusercontent.com/statistikZH/\
covid19zeitmarker/master/covid19zeitmarker.csv"
    )
    logger.debug("%s", "Pulling R_e data")
    # https://opendata.swiss/en/dataset/covid-19-schweiz/resource/92c50632-ecfa-4234-9c36-ea6121bb68ca
    ch_re_data = pd.read_csv("https://www.covid19.admin.ch/api/data/20220104-ph1baz76/sources/COVID19Re_geoRegion.csv")

    logger.debug("%s", "------Loading is completed ------")
    save_to_file("confirmed_df", confirmed_df)
    save_to_file("apple_mobility", apple_mobility)
    save_to_file("ch_lockdown_data", ch_lockdown_data)
    save_to_file("ch_re_data", ch_re_data)
    logger.debug("%s", "saved to cache!")

    return confirmed_df, apple_mobility, ch_lockdown_data, ch_re_data


def get_cached_data():
    c = 0
    try:
        logger.debug("%s", "Reading from cache...")
        confirmed_df = read_from_file("confirmed_df")
        apple_mobility = read_from_file("apple_mobility")
        ch_lockdown_data = read_from_file("ch_lockdown_data")
        ch_re_data = read_from_file("ch_re_data")
    except FileNotFoundError:
        c += 1
        if c > 3:
            raise RecursionError
        get_data()
        get_cached_data()
    return confirmed_df, apple_mobility, ch_lockdown_data, ch_re_data


def get_current_apple_url():
    response = requests.get(
        "https://covid19-static.cdn-apple.com/covid19-mobility-data/current/v3/index.json").json()
    uri = ("https://covid19-static.cdn-apple.com/"
           + response['basePath']
           + response['regions']['en-us']['csvPath'])
    logger.debug("%s", f"Apple URI: {uri}")
    return uri


def save_to_file(name, data):
    with open(f"assets/{name}.dat", "wb") as file:
        logger.debug("%s", f"Saving to file {file.name}")
        pickle.dump(data, file)


def read_from_file(name):
    with open(f"assets/{name}.dat", "rb") as file:
        logger.debug("%s", f"Reading from file {file.name}")
        return pickle.load(file)


# Get logger for main level, if this is is run as main, its properties will
# be set in the if name == main block, else those from get_data.py will be copied

logger = logging.getLogger("__main__")

if __name__ == "__main__":
    logger.setLevel(level=logging.DEBUG)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter(LOG_CONFIG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    get_data()
