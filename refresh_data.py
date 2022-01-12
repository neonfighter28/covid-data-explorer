import logging
import pandas as pd
import config
import requests
import pickle
from config import LOG_CONFIG, LOG_LEVEL


def get_new_data():
    # URLs
    # https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30120-1/fulltext
    url_cov_global = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    url_ch_cov_markers = "https://raw.githubusercontent.com/statistikZH/covid19zeitmarker/master/covid19zeitmarker.csv"
    url_apple_mobility_data = get_current_apple_url()
    url_ch_re_data = get_re_data_url()

    logger.info("%s", "------   Pulling data...   ------")

    confirmed_df = pd.read_csv(url_cov_global)
    logger.debug("%s", "Pulling mobility data...")
    apple_mobility = pd.read_csv(url_apple_mobility_data)
    logger.info("%s", "Pulling lockdown data...")
    ch_lockdown_data = pd.read_csv(url_ch_cov_markers)
    logger.debug("%s", "Pulling R_e data")
    ch_re_data = pd.read_csv(url_ch_re_data)

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
        get_new_data()
        get_cached_data()
    return confirmed_df, apple_mobility, ch_lockdown_data, ch_re_data


def get_current_apple_url():
    response = requests.get(
        "https://covid19-static.cdn-apple.com/covid19-mobility-data/current/v3/index.json"
        ).json()
    uri = ("https://covid19-static.cdn-apple.com/"
           + response['basePath']
           + response['regions']['en-us']['csvPath'])
    logger.debug("%s", f"Apple URI: {uri}")
    return uri


def get_re_data_url():
    response = requests.get("https://ckan.opendata.swiss/api/3/action/package_search?q=title:COVID19").json()

    # Structure of response
    # result:
    #   results:
    #       result["title_for_slug"] == "covid-19-schweiz"
    #       resources:
    #           resource["name"] == "COVID19Re_geoRegion"
    #           resource["format"] == "CSV"
    #           -> resource["download-url"] -> download url to be returned

    results = response["result"]["results"]

    for result in results:
        try:
            if "covid-19-schweiz" in result["title_for_slug"]:
                resources = result["resources"]
                break
        except KeyError:  # some dont have a slug title
            pass

    for resource in resources:
        if "COVID19Re_geoRegion" in resource["title"]["en"] and resource["format"].lower() == "csv":
            download_url = resource["download_url"]
            logger.info("%s", f"Download URL for R_e data: {download_url}")
            return download_url

def save_to_file(name, data):
    with open(f"assets/{name}.dat", "wb") as file:
        logger.debug("%s", f"Saving to file {file.name}")
        pickle.dump(data, file)


def read_from_file(name):
    with open(f"assets/{name}.dat", "rb") as file:
        logger.debug("%s", f"Reading from file {file.name}")
        return pickle.load(file)


# Get logger for main level, if this is is run as main, its properties will
# be set in the if name == main block, else those from get_data.py will be
# copied

logger = logging.getLogger("__main__")

if __name__ == "__main__":
    logger.setLevel(level=LOG_LEVEL)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter(LOG_CONFIG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    get_new_data()