import logging

import pandas as pd
import requests
import tqdm

from config import LOG_CONFIG, LOG_LEVEL


def download_file(url, filename="None"):
    try:
        with open(filename, "wb") as fp, requests.get(url, stream=True) as r:
            chunk_size = 1024
            for chunk in tqdm.tqdm(
                r.iter_content(chunk_size=chunk_size),
                total=int(int(r.headers["Content-Length"]) / chunk_size),
                unit="KB",
                desc=filename,
                leave=True,
            ):
                fp.write(chunk)
    except KeyError:
        with open(filename, "wb") as fp, requests.get(url, stream=True) as req:
            for chunk in tqdm.tqdm(req.iter_content(chunk_size=chunk_size), desc=filename, leave=True):
                fp.write(chunk)


def get_new_data() -> tuple[pd.DataFrame, ...]:
    # URLs
    # https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30120-1/fulltext
    url_ch_cov_markers = "https://raw.githubusercontent.com/statistikZH/covid19zeitmarker/master/covid19zeitmarker.csv"
    url_apple_mobility_data = get_current_apple_url()
    url_ch_re_data = get_re_data_url()
    url_owid = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    url_policies = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"

    logger.info("%s", "------   Pulling data...   ------")

    logger.debug("%s", "Pulling mobility data...")
    download_file(url_apple_mobility_data, filename="assets/apple_mobility.csv")
    logger.info("%s", "Pulling lockdown data...")
    download_file(url_ch_cov_markers, filename="assets/ch_lockdown_data.csv")
    logger.debug("%s", "Pulling R_e data")
    download_file(url_ch_re_data, filename="assets/ch_re_data.csv")
    logger.debug("%s", "Pulling OWID data")
    download_file(url_owid, filename="assets/owid_data.csv")
    logger.debug("%s", "Pulling policies")
    download_file(url_policies, filename="assets/policies.csv")

    logger.debug("%s", "------Loading is completed ------")


def get_cached_data() -> tuple[pd.DataFrame, ...]:
    try:
        logger.debug("%s", "Reading from cache...")
        apple_mobility = pd.read_csv("assets/apple_mobility.csv")
        ch_lockdown_data = pd.read_csv("assets/ch_lockdown_data.csv")
        ch_re_data = pd.read_csv("assets/ch_re_data.csv")
        owid_data = pd.read_csv("assets/owid_data.csv")
        policies = pd.read_csv("assets/policies.csv")
    except FileNotFoundError:
        (
            apple_mobility,
            ch_lockdown_data,
            ch_re_data,
            owid_data,
            policies,
        ) = get_new_data()
        get_cached_data()
    return apple_mobility, ch_lockdown_data, ch_re_data, owid_data, policies


def get_current_apple_url() -> str:
    response = requests.get(
        "https://covid19-static.cdn-apple.com/covid19-mobility-data/current/v3/index.json"
    ).json()
    uri = (
        "https://covid19-static.cdn-apple.com/"
        + response["basePath"]
        + response["regions"]["en-us"]["csvPath"]
    )
    logger.debug("%s", f"Apple URI: {uri}")
    return uri


def get_re_data_url() -> str:
    response = requests.get(
        "https://ckan.opendata.swiss/api/3/action/package_search?q=title:COVID19"
    ).json()

    # Structure of response
    # result:
    #   results:
    #       result["title_for_slug"] == "covid-19-schweiz"
    #       resources:
    #           resource["name"] == "COVID19Re_geoRegion"
    #           resource["format"] == "CSV"
    #           -> resource["download-url"] -> Download URL to be returned

    results = response["result"]["results"]

    for result in results:
        try:
            if "covid-19-schweiz" in result["title_for_slug"]:
                resources = result["resources"]
                break
        except KeyError:  # some dont have a slug title
            pass

    for resource in resources:
        if (
            "COVID19Re_geoRegion" in resource["title"]["en"]
            and resource["format"].lower() == "csv"
        ):
            download_url = resource["download_url"]
            logger.info("%s", f"Download URL for R_e data: {download_url}")
            return download_url


# Get logger for main level, if this is is run as main, its properties will
# be set in the if name == main block, else those from get_data.py will be
# copied

logger = logging.getLogger("__main__")

if __name__ == "__main__":
    # setup logger, if this is run as main
    logger.setLevel(level=LOG_LEVEL)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter(LOG_CONFIG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # refresh data
    get_new_data()
