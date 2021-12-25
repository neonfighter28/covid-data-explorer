import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import set_matplotlib_formats

plt.style.use('seaborn-poster')

set_matplotlib_formats('retina')

warnings.filterwarnings("ignore")

__all__ = ["flatten", "get_data", "normalize", "daily_increase", "moving_average", "prep_apple_mobility_data", "interp_nans"]

# helper method for flattening the data, so it can be displayed on a bar graph
def flatten(arr):
    return [i[0] for i in arr.tolist()]

def normalize(data):
    return [i/max(interp_nans(data)) for i in data]

def save_to_file(name, data):
    with open(f"assets/{name}.dat", "wb") as file:
        pickle.dump(data, file)

def read_from_file(name):
    with open(f"assets/{name}.dat", "rb") as file:
        return pickle.load(file)

def get_data():
    try:
        confirmed_df = read_from_file("confirmed_df")
        apple_mobility = read_from_file("apple_mobility")
        #raise FileNotFoundError

    except FileNotFoundError:
        # SOURCE https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30120-1/fulltext
        confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv').replace("sub-region", "sub_region", inplace=True)
        apple_mobility = pd.read_csv("https://covid19-static.cdn-apple.com/covid19-mobility-data/2204HotfixDev22/v3/en-us/applemobilitytrends-2021-12-23.csv")

        print(confirmed_df)
        save_to_file("confirmed_df", confirmed_df)
        save_to_file("apple_mobility", apple_mobility)

    return confirmed_df, apple_mobility

def daily_increase(data):
    return [data if i ==0 else data[i]-data[i-1] for i in range(len(data))]

def moving_average(data, window_size):
    return [np.mean(data[i:i+window_size]) if i + window_size < len(data) else np.mean(data[i:len(data)]) for i in range(len(data))]

def prep_apple_mobility_data(apple_mobility, country) -> list[int, int]:

    try:
        default_mob_data_dict = {}
        for i, key in enumerate(apple_mobility):
            default_mob_data_dict[key] = []
    except KeyError:
        pass

    # Get corresponding data rows for country:
    indexes_of_datarows = [i for i, v in enumerate(apple_mobility.region) if v.upper() == country.upper()]

    datasets = []

    # Add Values to data structure
    for i in indexes_of_datarows:
        mob_data_dict = default_mob_data_dict.copy()
        for k, _ in mob_data_dict.items():
            mob_data_dict[k] = apple_mobility.loc[i][k]
        datasets.append(mob_data_dict)

    return [([(k, v) for index, (k, v) in enumerate(dataset.items()) if index > 5]) for dataset in datasets]

def interp_nans(x:[float],left=None, right=None, period=None)->[float]: # pylint: disable=invalid-name
    return list(
        np.interp(
            x=list(range(len(x))),
            xp=[i for i, yi in enumerate(x) if np.isfinite(yi)],
            fp=[yi for i, yi in enumerate(x) if np.isfinite(yi)],
            left=left,
            right=right,
            period=period
            )
        )
