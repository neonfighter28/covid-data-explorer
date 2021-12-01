import datetime
import math
import operator
import pickle
import random
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sys import argv
from helper import *

plt.style.use('seaborn-poster')
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('retina')
import warnings

warnings.filterwarnings("ignore")

# helper method for flattening the data, so it can be displayed on a bar graph
def flatten(arr):
    a = []
    arr = arr.tolist()
    for i in arr:
        a.append(i[0])
    return a

def save_to_file(name, data):
    with open(f"assets/{name}.dat", "wb") as file:
        pickle.dump(data, file)

def read_from_file(name):
    with open(f"assets/{name}.dat", "rb") as file:
        return pickle.load(file)

def get_data():
    try:
        confirmed_df = read_from_file("confirmed_df")
        deaths_df = read_from_file("deaths_df")
        latest_data = read_from_file("latest_data")
        us_medical_data = read_from_file("us_medical_data")
        apple_mobility = read_from_file("apple_mobility")
        #raise FileNotFoundError

    except FileNotFoundError:
        confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv').replace("sub-region", "sub_region", inplace=True)
        deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
        # recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
        latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/11-21-2021.csv')
        us_medical_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/11-21-2021.csv')
        apple_mobility = pd.read_csv("https://covid19-static.cdn-apple.com/covid19-mobility-data/2202HotfixDev21/v3/en-us/applemobilitytrends-2021-11-21.csv")

        print(confirmed_df)
        save_to_file("confirmed_df", confirmed_df)
        save_to_file("deaths_df", deaths_df)
        save_to_file("latest_data", latest_data)
        save_to_file("us_medical_data", us_medical_data)
        save_to_file("apple_mobility", apple_mobility)

    return confirmed_df, deaths_df, latest_data, us_medical_data, apple_mobility

def daily_increase(data):
    d = []
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d

def moving_average(data, window_size):
    moving_average = []
    for i in range(len(data)):
        if i + window_size < len(data):
            moving_average.append(np.mean(data[i:i+window_size]))
        else:
            moving_average.append(np.mean(data[i:len(data)]))
    return moving_average

def prep_apple_mobility_data(apple_mobility, country) -> list[int, int]:
    default_mob_data_dict = {}

    try:
        for i, key in enumerate(apple_mobility):
            default_mob_data_dict[key] = []
    except KeyError:
        pass

    index = []
    # Get corresponding data rows for country:
    for i, v in enumerate(apple_mobility.region):
        if v.upper() == country.upper():
            index.append(i)

    datasets = []
    # Add Values to data structure
    try:
        for i in index:
            mob_data_dict = default_mob_data_dict.copy()
            for k, v in mob_data_dict.items():
                mob_data_dict[k] = apple_mobility.loc[i][k]
            datasets.append(mob_data_dict)

    except KeyError:
        pass

    # Values to x and y axis
    x, y = [], []
    i = -1

    datasets_as_xy = []
    prev_dataset = None
    for dataset in datasets:
        i= 0
        temp = []
        temp2 = []
        for k, v in dataset.items():
            i+=1
            temp = []
            if i < 7:
                continue
            else:
                temp.append(k)
                temp.append(v)
            temp2.append(temp)
        datasets_as_xy.append(temp2)

    return datasets_as_xy