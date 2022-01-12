"""
Usage: py get-data.py

Goals:
Plot containing 4 lines:
1. Traffic Data of country -> walking | DONE
2. Traffic Data of country -> driving | DONE
3. Traffic Data of country -> transit | DONE
4. Traffic data normalized            | DONE
5. COVID Data of country              | DONE
6. R Value

6. New dataset on whether there was a lockdown or not

    Dates of lockdown:
    - 16.03.20-11.05.20,
    - 18.10.20-17.02.21,

    -> different levels of lockdowns?

Traffic Data needs to be normalized, to account for weekends/days off | DONE

-> Predict COVID Data for the next 2 weeks
-> Predict traffic Data for the next 2 weeks
"""
import datetime
import json
import logging
import time
from functools import cache
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from tqdm import tqdm

import refresh_data
from config import CACHE, COUNTRY, LOG_CONFIG, DATES_RE, LOG_LEVEL

plt.style.use('seaborn-poster')
timestart = time.perf_counter()

logger = logging.getLogger(__name__)
logger.setLevel(level=LOG_LEVEL)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter(LOG_CONFIG)
fh.setFormatter(fh_formatter)
logger.addHandler(fh)


@cache
def rround(*args, **kwargs):
    return round(*args, **kwargs)


class Helper:
    def flatten(arr):
        # helper method for flattening the data,
        # so it can be displayed on a bar graph
        return [i[0] for i in arr.tolist()]

    def average(num):
        # sum of an array divided by its length
        return sum(num) / len(num)

    def normalize(data):
        return [i/max(data) for i in data]

    def daily_increase(data):
        return [data if i == 0 else data[i] - data[i-1] for i in range(len(data))]


def moving_average(data, window_size=7):
    return [
        np.mean(data[i:i+window_size])
        if i + window_size < len(data)
        else np.mean(data[i:len(data)])
        for i in range(len(data))
    ]


def prep_apple_mobility_data(apple_mobility, country) -> list[int, int]:
    try:
        default_mob_data_dict = {}
        for key in apple_mobility:
            default_mob_data_dict[key] = []
    except KeyError:
        pass

    # Get corresponding data rows for country:
    indexes_of_datarows = [i for i, v in enumerate(
        apple_mobility.region) if v.upper() == country.upper()]

    datasets = []
    # Add Values to data structure
    for i in indexes_of_datarows:
        mob_data_dict = default_mob_data_dict.copy()
        for k, _ in mob_data_dict.items():
            mob_data_dict[k] = apple_mobility.loc[i][k]
        datasets.append(mob_data_dict)

    return [
        [(k, v) for index, (k, v) in enumerate(dataset.items()) if index > 5]
        for dataset in datasets
    ]


def interp_nans(x: [float], left=None, right=None, period=None) -> [float]:
    # Very resource intensive
    lst = list(
        np.interp(
            x=list(range(len(x))),
            xp=[i for i, yi in enumerate(x) if np.isfinite(yi)],
            fp=[yi for i, yi in enumerate(x) if np.isfinite(yi)],
            left=left,
            right=right,
            period=period
        )
    )

    return [rround(i, 1) for i in lst]


class Main:
    def __init__(self):
        self.__country = COUNTRY
        self.__cache = CACHE
        self.confirmed_df, self.apple_mobility, self.ch_lockdown_data, self.ch_re_data = refresh_data.get_cached_data()
        self.read_lockdown_data()

        self.datasets_as_xy = prep_apple_mobility_data(
            self.apple_mobility, self.country)

        self.plt = plt
        self.ax = plt.gca()
        self.ax2 = self.ax.twinx()
        self.data_x = self.get_x_data()

    @property
    def country(self):
        return self.__country

    @property
    def cache(self):
        return self.__cache

    def read_lockdown_data(self):
        self.ch_lockdown_data.drop('Link', axis=1, inplace=True)
        self.ch_lockdown_data = self.ch_lockdown_data[
            self.ch_lockdown_data.Kategorisierung != "Ferien"
        ]

    def _get_index_of_datarow(self):
        try:
            for index, value in enumerate(self.confirmed_df.loc):
                if self.confirmed_df.loc[index]["Country/Region"].upper() \
                        == self.country.upper():
                    break
        except KeyError:
            pass
        return index

    def _build_def_data(self):
        def_data = {}
        index = self._get_index_of_datarow()
        try:
            for key in self.confirmed_df:
                def_data[key] = []

            for k, v in def_data.items():
                def_data[k] = self.confirmed_df.loc[index][k]
        except KeyError:
            pass
        return def_data

    def get_r_value(self):
        self.ch_re_data = self.ch_re_data.loc[self.ch_re_data["geoRegion"] == "CH"]
        self.re_date = self.ch_re_data.date.to_list()
        re_mean = self.ch_re_data.median_R_mean.to_list()
        re_high = self.ch_re_data.median_R_highHPD.to_list()
        re_low = self.ch_re_data.median_R_lowHPD.to_list()
        self.re_mean = self.add_nans_to_start_of_list(re_mean)
        self.re_high = self.add_nans_to_start_of_list(re_high)
        self.re_low = self.add_nans_to_start_of_list(re_low)

    def add_nans_to_start_of_list(self, re, nan=DATES_RE):
        # The first 26 days are not included in this dataset
        x = [np.nan for i in range(nan)]
        for i in re:
            x.append(rround(i*100, 1))
        return x

    def get_lst(self):
        lst = [0 for i in range(3)]
        k_minus_1 = 0
        for index, (k, v) in enumerate(self._build_def_data().items()):
            if index < 5:
                lst.append(0)
            else:
                lst.append(v-k_minus_1)
                k_minus_1 = v
        return lst

    def format_plot(self):
        self.plt.grid()
        self.ax.set_ylabel(
            ' Increase of traffic routing requests in %, baseline at 100', size=20)
        self.ax.set_ylim(ymax=200)
        self.plt.xlabel('Days Since 1/22/2020', size=15)

    def get_x_data(self):
        for value in self.datasets_as_xy:
            return (list(zip(*value))[0])

    def plot(self):
        self.format_plot()
        lst = self.get_lst()
        logger.debug("%s", "Plotting traffic data")
        # Get average of all lists
        data_rows = []
        for z, value in enumerate(self.datasets_as_xy):
            data_y = interp_nans(list(zip(*value))[1])
            data_rows.append(moving_average(data_y, 7))

            match z:
                case 0:
                    self.plot_traffic_data(self.data_x, moving_average(data_y),
                                           color="#FE9402", label="Driving")
                case 1:
                    self.plot_traffic_data(self.data_x, data_y,
                                           color="#FE2D55", label="Transit")
                case 2:
                    self.plot_traffic_data(self.data_x, data_y,
                                           color="#AF51DE", label="Walking")
                case _:
                    self.plot_traffic_data(self.data_x, data_y,
                                           color="black")

        avg_traffic_data = moving_average(
            [sum(e)/len(e) for e in zip(*data_rows)], 7)
        self.ax.plot(self.data_x, avg_traffic_data, color="green",
                     label="Average mobility data")

        self.ax2.plot(
            self.data_x[2:],
            moving_average(lst),
            color="blue",
            label=f"Incidence {COUNTRY}, moving average"
        )
        self.ax2.set_ylim(ymax=Helper.average(sorted(lst, reverse=True)[:2]))
        self.lst = lst
        self.plt.xticks(size=10, rotation=180, ticks=[
            i*50 for i in range(len(self.data_x) % 50)])
        self.plt.yticks(size=10)
        self.plt.grid()
        self.get_r_value()
        self.ax.plot(self.re_mean)

        self.ax.fill_between(self.data_x, self.re_low, self.re_mean, alpha=0.5)
        self.ax.fill_between(self.data_x, self.re_high,
                             self.re_mean, alpha=0.5)
        self.plot_lockdown_data()
        self.ax.legend()
        self.ax2.legend()
        logger.info(print(time.perf_counter() - timestart))
        # Calculate pearson const.
        self.log_pearson_constant(avg_traffic_data=avg_traffic_data)
        self.plt.show()

    def plot_traffic_data(self, x, y, **kwargs):
        self.ax.plot(x, moving_average(y),
                     alpha=0.5, **kwargs)

    def plot_lockdown_data(self):
        logger.debug("%s", "Plotting lockdown data")
        if self.country.lower() == "switzerland":
            for index, date in enumerate(self.data_x):
                if str(date) in list(self.ch_lockdown_data.Datum):
                    # print(list(self.ch_lockdown_data.Datum).index(date))
                    # print(True)
                    pass
            self.plt.axvspan(
                63,  # 16.03.20
                119,
                color='red', alpha=0.5
            )

            self.plt.axvspan(279, 402, color="red", alpha=0.5)

    def log_pearson_constant(self, avg_traffic_data):
        # Calculate pearson const.
        n_traffic_data = Helper.normalize(moving_average(avg_traffic_data, 50))
        n_daily_incidence = Helper.normalize(moving_average(self.lst, 50))
        logger.debug(
            "%s", f"Pearson Constant: {pearsonr(n_traffic_data[2:], n_daily_incidence)}")


if __name__ == "__main__":
    timestart = time.perf_counter()

    cls = Main()

    cls.plot()
