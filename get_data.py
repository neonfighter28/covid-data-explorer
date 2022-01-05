"""
Usage: py get-data.py [country='switzerland'] [no_cache=False]

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
import logging
import pickle
import time
import asyncio
from sys import argv, setrecursionlimit
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import requests
plt.style.use('seaborn-poster')
timestart = time.perf_counter()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s %(lineno)d: - %(message)s')
fh.setFormatter(fh_formatter)
logger.addHandler(fh)
"""
setrecursionlimit(10000)

@cache
def round(*args, **kwargs):
    return round(*args, **kwargs)
"""


class Helper:
    def flatten(arr):
        # helper method for flattening the data,
        # so it can be displayed on a bar graph
        return [i[0] for i in arr.tolist()]

    def average(num):
        return sum(num) / len(num)

    def normalize(data):
        return [i/max(interp_nans(data)) for i in data]

    def save_to_file(name, data):
        with open(f"assets/{name}.dat", "wb") as file:
            logger.debug("%s", f"Saving to file {file.name}")
            pickle.dump(data, file)

    def read_from_file(name):
        with open(f"assets/{name}.dat", "rb") as file:
            logger.debug("%s", f"Reading from file {file.name}")
            return pickle.load(file)


def parse_args(country="switzerland", cache="False"):
    country = argv[1] if len(argv) > 1 else "switzerland"
    cache = bool(argv[2]) if len(argv) > 2 else True
    logger.debug("%s", f"Arguments {country = }, {cache = }")
    return country, cache

def get_data(cache):
    try:
        if not cache:
            raise FileNotFoundError
        logger.debug("%s", "reading from cache...")
        confirmed_df = Helper.read_from_file("confirmed_df")
        apple_mobility = Helper.read_from_file("apple_mobility")
        ch_lockdown_data = Helper.read_from_file("ch_lockdown_data")

    except FileNotFoundError:
        # SOURCE
        # https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30120-1/fulltext
        logger.debug("%s", "Pulling data...")

        confirmed_df = pd.read_csv(
            'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/ \
            master/csse_covid_19_data/csse_covid_19_time_series/ \
            time_series_covid19_confirmed_global.csv')
        logger.info("%s", "Pulling mobility data...")
        apple_mobility = pd.read_csv(get_current_apple_url())
        logger.info("%s", "Pulling lockdown data...")
        ch_lockdown_data = pd.read_csv(
            "https://raw.githubusercontent.com/statistikZH/ \
            covid19zeitmarker/master/covid19zeitmarker.csv"
        )
        # https://opendata.swiss/en/dataset/covid-19-schweiz/resource/92c50632-ecfa-4234-9c36-ea6121bb68ca
        ch_re_data = "https://www.covid19.admin.ch/api/data/20220104-ph1baz76/sources/COVID19Re_geoRegion.csv"

        logger.debug("%s", "------Loading is completed ------")
        Helper.save_to_file("confirmed_df", confirmed_df)
        Helper.save_to_file("apple_mobility", apple_mobility)
        Helper.save_to_file("ch_lockdown_data", ch_lockdown_data)
        logger.debug("%s", "saved to cache!")

    return confirmed_df, apple_mobility, ch_lockdown_data


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
        [
                (k, v) for index, (k, v) in enumerate(dataset.items())
                if index > 5
            ]
        for dataset in datasets
    ]


def interp_nans(x: [float], left=None, right=None, period=None) -> [float]:
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
    return [round(i, 1) for i in lst]

def get_current_apple_url():
    response = requests.get(
        "https://covid19-static.cdn-apple.com/covid19-mobility-data \
        /current/v3/index.json").json()
    return ("https://covid19-static.cdn-apple.com/"
            + response['basePath']
            + response['regions']['en-us']['csvPath'])

if __name__ == "__main__":
    country, cache = parse_args()


class Main:
    def __init__(self, country, cache):
        self.__country = country
        self.__cache = cache
        self.timestart = time.perf_counter()
        self.confirmed_df, self.apple_mobility, self.ch_lockdown_data = get_data(cache)
        self.read_lockdown_data()

        self.datasets_as_xy = prep_apple_mobility_data(self.apple_mobility, self.country)

        self.plt = plt
        self.ax = plt.gca()
        self.ax2 = self.ax.twinx()

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
        try:
            for key in confirmed_df:
                def_data[key] = []

            for k, v in def_data.items():
                def_data[k] = confirmed_df.loc[index][k]
        except KeyError:
            pass
        return def_data

    def get_r_value(self):
        pass

cls = Main(country, cache)

confirmed_df, apple_mobility, ch_lockdown_data = get_data(cache)
ch_lockdown_data = cls.ch_lockdown_data

cols = confirmed_df.keys()
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
dates = confirmed.keys()

# Get Covid data for country
index = cls._get_index_of_datarow()
def_data = cls._build_def_data()

# Convert total cases each day to daily incidence and calculate R
r_value = [0 for i in range(7)]
lst = [0 for i in range(5)]
k_minus_1 = 0
new_data_def = def_data.copy()
dd = []
for index, (k, v) in enumerate(def_data.items()):
    if index < 5:
        new_data_def[k] = v
        dd.append(0)
    else:
        new_data_def[k] = v - k_minus_1
        lst.append(v-k_minus_1)

        dd.append(v)
        k_minus_1 = v

dd2 = [0, 0]
y_m_1 = 0
for i in dd:
    try:
        dd2.append(i/y_m_1*100)
    except ZeroDivisionError:
        dd2.append(0)
    y_m_1 = i


print(dd)
print(moving_average(lst))


# Create Data Structures
datasets_as_xy = prep_apple_mobility_data(apple_mobility, country)

# adjusted_dates = adjusted_dates.reshape(1, -1)[0]
plt.figure(figsize=(16, 10))

ax = plt.gca()
ax2 = ax.twinx()

# Get average of all lists
data_rows = []

for z, value in tqdm(enumerate(datasets_as_xy)):
    data_x = [i[0] for i in value]
    data_y = interp_nans([i[1] for i in value])
    data_rows.append(moving_average(data_y, 7))

    match z:
        case 0:
            ax.plot(data_x, moving_average(data_y, 7),
                    color="#FE9402", label="Driving", alpha=0.5)
        case 1:
            ax.plot(data_x, moving_average(data_y, 7),
                    color="#FE2D55", label="Transit", alpha=0.5)
        case 2:
            ax.plot(data_x, moving_average(data_y, 7),
                    color="#AF51DE", label="Walking", alpha=0.5)
        case _:
            ax.plot(data_x, moving_average(data_y, 7), color="black")

avg_traffic_data = moving_average([sum(e)/len(e) for e in zip(*data_rows)], 7)
ax.plot(data_x, avg_traffic_data, color="green", label="Average mobility data")
print(r_value)
ax.plot(data_x, moving_average(interp_nans(dd2)), color="red")

ax.set_ylim(ymax=200)
ax2.plot(
    data_x[2:],
    moving_average(lst, 7),
    color="blue",
    label=f"Incidence {country}, moving average"
)
ax2.set_ylim(ymax=Helper.average(sorted(lst, reverse=True)[:2]))
plt.xlabel('Days Since 1/22/2020', size=15)
ax.set_ylabel(
    ' Increase of traffic routing requests in %, baseline at 100', size=20)
plt.xticks(size=10, rotation=180, ticks=[
    i*50 for i in range(len(data_x) % 50)])
plt.yticks(size=10)
plt.grid()
# print(avg_traffic_data[2:], moving_average(lst,7))

# Calculate pearson const.
n_traffic_data = Helper.normalize(moving_average(avg_traffic_data, 50))
n_daily_incidence = Helper.normalize(moving_average(lst, 50))

logging.info(
    "%s", f"Pearson Constant: {pearsonr(n_traffic_data[2:], n_daily_incidence)}")

if country.lower() == "switzerland":
    for index, date in enumerate(data_x):

        if str(date) in list(ch_lockdown_data.Datum):
            print(list(ch_lockdown_data.Datum).index(date))

            print(True)

    plt.axvspan(
        63,  # 16.03.20
        119,
        color='red', alpha=0.5
    )

    plt.axvspan(279, 402, color="red", alpha=0.5)
ax.legend()
ax2.legend()
# plt.legend([f"Traffic requests for {country}"], loc=9)
print(time.perf_counter() - timestart)
# exit(1)
plt.show()


"""
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)

days_in_future = 40
future_forecast = np.array(
    [i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forecast[:-10]

start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forecast_dates = []
for i in range(len(future_forecast)):
    future_forecast_dates.append(
        (start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

# plt.plot(future_forecast_dates, future_forecast)
# plt.show()

# slightly modify the data to fit the model better
# (regression models cannot pick the pattern)
days_to_skip = 500
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = \
    train_test_split(
        days_since_1_22[days_to_skip:],
        world_cases[days_to_skip:],
        test_size=0.08,
        shuffle=False
    )
"""