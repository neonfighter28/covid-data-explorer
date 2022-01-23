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

Use stringency dataset from OWID
"""

# TODO: Add Lockdown markers to plot

from datetime import datetime, timedelta
import logging
import sys
import time
from functools import cache

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr

import refresh_data
from config import DATES_RE, LOG_CONFIG, LOG_LEVEL

plt.style.use('seaborn-poster')
timestart = time.perf_counter()

logger = logging.getLogger("__main__")


class CovidPredException(BaseException):
    """Base Exception for this module"""


class CountryNotFound(CovidPredException):
    """Country wasn't found"""


@cache
def rround(*args, **kwargs):
    return round(*args, **kwargs)


def flatten(arr):
    # helper method for flattening the data,
    # so it can be displayed on a bar graph
    return [i[0] for i in arr.tolist()]


def flatten_list(arr):
    return [item for sublist in arr for item in sublist]


def average(num):
    # sum of an array divided by its length
    return sum(num) / len(num)


def normalize(data):
    return [i / max(data) for i in data]


def daily_increase(data):
    return [data if i == 0 else data[i] - data[i - 1] for i in range(len(data))]


def moving_average(data, window_size=7):
    return [
        np.mean(data[i:i + window_size])
        if i + window_size < len(data)
        else np.mean(data[i:len(data)])
        for i in range(len(data))
    ]


def prep_apple_mobility_data(apple_mobility, country) -> list[int, int]:

    # contains only dates
    empty_mob_data_dict = {key: None for key in apple_mobility}

    # Get corresponding data rows for country:
    datarows_for_country = [i for i, v in enumerate(
        apple_mobility.region) if v.upper() == country.upper()]

    datasets = []
    # Add Values to data structure
    for datarow in datarows_for_country:
        mob_data_dict = empty_mob_data_dict.copy()
        for k, _ in mob_data_dict.items():
            mob_data_dict[k] = apple_mobility.loc[datarow][k]
        datasets.append(mob_data_dict)

    return [
        [(k, v) for index, (k, v) in enumerate(dataset.items()) if index > 5]
        for dataset in datasets
    ]


def interp_nans(x: list[float], left=None, right=None, period=None) -> list[float]:
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


def add_nans_to_start_of_list(re, nan=DATES_RE):
    # The first 26 days are not included in this dataset
    x = [np.nan for _ in range(nan)]
    for i in re:
        x.append(rround(i * 100, 1))
    return x


class Data:
    """
    This Class handles all data
    """

    def __init__(self, country="switzerland", use_cache="True") -> None:
        self.country = country
        self.cache = use_cache
        logger.debug("%s", f"{self.country = }, {self.cache = }")

        self.confirmed_df, \
            self.apple_mobility, \
            self.ch_lockdown_data, \
            self.ch_re_data, \
            self.owid_data = refresh_data.get_cached_data()
        self.confirmed_daily = None
        self.datasets_as_xy = None
        self.avg_traffic_data = None
        self.re_value_other = None

        self.re_mean = None
        self.re_low = None
        self.re_high = None

        self._build_data()

    def _build_data(self):
        self.capitalized_country = self.get_capitalized_country()
        self.set_re_values_ch()
        self.set_re_value_other()
        self.read_lockdown_data()
        self.confirmed_daily = self.get_confirmed_daily()
        self.datasets_as_xy = prep_apple_mobility_data(
            self.apple_mobility, self.country)
        self.data_x = self.get_x_data()

        start_date = datetime(2020, 1, 1).date()
        end_date = datetime.today().date()
        delta = end_date - start_date

        self.dates = [(start_date + timedelta(days=i)) for i in range(delta.days)]
        self.dates_as_str = [str(date) for date in self.dates]

        # Depends on datasets_as_xy
        self.avg_traffic_data = self.get_avg_traffic_data()

    def set_re_values_ch(self):
        self.ch_re_data = self.ch_re_data.loc[self.ch_re_data["geoRegion"] == "CH"]
        # self.re_date = self.data.ch_re_data.date.to_list()
        re_mean = self.ch_re_data.median_R_mean.to_list()
        re_high = self.ch_re_data.median_R_highHPD.to_list()
        re_low = self.ch_re_data.median_R_lowHPD.to_list()
        self.re_mean = add_nans_to_start_of_list(re_mean)
        self.re_high = add_nans_to_start_of_list(re_high)
        self.re_low = add_nans_to_start_of_list(re_low)

    def set_re_value_other(self):
        self.re_value_other = self.owid_data[self.owid_data.location == self.capitalized_country]
        self.re_value_other = self.re_value_other.reproduction_rate.to_list()
        self.re_value_other = [i * 100 for i in self.re_value_other]

    def get_capitalized_country(self):
        return self.country[:1].upper() + self.country[1:]

    def read_lockdown_data(self):
        self.ch_lockdown_data.drop('Link', axis=1, inplace=True)
        self.ch_lockdown_data = self.ch_lockdown_data[
            self.ch_lockdown_data.Kategorisierung != "Ferien"
        ]

    def get_x_data(self):
        for value in self.datasets_as_xy:
            return list(zip(*value))[0]

    def get_confirmed_daily(self):
        confirmed_daily = [0 for _ in range(3)]
        k_minus_1 = 0
        for index, (_, value) in enumerate(self._build_def_data().items()):
            if index < 2:
                continue  # First two values are text
            confirmed_daily.append(value - k_minus_1)
            k_minus_1 = value
        return confirmed_daily

    def _build_def_data(self):
        index = self._get_index_of_datarow()
        try:
            def_data = {
                key: self.confirmed_df.loc[index][key] for key in self.confirmed_df}
        except KeyError:
            pass
        return def_data

    def _get_index_of_datarow(self):
        # Might throw KeyError?
        try:
            for index, _ in enumerate(self.confirmed_df.loc):
                if self.confirmed_df.loc[index]["Country/Region"].upper() \
                        == self.country.upper():
                    return index
        except KeyError as exc:
            raise CountryNotFound(f"Country '{self.country}' was not found") from exc

    def get_avg_traffic_data(self):
        # Get average of all lists
        data_rows = [moving_average(interp_nans(list(zip(*row))[1]))
                     for row in self.datasets_as_xy]

        return moving_average([sum(e) / len(e) for e in zip(*data_rows)])


class AxisHandler:
    """
    Class for returning new axes and its legends
    """
    _axis = {}

    @staticmethod
    def get_axis(name: str = None) -> plt.Axes:
        logger.debug("%s", f"Getting axis for {name = }")
        try:
            return AxisHandler._axis[name]
        except KeyError:
            if not AxisHandler._axis:
                axis = PlotHandler.plot.gca()
                AxisHandler._axis[name] = axis
                return axis

            new_ax = list(AxisHandler._axis.values())[-1].twinx()
            AxisHandler._axis[name] = new_ax
            return new_ax

    @staticmethod
    def get_legends() -> tuple[list[matplotlib.lines.Line2D], list[str]]:
        handles, labels = [], []
        for axis in AxisHandler._axis.values():
            for handle, label in zip(*axis.get_legend_handles_labels()):
                handles.append(handle)
                labels.append(label)
        return handles, labels



class PlotHandler:
    """
    Handles Plotting
    """
    plot = None

    def __init__(self, **kwargs):
        self.data = Data(**kwargs)

        if PlotHandler.plot:
            del PlotHandler.plot
        PlotHandler.plot = plt

        self.data.data_x = self.data.data_x
        self.ax_handler = AxisHandler()

        self.formatted = False

    def format_plot(self):
        """
        Format the plot as a matplotlib plot
        """
        if not self.formatted:
            PlotHandler.plot.xlabel('Days Since 1/22/2020', size=15)
            PlotHandler.plot.xticks(size=10, rotation=90, ticks=[
                i * 50 for i in range(int(len(self.data.data_x) / 2) % 50)])
            self.formatted = True

    def _format_axis(self, axis, case):
        match case:
            case "re_data":
                axis.set_ylim(ymin=0, ymax=200)

    def plot_cases(self):
        self.format_plot()

        axis = AxisHandler.get_axis(name="cases")
        axis.set_ylim(ymax=average(sorted(self.data.confirmed_daily, reverse=True)[:2]))
        axis.plot(
            self.data.data_x,
            moving_average(self.data.confirmed_daily),
            color="blue",
            label=f"Incidence {self.data.country}, moving average",
        )
        axis.grid(color="blue", axis="y", alpha=0.1)

    def plot_re_data(self):
        if self.data.country == "switzerland":
            self._plot_ch_re_data()
        else:
            self._plot_other_re_data()

    def _plot_ch_re_data(self):
        self.format_plot()
        axis = AxisHandler.get_axis(name="ch_re_data")
        self._format_axis(axis, "re_data")
        axis.plot(self.data.re_mean, label='Daily Reproduction Value smoothed for Switzerland')
        axis.grid(color="cyan", axis="y", alpha=0.5)

        axis.fill_between(self.data.data_x, self.data.re_low,
                          self.data.re_mean, alpha=0.5)
        axis.fill_between(self.data.data_x, self.data.re_high,
                          self.data.re_mean, alpha=0.5)

    def _plot_other_re_data(self):
        self.format_plot()
        axis = AxisHandler.get_axis(name="other_re_data")
        axis.set_ylim(ymin=0, ymax=200)
        axis.plot(self.data.re_value_other,
                  label=f"Daily Reproduction Value smoothed for {self.data.capitalized_country}")

    def show_plot(self, exit_after=False):
        handles, labels = AxisHandler.get_legends()
        PlotHandler.plot.legend(handles, labels, loc="best")
        PlotHandler.plot.show()
        if exit_after:
            sys.exit(0)

    def plot_traffic_data(self):
        self.format_plot()
        axis = AxisHandler.get_axis(name="traffic_data")
        PlotHandler.plot.xticks(size=10, rotation=90, ticks=[
            i * 25 for i in range(int(len(self.data.data_x) / 2) % 100)])
        logger.debug("%s", "Plotting traffic data")
        # Get average of all lists
        data_rows = []
        for index, value in enumerate(self.data.datasets_as_xy):
            data_y = interp_nans(list(zip(*value))[1])
            data_rows.append(moving_average(data_y, 7))

            match index:
                case 0:
                    self._plot_traffic_data(axis, self.data.data_x, moving_average(data_y),
                                            color="#FE9402", label="Driving (%)")
                case 1:
                    self._plot_traffic_data(axis, self.data.data_x, data_y,
                                            color="#FE2D55", label="Transit (%)")
                case 2:
                    self._plot_traffic_data(axis, self.data.data_x, data_y,
                                            color="#AF51DE", label="Walking (%)")
                case _:
                    self._plot_traffic_data(axis, self.data.data_x, data_y,
                                            color="black", label="unknown datapoint")
        axis.set_ylabel(
            ' Increase of traffic routing requests in %, baseline at 100', size=20)
        axis.set_ylim(ymax=200)

        axis.plot(self.data.data_x, self.data.avg_traffic_data, color="green",
                  label="Average mobility data")

    def _plot_traffic_data(self, axis, x, y, **kwargs):
        axis.plot(x, moving_average(y),
                  alpha=0.5, **kwargs)

    def plot_lockdown_data(self):
        self.format_plot()
        logger.debug("%s", "Plotting lockdown data")
        axis = AxisHandler.get_axis(name="lockdown_data")
        axis.set_yticks([])  # this needs no ticks
        axis.plot(self.data.data_x, [0 for _ in range(len(self.data.data_x))], alpha=0)
        if self.data.country.lower() == "switzerland":
            ausweitungen = []
            lockerungen = []
            dates = []
            ind = 0
            for date in self.data.dates_as_str:
                if str(date) in list(self.data.ch_lockdown_data.Datum):
                    i = list(self.data.ch_lockdown_data.Datum).index(date)
                    if self.data.ch_lockdown_data.Kategorisierung[ind] == "Ausweitung":
                        ausweitungen.append(date)
                    elif self.data.ch_lockdown_data.Kategorisierung[ind] == "Lockerung":
                        lockerungen.append(date)
                    dates.append(date)
                    ind += 1
            PlotHandler.plot.vlines(
                x=ausweitungen,
                ymin=0,
                ymax=max(self.data.confirmed_daily),
                color="red",
                linestyles="dashed"
            )
            PlotHandler.plot.vlines(
                x=lockerungen,
                ymin=0,
                ymax=max(self.data.confirmed_daily),
                color="green",
                linestyles="dashed"
            )
            for i, x in enumerate(dates):
                t = self.data.ch_lockdown_data.Beschreibung.to_list()[i]
                plt.text(x, max(self.data.confirmed_daily), t, rotation=90, verticalalignment="top")

    def log_pearson_constant(self):
        # Calculate pearson const.
        n_traffic_data = normalize(
            moving_average(self.data.avg_traffic_data, 50))
        n_daily_incidence = normalize(
            moving_average(self.data.confirmed_daily, 50))
        logger.debug(
            "%s", f"Pearson Constant: {pearsonr(n_traffic_data[2:], n_daily_incidence)}")


if __name__ == "__main__":
    logger.setLevel(level=LOG_LEVEL)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter(LOG_CONFIG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    timestart = time.perf_counter()

    cls = PlotHandler()

    cls.plot_cases()
    cls.plot_re_data()
    cls.plot_traffic_data()
    cls.show_plot()
