"""
Usage: run main.py instead of this
"""

from datetime import datetime, timedelta
import logging
import random
import sys
import time
import uuid
from functools import cache

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr

import refresh_data
from config import OPTIONS_SET_1

plt.style.use("seaborn-poster")
random.seed(19)  # 19 for Covid-19 :P
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
        np.mean(data[i : i + window_size])
        if i + window_size < len(data)
        else np.mean(data[i : len(data)])
        for i in range(len(data))
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
            period=period,
        )
    )

    return [rround(i, 1) for i in lst]


class ColorHandler:
    cmap = matplotlib.cm.get_cmap("Spectral")
    colors = {}

    @staticmethod
    def get_color(name="None"):
        try:
            return ColorHandler.colors[name]
        except KeyError:
            # The following random number is not used in any security context
            color = ColorHandler.cmap(random.random())  # nosec
            ColorHandler.colors[name] = color
            return color


class Data:
    """
    This Class handles all data
    """

    apple_mobility: pd.DataFrame
    ch_lockdown_data: pd.DataFrame
    ch_re_data: pd.DataFrame
    owid_data: pd.DataFrame
    policies: pd.DataFrame

    def __init__(self, country="switzerland", use_cache="True") -> None:
        if "-" in country:
            country = country.replace("-", " ")  # See HELP_COUNTRY
        self.country = country
        self.cache = use_cache
        logger.debug("%s", f"{self.country = }, {self.cache = }")

        (
            Data.apple_mobility,
            Data.ch_lockdown_data,
            Data.ch_re_data,
            Data.owid_data,
            Data.policies,
        ) = refresh_data.get_cached_data()

        self.capitalized_country = self.country[:1].upper() + self.country[1:]
        logger.debug("%s", f"{self.capitalized_country = }")
        self.policies_for_country = Data.policies[
            Data.policies.CountryName == self.capitalized_country
        ]
        self.policies_for_country = Data.policies[
            Data.policies.CountryName == self.capitalized_country
        ]
        Data.ChData.lockdown_data = Data.ch_lockdown_data[
            Data.ch_lockdown_data.Kategorisierung != "Ferien"
        ]

        self.owid_data_for_country = Data.owid_data[
            Data.owid_data.location == self.capitalized_country
        ]
        self.cases_for_country = self.owid_data_for_country.new_cases.fillna(
            0
        ).to_list()
        self.dates_owid = self.owid_data_for_country.date.to_list()

        Data.ch_re_data = Data.ch_re_data.loc[Data.ch_re_data["geoRegion"] == "CH"]
        Data.ChData.re_dates = Data.ch_re_data.date.to_list()
        # self.re_date = self.data.ch_re_data.date.to_list()

        Data.ChData.re_mean = Data.ch_re_data.median_R_mean.to_list()
        Data.ChData.re_high = Data.ch_re_data.median_R_highHPD.to_list()
        Data.ChData.re_low = Data.ch_re_data.median_R_lowHPD.to_list()

        self.re_value_other = Data.owid_data[
            Data.owid_data.location == self.capitalized_country
        ]
        self.re_value_other = self.re_value_other.reproduction_rate.to_list()
        self.re_value_other = [i * 100 for i in self.re_value_other]

        self.traffic_data_for_country = self.apple_mobility[
            self.apple_mobility.region == self.capitalized_country
        ]
        self.datasets_as_xy = [
            [item for index, item in enumerate(row) if index > 6]
            for row in self.traffic_data_for_country.itertuples()
        ]
        self.data_x = self.apple_mobility.columns.to_list()[6:]

        start_date = datetime(2020, 1, 1).date()
        end_date = datetime.today().date()
        delta = end_date - start_date

        self.dates = [(start_date + timedelta(days=i)) for i in range(delta.days)]
        self.dates_as_str = [str(date) for date in self.dates]

        # Depends on datasets_as_xy
        self.avg_traffic_data = moving_average(
            [sum(e) / len(e) for e in zip(*self.datasets_as_xy)]
        )

    class ChData:
        """
        Subclass swiss data
        """

        re_data: pd.DataFrame
        re_mean: list[float]
        re_high: list[float]
        re_low: list[float]

        re_dates: list[str]
        lockdown_data: pd.DataFrame


class AxisHandler:
    """
    Class for returning new axes and its legends
    """

    _axis = {}

    @staticmethod
    def get_axis(name: str = None, ymin=None, ymax=None) -> plt.Axes:
        if not name:
            logger.warn("%s", "No name supplied to get_axis, using UUID...")
            name = uuid.uuid4()
        logger.debug("%s", f"Getting axis for {name = }")
        try:
            return AxisHandler._axis[name]
        except KeyError:
            if not AxisHandler._axis:
                axis = PlotHandler.plot.gca()
                AxisHandler._axis[name] = axis
                return axis

            new_ax = list(AxisHandler._axis.values())[-1].twinx()
            new_ax.set_ylim(ymin, ymax)
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
    _current_country = 0
    countries: int

    def __init__(self, country=[]):
        self.data = []

        PlotHandler.countries = len(country)
        for c in country:
            self.data.append(Data(country=c))

        PlotHandler.plot = plt

        self.ax_handler = AxisHandler()

        self.formatted = False

    def format_plot(self):
        """
        Format the plot as a matplotlib plot
        """
        if not self.formatted:
            PlotHandler.plot.xlabel("Days Since 1/22/2020", size=15)
            PlotHandler.plot.xticks(
                size=10,
                rotation=90,
                ticks=[
                    i * 25
                    for i in range(
                        int(len(self.data[PlotHandler._current_country].dates_owid))
                        % 100
                    )
                ],
            )
            self.formatted = True

    def plot_arbitrary_values(self, value) -> NotImplemented:
        self.format_plot()
        if value not in OPTIONS_SET_1:  # Value needs to be a datarow of the dataset
            return NotImplemented

        axis = AxisHandler.get_axis(f"Arbitrary: {value}")
        for i in range(PlotHandler.countries):
            print(i)
            print(self.data[i])
            axis.plot(
                self.data[i].dates_owid,
                self.data[i].owid_data_for_country[value].to_list(),
                label=f"{self.data[i].capitalized_country} - {value}",
                color=ColorHandler.get_color(f"{value}_{self.data[i].country}"),
            )
        return None

    def plot_cases(self):
        self.format_plot()
        for i in range(PlotHandler.countries):
            axis = AxisHandler.get_axis(
                name="cases",
                ymax=average(sorted(self.data[i].cases_for_country, reverse=True)[:10]),
            )

            axis.plot(
                self.data[i].dates_owid,
                moving_average(self.data[i].cases_for_country),
                color=ColorHandler.get_color(
                    f"cases_{self.data[i].capitalized_country}"
                ),
                label=f"Incidence {self.data[i].capitalized_country}, moving average",
            )
        axis.grid(
            color=ColorHandler.get_color(f"cases_{self.data[i].capitalized_country}"),
            axis="y",
            alpha=0.1,
        )

    def plot_re_data(self):
        if self.data[PlotHandler._current_country].country == "switzerland":
            self._plot_ch_re_data()
        else:
            self._plot_other_re_data()

    def _plot_ch_re_data(self):
        self.format_plot()
        axis = AxisHandler.get_axis(name="ch_re_data", ymin=0, ymax=200)
        axis.plot(
            Data.ChData.re_mean,
            label="Daily Reproduction Value smoothed for Switzerland",
        )
        axis.grid(color="cyan", axis="y", alpha=0.5)

        axis.fill_between(
            Data.ChData.re_dates,
            Data.ChData.re_low,
            Data.ChData.re_mean,
            alpha=0.5,
        )
        axis.fill_between(
            Data.ChData.re_dates,
            Data.ChData.re_high,
            Data.ChData.re_mean,
            alpha=0.5,
        )

    def _plot_other_re_data(self):
        self.format_plot()
        axis = AxisHandler.get_axis(name="other_re_data", ymin=0, ymax=200)
        for i in range(PlotHandler.countries):
            axis.plot(
                self.data[i].re_value_other,
                label=f"Daily Reproduction Value smoothed for {self.data[i].capitalized_country}",
            )

    def show_plot(self, exit_after=False):
        handles, labels = AxisHandler.get_legends()
        PlotHandler.plot.legend(handles, labels, loc="best")
        PlotHandler.plot.show()
        if exit_after:
            sys.exit()

    def plot_stringency_index(self):
        axis = AxisHandler.get_axis("stringency_index", ymin=0, ymax=100)
        for i in range(PlotHandler.countries):
            axis.plot(
                self.data[i].policies_for_country.StringencyIndex.to_list(),
                label=f"Stringency Index for {self.data[i].capitalized_country}",
            )

    def plot_traffic_data(self, detailed=False):
        self.format_plot()
        axis = AxisHandler.get_axis(name="traffic_data", ymin=0, ymax=200)
        logger.debug("%s", "Plotting traffic data")

        for i in range(PlotHandler.countries):
            if detailed:
                for index, data_y in enumerate(self.data[i].datasets_as_xy):
                    data_y = interp_nans(data_y)
                    colors = [
                        ColorHandler.get_color(
                            f"{i}{index}{self.data[i].capitalized_country}"
                        )
                        for _ in range(3)
                    ]
                    match index:
                        case 0:
                            self._plot_traffic_data(
                                axis,
                                self.data[i].data_x,
                                moving_average(data_y),
                                color=colors[0],
                                label=f"Driving (%) [{self.data[i].capitalized_country}]",
                            )
                        case 1:
                            self._plot_traffic_data(
                                axis,
                                self.data[i].data_x,
                                moving_average(data_y),
                                color=colors[1],
                                label=f"Transit (%) [{self.data[i].capitalized_country}]",
                            )
                        case 2:
                            self._plot_traffic_data(
                                axis,
                                self.data[i].data_x,
                                moving_average(data_y),
                                color=colors[2],
                                label=f"Walking (%) [{self.data[i].capitalized_country}]",
                            )
                        case _:
                            self._plot_traffic_data(
                                axis,
                                self.data[i].data_x,
                                moving_average(data_y),
                                color="black",
                                label="unknown datapoint",
                            )
            axis.set_ylabel(
                " Increase of traffic routing requests in %, baseline at 100", size=20
            )
            axis.plot(
                self.data[i].data_x,
                interp_nans(self.data[i].avg_traffic_data),
                color=ColorHandler.get_color(
                    f"mob_data_avg_{self.data[i].capitalized_country}"
                ),
                label=f"Average mobility data [{self.data[i].capitalized_country}]",
            )

    def _plot_traffic_data(self, axis, x, y, **kwargs):
        axis.plot(x, moving_average(y), alpha=0.5, **kwargs)

    def plot_lockdown_data(self):
        self.format_plot()
        logger.debug("%s", "Plotting lockdown data")
        axis = AxisHandler.get_axis(name="lockdown_data")
        axis.set_yticks([])  # this needs no ticks
        axis.plot(
            self.data[PlotHandler._current_country].dates_owid,
            [0 for _ in range(len(self.data[PlotHandler._current_country].dates_owid))],
            alpha=0,
        )
        if self.data[PlotHandler._current_country].country.lower() == "switzerland":
            ausweitungen = []
            lockerungen = []
            dates = []
            ind = 0
            for date in self.data[PlotHandler._current_country].dates_as_str:
                if str(date) in list(Data.ChData.lockdown_data.Datum):
                    i = list(Data.ChData.lockdown_data.Datum).index(date)
                    if Data.ChData.lockdown_data.Kategorisierung[ind] == "Ausweitung":
                        ausweitungen.append(date)
                    elif Data.ChData.lockdown_data.Kategorisierung[ind] == "Lockerung":
                        lockerungen.append(date)
                    dates.append(date)
                    ind += 1
            PlotHandler.plot.vlines(
                x=ausweitungen,
                ymin=0,
                ymax=max(self.data[PlotHandler._current_country].cases_for_country),
                color="red",
                linestyles="dashed",
            )
            PlotHandler.plot.vlines(
                x=lockerungen,
                ymin=0,
                ymax=max(self.data[PlotHandler._current_country].cases_for_country),
                color="green",
                linestyles="dashed",
            )
            for i, x in enumerate(dates):
                t = self.data[
                    PlotHandler._current_country
                ].ch_lockdown_data.Beschreibung.to_list()[i]
                plt.text(
                    x,
                    max(self.data[PlotHandler._current_country].cases_for_country),
                    t,
                    rotation=90,
                    verticalalignment="top",
                )

    def log_pearson_constant(self):
        # Calculate pearson const.
        n_traffic_data = normalize(
            moving_average(self.data[PlotHandler._current_country].avg_traffic_data, 50)
        )
        n_daily_incidence = normalize(
            moving_average(
                self.data[PlotHandler._current_country].cases_for_country, 50
            )
        )
        logger.debug(
            "%s", f"Pearson Constant: {pearsonr(n_traffic_data[2:], n_daily_incidence)}"
        )


if __name__ == "__main__":
    print("Please run main.py instead")
