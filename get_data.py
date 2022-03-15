"""
Usage: run main.py instead of this
"""

from datetime import datetime, timedelta
import logging
import random
import sys
import uuid
from functools import cache

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import refresh_data
from config import OPTIONS_SET_1

plt.style.use("seaborn-poster")
random.seed(24)

logger = logging.getLogger("__main__")


class CovidPredException(BaseException):
    """Base Exception for this module"""


class CountryNotFound(CovidPredException):
    """Country wasn't found"""


# Cache wrapper for round function rounding results to improve speed
# at the cost of memory
@cache
def rround(*args, **kwargs):
    return round(*args, **kwargs)


def average(num):
    # sum of an array divided by its length
    return sum(num) / len(num)


def moving_average(data, window_size=7):
    return [
        np.mean(data[i : i + window_size]) if i + window_size < len(data) else np.mean(data[i : len(data)])
        for i in range(len(data))
    ]


def interp_nans(data: list[float], left=None, right=None, period=None) -> list[float]:
    # Very resource intensive
    lst = list(
        np.interp(
            x=list(range(len(data))),
            xp=[i for i, yi in enumerate(data) if np.isfinite(yi)],
            fp=[yi for yi in data if np.isfinite(yi)],
            left=left,
            right=right,
            period=period,
        )
    )

    return [rround(i, 1) for i in lst]


class ColorHandler:
    """class ColorHandler handles all colors for this module."""

    cmap_strong = matplotlib.cm.get_cmap("hsv")
    cmap_light = matplotlib.cm.get_cmap("Pastel1")
    colors = {}

    @staticmethod
    def get_color(name: str = "None", strong: bool = True):
        """Returns a color for each call and indexes them to dictionaries so they can be recalled later on

        Args:
            name (str, optional): Name for the color. Defaults to "None".
            strong (bool, optional): Whether it should be a strong color. Defaults to True.

        Returns:
            tuple: Tuple containing the color values
        """
        logger.debug("%s", f"{name = }, {strong = }")

        try:
            logger.debug("%s", f"Returning existing color {name}, {ColorHandler.colors[name]}")
            return ColorHandler.colors[name]
        except KeyError:
            if strong:
                color = ColorHandler.cmap_strong(random.random())
            else:
                color = ColorHandler.cmap_light(random.random())
            logger.debug("%s", f"Creating new {color = }, {strong = }")
            ColorHandler.colors[name] = color
            return ColorHandler.colors[name]


class Data:
    """
    This Class handles all data
    """

    delta = datetime.today().date() - datetime(2020, 1, 1).date()

    dates = [(datetime(2020, 1, 1).date() + timedelta(days=i)) for i in range(delta.days)]
    dates_as_str = [str(date) for date in dates]

    class ChData:
        """
        Subclass for swiss data
        """

        re_data: pd.DataFrame
        re_mean: list[float]
        re_high: list[float]
        re_low: list[float]

        re_dates: list[str]
        lockdown_data: pd.DataFrame

    (apple_mobility, ch_lockdown_data, ch_re_data, owid_data, policies) = refresh_data.get_cached_data()

    ChData.lockdown_data = ch_lockdown_data[ch_lockdown_data.Kategorisierung != "Ferien"]

    ch_re_data = ch_re_data.loc[ch_re_data["geoRegion"] == "CH"]
    ChData.re_dates = ch_re_data.date.to_list()

    ChData.re_mean = ch_re_data.median_R_mean.to_list()
    ChData.re_high = ch_re_data.median_R_highHPD.to_list()
    ChData.re_low = ch_re_data.median_R_lowHPD.to_list()

    def __init__(self, country: str = "switzerland") -> None:
        if "-" in country:
            country = country.replace("-", " ")  # See HELP_COUNTRY
        self.country = country
        logger.debug("%s", f"{self.country = }")

        self.capitalized_country = self.country[:1].upper() + self.country[1:]
        logger.debug("%s", f"{self.capitalized_country = }")
        self.policies_for_country = Data.policies[Data.policies.CountryName == self.capitalized_country]
        self.policies_for_country = self.policies_for_country[Data.policies.Jurisdiction == "NAT_TOTAL"]

        self.owid_data_for_country = Data.owid_data[Data.owid_data.location == self.capitalized_country]
        self.cases_for_country = self.owid_data_for_country.new_cases.fillna(0).to_list()
        self.dates_owid = self.owid_data_for_country.date.to_list()

        self.re_value_other = Data.owid_data[Data.owid_data.location == self.capitalized_country]
        self.re_value_other = self.re_value_other.reproduction_rate.to_list()

        self.traffic_data_for_country = self.apple_mobility[self.apple_mobility.region == self.capitalized_country]
        self.datasets_as_xy = [
            [item for index, item in enumerate(row) if index > 6] for row in self.traffic_data_for_country.itertuples()
        ]
        self.data_x = self.apple_mobility.columns.to_list()[6:]

        # Depends on datasets_as_xy
        self.avg_traffic_data = moving_average([sum(e) / len(e) for e in zip(*self.datasets_as_xy)])


class AxisHandler:
    """
    Class for returning new axes and its legends
    """

    _axes = {}

    @staticmethod
    def get_axis(name: str = None, ymin: int = None, ymax: int = None) -> plt.Axes:
        if not name:
            logger.warning("%s", "No name supplied to get_axis, using UUID...")
            name = uuid.uuid4()
        logger.debug("%s", f"Getting axis for {name = }")
        try:
            return AxisHandler._axes[name]
        except KeyError:
            # If no axis exists yet
            if not AxisHandler._axes:
                AxisHandler._axes[name] = PlotHandler.plot.gca()
                return AxisHandler._axes[name]

            new_ax = list(AxisHandler._axes.values())[-1].twinx()
            new_ax.set_ylim(ymin, ymax)
            AxisHandler._axes[name] = new_ax
            return AxisHandler._axes[name]

    @staticmethod
    def get_legends() -> tuple[list[matplotlib.lines.Line2D], list[str]]:
        handles, labels = [], []
        for axis in AxisHandler._axes.values():
            for handle, label in zip(*axis.get_legend_handles_labels()):
                handles += [handle]
                labels += [label]
        return handles, labels


class PlotHandler:
    """
    Handles Plotting
    """

    plot = None
    _current = 0
    countries: int = 0

    def __init__(self, country=None):
        self.data = [Data(country=c) for c in country]

        PlotHandler.countries = len(country)

        PlotHandler.plot = plt

        self.ax_handler = AxisHandler()

        self.formatted = False

        PlotHandler.plot.xlabel("Days Since 1/22/2020", size=15)
        PlotHandler.plot.xticks(size=10, rotation=90, ticks=[i * 50 for i in range(int(len(Data.dates_as_str)) % 100)])

    def plot_arbitrary_values(self, value: str) -> NotImplemented:
        if value not in OPTIONS_SET_1:  # Value needs to be a datarow of the dataset
            return NotImplemented

        axis = AxisHandler.get_axis(f"Arbitrary: {value}")
        for i in range(PlotHandler.countries):
            axis.plot(
                self.data[i].dates_owid,
                self.data[i].owid_data_for_country[value].to_list(),
                label=f"{self.data[i].capitalized_country} - {value}",
                color=ColorHandler.get_color(f"{value}_{self.data[i].country}"),
            )
        return None

    def plot_cases(self):
        for i in range(PlotHandler.countries):
            axis = AxisHandler.get_axis(
                name="cases",
                # Avg of top 10 values
                ymax=average(sorted(self.data[i].cases_for_country, reverse=True)[:10]),
            )
            axis.plot(
                self.data[i].dates_owid,
                moving_average(self.data[i].cases_for_country),
                color=ColorHandler.get_color(f"cases_{self.data[i].capitalized_country}{i}"),
                label=f"Incidence {self.data[i].capitalized_country}, moving average",
            )
        axis.grid(color=ColorHandler.get_color(f"cases_{self.data[i].capitalized_country}"), axis="y", alpha=0.1)

    def plot_re_data(self):
        axis = AxisHandler.get_axis(name="re_data", ymin=0, ymax=2)
        for i in range(PlotHandler.countries):
            if self.data[i].country == "switzerland":
                axis.plot(Data.ChData.re_mean, label="Daily Reproduction Value smoothed for Switzerland")
                axis.grid(color="grey", axis="y", alpha=0.5)

                axis.fill_between(Data.ChData.re_dates, Data.ChData.re_low, Data.ChData.re_mean, alpha=0.5)
                axis.fill_between(Data.ChData.re_dates, Data.ChData.re_high, Data.ChData.re_mean, alpha=0.5)
            else:
                axis.plot(
                    self.data[i].re_value_other,
                    label=f"Daily Reproduction Value smoothed for {self.data[i].capitalized_country}",
                )

    @staticmethod
    def show_plot(exit_after=False):
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
                color=ColorHandler.get_color(f"Stringency_{self.data[i].capitalized_country}"),
            )

    def plot_traffic_data(self, detailed=False):
        axis = AxisHandler.get_axis(name="traffic_data", ymin=0, ymax=200)
        logger.debug("%s", "Plotting traffic data")

        for i in range(PlotHandler.countries):
            if detailed:
                for index, data_y in enumerate(self.data[i].datasets_as_xy):
                    data_y = interp_nans(data_y)
                    colors = [
                        ColorHandler.get_color(f"{i}{index}{self.data[i].capitalized_country}", strong=False)
                        for _ in range(3)
                    ]
                    match index:
                        case 0:
                            self._plot_traffic_data(
                                axis,
                                self.data[i].data_x,
                                moving_average(data_y),
                                color=colors[index],
                                label=f"Driving (%) [{self.data[i].capitalized_country}]",
                            )
                        case 1:
                            self._plot_traffic_data(
                                axis,
                                self.data[i].data_x,
                                moving_average(data_y),
                                color=colors[index],
                                label=f"Transit (%) [{self.data[i].capitalized_country}]",
                            )
                        case 2:
                            self._plot_traffic_data(
                                axis,
                                self.data[i].data_x,
                                moving_average(data_y),
                                color=colors[index],
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
            axis.set_ylabel(" Increase of traffic routing requests in %, baseline at 100", size=20)
            axis.plot(
                self.data[i].data_x,
                interp_nans(self.data[i].avg_traffic_data),
                color=ColorHandler.get_color(f"mob_data_avg_{self.data[i].capitalized_country}"),
                label=f"Average mobility data [{self.data[i].capitalized_country}]",
            )

    @staticmethod
    def _plot_traffic_data(axis, x_ax, y_ax, **kwargs):
        axis.plot(x_ax, moving_average(y_ax), alpha=0.5, **kwargs)

    def plot_lockdown_data(self):
        logger.debug("%s", "Plotting lockdown data")
        axis = AxisHandler.get_axis(name="lockdown_data")
        axis.plot(
            self.data[PlotHandler._current].dates_owid,
            [0 for _ in range(len(self.data[PlotHandler._current].dates_owid))],
            alpha=0,
        )
        if self.data[PlotHandler._current].country.lower() == "switzerland":
            ausweitungen = []
            lockerungen = []
            dates = []
            ind = 0
            for date in self.data[PlotHandler._current].dates_as_str:
                if str(date) in list(Data.ChData.lockdown_data.Datum):
                    if Data.ChData.lockdown_data.Kategorisierung[ind] == "Ausweitung":
                        ausweitungen += [date]
                    elif Data.ChData.lockdown_data.Kategorisierung[ind] == "Lockerung":
                        lockerungen += [date]
                    dates += [date]
                    ind += 1
            PlotHandler.plot.vlines(
                x=ausweitungen,
                ymin=0,
                ymax=max(self.data[PlotHandler._current].cases_for_country),
                color="red",
                linestyles="dashed",
            )
            PlotHandler.plot.vlines(
                x=lockerungen,
                ymin=0,
                ymax=max(self.data[PlotHandler._current].cases_for_country),
                color="green",
                linestyles="dashed",
            )
            for i, date in enumerate(dates):
                description = self.data[PlotHandler._current].ch_lockdown_data.Beschreibung.to_list()[i]
                plt.text(
                    date,
                    max(self.data[PlotHandler._current].cases_for_country),
                    description,
                    rotation=90,
                    verticalalignment="top",
                )
        else:
            logger.warning("No lockdown data available for this country")


if __name__ == "__main__":
    print("Please run main.py instead")
