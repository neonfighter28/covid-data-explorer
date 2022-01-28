#!/bin/python

from dataclasses import dataclass
import json
import logging
import sys
from help import HELP_ARBITRARY, HELP_DATA, HELP_COUNTRY, USAGE

import get_data
from config import LOG_CONFIG, LOG_LEVEL

COMMANDS = ["plt", "plot", "help", "h"]  # Valid commands


class InputFailure(BaseException):
    """Raised if input sequence needs to be repeated"""


@dataclass(frozen=True, order=True)
class Argument:
    name: str
    value: str


def split_input(input_string) -> tuple[str, list[str]]:
    try:
        command = input_string.split()[0]
    except IndexError:
        print("Handling IndexError")
        main(failure=True)
    args = input_string.split()[1:]
    logger.debug("%s", f"COMMAND {command}")
    logger.debug("%s", f"ARGS {args}")
    if command not in COMMANDS:
        logger.fatal("%s", f"Command invalid {command}")
        print("Command invalid, retry")
        main(failure=True)  # Retry
    return command, args


def unpack_args(args) -> Argument(str, str):
    """Takes a list of arguments and returns a list of tuples"""
    try:
        unpacked_args = [
            Argument(value, args[index + 1])
            for index, value in enumerate(args)
            if index % 2 == 0
        ]

        return unpacked_args
    except IndexError as exc:
        raise InputFailure from exc


class InputHandler:
    """
    Handles User Input and dispatches command
    """

    def __init__(self, command, args):
        self.command, self.args = command, args

        match self.command:
            case "plot" | "plt":
                self.plot_graphs(self.args)
            case "help" | "h":
                if self.args:
                    match args[0]:
                        case "data":
                            print(HELP_DATA)
                        case "country":
                            print(HELP_COUNTRY)
                        case "arbitrary":
                            print(HELP_ARBITRARY)
                main(failure=True, show=False)

            case _:
                raise ValueError(f"Bad command {self.command}")

    def plot_graphs(self, args):
        """
        According to the usage, the passed-through arguments are supposed to be in the following order:
        [COUNTRY] [TIMEA] [TIMEB] [data]
        whereas data is a list of lines to be plotted

        This function calls to the lower-level module get_data and adds lines to plot
        """

        self.start_date = None
        self.end_date = None
        self.country = None
        self.arguments = unpack_args(args)
        self.show_plot = True
        data_arguments = None

        self.is_crosscountry = False
        self.countries: list = []

        for argument in self.arguments:
            match argument.name:
                case "--country" | "-c":
                    self.country = argument.value
                case "--startdate" | "-sd":
                    self.start_date = argument.value
                    raise NotImplementedError
                case "--enddate" | "-ed":
                    self.end_date = argument.value
                    raise NotImplementedError
                case "--cross-country" | "-cc":
                    self.is_crosscountry = True
                    self.countries = argument.value.split("+")
                case "--data" | "-d":
                    data_arguments = argument.value
                case "--show" | "-s":
                    self.show_plot = json.loads(
                        argument.value.lower()
                    )  # Load string safely as bool

        self.country = ["switzerland"] if not self.country else [self.country]
        self.country = self.countries if self.is_crosscountry else self.country
        self.start_date = None if not self.start_date else self.start_date
        self.end_date = None if not self.end_date else self.end_date
        data_arguments = None if not data_arguments else data_arguments
        if not data_arguments:  # Need data to plot
            raise InputFailure
        logger.debug(
            "%s",
            f"{self.country = }, {self.start_date = }, {self.end_date = }, {data_arguments = }",
        )
        self.connection = get_data.PlotHandler(country=self.country)

        for argument in data_arguments.split(
            "+"
        ):  # Data arguments are supposed to be separated by a "+"
            logger.debug("%s", argument)
            match argument.lower():
                case "cs" | "cases":
                    logger.debug("%s", "Plotting Cases...")
                    self.connection.plot_cases()
                case "re" | "reproduction":
                    logger.debug("%s", "Plotting R_e values")
                    self.connection.plot_re_data()
                case "mb" | "mobility":
                    logger.debug("%s", "Plotting average mobility data")
                    self.connection.plot_traffic_data()
                case "mbd" | "mobilitydetailed":
                    logger.debug("%s", "Plotting detailed mobility data")
                    self.connection.plot_traffic_data(detailed=True)
                case "ld" | "lockdown":
                    logger.debug("%s", "Plotting lockdown data")
                    self.connection.plot_lockdown_data()
                case "str" | "stringency":
                    logger.debug("%s", "Plotting Stringency Index")
                    self.connection.plot_stringency_index()
                case _:
                    logger.debug("%s", f"Plotting values for {argument}")
                    if (
                        self.connection.plot_arbitrary_values(argument)
                        == NotImplemented
                    ):
                        logger.warning("%s", f"Data not found for {argument}")

        if not self.connection.ax_handler._axis:
            raise InputFailure(f"Bad -d argument {argument}")
        if self.show_plot:
            logger.debug("%s", "Showing plot...")
            self.connection.show_plot(exit_after=False)
        else:
            logger.debug("%s", "Not showing plot")

        del self.connection

        # raise InputFailure  # Repeat Input sequence until Ctrl+C is pressed or exit is entered


def main(failure=False, show=True):
    if not sys.argv[1:] or failure:
        command, args = ret_input(show)
        InputHandler(command, args)
    else:
        stdin = ""
        for i, arg in enumerate(sys.argv):
            if i == 0:
                continue
            stdin = stdin + arg + " "
        print("Handling Input from STDIN")
        command, args = split_input(stdin)
        InputHandler(command, args)


def ret_input(show=True) -> str:
    try:
        if show:
            print(USAGE)
        user_in = input("covid-sets >>>> ")
        if user_in == "exit":
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("\nCtrl+C pressed, exiting...")
        sys.exit()
    command, args = split_input(user_in)
    return command, args


def rec_until_keyboard_interrupt():
    # InputFailure is raised if any Input is malformed/arguments do not exist
    try:
        main()
    except InputFailure:
        rec_until_keyboard_interrupt()


logger = logging.getLogger(name="__main__")
if __name__ == "__main__":
    logger.setLevel(level=LOG_LEVEL)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter(LOG_CONFIG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    rec_until_keyboard_interrupt()
