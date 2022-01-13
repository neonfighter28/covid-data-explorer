import json
import logging
from sys import argv
from typing import Tuple

import get_data
from config import LOG_CONFIG, LOG_LEVEL

COMMANDS = ["plt", "plot"]

class InputFailure(BaseException):
    """Raised if input sequence needs to be repeated"""

class Argument:
    """
    Argument class containing the attributes name and value
    """
    def __init__(self, name, value) -> None:
        self.__name = name
        self.__value = value

    @property
    def name(self):
        return self.__name

    @property
    def value(self):
        return self.__value

    def __hash__(self) -> int:
        return hash(self.__name) + hash(self.__value)

    def __repr__(self) -> str:
        return f"Argument: {self.__name}, Value: {self.__value}"


class InputHandler():
    """
    Handles User Input and dispatches command
    """
    def __init__(self, user_in):
        self.input = user_in
        try:
            self.command = self.input.split()[0]
        except IndexError:
            self.__init__(ret_input())

        self.args = self.input.split()[1:]
        logger.debug("%s", f"COMMAND {self.command}")
        logger.debug("%s", f"ARGS {self.args}")

        try:
            assert self.command in COMMANDS
        except AssertionError:
            logger.fatal("%s", f"Command invalid {self.command}")
            print("Command invalid, retry")
            self.__init__(ret_input())

        match self.command:
            case "plot" | "plt":
                self.plot_graphs(self.args)
            case _:
                raise ValueError

    def plot_graphs(self, args):
        """
        According to the usage, the passed-through arguments are supposed to be in the following order:
        [COUNTRY] [TIMEA] [TIMEB] [data]
        where as data is a list of lines to be plotted
        """

        self.start_date = None
        self.end_date = None
        self.country = None
        self.arguments = self.arg_handler(args)
        self.show_plot = True
        self.data = None

        for argument in self.arguments:
            logger.debug("%s", f"{argument = }")
            match argument.name:
                case "--country" | "-c":
                    self.country = argument.value
                case "--startdate" | "-sd":
                    self.start_date = argument.value
                    raise NotImplementedError
                case "--enddate" | "-ed":
                    self.end_date = argument.value
                    raise NotImplementedError
                case "--data" | "-d":
                    self.data = argument.value
                case "--show" | "-s":
                    self.show_plot = json.loads(argument.value.lower()) # Assert it is a boolean

        self.country = "switzerland" if not self.country else self.country
        self.start_date = None if not self.start_date else self.start_date
        self.end_date = None if not self.end_date else self.end_date
        self.data = None if not self.data else self.data
        logger.debug("%s", f"{self.country = }, {self.start_date = }, {self.end_date = }, {self.data = }")
        self.connection = get_data.Main()
        self.data = self.data.split("+")
        print(self.data)
        for argument in self.data:
            logger.debug("%s", argument)
            match argument:
                case "cs" | "cases":
                    logger.debug("%s", "Plotting Cases...")
                    self.connection.plot_cases()
                case "re" | "reproduction":
                    logger.debug("%s", "Plotting R_e values")
                    self.connection.plot_re_data()
                case "mb" | "mobility":
                    logger.debug("%s", "Plotting mobility data")
                    self.connection.plot_traffic_data()
                case _:
                    logger.warning("%s", f"Data not found for {argument}")

        if self.show_plot:
            logger.debug("%s", "Showing plot...")
            self.connection.show_plot()
        else:
            logger.debug("%s", "Not showing plot")

    def arg_handler(self, args) -> Argument(str, str):
        """Takes a list of arguments and returns a list of tuples"""
        collector = []
        for index, value in enumerate(args):
            if index % 2 == 0:
                collector.append(Argument(value, args[index+1]))
        print(f"{collector = }")
        return collector


def main():
    if not argv:
        InputHandler(ret_input())
    else:
        stdin = ""
        for i, arg in enumerate(argv):
            if i == 0:
                continue
            stdin = stdin + arg + " "
        InputHandler(stdin)


def ret_input() -> str:
    print(
"""
Usage/Syntax:
i   | init
plt | plot
    --country   | -c    [COUNTRY]       | Default: switzerland
    --startdate | -sd   [DD.MM.YY]      | Default: None
    --enddate   | -ed   [DD.MM.YY]      | Default: None
    --show      | -s    [bool]          | Default: True Whether to show the plot
    --data      | -d    [ARG+ARG+...]    | Default: None
        - re | reproduction -> Plots Reproduction Value
        - cs | cases        -> Cases for country
        - mb | mobility     -> Plots Apple Mobility Data
"""
    )
    user_in = input("covid-sets >>>> ")
    return user_in


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
