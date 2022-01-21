import json
import logging
import sys

import get_data
from config import LOG_CONFIG, LOG_LEVEL

COMMANDS = ["plt", "plot", "set"]  # Valid commands


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


def split_input(input_string) -> tuple[str, list[str]]:
    try:
        command = input_string.split()[0]
    except IndexError:
        print("Handling IndexError")
        main(failure=True)
    args = input_string.split()[1:]
    logger.debug("%s", f"COMMAND {command}")
    logger.debug("%s", f"ARGS {args}")
    try:
        assert command in COMMANDS
    except AssertionError:
        logger.fatal("%s", f"Command invalid {command}")
        print("Command invalid, retry")
        main(failure=True)  # Retry
    return command, args


def unpack_args(args) -> Argument(str, str):
    """Takes a list of arguments and returns a list of tuples"""
    try:
        unpacked_args = [
            Argument(value, args[index+1])
            for index, value in enumerate(args)
            if index % 2 == 0
        ]

        return unpacked_args
    except IndexError as exc:
        raise InputFailure from exc


class InputHandler():
    """
    Handles User Input and dispatches command
    """

    def __init__(self, command, args):
        self.command, self.args = command, args

        match self.command:
            case "plot" | "plt":
                self.plot_graphs(self.args)
            case _:
                raise ValueError

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
                    data_arguments = argument.value
                case "--show" | "-s":
                    self.show_plot = json.loads(
                        argument.value.lower())  # Load string as bool

        self.country = "switzerland" if not self.country else self.country
        self.start_date = None if not self.start_date else self.start_date
        self.end_date = None if not self.end_date else self.end_date
        data_arguments = None if not data_arguments else data_arguments
        logger.debug(
            "%s", f"{self.country = }, {self.start_date = }, {self.end_date = }, {data_arguments = }")
        self.connection = get_data.PlotHandler(country=self.country, )
        data_arguments = data_arguments.split("+")

        for argument in data_arguments:
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
                case "ld" | "lockdown":
                    logger.debug("%s", "Plotting lockdown data")
                    self.connection.plot_lockdown_data()
                case _:
                    logger.warning("%s", f"Data not found for {argument}")

        if self.show_plot:
            logger.debug("%s", "Showing plot...")
            self.connection.show_plot(exit_after=False)
        else:
            logger.debug("%s", "Not showing plot")


def main(failure=False):
    if not sys.argv[1:] or failure:
        command, args = ret_input()
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


def ret_input() -> str:
    try:
        print(
            """
    Usage/Syntax:
    plt | plot
        --country   | -c    [COUNTRY]       | Default: switzerland
        --startdate | -sd   [DD.MM.YY]      | Default: None
        --enddate   | -ed   [DD.MM.YY]      | Default: None
        --show      | -s    [bool]          | Default: True Whether to show the plot
        --data      | -d    [ARG+ARG+...]    | Default: None
            - re | reproduction -> Plots Reproduction Value
            - cs | cases        -> Cases for country
            - mb | mobility     -> Plots Apple Mobility Data
            - ld | lockdown     -> Plots Lockdown Data (CH exclusive)
===================================================================================="""
        )
        user_in = input("covid-sets >>>> ")
    except KeyboardInterrupt:
        print("\nCtrl+C pressed, exiting...")
        sys.exit(1)
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
