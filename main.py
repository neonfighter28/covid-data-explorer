import logging
from typing import Tuple
from sys import argv

from config import LOG_CONFIG, LOG_LEVEL

COMMANDS = ["plt", "plot"]
ARGS = ["cs", "cases", "re", "reproduction", "mb", "mobility"]

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
        self.command = self.input.split()[0]
        self.args = self.input.split()[1:]
        logger.debug("%s", f"COMMAND {self.command}")
        logger.debug("%s", f"ARGS {self.args}")

        try:
            assert self.command in COMMANDS
            for arg in self.args:
                assert arg in ARGS
        except AssertionError:
            logger.fatal("%s", f"Arguments invalid {self.args, self.command}")
            print("Arguments invalid, retry")
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
        where as data is a list of lines to be plotted, defined in the global PLT
        """

        self.arguments = self.arg_handler(args)

    def arg_handler(self, args) -> Argument(str, str):
        """Takes a list of arguments and returns a list of tuples"""
        collector = []
        for index, value in enumerate(args):
            if index % 2 == 0:
                collector.append(Argument(value[index], value[index+1]))
        return collector





def main():
    if argv is False:
        InputHandler(ret_input())
    else:
        stdin = ""
        for i, arg in enumerate(argv):
            if i == 0:
                continue
            stdin = stdin + arg + " "
        InputHandler(stdin)


def ret_input():
    print(
"""
Usage/Syntax:
i   | init
plt | plot
    --country [COUNTRY]
    --timeA   [DD.MM.YY]
    --timeB   [DD.MM.YY]
    --data    [list(DATA)]

data can be
    - re | reproduction -> Plots Reproduction Value
    - cs | cases        -> Cases for country
    - mb | mobility     -> Plots Apple Mobility Data
"""
    )
    user_in = input("IN >>>> ")
    return user_in


logger = logging.getLogger(name="__main__")
if __name__ == "__main__":
    logger.setLevel(level=LOG_LEVEL)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter(LOG_CONFIG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    main()
