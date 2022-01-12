import logging
from config import LOG_LEVEL, LOG_CONFIG

COMMANDS = ["plt", "plot"]
ARGS = ["cs", "cases", "re", "reproduction", "mb", "mobility"]

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

        self.country = args[0]



def main():
    InputHandler(ret_input())


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
    user_in = input()
    return user_in


logger = logging.getLogger(name="__main__")
if __name__ == "__main__":
    logger.setLevel(level=LOG_LEVEL)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter(LOG_CONFIG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    main()
