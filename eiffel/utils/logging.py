"""Logging module."""

import logging
from enum import Enum
from functools import wraps

from flwr.common.logger import logger as flwr_logger

# flwr_logger.removeHandler(flwr_logger.handlers[0])


class VerbLevel(Enum):
    """Verbosity level.

    This class defines the verbosity level for Eiffel clients, which is then passed
    directly to Keras' model training API.

    From https://keras.io/api/models/model_training_apis/#fit-method:

        verbose: 'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 =
        one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with
        ParameterServerStrategy. Note that the progress bar is not particularly useful
        when logged to a file, so verbose=2 is recommended when not running
        interactively (eg, in a production environment).
    """

    AUTO = "auto"
    SILENT = 0
    INPLACE = 1
    VERBOSE = 2

    def __str__(self):
        """Return the string representation of the verbosity level."""
        return str(self.value)


class ColoredFormatter(logging.Formatter):
    """Formatter for colored log output.

    This class colors the output of the log messages according to their level. It also
    provides way to display more details about the loggers, such as the name of the
    file and the line number.
    """

    details = False

    shallow = "\x1b[38;5;8m"
    bold = "\x1b[1m"
    reset = "\x1b[0m"
    cyan = "\x1b[38;5;14m"

    COLORS = {
        logging.DEBUG: f"{bold}\x1b[38;5;8m",  # grey
        logging.INFO: "\x1b[38;5;14m",  # cyan
        logging.WARNING: "\x1b[38;5;11m",  # yellow
        logging.ERROR: "\x1b[38;5;9m",  # red
        logging.CRITICAL: f"{bold}\x1b[38;5;9m",  # bold red
    }

    def __init__(self, *args, verbose_loggers: bool | str | list = False, **kwargs):
        """Initialize the formatter.

        Parameters
        ----------
        verbose_loggers : bool | str | list, optional
            Controls the loggers to pass to verbose mode. If False, no logger will be
            verbose. If True, all loggers will be verbose. If a string, only the
            specified logger will be verbose. If a list of strings, each logger in the
            list will be verbose. By default False.
        """
        super().__init__(*args, **kwargs)

        if isinstance(verbose_loggers, bool) and verbose_loggers:
            self.details = True
        elif isinstance(verbose_loggers, str) and verbose_loggers.lower() in __name__:
            self.details = True
        elif isinstance(verbose_loggers, list) and any(
            x.lower() in __name__ for x in verbose_loggers
        ):
            self.details = True

    def format(self, record):
        """Format the log message.

        Refer to the logging.Formatter class for more information.
        """
        log_fmt = (
            f"{self.shallow}%(asctime)s{self.reset}"
            + f" [{self.COLORS[record.levelno]}%(levelname)s{self.reset}]"
            + f" [{self.shallow}%(name)s{self.reset}]"
            + (" :%(lineno)d" if self.details else "")
            + " > %(message)s"
        )
        formatter = logging.Formatter(log_fmt)
        message = formatter.format(record)
        if not self.details:
            package_name = record.name.split(".")[0]
            return message.replace(record.name, package_name)
        return message


def logged(function):
    """Log function calls automatically via a decorator.

    Parameters
    ----------
    function : callable
        Function to log.

    Returns
    -------
    function : callable
        Wrapped function.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        logging.getLogger(__name__).debug(
            "Called `%s` with args: %s",
            function.__name__,
            {**dict(zip(function.__code__.co_varnames, args)), **kwargs},
        )

        result = function(*args, **kwargs)
        return result

    return wrapper
