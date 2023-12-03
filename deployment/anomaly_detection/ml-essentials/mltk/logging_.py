import logging
import sys
from datetime import datetime
from typing import *

__all__ = ['configure_logging', 'clear_logging', 'print_with_time']

LOG_FORMAT: str = '[%(asctime)-15s] %(levelname)-8s %(message)s'


def configure_logging(level: str = 'INFO',
                      propagate: bool = False,
                      output_stream=sys.stdout,
                      fmt: str = LOG_FORMAT
                      ) -> None:
    """
    Configure the Python logging facility for the `mltk` package.

    Args:
        level: The log level.
        propagate: Whether or not to propagate log messages to parent loggers?
        output_stream: The output stream, where to print the logs.
        fmt: The log message format.
    """
    logger = logging.getLogger('mltk')

    logger.setLevel(level)
    logger.propagate = propagate

    # initialize the handler
    logger.handlers.clear()
    handler = logging.StreamHandler(output_stream)
    handler.setFormatter(logging.Formatter(fmt=fmt))
    logger.addHandler(handler)


def clear_logging() -> None:
    """Clear all logging configs for the `mltk` package."""
    logger = logging.getLogger('mltk')
    logger.propagate = True
    logger.setLevel(logging.NOTSET)
    logger.handlers.clear()


configure_logging()  # configure logging by default settings


def print_with_time(message: str,
                    print_func: Callable[[str], None] = print):
    """
    Print a line of message with time time in front of it.

    Args:
        message: The message to be printed.
        print_func: The print function.
    """
    from .formatting import format_as_asctime
    dt_str = format_as_asctime(datetime.now())
    message = f'[{dt_str}] {message}'
    print_func(message)
