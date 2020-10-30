"""Utilities for zest."""

import contextlib
import logging

from . import settings


def configure_logging(verbose: bool = False) -> logging.Handler:
    """Configure logging and return the log handler.

    This function is useful in scripts when logging should be set up
    with a basic configuration.

    Parameters
    ----------
    verbose : bool, optional (default=False)
        If ``True`` set the log level to DEBUG, else set it to INFO.

    Returns
    -------
    logging.Handler
        The log handler set up by this function.
    """
    # unset the log level from root (defaults to WARNING)
    logging.root.setLevel(logging.NOTSET)

    # set up the log handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))

    # attach the log handler to root
    logging.root.addHandler(handler)

    return handler
