"""Timing utilities."""

import logging
import time
import functools

logger = logging.getLogger(__name__)


def timeit(func):
    """Time a function."""

    @functools.wraps(func)
    def timed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__qualname__} ({args, kwargs}) took {elapsed:.3f}s")
        return result

    return timed
