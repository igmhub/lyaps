"""This module provides general functions.

These are:
    - userprint
See the respective docstrings for more details
"""

import sys


def userprint(*args, **kwds):
    """Defines an extension of the print function.

    Args:
        *args: arguments passed to print
        **kwargs: keyword arguments passed to print
    """
    print(*args, **kwds)
    sys.stdout.flush()
