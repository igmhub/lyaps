import sys


def userprint(*args, **kwds):
    """Defines an extension of the print function.

    Args:
        *args: arguments passed to print
        **kwargs: keyword arguments passed to print
    """
    print(*args, **kwds)
    sys.stdout.flush()
