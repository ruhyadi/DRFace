"""
Timer class for timing code execution.
"""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import time
from datetime import datetime, timezone


class Timer:
    """Timer class for timing code execution."""
    def __init__(self) -> None:
        """Initialize timer class."""
        pass

    def now(self, utc: bool = False, T: bool = True) -> str:
        """
        Get current date and time.

        Args:
            utc (bool): If True, return UTC time, else return local time.
            T (bool): If True, return time with T, else return time without T.

        Returns:
            str: Current date and time in %Y-%m-%dT%H:%M:%S format.

        Examples:
            >>> timer = Timer()
            >>> timer.now()
            2022-12-11T14:54:22
            >>> timer.now(utc=True)
            2022-12-11T14:54:22
            >>> timer.now(T=False)
            2022-12-11 14:54:22

        """
        if utc:
            if T:
                return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if T:
            return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def now_iso(self, utc: bool = False) -> str:
        """
        Get current timestamp in ISO format.

        Args:
            utc (bool): If True, return UTC time, else return local time.

        Returns:
            str: Current timestamp in ISO format. %Y-%m-%dT%H:%M:%S.%fZ

        Examples:
            >>> timer = Timer()
            >>> timer.now_iso()
            2022-12-11T14:54:22.000000Z
            >>> timer.now_iso(utc=True)
            2022-12-11T14:54:22.000000Z

        """
        if utc:
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")


if __name__ == "__main__":
    """Debugging."""
    timer = Timer()
    print(timer.now())
    print(timer.now(utc=True))
    print(timer.now(T=False))
    print(timer.now_iso())
    print(timer.now_iso(utc=True))