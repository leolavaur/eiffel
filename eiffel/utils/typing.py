"""Typing definitions for Eiffel."""

from typing import TypedDict

from numpy.typing import NDArray as NDArray

EiffelCID = str
ConfigDict = dict[str, str | int | float | bool | bytes]


class MetricsDict(dict[str, float]):
    """A dictionary with Eiffel-specific typing."""

    _authorized_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "missrate",
        "fallout",
        "loss",
    ]

    def __init__(self, *args, **kwargs):
        """Initialize a MetricsDict."""
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if k not in self._authorized_keys:
                raise ValueError(f"Invalid key '{k}' in MetricsDict")
            if not isinstance(v, float):
                raise ValueError(f"Invalid value '{v}' in MetricsDict, must be float")

    def __setitem__(self, __key: str, __value: float) -> None:
        """Set an item in the dictionary."""
        if __key not in self._authorized_keys:
            raise ValueError(f"Invalid key '{__key}' in MetricsDict")
        if not isinstance(__value, float):
            raise ValueError(f"Invalid value '{__value}' in MetricsDict, must be float")
        return super().__setitem__(__key, __value)
