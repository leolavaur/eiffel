"""Hydra utils."""

from typing import Any, Callable, Type

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig


def instantiate_or_return(obj: Any, typ: Type) -> Any:
    """Instantiate an object if it is not already an instance of the given type.

    Parameters
    ----------
    obj : Any
        Object to instantiate. If it is already an instance of the given type, it is
        returned as is. Otherwise, `obj` is considered as a config object and is
        instantiated.
    typ : Type
        Type of the object to instantiate.

    Returns
    -------
    Any
        Instantiated object of type `typ`.
    """
    if isinstance(obj, typ):
        return obj
    if isinstance(obj, (list, ListConfig, dict, DictConfig)):
        ret = instantiate(obj)
        if isinstance(ret, typ):
            return ret
        raise ValueError(f"Expected {typ}, got {type(ret)}")
    raise ValueError(f"Unexpected object {obj}, was expecting {typ}.")
