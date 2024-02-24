"""Utilities to process Eiffel metrics."""

import logging
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from eiffel.core.results import Results

logger = logging.getLogger(__name__)

# Result analysis
# ---------------


def average(ss: list[dict] | dict[str, dict]) -> dict:
    """Compute the mean, metric per metric and for each round, of a list of Series.

    Parameters
    ----------
    ss : List[Series] | dict[str, Series]
        Series to average. If a dict is given, the values are considered to be the
        Series, and the keys are ignored.

    Returns
    -------
    Series
        The mean of the Series.
    """
    lst = ss if isinstance(ss, list) else list(ss.values())

    if len(lst) == 0:
        raise ValueError("lst must not be empty")

    return dict_avg(lst)


def load_metric(
    path: str,
    dotpath: str,
    attr: str = "distributed",
    with_malicious: bool = False,
) -> list[float]:
    """Load the metrics from the given path."""
    res = getattr(Results.from_path(path), attr)
    if not with_malicious:
        res = {k: v for k, v in res.items() if "malicious" not in k}
    return [get_value(d, dotpath) for d in average(res).values()]


def search_results(path: str, sort: bool = True, **conditions: str) -> list[str]:
    """
    Search corresponding results from the given path.

    Parameters
    ----------
    path : str
        The path to the results.
    conditions : dict[str, str]
        The conditions to search the results. The keys are the names of hydra options
        that have been set, and the values are the values of the options. For example,
        if conditions is set to `{"distribution": "10-0"}`, then the fonctions will load
        all experiments that have been run with `distribution=10-0`. Values are regex
        patterns, enclosed in `^[...]$` afterwards. For example, `{"distribution":
        "10-.*"}`will match all experiments that have been run with `distribution`
        starting with `10-`, while `{"scenario": "continuous-10"}` will match only
        "continuous-10" and not "continuous-100".

    Returns
    -------
    list[str]
        The paths to the results matching the conditions.
    """
    for _, v in conditions.items():
        v = str(v)
    p = Path(path)
    if not p.is_dir():
        raise ValueError(f"{path} is not a directory.")

    metrics: list[str] = []

    for d in p.iterdir():
        if not d.is_dir():
            continue
        options = {
            k.strip("+"): v for k, v in [p.split("=") for p in d.name.split(",")]
        }
        if all(
            re.match(rf"^{v}$", options.get(k, "")) is not None
            for k, v in conditions.items()
        ):
            metrics.append(d.as_posix())

    return sorted(metrics) if sort else metrics


def load_asr(
    path: str, target: list[str] = [], reference: list[float] = []
) -> list[float]:
    r"""Compute the ASR from the given path.

    The ASR (Attack Success Rate) measures the impact of an attack on the legitimate
    participants of the federation. Its definition depends on the attack:

    - Foruntargeted attacks, the ASR is the mean missclassification rate of the benign
      participants, or $1 - \text{accuracy}$.
    - For targeted attacks, the ASR is the mean missrate obtained by benign participants
      in the targeted label(s).

    The above is refered to as the AASR (Absolute ASR). The ASR can also be computed
    relatively to a reference scenario (typically the same run without malicious
    participants), in which case it is scaled between this reference and 1. This is
    refered to as the RASR (Relative ASR).

    Parameters
    ----------
    path : str
        The path to the results.
    target : list[str], optional
        The targeted labels, by default []. No target (empty list) is interpreted as an
        untargeted attack. The labels must be provided as they are in the dataset; the
        selection is case-sensitive.
    reference : str, optional
        The reference scenario, by default None. If None, the AASR is returned.

    Returns
    -------
    list[float]
        The RASR over time (round per round) if a reference is given, the AASR
        otherwise.
    """
    if len(target) > 0:
        aasrs = []
        for t in target:
            aasrs.append(load_metric(path, dotpath=f"{t}.missrate"))

        aasr = np.mean(aasrs, axis=0)
    else:
        aasr = 1 - np.array(load_metric(path, dotpath="global.accuracy"))

    if reference:
        ref = np.array(reference)
        rasr = ((np.maximum(ref, aasr) - ref) / (1 - ref)).astype(float)
        if any(x for x in rasr if x > 1):
            pass
        return rasr.tolist()
    return aasr.astype(float).tolist()


# Multirun analysis
# -----------------


def choices(path: str) -> dict[str, list[str]]:
    """Return the available choices for each condition."""
    if not Path(path).is_dir():
        raise ValueError(f"{path} is not a directory.")

    dirs = [d for d in Path(path).iterdir() if d.is_dir()]

    choices: dict[str, list[str]] = {}

    for d in dirs:
        options = {
            k.strip("+"): v for k, v in [p.split("=") for p in d.name.split(",")]
        }
        for k, v in options.items():
            if k not in choices:
                choices[k] = []
            if v not in choices[k]:
                choices[k].append(v)

    n_combinations = int(np.prod([len(v) for v in choices.values()]))
    if (n_dirs := len(dirs)) < n_combinations:
        logger.warning(
            f"Not all theoritical combinations are covered: {n_combinations} possibles,"
            f" {n_dirs} found."
        )
    return choices


def display_choices(d: dict[str, list[str]]) -> None:
    """Display the choices."""
    display(
        HTML(
            "<style>table td, table th, table tr {text-align:left !important;}</style>"
            + "<table><tr><th>Key</th><th>Values</th></tr>"
            + "".join(
                f"<tr><td>{k}</td><td>{', '.join(v)}</td></tr>" for k, v in d.items()
            )
            + "</table>"
        )
    )


# Utilities
# ---------


def get_value(d: dict, dotpath: str) -> Any:
    """Get the value of a dict using a dotpath."""
    for key in dotpath.split("."):
        if key not in d:
            raise ValueError(f"Key {key} not found in {d}.")
        d = d[key]
    return d


def dict_avg(ss: list[dict]) -> dict:
    """Recursively average a list of dicts."""
    d = {}
    for k in ss[0].keys():
        if isinstance(ss[0][k], dict):
            d[k] = dict_avg([s[k] for s in ss])
        else:
            d[k] = np.mean([s[k] for s in ss])
    return d
