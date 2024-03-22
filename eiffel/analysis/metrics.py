"""Utilities to process Eiffel metrics."""

import logging
import re
from pathlib import Path
from typing import Any, Iterable, cast, overload

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
    """Calculate the average of a list of dictionaries or a dictionary of dictionaries.

    Parameters
    ----------
    ss : list[dict] | dict[str, dict]
        The input data. It can be either a list of dictionaries or a dictionary of
        dictionaries. In the latter case, the keys are ignored and the values are
        assumed to be the dictionaries to average.

    Returns
    -------
    dict
        The average of the input data.
    """
    lst = ss if isinstance(ss, list) else list(ss.values())

    if len(lst) == 0:
        raise ValueError("lst must not be empty")

    return dict_avg(lst)


@overload
def load_metric(
    path: str, dotpath: str, attr: str = ..., with_malicious: bool = ...
) -> list[float]: ...


@overload
def load_metric(
    path: str, dotpath: Iterable[str], attr: str = ..., with_malicious: bool = ...
) -> dict[str, list[float]]: ...


def load_metric(
    path: str,
    dotpath: Iterable[str] | str = "global.accuracy",
    attr: str = "distributed",
    with_malicious: bool = False,
) -> list[float] | dict[str, list[float]]:
    """Load the metrics from the given path.

    Parameters
    ----------
    path : str
        The path to the results.
    dotpaths : list[str] | str
        The dotpaths to the metrics to load. If a list is provided, the function will
        return a dictionary with the dotpaths as keys and the corresponding metrics as
        values. If a single dotpath is provided, the function will return the metric
        over time (round per round).
    attr : str, optional
        The attribute to load from the results, by default "distributed".
    with_malicious : bool, optional
        Whether to include the malicious participants in the results, by default False.

    Returns
    -------
    list[float] | dict[str, list[float]]
        The metric over time (round per round) if a single dotpath is provided, a
        dictionary with the dotpaths as keys and the corresponding metrics as values
        otherwise.
    """
    res = getattr(Results.from_path(path), attr)
    if not with_malicious:
        res = {k: v for k, v in res.items() if "malicious" not in k}
    avgs = average(res).values()
    if isinstance(dotpath, str):
        return [get_value(d, dotpath) for d in avgs]
    return {dotpath: [get_value(d, dotpath) for d in avgs] for dotpath in dotpath}


def load_df(
    paths: Iterable[str],
    dotpath: str = "global.accuracy",
    attr: str = "distributed",
    with_malicious: bool = False,
) -> pd.DataFrame:
    """Load the selected metrics from the given paths and return them as a DataFrame.

    Each path is expected to be a directory containing the results of a single run. The
    function will load the metrics from each path and return them as a DataFrame, with
    each runs' metrics as rows and the rounds as columns. The DataFrame is indexed by
    the run's distinguishing options. The common options are used as the dataset's name.
    If the path's name does not contain options (as exported by Hydra), the path itself
    is used as the index.

    Parameters
    ----------
    paths : list[str]
        The paths to the results.
    dotpath : str
        The dotpath to the metrics to load.
    attr : str, optional
        The attribute to load from the results, by default "distributed".

    Returns
    -------
    pd.DataFrame
        The metrics over time (round per round) as a DataFrame.
    """
    names = [Path(p).name for p in paths]
    common, distinguishing = conditions(names)
    metrics = [load_metric(p, dotpath, attr, with_malicious) for p in paths]
    if not all(len(m) == len(metrics[0]) for m in metrics):
        raise ValueError("Metrics have different lengths.")
    df = pd.DataFrame(metrics, index=distinguishing)
    df.columns.name = "Round"
    df.Name = common
    return df


def scale_df(df: pd.DataFrame, length: int = 0) -> pd.DataFrame:
    """Scale the DataFrame to the given length.

    Each result length varies depending on the number of rounds. This function scales
    results to the given length by numpy interpolation. If no length is given, the
    maximum length in the DataFrame is used.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to scale.
    length : int, optional
        Length to scale to, by default 0.

    Returns
    -------
    pd.DataFrame
        Scaled DataFrame.
    """
    if length == 0:
        length = max(df.count(axis=1))
    if length <= 0:
        raise ValueError("Length must be positive.")
    ret = pd.DataFrame(index=df.index, columns=range(length))
    x_target = np.arange(0, length)
    for i, row in df.iterrows():
        x = np.arange(0, row.count())
        interp = np.interp(x_target, x * length / row.count(), row[row.notna()])
        assert len(interp) == length
        ret.loc[cast(str, i)] = interp
    return ret.astype(float)


def conditions(names: list[str]) -> tuple[str, list[str]]:
    """Extract the conditions from the given names.

    Parameters
    ----------
    names : list[str]
        The names to extract the conditions from, as exported by Hydra. Conditions are
        of the form `(+)?name=value`, where `name` is the name of the option and `value`
        is the value of the option.

    Returns
    -------
    str
        The common conditions, separated by commas.
    list[str]
        The distinguishing conditions for each name. The order is the same as the order
        of the input names.
    """
    options = [[o.strip("+") for o in n.split(",")] for n in names]
    common = set(options[0])

    for o in options[1:]:
        common &= set(o)

    distinguishing = [",".join(list(set(o) - common)) for o in options]
    return ",".join(common), distinguishing


def search_results(path: str, sort: bool = True, **conditions: str | int) -> list[str]:
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

    if len(metrics) == 0:
        raise ValueError(f"No results found with conditions {conditions}.")

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
            aasrs.append(
                load_metric(path, dotpath=f"{t}.missrate", with_malicious=False)
            )

        aasr = np.mean(aasrs, axis=0)
    else:
        aasr = 1 - np.array(
            load_metric(path, dotpath="global.accuracy", with_malicious=False)
        )

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
            f"{path}: Not all theoritical combinations are covered:"
            f" {n_combinations} possibles, {n_dirs} found."
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
        elif isinstance(ss[0][k], list):
            d[k] = np.mean([s[k] for s in ss], axis=0).tolist()
        else:
            d[k] = np.mean([s[k] for s in ss])
    return d
