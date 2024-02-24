"""Plotting and data analysis utilities."""

import json
import re
from collections import Counter
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

MARKERS = ["o", "D", "v", "*", "+", "^", "p", ".", "P", "<", ">", "X"]
LINESTYLES = ["-", "--", "-.", ":"]
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def load_metric(path: str, metric="accuracy", with_malicious=False) -> list[float]:
    """
    Load the metrics from the given path.
    """
    p = Path(path)
    j = json.loads(p.read_text())
    client_metrics: dict[str, list[float]] = {}
    if not with_malicious:
        j = {k: v for k, v in j.items() if "malicious" not in k}
    for k, v in j.items():
        client_metrics[k] = [m["global"][metric] for m in v.values()]
    metrics_zip = zip(*client_metrics.values())
    return [sum(m) / len(m) for m in metrics_zip]


def load_metrics(
    *paths: str, metric="accuracy", with_malicious=False
) -> dict[str, tuple[list[float], list[float]]]:
    """
    Load the metrics from the given paths.
    """
    metrics = {}
    for path in paths:
        p = Path(path)
        f_dist = (p / "distributed.json").as_posix()
        f_fit = (p / "fit.json").as_posix()
        metrics[p.name] = (
            load_metric(f_fit, metric, with_malicious),
            load_metric(f_dist, metric, with_malicious),
        )
    return metrics


def search_metrics(path: str, sort: bool = True, **conditions: str) -> list[str]:
    """
    Search the metrics from the given path.

    Parameters
    ----------
    path : str
        The path to the metrics.
    conditions : dict[str, str]
        The conditions to search the metrics. The keys are the names of hydra options
        that have been set, and the values are the values of the options. For example,
        if conditions is set to `{"distribution": "10-0"}`, then the fonctions will load
        all experiments that have been run with `distribution=10-0`. Values can also be
        regex patterns. For example, if conditions is set to `{"distribution":
        "10-.*"}`, then the fonctions will load all experiments that have been run with
        `distribution` starting with `10-`.

    Returns
    -------
    list[str]
        The paths to the metrics.
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
            re.match(v, options.get(k, "")) is not None for k, v in conditions.items()
        ):
            metrics.append(d.as_posix())

    return sorted(metrics) if sort else metrics


def avg(cond: str, lines: dict[str, list[float]]) -> dict[str, list[float]]:
    """
    Average the lines on the specified condition.
    """
    # extract the existing values of the condition
    values: list[str] = []
    for k in lines:
        conditions = k.split(",")
        for c in conditions:
            if c.startswith(cond):
                values.append(c.split("=")[1])
    values = list(set(values))

    new_lines = {}
    for variant in values:
        for k in lines:
            if f"{cond}={variant}" in k:
                # remove the condition from the name
                new_name = ",".join(
                    [c for c in k.split(",") if c != f"{cond}={variant}"]
                )
                if f"{cond}={variant}" not in new_lines:
                    new_lines[f"{cond}={variant}"] = {}
                new_lines[f"{cond}={variant}"][new_name] = lines[k]

    avgs = {}
    for name in next(iter(new_lines.values())):
        avgs[name] = []
        for variant in new_lines:
            avgs[name].append(new_lines[variant][name])
        avgs[name] = [sum(m) / len(m) for m in zip(*avgs[name])]
    return avgs


def plot(
    *plotables: dict[str, tuple[list[float], list[float]]],
    direction: str = "horizontal",
) -> None:
    """
    Plot the given lines.
    """

    def _normalize(lst: list[float], length: int) -> list[float]:
        """Normalize list length by repeating values."""
        if len(lst) < length:
            mul = int(length / len(lst))
            return list(chain(*(([value] * mul) for value in lst)))
        return lst

    max_len = max(
        len(lst)
        for line_pairs in plotables
        for pair in line_pairs.values()
        for lst in pair
    )
    new_plotables: list[dict[str, tuple[list[float], list[float]]]] = []
    for line_pairs in plotables:
        new_line_pairs = {}
        for k, pair in line_pairs.items():
            # strip "+"s from the name
            if "+" in k:
                k = ",".join([c.strip("+") for c in k.split(",")])
            new_line_pairs[k] = (
                _normalize(pair[0], max_len),
                _normalize(pair[1], max_len),
            )

        new_plotables.append(new_line_pairs)

    width, height = plt.rcParams.get("figure.figsize", (6.4, 4.8))

    if direction == "horizontal":
        fig, axs = plt.subplots(
            1,
            len(new_plotables),
            figsize=(width * len(new_plotables), height),
            sharex=True,
            sharey=True,
        )
    elif direction == "vertical":
        fig, axs = plt.subplots(
            len(new_plotables),
            1,
            figsize=(width, height * len(new_plotables)),
            sharex=True,
            sharey=True,
        )
    else:
        raise ValueError(f"Unknown direction: {direction}")

    if len(new_plotables) == 1:
        axs = [axs]

    for ax, lines in zip(axs, new_plotables):
        ax: Axes
        y = None
        if len(lines) == 1:
            m = re.match(r".*epochs=(\w+).*", list(lines.keys())[0])
            if m is not None:
                epochs_str = m.group(1)
                if m := re.match(r"(\d+)e", epochs_str) or re.match(
                    r"\d+_(\d+)x\d+", epochs_str
                ):
                    print(m)
                    epochs = int(m.group(1))
                    y = list(range(0, len(list(lines.values())[0][0]) * epochs, epochs))

        # extract the common conditions in the names
        conditions_count: dict[str, int] = {}
        for k in lines:
            conditions = k.split(",")
            for c in conditions:
                conditions_count[c] = conditions_count.get(c, 0) + 1
        common = [k for k, v in conditions_count.items() if v == len(lines)]

        # remove the common conditions from the names
        new_lines: dict[str, tuple[list[float], list[float]]] = {}
        for k, v in lines.items():
            conditions = [c for c in k.split(",") if c not in common]
            new_lines[",".join(conditions)] = v

        # plot the lines
        mark_color = zip(MARKERS, COLORS)
        for k, (fit, dist) in new_lines.items():
            marker, color = next(mark_color)

            if y is not None:
                ax.plot(
                    y,
                    fit,
                    label=k + " (fit)",
                    # marker=marker,
                    color=color,
                    linestyle="--",
                )
                ax.plot(
                    y,
                    dist,
                    label=k + " (dist)",
                    # marker=marker,
                    color=color,
                    linestyle="-",
                )
            else:
                ax.plot(
                    fit,
                    label=k + " (fit)",
                    # marker=marker,
                    color=color,
                    linestyle="--",
                )
                ax.plot(
                    dist,
                    label=k + " (dist)",
                    # marker=marker,
                    color=color,
                    linestyle="-",
                )

        ax.yaxis.set_tick_params(labelleft=True)

        if direction == "horizontal":
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),
                ncol=min(len(new_lines), 3),
            )
        elif direction == "vertical":
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1, 1),
            )
        else:
            raise ValueError(f"Unknown direction: {direction}")
        ax.set_xlabel("Epoch")
        ax.set_title(" ".join(common), wrap=True)

    # factorize titles if possible
    titles = [ax.get_title() for ax in axs]
    conds = chain(*[t.split(" ") for t in titles])
    t = " ".join([k for k, v in Counter(conds).items() if v > 1])
    fig.suptitle(t, wrap=True)
    for ax in axs:
        # set the remaining conditions
        ax.set_title(" ".join([c for c in ax.get_title().split(" ") if c not in t]))
    plt.show()
