"""Plotting and data analysis utilities."""

import itertools
import json
import re
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Any, Generator, NamedTuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

MARKERS = itertools.cycle(["o", "D", "v", "*", "+", "^", "p", ".", "P", "<", ">", "X"])
LINESTYLES = itertools.cycle(["-", "--", "-.", ":"])
COLORS = itertools.cycle(
    [
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
)


class Plotable(NamedTuple):
    """A plotable object."""

    name: str
    values: list[float]

    def __getitem__(self, s: slice) -> "Plotable":
        """Get a slice of the plotable."""
        return Plotable(self.name, self.values[s])


def envelope(
    plotables: list[Plotable], ax: Axes | None = None, color: str = "orange"
) -> None:
    """Plot statistical envelopes of the given plotables.

    Parameters
    ----------
    plotables : list[Plotable]
        List of Plotables objects of equal length.
    """
    if not all(len(p.values) == len(plotables[0].values) for p in plotables):
        raise ValueError("All Plotables must have the same length")

    # stack the values
    data = np.column_stack([p.values for p in plotables])

    # Compute mean, standard deviation, and quartiles along the columns
    mean = np.mean(data, axis=1)
    sigma = np.std(data, axis=1)
    low_sigma = mean - sigma
    high_sigma = mean + sigma
    max_ = np.max(data, axis=1)
    min_ = np.min(data, axis=1)

    # plot
    if ax is None:
        ax = plt.gca()

    x = list(range(1, len(mean) + 1))
    # plot the mean
    ax.plot(x, mean, label="mean", color=color, linewidth=3)
    ax.plot(x, low_sigma, color=color)
    ax.plot(x, high_sigma, color=color)
    ax.plot(x, min_, color=color, linestyle="--")
    ax.plot(x, max_, color=color, linestyle="--")

    # plot the envelope
    ax.fill_between(
        x,
        low_sigma,
        high_sigma,
        alpha=0.2,
        facecolor=color,
    )
    # plot the quartiles
    ax.fill_between(
        x,
        min_,
        max_,
        alpha=0.4,
        facecolor=color,
    )


def scale(plotables: list[Plotable], length: int) -> list[Plotable]:
    """Scale the plotables to the given length.

    Each result length varies depending on the number of rounds. This function scales
    results to the given length by repeating values. Thus, the results can only be
    scaled to a multiple of the original length.

    Parameters
    ----------
    plotables : Plotable
        Plotables to scale.
    length : int
        Length to scale to. Must be a multiple of the original length.

    Returns
    -------
    list[Plotable]
        Scaled plotables.
    """
    if length <= 0:
        raise ValueError("Length must be positive.")
    ret = []
    for p in plotables:
        if length % len(p.values) != 0:
            raise ValueError(
                f"Length {length} is not a multiple of {len(p.values)} for {p.name}."
            )
        mul = int(length / len(p.values))
        ret.append(Plotable(p.name, list(chain(*(([v] * mul) for v in p.values)))))

    return ret


# The following code is kept here for reference, but will be removed in the future.
if False:

    def old(
        *plotables: dict[str, tuple[list[float], list[float]]],
        direction: str = "horizontal",
    ) -> None:
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
                        y = list(
                            range(0, len(list(lines.values())[0][0]) * epochs, epochs)
                        )

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
