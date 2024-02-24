"""Plotting Hydra callbacks for Eiffel."""

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from omegaconf import DictConfig
from scipy.interpolate import BSpline, make_interp_spline


class PlotterCallback(Callback):
    """Plot callback.

    This class implements Hydra's callback mechanism to plot the FL runs.

    For reference, see: https://hydra.cc/docs/experimental/callbacks/.
    """

    def __init__(self, output: str, input: str = "metrics.json") -> None:
        self.input = input
        self.output = output

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        """Call when a job ends."""
        try:
            f = Path(self.input).read_text()
            metrics = json.loads(f)
            self._plot(metrics, output=self.output)
        except FileNotFoundError:
            print(f"File {self.input} not found.")

    def _plot(self, metrics: dict, output: str) -> None:
        """Generate the plot."""
        raise NotImplementedError


class PlotCallback(PlotterCallback):
    """Plot callback.

    This callback plots a selected metric for each run. By default, it plots the mean of
    the selected metric in blue and, if attackers are present, the mean of the same
    metric for the attackers in red.
    """

    def __init__(
        self,
        *,
        metric: str = "accuracy",
        smooth: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.smooth = smooth

    def _plot(self, metrics: dict, output: str) -> None:
        """Generate the plot."""
        benign_metrics = []
        attacker_metrics = []

        for cid, cmetrics in metrics.items():
            selection = [m[self.metric] for _, m in cmetrics.items()]
            if "malicious" in cid:
                attacker_metrics.append(selection)
            else:
                benign_metrics.append(selection)

        benign_metrics = list(zip(*benign_metrics))
        attacker_metrics = list(zip(*attacker_metrics))

        benign_metrics = [sum(m) / len(m) for m in benign_metrics]
        attacker_metrics = [sum(m) / len(m) for m in attacker_metrics]
        rounds = [r + 1 for r in range(len(benign_metrics))]

        benign_plot = (rounds, benign_metrics)
        attacker_plot = (rounds, attacker_metrics)

        if self.smooth:
            # 300 represents number of points to make between min and max
            lin_x = np.linspace(min(rounds), max(rounds), 300)
            benign_spl = make_interp_spline(
                rounds, benign_metrics, k=2
            )  # type: BSpline
            benign_smooth = benign_spl(lin_x)

            benign_plot = (lin_x, benign_smooth)

            if attacker_metrics:
                attacker_spl = make_interp_spline(rounds, attacker_metrics, k=2)
                attacker_smooth = attacker_spl(lin_x)
                attacker_plot = (lin_x, attacker_smooth)

        plt.figure()
        plt.title(f"Mean {self.metric}")
        plt.xlabel("Rounds")
        plt.ylabel(self.metric.title())
        plt.plot(*benign_plot, label="Benign")
        if attacker_metrics:
            plt.plot(*attacker_plot, label="Attacker")
        plt.legend()
        plt.savefig(Path(output))
        plt.close()
