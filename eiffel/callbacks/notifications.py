"""Notifications callbacks."""

import json
import textwrap
import time
from typing import Any

import requests
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from omegaconf import DictConfig


class MattermostCallback(Callback):
    """Mattermost incomming webhook callback.

    This class implements Hydra's callback mechanism to send messages to a Mattermost
    channel on certain events, such as when a job ends.

    For reference, see: https://hydra.cc/docs/experimental/callbacks/

    TODO: link to the generated files (https://github.com/microsoft/vscode-remote-release/issues/656)
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.jobs: list[tuple[JobReturn, float]] = []
        self.timings = {}

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Call when run starts."""
        pass

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Call when multirun starts."""
        pass

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Call when a job starts."""
        id = config.hydra.job.get("id", None) or config.hydra.job.name
        self.timings[id] = time.time()

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        """Call when a job ends."""
        id = config.hydra.job.get("id", None) or config.hydra.job.name
        t = time.time() - self.timings[id]
        self.jobs.append((job_return, t))

    def on_run_end(self, config: DictConfig, **kwargs: Any) -> None:
        """Call when run ends."""
        self._send_notification(config, **kwargs)

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """Call when multirun ends."""
        self._send_notification(config, **kwargs)

    def _send_notification(self, config: DictConfig, **kwargs: Any):
        """Send a message to Mattermost.

        Args:
            config: Hydra's configuration object.
        """
        jobs = "\n".join(
            [
                f"| **{config.hydra.job.name}**  | `{j.status}` |"
                f" {t:.2f} seconds | `{j.overrides}` | `{j.working_dir}` |"
                for j, t in self.jobs
            ]
        )

        r = requests.post(
            url=self.url,
            data=json.dumps(
                {
                    "text": textwrap.dedent(
                        f"""
                        Hey @here!
                        The experiment has finished. The following jobs were run:
                        
                        | Job | Status | Time | Overrides | Results |
                        | --- | ------ | ---- | --------- | ------- |
                        """
                    )
                    + jobs,
                }
            ),
        )
        r.raise_for_status()
