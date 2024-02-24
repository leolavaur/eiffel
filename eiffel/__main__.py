"""Entrypoint for the Eiffel CLI."""

# import eiffel

import json
import logging
import textwrap
from pathlib import Path
from typing import Any

import hydra
import tensorflow as tf
from flwr.common.logger import logger as flwr_logger
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import InterpolationToMissingValueError, MissingMandatoryValue

from eiffel.core.experiment import Experiment
from eiffel.utils import set_seed


def collect_missing(cfg: Any, missing=[]) -> list:
    """Collect missing fields.

    Recursively collect missing fields from a configuration object using the
    InterpolationToMissingValueError and MissingMandatoryValue exceptions.
    """
    if isinstance(cfg, ListConfig):
        for item in cfg:
            collect_missing(item, missing)
    elif isinstance(cfg, DictConfig):
        for k in cfg.keys():
            try:
                collect_missing(cfg[k], missing)
            except MissingMandatoryValue as e:
                missing.append(k)
            except InterpolationToMissingValueError as e:
                pass
    return missing


@hydra.main(version_base="1.3", config_path="conf", config_name="eiffel")
def main(cfg: DictConfig):
    """Entrypoint for the Eiffel CLI."""
    log = logging.getLogger(__name__)
    set_seed(cfg.seed)

    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.handlers.clear()
        logger.propagate = True

    enable_tf_gpu_growth()

    log.info("Starting Eiffel")
    if cfg.is_empty():
        log.critical("Empty configuration.")
        exit(1)

    # missings = collect_missing(cfg)
    missings = OmegaConf.missing_keys(cfg)
    if missings:
        log.critical(f"Missing fields: {missings}")
        exit(1)

    log.debug(
        "Dumping configuration.\n"
        + textwrap.indent(OmegaConf.to_yaml(cfg, resolve=True), "\t")
    )

    log.info(f"{len(tf.config.list_physical_devices('GPU'))} GPUs available.")

    # _convert_ is required for nested instantiations
    # see:
    # - https://github.com/facebookresearch/hydra/issues/2591#issuecomment-1518170214
    # - https://github.com/facebookresearch/hydra/issues/1719
    ex: Experiment = instantiate(cfg.experiment, _convert_="object")
    Path("./stats.json").write_text(json.dumps(ex.data_stats(), indent=4))
    ex.run()
    res = ex.results
    res.save("fit")
    res.save("distributed")
    log.info("Run completed.")
    log.info(f"Results saved in: {Path.cwd()}.")


if __name__ == "__main__":
    import sys

    gettrace = getattr(sys, "gettrace", None)

    if gettrace is not None and gettrace():
        # Running in a debugger.
        # Paste here the command that you want to debug.
        cmd = (
            "eiffel --config-dir exps/assessment/similarity-1/conf"
            " +datasets=nfv2/sampled/cicids +distribution=5-5 +epochs=100/10x10"
            " +scenario=continuous-100 +target=untargeted batch_size=512"
            " partitioner=kmeans_drop_2 seed=421"
        )
        sys.argv = cmd.split(" ")

    flwr_logger.setLevel(logging.INFO)
    try:
        main()
    except Exception as e:
        logging.getLogger(__name__).exception(e)
        raise
