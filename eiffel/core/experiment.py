"""Eiffel engine."""

import functools
import json
import logging
import math
from copy import deepcopy
from functools import partial, reduce
from types import NoneType
from typing import Any, Callable, Type

import numpy as np
import psutil
import ray
import tensorflow as tf
from flwr.common import ndarrays_to_parameters
from flwr.server import Server, ServerConfig
from flwr.server.strategy import FedAvg, Strategy
from flwr.simulation import start_simulation
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from hydra.utils import instantiate
from keras.models import Model
from omegaconf import DictConfig, ListConfig

from eiffel.core.errors import ConfigError
from eiffel.datasets.dataset import Dataset
from eiffel.datasets.partitioners import DumbPartitioner, Partitioner
from eiffel.datasets.poisoning import PoisonIns
from eiffel.utils import set_seed
from eiffel.utils.hydra import instantiate_or_return
from eiffel.utils.time import timeit
from eiffel.utils.typing import ConfigDict, MetricsDict

from .client import mk_client, mk_client_init_fn
from .pool import Pool
from .results import Results

logger = logging.getLogger(__name__)


class Experiment:
    """Eiffel experiment.

    Attributes
    ----------
    server : flwr.server.Server
        The Flower server. It administrates the entire FL process, and is responsible
        for aggregating the models, based on the function provided in the
        `flwr.server.Strategy` object.
    strategy : flwr.server.Strategy
        The strategy used by the Flower server to aggregate the models. It is passed to
        Flower's `start_simulation` function.
    pools : list[Pool]
        The different client pools. Each pool is a collection of clients that share the
        same dataset and attack type.
    n_clients : int
        The total number of clients in the experiment.
    """

    n_clients: int
    n_rounds: int
    n_concurrent: int
    seed: int

    server: Server | None
    strategy: Strategy
    pools: list[Pool]

    @timeit
    def __init__(
        self,
        seed: int,
        num_rounds: int,
        num_epochs: int,
        batch_size: int,
        model_fn: Callable[..., tf.keras.Model] | DictConfig,
        pools: list[Pool | DictConfig],
        datasets: list[Dataset | DictConfig],
        attacks: list[PoisonIns | dict | DictConfig],
        strategy: partial[Strategy] | Strategy | None = None,
        server: Server | None = None,
        partitioner: Partitioner | DictConfig | None = None,
    ):
        """Initialize the experiment.

        The `expriment` object is a wrapper around the Flower server. It is responsible
        for instantiating the server, the clients, and the strategy. It also handles the
        data partitioning and the attack configuration.

        Initialization relies mostly on three configurations objects obtained from
        Hydra, and thereafter mapped together to create the experiment's setup:

        - `pools`: the list of client pools. Each pool is a DictConfig object
            containing, or an instantiated Pool object. If a dictionary, it should
            contain, at the very least, the number of clients in the pools as:
            `{n_benign: int, n_malicious: int}`.
        - `datasets`: the list of datasets used by the clients. Each dataset is a
          DictConfig object that can be passed to Hydra's instantiation logic for a
          `load_data` fonction, or a Dataset object.
        - `attacks`: the attack configuration as: `{type: str, profile: str}`, or a list
          of PoisonIns objects.

        The number of pools is defined by the length of the `pools` list. If a single
        element is provided, ie. if the length of the list is 1, then the attack or
        dataset is used by all pools. Otherwise, the length of the list should be equal
        to the number of pools.

        If the number of datasets is 1, then the evaluation can be done centrally by the
        server. Otherwise, the evaluation is done by each client, and the distributed
        metrics are aggregated afterwards. This can be disabled using the
        `distributed_evaluation` flag, which is set to `False` by default.

        Parameters
        ----------
        seed : int
            The seed for reproducibility.
        num_rounds : int
            The number of rounds to run.
        num_epochs : int
            The number of epochs to run on each client.
        batch_size : int
            The batch size to use.
        model_fn : Callable[..., tf.keras.Model] | DictConfig
            A function that returns a compiled Keras model. Each client process will run
            this function to instantiate its model. If a DictConfig object is provided,
            the function is instantiated using Hydra's instantiation logic and MUST
            return a `functool.partial` object, using `_partial_: True`. Overall, it is
            recommanded to pass a partial function.
        pools : list[Pool | DictConfig]
            The list of client pools.
        datasets : list[Dataset | DictConfig]
            The datasets to use per pool.
        attacks : list[PoisonIns | dict | DictConfig]
            The attacks to use per pool.
        strategy : Strategy | DictConfig | None, optional
            The Flower-compatible strategy to use. Defaults to
            `flwr.server.strategy.FedAvg` if None.
        server : Server | DictConfig | None, optional
            The Flower server. Defaults to the default Flower server if None.
        partitioner : Partitioner | DictConfig | None, optional
            The partitioner to use. Defaults to `DumbPartitioner` if None.

        Raises
        ------
        ConfigError
            If the number of pools is not equal to the number of datasets or the number
            of attacks.
        """
        self.seed = seed
        # set_seed(seed)

        self.server = server
        self.n_rounds = num_rounds
        self.pools = []

        pools = obj_to_list(pools)
        attacks = obj_to_list(attacks, expected_length=len(pools))
        datasets = obj_to_list(datasets, expected_length=len(pools))

        pools_mapping = zip(pools, attacks, datasets)

        for pool, attack, dataset in pools_mapping:
            if not isinstance(dataset, Dataset):
                dataset = instantiate_or_return(dataset, Dataset)

            if not isinstance(attack, (PoisonIns, NoneType)):
                if isinstance(attack, dict | DictConfig):
                    if "n_rounds" not in attack:
                        attack["n_rounds"] = num_rounds
                    attack = PoisonIns.from_dict(
                        dict(attack), default_target=dataset.default_target
                    )
                else:
                    raise TypeError(
                        "`attack` must be a PoisonIns, a valid PoisonIns "
                        f"configuration dictionary, or None; got {type(attack)}."
                    )

            if isinstance(pool, (DictConfig, dict)):
                if "_target_" in pool:
                    pool = instantiate(
                        pool,
                        partitioner=instantiate_or_return(partitioner, partial),
                        dataset=instantiate_or_return(dataset, Dataset),
                        attack=attack,
                        model_fn=instantiate_or_return(model_fn, partial),
                        seed=self.seed,
                    )
                else:
                    pool = Pool(
                        dataset=instantiate_or_return(dataset, Dataset),
                        model_fn=instantiate_or_return(model_fn, partial),
                        partitioner=instantiate_or_return(partitioner, partial),
                        attack=attack,
                        seed=self.seed,
                        **{str(k): v for k, v in pool.items()},
                    )
            elif not isinstance(pool, Pool):
                raise ConfigError(
                    f"Invalid pool type: {type(pool)}. Expected a Pool object."
                )
            self.pools.append(pool)

        self.n_clients = sum([len(p) for p in self.pools])

        if strategy is None:
            strategy = FedAvg()

        if isinstance(strategy, partial):
            self.strategy = strategy(
                min_fit_clients=self.n_clients,
                min_evaluate_clients=self.n_clients,
                min_available_clients=self.n_clients,
                on_fit_config_fn=mk_config_fn({
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                }),
                evaluate_metrics_aggregation_fn=aggregate_metrics_fn,
                fit_metrics_aggregation_fn=aggregate_metrics_fn,
                on_evaluate_config_fn=mk_config_fn(
                    {"batch_size": batch_size}, stats_when=self.n_rounds
                ),
                initial_parameters=get_random_weights(model_fn, datasets[0].X.shape[1]),
            )
        else:
            self.strategy = strategy

        assert isinstance(self.strategy, Strategy), (
            "Invalid strategy type: "
            f"{type(self.strategy)}. Expected a flwr.server.Strategy object."
        )

    def run(self, **ray_kwargs) -> None:
        """Run the experiment."""
        init_kwargs = (ray_kwargs or {}) | {
            "ignore_reinit_error": True,
            # "local_mode": True,
            "num_gpus": len(tf.config.list_physical_devices("GPU")),
        }
        ray.init(**init_kwargs)

        for pool in self.pools:
            pool.deploy()

        mappings = reduce(lambda a, b: a | b, [p.gen_mappings() for p in self.pools])

        fn = functools.partial(
            mk_client,
            mappings=mappings,
            seed=self.seed,
        )

        self.hist = start_simulation(
            client_fn=fn,
            num_clients=self.n_clients,
            config=ServerConfig(num_rounds=self.n_rounds),
            strategy=self.strategy,
            client_resources=compute_client_resources(self.n_clients),
            actor_kwargs={"on_actor_init_fn": mk_client_init_fn(seed=self.seed)},
            clients_ids=reduce(lambda a, b: a + b, [p.ids for p in self.pools]),
            server=self.server,
            keep_initialised=True,
        )

        ray.shutdown()

    @property
    def results(self) -> Results:
        """Return the experiment's results."""
        return Results.from_flwr(self.hist)

    def data_stats(self) -> dict[str, dict[str, int]]:
        """Return the data statistics for each pool."""
        return {p.pool_id: p.shards_stats for p in self.pools}


def get_random_weights(model_config: DictConfig, n_features: int) -> list[np.ndarray]:
    """Get random weights for a model."""
    model_fn = instantiate_or_return(model_config, partial)
    model: Model = model_fn(n_features)
    return ndarrays_to_parameters(model.get_weights())


def mk_config_fn(
    config: ConfigDict, stats_when: int = -1
) -> Callable[[int], ConfigDict]:
    """Return a function which creates a config for the given round.

    Optionally, the function can be configured to return a config with the `stats` flag
    enabled for a given round. This is useful to compute attack-wise statistics.

    Parameters
    ----------
    config : ConfigDict
        The configuration to return.
    stats_when : int, optional
        The round for which to enable the `stats` flag. Defaults to -1, which disables
        the flag entirely.
    """
    if stats_when > 0:

        def config_fn(r: int) -> ConfigDict:
            cfg = config | {"round": r}
            if r == stats_when:
                return cfg | {"stats": True}
            return cfg

        return config_fn

    return lambda r: config | {"round": r}


def compute_client_resources(
    n_concurrent: int, headroom: float = 0.1
) -> dict[str, float]:
    """Compute the number of CPUs and GPUs to allocate to each client.

    Parameters
    ----------
    n_concurrent : int
        The number of concurrent clients.
    headroom : float, optional
        The headroom to leave for the system. Defaults to 0.1.

    Returns
    -------
    dict[str, float]
        The number of CPUs and GPUs to allocate to each client.
    """
    available_cpus = psutil.cpu_count() * (1 - headroom)
    available_gpus = len(tf.config.list_physical_devices("GPU"))
    if n_concurrent > available_cpus:
        logger.warning(
            f"Number of concurrent clients ({n_concurrent}) is greater than the number"
            f" of available CPUs ({available_cpus}). Some clients will be run"
            " sequentially."
        )
    return {
        "num_cpus": math.floor(max(1, available_cpus / n_concurrent)),
        "num_gpus": available_gpus / min(n_concurrent, available_cpus),
    }


def obj_to_list(
    config_obj: ListConfig | DictConfig | list,
    expected_length: int = 0,
) -> list:
    """Convert a DictConfig or ListConfig object to a list."""
    if not isinstance(config_obj, (ListConfig, list, DictConfig, dict, NoneType)):
        raise ConfigError(
            f"Invalid config object: {type(config_obj)}. Expected a list or dictionary."
        )

    if not isinstance(config_obj, (list, ListConfig)):
        config_obj = [config_obj]

    if expected_length > 0:
        if len(config_obj) > 1 and len(config_obj) != expected_length:
            raise ConfigError(
                "The number of items in config_obj should be equal to"
                f" {expected_length}, or 1."
            )

        elif len(config_obj) == 1:
            config_obj = list(config_obj) * expected_length

    return list(config_obj)


def aggregate_metrics_fn(metrics_mapping: list[tuple[int, MetricsDict]]) -> MetricsDict:
    """Collect all metrics client-per-client.

    Eiffel processes metrics after the experiment's ending, which permits more versatile
    analytics. However, Flower expects a single metrics dictionary. This serializes each
    client's metrics into a single dictionary, indexed by the client's ID and containing
    collected metrtics (recall and missrate) for each attack class, as well as global
    metrics for the entire test set: accuracy, precision, recall, F1-score, missrate,
    and fallout.

    Parameters
    ----------
    metrics_mapping : list[tuple[int, MetricsDict]]
        A list of tuples containing the number of samples in the testing set and the
        collected metrics for each client.

    Returns
    -------
    MetricsDict
        A single dictionary containing all metrics.
    """
    round_metrics: MetricsDict = {}
    for _, m in metrics_mapping:
        met = {}
        for k, v in m.items():
            try:
                met[k] = json.loads(str(v))
            except json.JSONDecodeError:
                met[k] = v

        cid = str(met.pop("_cid"))
        round_metrics[cid] = json.dumps(met)

    return round_metrics
