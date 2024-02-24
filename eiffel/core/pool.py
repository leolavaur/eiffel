"""Client pool API for Eiffel."""

import random
from typing import Callable, Optional

import keras
import tensorflow as tf
from hydra.utils import call
from omegaconf import DictConfig
from ray import ObjectRef
from ray.actor import ActorHandle

from eiffel.core.errors import ConfigError
from eiffel.datasets.dataset import Dataset, DatasetHandle
from eiffel.datasets.partitioners import DumbPartitioner, Partitioner
from eiffel.datasets.poisoning import PoisonIns, PoisonTask
from eiffel.utils.time import timeit

from ..utils.typing import EiffelCID


class Pool:
    """Pool of clients.

    A pool is a collection of clients that share the same dataset and attack type.

    Attributes
    ----------
    pool_id : str
        The pool ID.
    attack : PoisonIns
        The attack to perform. If `None`, the pool is benign.
    shards : dict[EiffelCID, tuple[Dataset, Dataset]]
        The different client partitions. The keys are the client IDs, and the values
        are tuples of the training and test datasets.
    """

    # pool_id: str
    # attack: PoisonIns | None
    # shards: dict[EiffelCID, tuple[Dataset, Dataset]]
    # holders: dict[EiffelCID, ActorHandle]
    # model_fn: Callable[..., tf.keras.Model]

    @timeit
    def __init__(
        self,
        dataset: Dataset | DictConfig,
        model_fn: Callable[..., tf.keras.Model],
        n_benign: int,
        *,
        n_malicious: int = 0,
        attack: PoisonIns | dict | None = None,
        pool_id: str | None = None,
        test_ratio: float = 0.2,
        common_test: bool = True,
        partitioner: Partitioner | Callable | None = None,
        seed: int,
    ) -> None:
        """Initialize the pool.

        Parameters
        ----------
        dataset : Dataset | str
            The dataset used by the clients. If a DictConfig is provided, it should be a
            valid OmegaConf configuration that can be passed to Hydra's instantiation
            logic, and return a `Dataset` object. Otherwise, it should be a `Dataset`
            object.
        benign : int
            The number of benign clients in the pool.
        malicious : int, optional
            The number of malicious clients in the pool. Defaults to 0.
        attack : Optional[dict | PoisonIns], optional
            The attack to perform. If a dictionary is provided, it should be a valid
            dictionary
        """
        self.seed = seed
        self.model_fn = model_fn
        self.attack = attack
        self.holders = {}

        if not pool_id:
            alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
            pool_id = "".join(random.choices(alphabet, k=6))
        self.pool_id = pool_id

        benign_cids = [f"{pool_id}_benign_{i}" for i in range(n_benign)]
        if n_malicious > 0 and attack is None:
            raise ValueError(
                "Invalid conditions for attack scenarios: "
                f"`{n_malicious = }`, yet attack is `None`."
            )
        malicious_cids = [f"{pool_id}_malicious_{i}" for i in range(n_malicious)]

        if not isinstance(dataset, Dataset):
            dataset = call(dataset)
        _test, _train = dataset.split(at=test_ratio, seed=self.seed)

        if not partitioner:
            partitioner = DumbPartitioner

        partitioner = partitioner(n_partitions=n_benign + n_malicious, seed=self.seed)

        partitioner.load(_train)
        _train_shards = partitioner.all()
        if common_test:
            _test_shards = [_test.copy() for _ in _train_shards]
        else:
            partitioner.load(_test)
            _test_shards = partitioner.all()

        self.shards = {}
        for cid in benign_cids:
            self.shards[cid] = (_train_shards.pop(), _test_shards.pop())

        if attack:
            assert isinstance(attack, PoisonIns)

            p_task = self.attack.base
            for cid in malicious_cids:
                _train_shard = _train_shards.pop()
                _train_shard.poison(
                    p_task.fraction,
                    p_task.operation,
                    target_classes=self.attack.target,
                    seed=self.seed,
                )
                self.shards[cid] = (_train_shard, _test_shards.pop())

    def __len__(self) -> int:
        """Return the number of clients in the pool."""
        return len(self.shards)

    def __contains__(self, cid: EiffelCID) -> bool:
        """Return whether the pool contains the client."""
        return cid in self.shards

    @timeit
    def deploy(self) -> None:
        """Deploy the dataset onto the Ray object store."""
        if not self.holders:
            self.holders = {}
        for cid, (train, test) in self.shards.items():
            if cid in self.holders:
                raise ValueError(f"Client `{cid}` already deployed.")

            self.holders[cid] = DatasetHandle.remote({"train": train, "test": test})

    def deployed(self) -> bool:
        """Return whether the pool is deployed."""
        return len(self.holders) == len(self.shards)

    def gen_mappings(self) -> dict[EiffelCID, tuple[ObjectRef, PoisonIns, keras.Model]]:
        """Generate mappings between CIDs, and their handle and poisoning instructions.

        Returns
        -------
        dict[EiffelCID, tuple[ObjectRef, PoisonIns]]
            The mappings. Keys are the client IDs, and values are tuples of each
            client's dataset handle and poisoning instructions.
        """
        if not self.deployed():
            raise RuntimeError(
                "Attempting to access the handles of an undeployed pool."
            )

        mappings = {}
        for cid, handle in self.holders.items():
            mappings[cid] = (
                handle,
                self.attack if "malicious" in cid else None,
                self.model_fn,
            )
        return mappings

    @property
    def shards_stats(self) -> dict:
        """Return the pool data statistics."""
        return {
            cid: {"train": train.stats, "test": test.stats}
            for cid, (train, test) in self.shards.items()
        }

    @property
    def ids(self) -> list[EiffelCID]:
        """Return the list of client IDs."""
        return list(self.shards.keys())
