"""Tests for eiffel.core.client."""

import os

import ray

from eiffel.core.client import EiffelClient, mk_client
from eiffel.datasets.dataset import DatasetHandle
from eiffel.datasets.nfv2 import load_data
from eiffel.models.supervized import mk_popoola_mlp

SEED = 1138


def mk_mock_client():
    """Make a mock client."""
    print(os.getcwd())
    dataset = load_data("data/nfv2/sampled/cicids.csv.gz", seed=SEED)
    train, test = dataset.split(0.8, seed=SEED)
    data_holder = DatasetHandle.remote({"train": train, "test": test})
    client = mk_client(
        cid="mock_client",
        mappings={
            "mock_client": (
                data_holder,
                None,
                mk_popoola_mlp(train.X.shape[1]),
            )
        },
        seed=SEED,
    )

    return client


def test_evaluate():
    """Test the evaluate method.

    The evaluate method should return the metrics on the test set.
    """
    pass
