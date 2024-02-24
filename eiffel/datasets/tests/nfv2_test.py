"""Tests for dataset/nfv2.py."""

import tempfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from eiffel.datasets import Dataset
from eiffel.datasets.nfv2 import RM_COLS, NFV2Dataset, load_data
from eiffel.datasets.poisoning import PoisonIns, PoisonOp, PoisonTask


def test_load_data():
    """Test load_data()."""
    # mock the dataset with random data in a temporary directory
    cols = RM_COLS + ["col1", "col2"]

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = f"{tmpdir}/nfv2.csv"

        mock_df = pd.DataFrame(np.random.rand(100, len(cols)), columns=cols)
        # fill the "Attack" column with random values in {"Benign", "Botnet", "Dos",
        # "DDoS"}
        mock_df["Attack"] = np.random.choice(
            ["Benign", "Botnet", "Dos", "DDoS"], size=len(mock_df)
        )
        mock_df = mock_df.astype({"Attack": "category"})
        mock_df["Label"] = mock_df["Attack"] == "Benign"
        mock_df.to_csv(data_path, index=False)

        # Test1: load the whole dataset
        d = load_data(data_path)

        assert isinstance(d, Dataset)
        assert len(d) == len(mock_df)
        assert set(d.X.columns) == set(cols) - set(RM_COLS)

        # Test2: load with train/test split
        train, test = d.split(at=0.8)
        train2, test2 = d.split(at=0.8, seed=1138)
        train3, test3 = d.split(at=0.8, seed=1138)

        assert len(train) == 0.8 * len(mock_df)
        assert len(test) == 0.2 * len(mock_df)

        assert len(train) == len(train2) == len(train3)
        assert len(test) == len(test2) == len(test3)

        assert not train.X.equals(train2.X) and not test.X.equals(test2.X)
        assert train2.X.equals(train3.X) and test2.X.equals(test3.X)

        assert isinstance(train, Dataset)


def mk_mockset(seed: int | None = None) -> NFV2Dataset:
    """Build a mock dataset for testing."""
    if seed is not None:
        np.random.seed(seed)

    # Make a mock dataset
    m = pd.DataFrame()
    m["Attack"] = np.random.choice(["Benign", "Botnet", "DoS", "DDoS"], size=100)
    y = pd.Series(m["Attack"] != "Benign", name="Label")

    return NFV2Dataset(
        pd.DataFrame(np.random.rand(100, 10), columns=[f"col{i}" for i in range(10)]),
        y,
        m,
    )


def test_poison_targeted():
    """Test poison()."""
    SEED = 1138

    np.random.seed(SEED)

    mock_d = mk_mockset(SEED)

    dos_n = sum(mock_d.m["Attack"] == "DoS")

    # Test1: poisoning on 10% of target
    n = mock_d.poison(
        *PoisonTask(0.1),
        target_classes=["DoS"],
        seed=SEED,
    )
    p_dos_n = sum(mock_d.y[mock_d.m["Attack"] == "DoS"])
    assert p_dos_n == np.floor(0.9 * dos_n)  # floor because of ceil in `poison()`

    # Test2: poisoning on 10% of target; again -> 20% should be poisoned
    n = mock_d.poison(
        *PoisonTask(0.1),
        target_classes=["DoS"],
        seed=SEED,
    )
    p_dos_n = sum(mock_d.y[mock_d.m["Attack"] == "DoS"])
    assert p_dos_n == np.floor(0.8 * dos_n)

    # Test3: decrease poisoning by 10% of target -> 10% should be poisoned
    n = mock_d.poison(
        *PoisonTask(0.1, PoisonOp.DEC),
        target_classes=["DoS"],
        seed=SEED,
    )
    p_dos_n = sum(mock_d.y[mock_d.m["Attack"] == "DoS"])
    assert p_dos_n == np.floor(0.9 * dos_n)


def test_poison_untargeted():
    """Test poison()."""
    SEED = 1138

    np.random.seed(SEED)

    mock_d = mk_mockset(SEED)

    # Test1: poisoning on 10% of target
    n = mock_d.poison(
        *PoisonTask(0.1),
        seed=SEED,
    )
    n_poisoned = sum(
        (~mock_d.y)
        & (mock_d.m["Attack"] != "Benign")  # labelled as benign but is malicious
    ) + sum(
        mock_d.y
        & (mock_d.m["Attack"] == "Benign")  # labelled as malicious but is benign
    )

    assert n_poisoned == sum(mock_d.m["Poisoned"])
    assert n_poisoned == np.floor(0.1 * len(mock_d))

    # Test2: poisoning on 10% of target; again -> 20% should be poisoned
    n = mock_d.poison(
        *PoisonTask(0.1),
        seed=SEED,
    )

    n_poisoned = sum(
        (~mock_d.y)
        & (mock_d.m["Attack"] != "Benign")  # labelled as benign but is malicious
    ) + sum(
        mock_d.y
        & (mock_d.m["Attack"] == "Benign")  # labelled as malicious but is benign
    )

    assert n_poisoned == sum(mock_d.m["Poisoned"])
    assert n_poisoned == np.floor(0.2 * len(mock_d))

    # Test3: decrease poisoning by 10% of target -> 10% should be poisoned
    n = mock_d.poison(
        *PoisonTask(0.1, PoisonOp.DEC),
        seed=SEED,
    )

    n_poisoned = sum(
        (~mock_d.y)
        & (mock_d.m["Attack"] != "Benign")  # labelled as benign but is malicious
    ) + sum(
        mock_d.y
        & (mock_d.m["Attack"] == "Benign")  # labelled as malicious but is benign
    )

    assert n_poisoned == sum(mock_d.m["Poisoned"])
    assert n_poisoned == np.floor(0.1 * len(mock_d))


if __name__ == "__main__":
    test_poison_untargeted()
