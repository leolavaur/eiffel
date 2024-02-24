"""Tests for dataset.py."""

import tempfile

import numpy as np
import pandas as pd
import pytest

from eiffel.datasets.dataset import Dataset
from eiffel.datasets.nfv2 import RM_COLS, load_data

SEED = 1138


def test_Dataset():
    """Test Dataset."""
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

        # Test1: split the dataset
        train, test = d.split(at=0.8, stratify=d.m["Attack"], seed=SEED)
        assert isinstance(train, Dataset)
        assert isinstance(test, Dataset)
        assert train is not test
        assert len(train) == 80


if __name__ == "__main__":
    pytest.main([__file__])
