"""NF-V2 Dataset utilities.

This module contains functions to load and prepare the NF-V2 dataset for Deep Learning
applications. The NF-V2 dataset is a collection of 4 datasets with a standardised set of
features. The datasets are:
    * CSE-CIC-IDS-2018
    * UNSW-NB15
    * ToN-IoT
    * Bot-IoT

The NF-V2 dataset is available at: https://staff.itee.uq.edu.au/marius/NIDS_datasets/

Part of the code in this module is inspired on the code from Bertoli et al. (2022), who
tested Federated Learning on the NF-V2 dataset. See:
https://github.com/c2dc/fl-unsup-nids


File structure
--------------

Wether it is downloaded using Eiffel or not, the NF-V2 dataset is expected to follow the
following file structure. You should rename the files to match the structure if you
download the datasets manually.
```
.../nfv2
    ├── origin
    │   ├── botiot.csv.gz
    │   ├── cicids.csv.gz
    │   ├── toniot.csv.gz
    │   └── nb15.csv.gz
    ├── sampled
    │   ├── botiot.csv.gz
    │   └── ...
    └── reduced
        └── ...
```

References
----------
    * Sarhan, M., Layeghy, S. & Portmann, M., Towards a Standard Feature Set for Network
      Intrusion Detection System Datasets. Mobile Netw Appl (2021).
      https://doi.org/10.1007/s11036-021-01843-0 
    * Bertoli, G., Junior, L., Santos, A., & Saotome, O., Generalizing intrusion
      detection for heterogeneous networks: A stacked-unsupervised federated learning
      approach. arXiv preprint arxiv:2209.00721 (2022). https://arxiv.org/abs/2209.00721
"""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional

import numpy as np
import pandas as pd
from omegaconf import ListConfig
from sklearn.preprocessing import MinMaxScaler

from eiffel.datasets import DEFAULT_SEARCH_PATH, Dataset

from .poisoning import PoisonOp

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_PATH = DEFAULT_SEARCH_PATH / "nfv2"


# Columns to drop from the dataset.
# ---------------------------------
# The sampled and reduced datasets contain an additional column called `Dataset` which
# must be dropped as well. Columns can either be completely discarded or saved in the
# metadata array (m).
RM_COLS = [
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "Label",
    "Attack",
]


class NFV2Dataset(Dataset):
    """NF-V2 Dataset."""

    _stratify_column: ClassVar[str] = "Attack"

    def poison(
        self: Dataset,
        ratio: float,
        op: PoisonOp,
        *,
        seed: int,
        target_classes: Optional[List[str]] = None,
    ) -> int:
        """Poison a dataset by apply a function to a given number of samples.

        Parameters
        ----------
        n: int
            Number of samples to poison in the target. If `target` is None, the whole
            dataset is poisoned.
        op: PoisonOp
            Poisoning operation to apply. Either PoisonOp.INC or PoisonOp.DEC.
        target_classes: Optional[List[str]]
            List of classes to poison. If None, all classes are poisoned, including
            benign samples. If ["*"], all attacks are poisoned, excluding benign
            samples. Otherwise, only the attacks in the list are poisoned.
        seed: Optional[int]
            Seed for reproducibility.

        Returns
        -------
        int
            The number of samples that have been modified.
        """
        if seed is None:
            logger.warn(
                "No seed provided for poisoning. Results will not be reproducible."
            )

        d = self.copy()

        assert target_classes is None or (
            isinstance(target_classes, List | ListConfig)
            and all(isinstance(c, str) for c in target_classes)
        ), "Invalid value for `target_classes`. Must be a list of strings or None."

        if target_classes is None:
            # If targeted means all dataset (including benign samples)
            # target is a boolean Series using the same index as the dataset
            target = pd.Series(True, index=d.y.index)

        elif target_classes == ["*"]:
            # If targeted means all attacks (excluding benign samples)
            target = d.m["Attack"] != "Benign"

        else:
            target = d.m["Attack"].isin(target_classes)

        n = np.ceil(sum(target) * ratio).astype(int)
        if n > sum(target):
            raise ValueError(
                f"Invalid value for `ratio`: ratio * len(target) = {n}."
                "Must be less or equal to len(target)."
            )

        if len(target) != len(self):
            raise ValueError(
                "Invalid value for `target`. Must be of the same length as the dataset."
            )

        if target.dtype != bool:
            raise ValueError("Invalid value for `target`. Must be a boolean Series.")

        # get poisoning metadata
        if "Poisoned" not in d.m.columns:
            d.m["Poisoned"] = False

        if op == PoisonOp.DEC:
            target = target & d.m["Poisoned"]
        else:
            target = target & ~d.m["Poisoned"]

        # indices of the samples to poison (cap n at the number of available samples)
        n = min(n, sum(target))
        idx = d.y[target].sample(n=n, random_state=seed).index.to_list()

        # apply the poisoning operation
        # cast to int to avoid future deprecation in NumPy
        d.y.loc[idx] = d.y[idx].apply(lambda x: int(not x))
        if op == PoisonOp.DEC:
            d.m.loc[idx, "Poisoned"] = False
        else:
            d.m.loc[idx, "Poisoned"] = True

        # save
        self.X = d.X
        self.y = d.y
        self.m = d.m

        # clean up
        del target
        del d

        return len(idx)


def load_data(
    path: str,
    *,
    seed: int,
    search_path: str | Path | None = None,
    shuffle: bool = True,
    **kwargs,
) -> NFV2Dataset:
    """Load a dataset.

    Parameters
    ----------
    path : str
        Key of the dataset to load. Can be a shortcut key or a path to a CSV file.
    search_path : str or Path, optional
        Path to the directory containing the dataset. If not given, the dataset is
        loaded from the default path.
    seed : int, optional
        Seed for shuffling the dataset.
    shuffle : bool, optional
        If `True`, the dataset is shuffled before being split.

    Returns
    -------
    NFV2Dataset
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset is not found at the given path.
    """
    # PATH MANAGEMENT
    # ---------------

    ppath = Path(path)

    if ppath.exists() and ppath.is_file():
        csv_path = path

    else:
        if search_path is None:
            search_path = DEFAULT_SEARCH_PATH
        else:
            search_path = Path(search_path)

        if not search_path.exists():
            raise FileNotFoundError(
                f"Search path does not exist: {search_path}",
                {"search_path": search_path},
            )

        csv_path = search_path / ppath

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: '{ppath}'",
                {"search_path": search_path, "csv_path": csv_path},
            )

    df = pd.read_csv(csv_path, engine="pyarrow")

    # DATA PREPROCESSING
    # ------------------
    # shuffle the dataset
    if shuffle:
        df = df.sample(frac=1, random_state=seed)

    # drop the "Dataset" column if it exists
    if "Dataset" in df.columns:
        df = df.drop(columns=["Dataset"])

    # select the columns to compose the Dataset object
    X = df.drop(columns=RM_COLS)
    y = df["Label"]
    m = df[RM_COLS]

    # convert classes to numerical values
    X = pd.get_dummies(X)

    # normalize the data
    scaler = MinMaxScaler()
    scaler.fit(X)
    X[X.columns] = scaler.transform(X)

    return NFV2Dataset(X, y, m, **kwargs)


def mk_nfv2_mockset(size: int, iid: bool, seed: int) -> NFV2Dataset:
    """Create a mock NF-V2 dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = f"{tmpdir}/nfv2.csv"

        cols = RM_COLS + ["col1", "col2"]
        mock_df = pd.DataFrame(np.random.rand(size, len(cols)), columns=cols)

        # fill the "Attack" column with random values in {"Benign", "Botnet", "Dos",
        # "DDoS"}
        if iid:
            if size % 4 != 0:
                # if the dataset is not divisible by 4, the classes will not be balanced
                raise ValueError(
                    "Cannot create an IID dataset with a size that is not divisible"
                    " by 4.",
                    size,
                )
            mock_df["Attack"] = np.array(
                ["Benign", "Botnet", "Dos", "DDoS"] * (size // 4)
            )
        else:
            mock_df["Attack"] = np.random.choice(
                ["Benign", "Botnet", "Dos", "DDoS"], size=len(mock_df)
            )
        mock_df = mock_df.astype({"Attack": "category"})
        mock_df["Label"] = mock_df["Attack"] == "Benign"
        mock_df.to_csv(data_path, index=False)

        return load_data(data_path, seed=seed, shuffle=True)
