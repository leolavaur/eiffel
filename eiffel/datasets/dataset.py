"""Common dataset classes and functions.

This module contains the common dataset classes and functions used by the Eiffel
framework. It provides a unified interface for loading datasets, regardless of their
format.
"""

import math
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import gettempdir
from typing import ClassVar, Hashable, List, Optional, Tuple, cast, overload

import numpy as np
import pandas as pd
import ray
from keras.utils import Sequence
from sklearn.model_selection import train_test_split

from eiffel.utils.typing import ConfigDict

from .poisoning import PoisonOp


class BatchLoader(Sequence):
    """Generator of batches for training."""

    X: pd.DataFrame
    target: pd.DataFrame | pd.Series | None

    batch_size: int

    def __init__(
        self,
        batch_size: int,
        X: pd.DataFrame,
        *,
        seed: int,
        target: pd.DataFrame | pd.Series | None = None,
        shuffle: bool = False,
    ):
        """Initialise the BatchLoader."""
        self.batch_size = batch_size

        self.X = X
        self.target = target if target is not None else X.copy()

        if shuffle:
            indices = np.arange(len(X))
            np.random.seed(seed)
            np.random.shuffle(indices)

            self.X = X.iloc[indices]
            self.target = self.target.iloc[indices]

    def __len__(self):
        """Return the number of batches."""
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        """Return the batch at the given index."""
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch_size.
        high = min(low + self.batch_size, len(self.X))

        batch_x = self.X[low:high]
        if self.target is None:
            return batch_x

        batch_target = self.target[low:high]

        # A Sequence should apparently return a tuple of NumPy arrays, as DataFrames
        # cause errors in the fit() method.
        return batch_x.to_numpy(), batch_target.to_numpy()


@dataclass
class Dataset:
    """Dataset class."""

    X: pd.DataFrame
    y: pd.Series
    m: pd.DataFrame

    key: str | None = None

    _default_target: list[str] = field(init=True, default_factory=list)

    # Class attributes
    _stratify_column: ClassVar[str]
    _stratify_df: ClassVar[str] = "m"

    def __post_init__(self):
        """Check the validity of the stratify options.

        The following attributes must be set by the child class:
        * `_stratify_column`: the column to use for stratification.

        """
        if not hasattr(self, "_stratify_column") or not self._stratify_column:
            raise ValueError("Stratify column not set.")
        if self._stratify_column and self._stratify_column not in getattr(
            self, self._stratify_df
        ):
            raise ValueError(
                f"Stratify column {self._stratify_column} not found in DataFrame "
                f"{self._stratify_df}"
            )

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.X)

    def __eq__(self, other):
        """Compare two datasets."""
        if not isinstance(other, Dataset):
            return False
        return (
            self.X.equals(other.X) and self.y.equals(other.y) and self.m.equals(other.m)
        )

    def __add__(self, other: "Dataset") -> "Dataset":
        """Concatenate two datasets."""
        return self.__class__(
            pd.concat([self.X, other.X], axis=0),
            pd.concat([self.y, other.y], axis=0),
            pd.concat([self.m, other.m], axis=0),
        )

    def __getitem__(
        self, key: int | slice | list | pd.Series | np.ndarray
    ) -> "Dataset":
        """Return the given slice of X, y and m.

        Parameters
        ----------
        key : int | slice | list
            Index or slice to return.

        Returns
        -------
        Dataset
            Dataset containing the given slice.
        """
        assert (
            isinstance(key, int)
            or isinstance(key, slice)
            or isinstance(key, list)
            or isinstance(key, pd.Series)
            or isinstance(key, np.ndarray)
        ), f"Invalid key type '{type(key)}' for Dataset.__getitem__"
        return self.__class__(self.X[key], self.y[key], self.m[key])

    def to_sequence(
        self,
        batch_size: int,
        target: int | None = None,
        *,
        seed: int,
        shuffle: bool = False,
    ) -> BatchLoader:
        """Convert the dataset to a BatchLoader object.

        Parameters
        ----------
        batch_size : int
            Size of the batches.
        target : int, optional
            Target to use for the batches, defaults to None. 0 for X, 1 for y, 2 for m.

        Returns
        -------
        BatchLoader
            The dataset as a batch sequence that can be processed by the Keras API.

        """
        if target is None:
            return BatchLoader(batch_size, self.X, seed=seed, shuffle=shuffle)
        if 0 <= target <= 2:
            return BatchLoader(
                batch_size,
                self.X,
                target=self.to_tuple()[target],
                seed=seed,
                shuffle=shuffle,
            )
        raise IndexError("If not None, parameter `target` must be in [0, 2]")

    def to_tuple(self):
        """Return the dataset as a tuple.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
            Tuple of the dataset.
        """
        return self.X, self.y, self.m

    def split(
        self,
        at: float,
        seed: int,
        stratify: Optional[pd.Series] = None,
        drop: bool = True,
    ) -> Tuple["Dataset", "Dataset"]:
        """Split the dataset into a training and a test set.

        This function uses `train_test_split` from scikit-learn to split the dataset
        into two parts, using the given ratio. If `stratify` is not None, it will be
        used to stratify the split (ie. the proportion of the classes will be
        preserved). Note that using stratify requires all classes to contain at least 2
        samples. If a class contains only one sample, it will be droped, unless `drop`
        is set to False, in which case an exception will be raised.

        Parameters
        ----------
        at : float
            Ratio where to split the dataset. Must be in [0, 1]. The first dataset will
            contain `at`% of the samples, the second one will contain the remaining.
        seed : int, optional
            Seed for the random number generator, by default None.
        stratify : pd.Series, optional
            Series to use for stratification, by default None.

        Returns
        -------
        Tuple[Dataset, Dataset]
            Tuple of the training and test sets.
        """
        if stratify is not None and any(c < 2 for c in stratify.value_counts().values):
            if not drop:
                raise ValueError(
                    "Stratification requires all classes to contain at least 2 samples."
                )
            drop_classes = stratify.value_counts()[stratify.value_counts() < 2].index
            idx_to_drop = stratify[stratify.isin(drop_classes)].index
            self.drop(idx_to_drop)
            stratify = stratify.drop(idx_to_drop)

        X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
            *self.to_tuple(),
            train_size=at,
            random_state=seed,
            stratify=np.array(stratify) if stratify is not None else None,
        )

        train_d = deepcopy(self.__dict__)
        train_d["X"], train_d["y"], train_d["m"] = X_train, y_train, m_train
        test_d = deepcopy(self.__dict__)
        test_d["X"], test_d["y"], test_d["m"] = X_test, y_test, m_test

        return (self.__class__(**train_d), self.__class__(**test_d))

    def copy(self) -> "Dataset":
        """Return a copy of the dataset."""
        return self.__class__(
            **deepcopy(self.__dict__),
        )

    def shuffle(self, seed: int):
        """Shuffle the dataset."""
        indices = np.arange(len(self.X))
        np.random.seed(seed)
        np.random.shuffle(indices)

        self.X = self.X.iloc[indices]
        self.y = self.y.iloc[indices]
        self.m = self.m.iloc[indices]

    def drop(self, indices: list[int] | pd.Index):
        """Drop the given indices from the dataset."""
        self.X = self.X.drop(indices)
        self.y = self.y.drop(indices)
        self.m = self.m.drop(indices)

    def poison(
        self,
        ratio: float,
        op: PoisonOp,
        *,
        seed: int,
        target_classes: Optional[List[str]] = None,
    ) -> int:
        """Increase or decrease the proportion of poisoned samples in the dataset.

        This function MUST be implemented by the child class to allow poisoning.

        Parameters
        ----------
        n: int
            Number of samples to poison in the target. If `target` is None, the whole
            dataset is poisoned.
        op: PoisonOp
            Poisoning operation to apply. Either PoisonOp.INC or PoisonOp.DEC.
        target_classes: Optional[List[str]]
            List of classes to poison. If None, all classes are poisoned, including
            benign samples.
        seed: Optional[int]
            Seed for reproducibility.

        Returns
        -------
        int
            The number of samples that have been modified.

        Raises
        ------
        NotImplementedError
            If the function is not implemented by the child class.
        """
        raise NotImplementedError(
            f"{self.__class__}.poison(): function not implemented."
        )

    @property
    def default_target(self) -> list[str] | None:
        """Return the default target for poisoning."""
        if len(self._default_target) == 0:
            raise ValueError("No default target set for this dataset.")
        return self._default_target

    @property
    def stats(self) -> dict[str, int]:
        """Return the data statistics the dataset."""
        return (
            cast(pd.Series, getattr(self, self._stratify_df)[self._stratify_column])
            .value_counts()
            .to_dict()
        )


@ray.remote
class DatasetHandle:
    """Dataset holder to store datasets in the Ray object store.

    This class is used to store stateful datasets in the Ray object store. It is used to
    preserve the state of the dataset between the different steps of the pipeline, as
    Flower clients (up tp 1.5.0 at least) are ephemeral and do not preserve their state.

    The class provides almost the same public API as `dict`, and can be used as such,
    except for magic methods. Datasets are stored using the dict API.

    Examples
    --------
    >>> train, test = load_data("cifar10")
    >>> ray.init()
    >>> handle = DatasetHandle.remote({"train": train, "test": test})
    >>> handle.get.remote("train")
    Dataset(...)
    >>> handle["train"]
    TypeError: 'DatasetHandle' object is not subscriptable
    >>> handle.keys.remote()
    dict_keys(['train', 'test'])
    >>> train, test = load_data("cifar100")
    >>> handle.update.remote({"train": train, "test": test})

    Attributes
    ----------
    _dict : dict
        Dictionary containing the datasets.
    """

    _dict: dict = {}

    def __init__(self, *args, **kwargs):
        """Initialize the DatasetHandle."""
        self._dict = dict(*args, **kwargs)

    def clear(self):
        """Clear the inner dictionary."""
        self._dict.clear()

    def copy(self):
        """Return a copy of the inner dictionary."""
        return self.__class__(self._dict.copy())

    def get(self, *args, **kwargs):
        """Return the value for the given key."""
        return self._dict.get(*args, **kwargs)

    def items(self):
        """Return the items of the inner dictionary."""
        return self._dict.items()

    def keys(self):
        """Return the keys of the inner dictionary."""
        return self._dict.keys()

    def pop(self, *args, **kwargs):
        """Pop the given key from the inner dictionary."""
        return self._dict.pop(*args, **kwargs)

    def popitem(self):
        """Pop an item from the inner dictionary."""
        return self._dict.popitem()

    def update(self, *args, **kwargs):
        """Update the inner dictionary."""
        return self._dict.update(*args, **kwargs)

    def values(self):
        """Return the values of the inner dictionary."""
        return self._dict.values()

    def poison(self, key: Hashable, *args, **kwargs) -> int:
        """Poison the held dataset."""
        return self._dict[key].poison(*args, **kwargs)
