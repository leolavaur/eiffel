"""Dataset partitioners."""

import math
from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np

from .dataset import Dataset


class Partitioner(metaclass=ABCMeta):
    """Abstract partitioner class."""

    n_partitions: int
    partitions: list[Dataset]

    def __init__(self, *, n_partitions: int, seed: int) -> None:
        """Initialize the partitioner."""
        self.n_partitions = n_partitions
        self.partitions = []
        self.seed = seed

        np.random.seed(self.seed)

    def __len__(self) -> int:
        """Return the number of partitions."""
        return len(self.partitions)

    def __getitem__(self, idx: int | slice) -> Dataset | list[Dataset]:
        """Select a subset of partitions from the partitioner."""
        return self.partitions[idx]

    @abstractmethod
    def _partition(self, dataset: Dataset) -> None:
        """Partition the dataset.

        This function should be implemented by the subclasses.
        """
        raise NotImplementedError

    def load(self, dataset: Dataset) -> None:
        """Load and partition the dataset."""
        self.dataset = dataset
        self._partition(dataset)
        assert len(self.partitions) == self.n_partitions, (
            "Partitioner did not produce the expected number of partitions. "
            f"Expected {self.n_partitions}, got {len(self.partitions)}"
        )

    def all(self) -> list[Dataset]:
        """Return all the partitions."""
        return self.partitions

    def one(self, n: int) -> Dataset:
        """Return one partition."""
        return self.partitions[n]

    def pop(self) -> Dataset:
        """Return and remove the last partition."""
        return self.partitions.pop()


class DumbPartitioner(Partitioner):
    """Dumb partitioner class.

    The dumb partitioner does so by splitting the dataset into `n_partitions` chunks
    based on their indices. There is no guarantee that the partitions will be balanced.
    """

    def _partition(self, dataset: Dataset) -> None:
        """Partition the dataset."""
        partition_size = math.floor(len(dataset.X) / self.n_partitions)
        self.partitions = []
        for i in range(self.n_partitions):
            idx_from, idx_to = i * partition_size, (i + 1) * partition_size

            self.partitions.append(dataset[idx_from:idx_to])


class IIDPartitioner(Partitioner):
    """IID partitioner class.

    The IID partitioner ensures that the partitions are balanced using
    `train_test_split()` from scikit-learn.
    """

    class_column: str
    df_key: str

    def __init__(
        self,
        *args,
        class_column: str,
        df_key: str = "m",
        **kwargs,
    ) -> None:
        """Initialize the IID partitioner."""
        self.class_column = class_column
        self.df_key = df_key
        super().__init__(*args, **kwargs)

    def _partition(self, dataset: Dataset) -> None:
        """Partition the dataset."""
        if not hasattr(dataset, self.df_key):
            raise KeyError(
                f"Dataset does not contain a DataFrame with key {self.df_key}"
            )

        if self.class_column not in getattr(dataset, self.df_key).columns:
            raise KeyError(
                f"Dataset does not contain a column named {self.class_column}"
            )

        self.partitions = []
        n = self.n_partitions
        for _ in range(self.n_partitions):
            if n == 1:
                self.partitions.append(dataset)
                break

            ratio = 1 / n
            n -= 1

            d, rest = dataset.split(
                at=ratio,
                stratify=getattr(dataset, self.df_key)[self.class_column],
                seed=self.seed,
            )

            self.partitions.append(d)
            dataset = rest


class NIIDClassPartitioner(Partitioner):
    """NIID partitioner class.

    This partitioner will select `n_drop` classes from each partition, except for the
    ones that are identified in `preserved_classes`, and drop all the samples that
    belong to these classes. Note that droping samples means that partitions will be
    smaller than `len(dataset) / n_partitions`.
    """

    def __init__(
        self,
        class_column: str,
        preserved_classes: list[str],
        *args,
        n_drop: int = 1,
        n_keep: int = 0,
        df_key: str = "m",
        **kwargs,
    ) -> None:
        """Initialize the NIID partitioner."""
        self.class_column = class_column
        self.preserved_classes = preserved_classes
        self.n_drop = n_drop
        self.n_keep = n_keep
        self.df_key = df_key
        super().__init__(*args, **kwargs)

    def _partition(self, dataset: Dataset) -> None:
        """Partition the dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset to partition.
        n_partition : int
            Number of partitions.
        class_column : str
            Name of the column containing the class labels.
        preserved_classes : list[str]
            List of class values to preserve. Class dropping will not be applied to
            these classes.
        n_drop : int, optional
            Number of classes to drop per client, by default 1.
        df_key : str, optional
            Key of the DataFrame in the Dataset object containing the `class_column`, by
            default "m".

        Raises
        ------
        ValueError
            If the number of classes to drop is greater than the number of classes in
            the dataset minus the number of classes to preserve.
        KeyError
            If the class column is not present in the dataset, if the class column is
            not a column of the DataFrame, or if the class values are not present in the
            dataset.
        """
        if not hasattr(dataset, self.df_key):
            raise KeyError(
                f"Dataset does not contain a DataFrame with key {self.df_key}"
            )

        if self.class_column not in getattr(dataset, self.df_key).columns:
            raise KeyError(
                f"Dataset does not contain a column named {self.class_column}"
            )

        available_classes = getattr(dataset, self.df_key)[self.class_column].unique()

        # Deprecated: this check is not necessary, as the partitioner will only drop
        # classes that are present in the dataset anyway.
        #
        # if self.preserved_classes and any(
        #     c not in available_classes for c in self.preserved_classes
        # ):
        #     raise KeyError(
        #         "Dataset does not contain all the class values in"
        #         f" {self.preserved_classes}"
        #     )

        if self.n_drop > (len(available_classes) - len(self.preserved_classes)):
            raise ValueError(
                f"Cannot drop {self.n_drop} classes, only "
                f"{len(available_classes) - len(self.preserved_classes)} "
                "classes are available."
            )

        if self.n_drop > 0 and self.n_keep > 0:
            raise ValueError(
                f"Cannot use `n_drop` and `n_keep` at the same time ({self.n_drop=},"
                f" {self.n_keep=})."
            )

        dropable = [c for c in available_classes if c not in self.preserved_classes]

        self.partitions = []
        pt = IIDPartitioner(
            class_column=self.class_column,
            n_partitions=self.n_partitions,
            df_key=self.df_key,
            seed=self.seed,
        )
        pt.load(dataset)
        parts = pt.all()

        if self.n_drop > 0:
            for p in parts:
                drop = np.random.choice(dropable, self.n_drop, replace=False)
                mask = getattr(p, self.df_key)[self.class_column].isin(drop)
                p.drop(mask[mask].index)  # select only the rows in mask that are True
                self.partitions.append(p)

        elif self.n_keep > 0:
            for p in parts:
                keep = np.random.choice(
                    dropable,
                    self.n_keep,
                    replace=False,
                )
                mask = getattr(p, self.df_key)[self.class_column].isin(keep)
                p.drop(mask[~mask].index)
                self.partitions.append(p)

        else:
            # When both `n_drop` and `n_keep` are 0, the partitioner behaves like an
            # IIDPartitioner.
            self.partitions = parts
