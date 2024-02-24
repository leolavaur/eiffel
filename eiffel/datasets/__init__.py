"""Eiffel dataset module.

This module provides a unified interface for managing DataFrame-based datasets in
federated contexts. It also provides a poisoning API to implement data-poisoning attacks
in FL.

Along with the API, this package also contains submodules that implement actual datasets
for IDS experiments. These datasets can be used to easily setup experiments.

Dataset classes
---------------
Dataset
    A dataset class wrapping `X` and `y` DataFrames. It provides helpers to manipulate
    the data, such as splitting, partitioning, etc. It contains also a `m` DataFrame
    that holds metadata about the dataset.
BatchLoader
    A batch generator for training TensorFlow models using an iterator.
DatasetHolder
    A Ray actor class that holds a dataset.

Partitioners
------------
IIDPartitioner
    A partitioner that splits a dataset into IID partitions.
NIIDClassPartitioner
    A partitioner that splits a dataset into NIID partitions based on the class labels.
"""

import pathlib
import tempfile

from .dataset import BatchLoader, Dataset, DatasetHandle
from .partitioners import IIDPartitioner, NIIDClassPartitioner

DEFAULT_SEARCH_PATH = pathlib.Path(tempfile.gettempdir()) / "eiffel-data"

__all__ = [
    "BatchLoader",
    "Dataset",
    "DatasetHandle",
    "IIDPartitioner",
    "NIIDClassPartitioner",
    "DEFAULT_SEARCH_PATH",
]
