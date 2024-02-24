"""Tests for partitioners.py."""

import pytest

from eiffel.datasets.nfv2 import mk_nfv2_mockset
from eiffel.datasets.partitioners import (
    DumbPartitioner,
    IIDPartitioner,
    NIIDClassPartitioner,
)

SEED = 1138


def test_partitioners():
    """Test partitioners."""
    d = mk_nfv2_mockset(100, iid=True, seed=SEED)
    classes = d.m["Attack"].unique()

    # Test0: Dumb partitioner
    pt = DumbPartitioner(n_partitions=10, seed=SEED)
    pt.load(d)
    parts = pt.all()

    assert len(parts) == 10
    assert all(len(p) == 10 for p in parts)

    # Test1: IID partitioner
    pt = IIDPartitioner(n_partitions=10, class_column="Attack", seed=SEED)
    pt.load(d)
    parts = pt.all()

    assert len(parts) == 10
    assert all(len(p) == 10 for p in parts)
    assert all(set(p.m["Attack"].unique()) == set(classes) for p in parts)

    # Test2: NIID partitioner
    pt = NIIDClassPartitioner(
        n_partitions=10, class_column="Attack", preserved_classes=["Benign"], seed=SEED
    )
    pt.load(d)
    parts = pt.all()
    assert len(parts) == 10
    assert all(set(p.m["Attack"].unique()) != set(classes) for p in parts)


if __name__ == "__main__":
    pytest.main([__file__])
