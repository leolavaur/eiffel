"""Tests for dataset/poisoning.py."""

import pandas as pd
import pytest

from eiffel.datasets.poisoning import (
    PoisonIns,
    PoisonOp,
    PoisonTask,
    parse_poisoning_selector,
)


def test_parse_poisoning_selector():
    """Test parse_poisoning_selector()."""
    assert parse_poisoning_selector("0.5", 10) == (PoisonTask(0.5), {})
    assert parse_poisoning_selector("1", 10) == (PoisonTask(1.0), {})
    assert parse_poisoning_selector("0.0+0.1[:5]", 10) == (
        PoisonTask(0.0),
        {
            1: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
            2: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
            3: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
            4: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
            5: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
        },
    )
    assert parse_poisoning_selector("0.1+0.1[:]", 3) == (
        PoisonTask(0.1),
        {
            1: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
            2: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
            3: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
        },
    )
    assert parse_poisoning_selector("0.1+0.1[1:]", 3) == (
        PoisonTask(0.1),
        {
            1: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
            2: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
            3: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
        },
    )
    assert parse_poisoning_selector("1-0.1[2:3]", 4) == (
        PoisonTask(1.0),
        {
            2: PoisonTask(fraction=0.1, operation=PoisonOp.DEC),
            3: PoisonTask(fraction=0.1, operation=PoisonOp.DEC),
        },
    )
    with pytest.raises(IndexError):
        _ = parse_poisoning_selector("1-0.1[2:6]", 4)

    assert parse_poisoning_selector("1-0.1[2:6]", 10) == (
        PoisonTask(1.0),
        {
            2: PoisonTask(fraction=0.1, operation=PoisonOp.DEC),
            3: PoisonTask(fraction=0.1, operation=PoisonOp.DEC),
            4: PoisonTask(fraction=0.1, operation=PoisonOp.DEC),
            5: PoisonTask(fraction=0.1, operation=PoisonOp.DEC),
            6: PoisonTask(fraction=0.1, operation=PoisonOp.DEC),
        },
    )
    assert parse_poisoning_selector(r"0.1+0.1{3,5}", 10) == (
        PoisonTask(0.1),
        {
            3: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
            5: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
        },
    )
    assert parse_poisoning_selector(r"0.1+0.1[2:5]-0.1[7:9]", 10) == (
        PoisonTask(0.1),
        {
            2: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
            3: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
            4: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
            5: PoisonTask(fraction=0.1, operation=PoisonOp.INC),
            7: PoisonTask(fraction=0.1, operation=PoisonOp.DEC),
            8: PoisonTask(fraction=0.1, operation=PoisonOp.DEC),
            9: PoisonTask(fraction=0.1, operation=PoisonOp.DEC),
        },
    )
    assert parse_poisoning_selector(r"0.5+0.1{1}-0.2[:1]", 10) == (
        PoisonTask(0.5),
        {
            1: PoisonTask(fraction=0.2, operation=PoisonOp.DEC),
        },
    )

    with pytest.raises(IndexError):
        _ = parse_poisoning_selector("0.5+0.1[12:4]", 1)

    with pytest.raises(ValueError):
        _ = parse_poisoning_selector("0.5+0.1[1:4:2]", 1)
        _ = parse_poisoning_selector("0.", 1)
        _ = parse_poisoning_selector("toto", 1)
