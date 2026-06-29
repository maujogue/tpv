"""Tests for PhysioNet run loading helpers."""

from pathlib import Path

import pytest

from tpv.loader import (
    ExperimentType,
    get_experiment_type,
    normalize_subject_id,
    subject_run_path,
)


def test_get_experiment_type_maps_repeated_task_runs() -> None:
    """Runs 4, 8, and 12 should be the same imagery task family."""

    assert get_experiment_type(4) is ExperimentType.IMAGERY_LEFT_RIGHT
    assert get_experiment_type(8) is ExperimentType.IMAGERY_LEFT_RIGHT
    assert get_experiment_type(12) is ExperimentType.IMAGERY_LEFT_RIGHT


def test_get_experiment_type_rejects_unknown_run() -> None:
    """Run ids outside 1-14 should fail fast."""

    with pytest.raises(ValueError, match="Unknown run_id"):
        get_experiment_type(15)


def test_subject_run_path_uses_physionet_names() -> None:
    """Subject and run ids should match EDF file naming."""

    assert subject_run_path(1, 4, Path("raw")) == Path("raw/S001/S001R04.edf")
    assert subject_run_path("S109", 14, Path("raw")) == Path("raw/S109/S109R14.edf")


def test_normalize_subject_id_rejects_out_of_range_values() -> None:
    """Only PhysioNet subjects 1-109 are valid."""

    with pytest.raises(ValueError, match="between 1 and 109"):
        normalize_subject_id(110)
