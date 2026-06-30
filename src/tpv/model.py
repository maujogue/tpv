"""Serialisable wrapper that pairs a fitted pipeline with its training metadata."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


@dataclasses.dataclass
class ModelMetadata:
    experiment: str
    channels: list[str] | None  # None means all channels
    l_freq: float
    h_freq: float
    tmin: float
    tmax: float
    subject: int
    n_components: int
    test_size: float = 0.2
    random_state: int = 42


@dataclasses.dataclass
class ModelBundle:
    pipeline: Pipeline
    metadata: ModelMetadata


def model_filename_suffix(meta: ModelMetadata) -> str:
    """Return a descriptive suffix for the saved model filename."""

    channel_tag = "allch" if meta.channels is None else f"{len(meta.channels)}ch"
    exp_tag = meta.experiment.lower()
    return f"_{exp_tag}_{channel_tag}_s{meta.subject}"
