"""Playback helpers that simulate an online EEG prediction stream."""

from collections.abc import Iterator
from time import perf_counter, sleep
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


class EpochPredictor(Protocol):
    """Minimal interface needed for streamed epoch prediction."""

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int_]:
        """Predict one label per epoch in ``X``."""


class StreamPrediction(BaseModel):
    """Prediction result for one streamed EEG chunk.

    Args:
        index: Chunk index in playback order.
        label: Predicted class label.
        latency_seconds: Time spent inside the processing pipeline.
    """

    index: int
    label: int
    latency_seconds: float = Field(ge=0.0)


def playback_epochs(
    epochs_data: NDArray[np.float64], interval_seconds: float = 0.0
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """Yield epochs one by one to simulate a data stream.

    Args:
        epochs_data: EEG chunks with shape ``(n_epochs, n_channels, n_times)``.
        interval_seconds: Optional delay between chunks.

    Yields:
        Pairs of ``(index, chunk)`` in playback order.

    Raises:
        ValueError: If ``epochs_data`` is not three-dimensional.
    """

    chunks = np.asarray(epochs_data, dtype=np.float64)
    if chunks.ndim != 3:
        raise ValueError("epochs_data must have shape (n_epochs, n_channels, n_times)")
    for index, chunk in enumerate(chunks):
        if interval_seconds > 0.0 and index > 0:
            sleep(interval_seconds)
        yield index, chunk


def predict_stream(
    pipeline: EpochPredictor,
    stream: Iterator[tuple[int, NDArray[np.float64]]],
    max_latency_seconds: float = 2.0,
) -> Iterator[StreamPrediction]:
    """Predict labels for streamed chunks within a latency budget.

    Args:
        pipeline: Fitted sklearn pipeline accepting one epoch batch at a time.
        stream: Iterator producing ``(index, chunk)`` pairs.
        max_latency_seconds: Maximum allowed processing time per chunk.

    Yields:
        Prediction results in stream order.

    Raises:
        TimeoutError: If one chunk takes longer than ``max_latency_seconds``.
    """

    for index, chunk in stream:
        started_at = perf_counter()
        label = int(pipeline.predict(chunk[np.newaxis, :, :])[0])
        latency = perf_counter() - started_at
        if latency > max_latency_seconds:
            raise TimeoutError(
                f"prediction for chunk {index} took {latency:.3f}s, "
                f"above the {max_latency_seconds:.3f}s limit"
            )
        yield StreamPrediction(index=index, label=label, latency_seconds=latency)
