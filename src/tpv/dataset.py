"""Convert MNE epochs into arrays used by the treatment pipeline."""

import mne
import numpy as np
from numpy.typing import NDArray

REST_EVENT = "T0"


def motion_epochs_to_arrays(
    epochs_by_run: list[mne.Epochs],
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Return EEG chunks and labels for motion events only.

    PhysioNet motor imagery runs include ``T0`` rest events. CSP is a binary
    motion classifier here, so training and playback keep only ``T1`` and ``T2``.

    Args:
        epochs_by_run: Epoched runs containing MNE event ids.

    Returns:
        Tuple of ``(X, y)`` where ``X`` has shape
        ``(n_epochs, n_channels, n_times)`` and ``y`` contains the event ids.

    Raises:
        ValueError: If no motion epochs are available.
    """

    chunks: list[NDArray[np.float64]] = []
    labels: list[NDArray[np.int_]] = []
    for epochs in epochs_by_run:
        rest_id = epochs.event_id.get(REST_EVENT)
        run_data = epochs.get_data(copy=True)
        run_labels = epochs.events[:, 2]
        if rest_id is None:
            mask = np.ones(run_labels.shape, dtype=bool)
        else:
            mask = run_labels != rest_id
        if np.any(mask):
            chunks.append(run_data[mask])
            labels.append(run_labels[mask])

    if not chunks:
        raise ValueError("no motion epochs found; expected T1/T2 events")
    return np.concatenate(chunks), np.concatenate(labels)
