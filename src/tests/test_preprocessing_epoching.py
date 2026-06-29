"""Tests for the first EEG preprocessing phase."""

import mne
import numpy as np
import pytest

from tpv.epoching import EpochConfig, make_epochs
from tpv.events import extract_events
from tpv.preprocessing import FilterConfig, filter_raw


def make_raw_with_annotations() -> mne.io.RawArray:
    """Create a tiny annotated EEG recording for tests."""

    sampling_frequency = 100.0
    times = np.arange(0, 3, 1 / sampling_frequency)
    signal = np.vstack(
        [
            np.sin(2 * np.pi * 10 * times),
            np.cos(2 * np.pi * 12 * times),
        ]
    )
    info = mne.create_info(["C3", "C4"], sampling_frequency, ch_types="eeg")
    raw = mne.io.RawArray(signal, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(onset=[1.0, 2.0], duration=[0.0, 0.0], description=["T1", "T2"])
    )
    return raw


def test_filter_raw_returns_filtered_copy() -> None:
    """Filtering should not mutate the caller's raw object."""

    raw = make_raw_with_annotations()
    filtered = filter_raw(raw, FilterConfig(l_freq=7.0, h_freq=30.0))

    assert filtered is not raw
    assert filtered.info["highpass"] == pytest.approx(7.0)
    assert filtered.info["lowpass"] == pytest.approx(30.0)
    assert raw.info["highpass"] == pytest.approx(0.0)


def test_filter_config_rejects_inverted_band() -> None:
    """The band-pass lower cutoff must stay below the upper cutoff."""

    with pytest.raises(ValueError, match="l_freq"):
        FilterConfig(l_freq=30.0, h_freq=7.0)


def test_extract_events_reads_annotations() -> None:
    """Annotations should become MNE event rows and id mappings."""

    events, event_id = extract_events(make_raw_with_annotations())

    assert events.shape == (2, 3)
    assert event_id == {"T1": 1, "T2": 2}


def test_make_epochs_uses_configured_window() -> None:
    """Epoch extraction should produce one example per annotation."""

    epochs = make_epochs(make_raw_with_annotations(), EpochConfig(tmin=-0.1, tmax=0.2))

    assert len(epochs) == 2
    assert epochs.get_data(copy=False).shape[:2] == (2, 2)


def test_epoch_config_rejects_empty_window() -> None:
    """Epoch windows need positive duration."""

    with pytest.raises(ValueError, match="tmin"):
        EpochConfig(tmin=1.0, tmax=1.0)
