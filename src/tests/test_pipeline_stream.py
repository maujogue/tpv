"""Tests for the treatment pipeline and playback prediction."""

import time
import mne

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from tpv.dataset import motion_epochs_to_arrays
from tpv.pipeline import CSPTransformer, PipelineConfig, build_treatment_pipeline
from tpv.stream import playback_epochs, predict_stream


class SlowClassifier(BaseEstimator, ClassifierMixin):
    """Classifier that sleeps to exercise the streaming deadline."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SlowClassifier":
        """Return the fitted classifier.

        Args:
            X: Feature matrix.
            y: Labels.

        Returns:
            The classifier itself.
        """

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Sleep and return a dummy label.

        Args:
            X: Feature matrix.

        Returns:
            One zero label per input row.
        """

        time.sleep(0.02)
        return np.zeros(X.shape[0], dtype=int)


def make_separable_epochs() -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic two-class EEG epochs for tests."""

    rng = np.random.default_rng(42)
    times = np.linspace(0.0, 1.0, 80, endpoint=False)
    epochs: list[np.ndarray] = []
    labels: list[int] = []
    for _ in range(12):
        signal = np.vstack(
            [
                3.0 * np.sin(2 * np.pi * 10 * times),
                0.4 * np.sin(2 * np.pi * 12 * times),
                0.2 * np.sin(2 * np.pi * 14 * times),
                0.2 * np.sin(2 * np.pi * 16 * times),
            ]
        )
        epochs.append(signal + 0.01 * rng.standard_normal(signal.shape))
        labels.append(1)
    for _ in range(12):
        signal = np.vstack(
            [
                0.4 * np.sin(2 * np.pi * 10 * times),
                3.0 * np.sin(2 * np.pi * 12 * times),
                0.2 * np.sin(2 * np.pi * 14 * times),
                0.2 * np.sin(2 * np.pi * 16 * times),
            ]
        )
        epochs.append(signal + 0.01 * rng.standard_normal(signal.shape))
        labels.append(2)
    return np.asarray(epochs), np.asarray(labels)


def make_epochs_with_rest() -> mne.Epochs:
    """Create epochs containing rest and two motion annotations."""

    sampling_frequency = 100.0
    data = np.zeros((2, 400))
    info = mne.create_info(["C3", "C4"], sampling_frequency, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 2.0, 3.0],
            duration=[0.0, 0.0, 0.0],
            description=["T0", "T1", "T2"],
        )
    )
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    return mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.0,
        tmax=0.1,
        baseline=None,
        preload=True,
        verbose=False,
    )


def test_motion_epochs_to_arrays_drops_rest_events() -> None:
    """Training arrays should keep only T1 and T2 motion chunks."""

    X, y = motion_epochs_to_arrays([make_epochs_with_rest()])

    assert X.shape[0] == 2
    assert set(y.tolist()) == {2, 3}


def test_csp_transformer_outputs_configured_feature_count() -> None:
    """CSP should reduce epochs to the configured number of features."""

    X, y = make_separable_epochs()
    features = CSPTransformer(n_components=4).fit_transform(X, y)

    assert features.shape == (24, 4)
    assert np.isfinite(features).all()


def test_treatment_pipeline_learns_separable_epochs() -> None:
    """The sklearn pipeline should combine CSP and classifier end to end."""

    X, y = make_separable_epochs()
    pipeline = build_treatment_pipeline(PipelineConfig(n_components=4))

    pipeline.fit(X, y)

    assert np.array_equal(pipeline.predict(X), y)


def test_predict_stream_returns_one_result_per_chunk() -> None:
    """Playback prediction should preserve chunk order and labels."""

    X, y = make_separable_epochs()
    pipeline = build_treatment_pipeline(PipelineConfig(n_components=4)).fit(X, y)

    predictions = predict_stream(
        pipeline, playback_epochs(X[:3]), max_latency_seconds=2.0
    )

    assert [prediction.index for prediction in predictions] == [0, 1, 2]
    assert [prediction.label for prediction in predictions] == [1, 1, 1]
    assert all(prediction.latency_seconds < 2.0 for prediction in predictions)


def test_predict_stream_rejects_slow_predictions() -> None:
    """Streaming must fail when a chunk misses the two-second contract."""

    X, _ = make_separable_epochs()

    with pytest.raises(TimeoutError, match="chunk 0"):
        predict_stream(
            SlowClassifier(), playback_epochs(X[:1]), max_latency_seconds=0.001
        )


def test_pipeline_config_requires_even_csp_components() -> None:
    """CSP components are selected symmetrically from both classes."""

    with pytest.raises(ValueError, match="n_components"):
        PipelineConfig(n_components=3)
