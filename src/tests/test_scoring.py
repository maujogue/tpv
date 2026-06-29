"""Tests for train, validation, and test evaluation."""

import numpy as np
import pytest

from tpv.pipeline import PipelineConfig, build_treatment_pipeline
from tpv.scoring import SplitConfig, evaluate_pipeline


def make_separable_epochs() -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic two-class EEG epochs for scoring tests."""

    rng = np.random.default_rng(7)
    times = np.linspace(0.0, 1.0, 80, endpoint=False)
    epochs: list[np.ndarray] = []
    labels: list[int] = []
    for _ in range(20):
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
    for _ in range(20):
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


def test_evaluate_pipeline_reports_validation_and_test_scores() -> None:
    """Evaluation should score the whole pipeline without training on test epochs."""

    X, y = make_separable_epochs()
    pipeline = build_treatment_pipeline(PipelineConfig(n_components=4))

    fitted_pipeline, scores, predictions = evaluate_pipeline(
        pipeline, X, y, SplitConfig(n_splits=4, random_state=3)
    )

    assert len(scores.cross_val_scores) == 4
    assert scores.n_train == 24
    assert scores.n_validation == 8
    assert scores.n_test == 8
    assert scores.validation_accuracy == 1.0
    assert scores.test_accuracy == 1.0
    assert len(predictions) == scores.n_test
    assert all(prediction.equal for prediction in predictions)
    assert np.array_equal(fitted_pipeline.predict(X), y)


def test_split_config_requires_complete_dataset_ratios() -> None:
    """Train, validation, and test ratios must cover each epoch once."""

    with pytest.raises(ValueError, match="must sum to 1.0"):
        SplitConfig(train_size=0.5, validation_size=0.2, test_size=0.2)
