"""Train, validation, and test scoring for EEG pipelines.

This module keeps the evaluation split explicit: tune on train/validation data,
then report accuracy once on held-out test epochs that the fitted model never saw.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


class SplitConfig(BaseModel):
    """Configuration for train, validation, and test evaluation."""

    train_size: float = Field(default=0.6, gt=0.0, lt=1.0)
    validation_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    n_splits: int = Field(default=5, ge=2)
    random_state: int = Field(default=42)

    @model_validator(mode="after")
    def validate_total_size(self) -> "SplitConfig":
        """Validate that split ratios cover the whole dataset once."""

        total = self.train_size + self.validation_size + self.test_size
        if not np.isclose(total, 1.0):
            raise ValueError("train, validation, and test sizes must sum to 1.0")
        return self


class EvaluationResult(BaseModel):
    """Scores produced by a train/validation/test evaluation."""

    cross_val_scores: list[float]
    cross_val_mean: float
    validation_accuracy: float
    test_accuracy: float
    n_train: int
    n_validation: int
    n_test: int


class PredictionResult(BaseModel):
    """Prediction compared with its expected label for one test epoch."""

    index: int
    prediction: int
    truth: int
    equal: bool


def evaluate_pipeline(
    pipeline: Pipeline,
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    config: SplitConfig | None = None,
) -> tuple[Pipeline, EvaluationResult, list[PredictionResult]]:
    """Evaluate a processing pipeline with explicit held-out data.

    Args:
        pipeline: Unfitted sklearn pipeline containing preprocessing and classifier steps.
        X: EEG epochs with shape ``(n_epochs, n_channels, n_times)``.
        y: Class labels with shape ``(n_epochs,)``.
        config: Split and cross-validation configuration.

    Returns:
        Fitted pipeline, aggregate scores, and held-out test predictions.

    Raises:
        ValueError: If the arrays do not contain matching epoch and label counts.
    """

    epochs = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y)
    if epochs.ndim != 3:
        raise ValueError("X must have shape (n_epochs, n_channels, n_times)")
    if labels.shape != (epochs.shape[0],):
        raise ValueError("y must contain one label per epoch")

    split_config = config or SplitConfig()
    train_validation_size = split_config.train_size + split_config.validation_size
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        epochs,
        labels,
        train_size=train_validation_size,
        stratify=labels,
        random_state=split_config.random_state,
    )
    validation_ratio = split_config.validation_size / train_validation_size
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=validation_ratio,
        stratify=y_train_validation,
        random_state=split_config.random_state,
    )

    cross_validator = StratifiedKFold(
        n_splits=split_config.n_splits,
        shuffle=True,
        random_state=split_config.random_state,
    )
    cross_val_scores = cross_val_score(
        clone(pipeline), X_train_validation, y_train_validation, cv=cross_validator
    )

    validation_pipeline = clone(pipeline)
    validation_pipeline.fit(X_train, y_train)
    validation_accuracy = float(
        accuracy_score(y_validation, validation_pipeline.predict(X_validation))
    )

    fitted_pipeline = clone(pipeline)
    fitted_pipeline.fit(X_train_validation, y_train_validation)
    test_predictions = fitted_pipeline.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, test_predictions))
    predictions = [
        PredictionResult(
            index=index,
            prediction=int(prediction),
            truth=int(truth),
            equal=bool(prediction == truth),
        )
        for index, (prediction, truth) in enumerate(
            zip(test_predictions, y_test, strict=True)
        )
    ]

    return (
        fitted_pipeline,
        EvaluationResult(
            cross_val_scores=[float(score) for score in cross_val_scores],
            cross_val_mean=float(np.mean(cross_val_scores)),
            validation_accuracy=validation_accuracy,
            test_accuracy=test_accuracy,
            n_train=int(y_train.shape[0]),
            n_validation=int(y_validation.shape[0]),
            n_test=int(y_test.shape[0]),
        ),
        predictions,
    )
