"""Sklearn pipeline components for EEG motor imagery classification.

The pipeline consumes epoched EEG arrays with shape
``(n_epochs, n_channels, n_times)``. CSP reduces each epoch to a small set of
log-variance features, then a sklearn classifier decides which motion label the
chunk represents.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


class PipelineConfig(BaseModel):
    """Configuration for the treatment pipeline.

    Args:
        n_components: Number of CSP spatial filters kept as features.
    """

    n_components: int = Field(default=4, ge=2, description="Number of CSP filters")

    @model_validator(mode="after")
    def validate_even_components(self) -> "PipelineConfig":
        """Validate that CSP keeps symmetric filters from both classes.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If ``n_components`` is odd.
        """

        if self.n_components % 2 != 0:
            raise ValueError("n_components must be even")
        return self


class CSPTransformer(BaseEstimator, TransformerMixin):
    """Common Spatial Patterns transformer following sklearn's interface.

    CSP learns spatial filters that maximize normalized variance for one class
    while minimizing it for the other. ``transform`` returns log-variance
    features suitable for a standard sklearn classifier.
    """

    filters_: NDArray[np.float64]
    classes_: NDArray[np.int_]

    def __init__(self, n_components: int = 4) -> None:
        """Store CSP configuration.

        Args:
            n_components: Number of spatial filters to retain. Must be even.
        """

        self.n_components = n_components

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int_]) -> "CSPTransformer":
        """Learn CSP filters from labelled epochs.

        Args:
            X: EEG epochs with shape ``(n_epochs, n_channels, n_times)``.
            y: Binary class labels with shape ``(n_epochs,)``.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If inputs have incompatible shapes or non-binary labels.
        """

        epochs = self._validate_epochs(X)
        labels = np.asarray(y)
        if labels.shape != (epochs.shape[0],):
            raise ValueError("y must contain one label per epoch")
        classes = np.unique(labels)
        if classes.shape[0] != 2:
            raise ValueError("CSP requires exactly two classes")
        if self.n_components < 2 or self.n_components % 2 != 0:
            raise ValueError("n_components must be an even integer >= 2")
        if self.n_components > epochs.shape[1]:
            raise ValueError("n_components cannot exceed the number of channels")

        covariance_a = self._mean_normalized_covariance(epochs[labels == classes[0]])
        covariance_b = self._mean_normalized_covariance(epochs[labels == classes[1]])
        composite = covariance_a + covariance_b
        eigenvalues, eigenvectors = np.linalg.eig(
            np.linalg.pinv(composite) @ covariance_a
        )
        order = np.argsort(np.abs(eigenvalues.real - 0.5))[::-1]

        self.filters_ = eigenvectors.real[:, order[: self.n_components]].T
        self.classes_ = classes
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project epochs and return CSP log-variance features.

        Args:
            X: EEG epochs with shape ``(n_epochs, n_channels, n_times)``.

        Returns:
            Feature matrix with shape ``(n_epochs, n_components)``.

        Raises:
            ValueError: If ``fit`` has not been called or inputs are invalid.
        """

        if not hasattr(self, "filters_"):
            raise ValueError("CSPTransformer must be fitted before transform")
        epochs = self._validate_epochs(X)
        projected = np.asarray([self.filters_ @ epoch for epoch in epochs])
        variances = np.var(projected, axis=2)
        normalized = variances / variances.sum(axis=1, keepdims=True)
        return np.log(normalized)

    @staticmethod
    def _validate_epochs(X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Validate and cast an epoch array.

        Args:
            X: Candidate epoch array.

        Returns:
            Float epoch array.

        Raises:
            ValueError: If the array is not three-dimensional.
        """

        epochs = np.asarray(X, dtype=np.float64)
        if epochs.ndim != 3:
            raise ValueError("X must have shape (n_epochs, n_channels, n_times)")
        return epochs

    @staticmethod
    def _mean_normalized_covariance(epochs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the average trace-normalized covariance matrix.

        Args:
            epochs: Epochs from one class.

        Returns:
            Mean covariance matrix for the class.
        """

        covariances = []
        for epoch in epochs:
            covariance = epoch @ epoch.T
            covariances.append(covariance / np.trace(covariance))
        return np.mean(covariances, axis=0)


def build_treatment_pipeline(config: PipelineConfig | None = None) -> Pipeline:
    """Build the sklearn treatment pipeline required by the subject.

    Args:
        config: Optional pipeline configuration. Defaults to four CSP filters.

    Returns:
        Sklearn pipeline with CSP dimensionality reduction and LDA classification.
    """

    pipeline_config = config or PipelineConfig()
    return Pipeline(
        steps=[
            ("csp", CSPTransformer(n_components=pipeline_config.n_components)),
            ("classifier", LinearDiscriminantAnalysis()),
        ]
    )
