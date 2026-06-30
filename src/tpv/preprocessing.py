"""Preprocess raw EEG signals before epoch extraction.

Motor imagery studies usually keep the mu and beta rhythms. The default 8-40 Hz
band-pass filter keeps those rhythms while removing slow drift and high-frequency noise.
"""

import mne
from pydantic import BaseModel, Field, model_validator


class FilterConfig(BaseModel):
    """Band-pass filtering configuration.

    Args:
        l_freq: Lower cutoff frequency in Hz, or ``None`` for no high-pass.
        h_freq: Upper cutoff frequency in Hz, or ``None`` for no low-pass.
    """

    l_freq: float | None = Field(
        default=8.0, ge=0.0, description="Lower cutoff frequency"
    )
    h_freq: float | None = Field(
        default=40.0, gt=0.0, description="Upper cutoff frequency"
    )

    @model_validator(mode="after")
    def validate_band(self) -> "FilterConfig":
        """Validate that the frequency band is ordered.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If both cutoffs are set and ``l_freq >= h_freq``.
        """

        if (
            self.l_freq is not None
            and self.h_freq is not None
            and self.l_freq >= self.h_freq
        ):
            raise ValueError("l_freq must be lower than h_freq")
        return self


def filter_raw(
    raw: mne.io.BaseRaw, config: FilterConfig | None = None
) -> mne.io.BaseRaw:
    """Return a filtered copy of a raw EEG run.

    Args:
        raw: MNE raw EEG recording.
        config: Optional filter configuration. Defaults to 8-40 Hz.

    Returns:
        Filtered copy of ``raw``. The input object is not modified.
    """

    filter_config = config or FilterConfig()
    filtered = raw.copy()
    filtered.filter(
        l_freq=filter_config.l_freq, h_freq=filter_config.h_freq, verbose=False
    )
    return filtered


def filter_runs(
    runs: list[mne.io.BaseRaw],
    config: FilterConfig | None = None,
) -> list[mne.io.BaseRaw]:
    """Filter several raw runs with the same band-pass settings.

    Args:
        runs: Raw EEG recordings.
        config: Optional filter configuration. Defaults to 8-40 Hz.

    Returns:
        Filtered copies in the same order as ``runs``.
    """

    return [filter_raw(run, config) for run in runs]
