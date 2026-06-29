"""Create labelled training examples from continuous EEG recordings."""

import mne
from pydantic import BaseModel, Field, model_validator

from tpv.events import extract_events


class EpochConfig(BaseModel):
    """Configuration for epoch extraction around annotated events.

    Args:
        tmin: Start time relative to the event in seconds.
        tmax: End time relative to the event in seconds.
        baseline: Optional baseline correction interval.
        preload: Whether epoch data is loaded into memory immediately.
    """

    tmin: float = Field(default=0.5, description="Epoch start relative to event")
    tmax: float = Field(default=2.5, description="Epoch end relative to event")
    baseline: tuple[float, float] | None = Field(
        default=None, description="Baseline interval"
    )
    preload: bool = Field(default=True, description="Load epoch data eagerly")

    @model_validator(mode="after")
    def validate_window(self) -> "EpochConfig":
        """Validate that the epoch window has positive duration.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If ``tmin`` is not lower than ``tmax``.
        """

        if self.tmin >= self.tmax:
            raise ValueError("tmin must be lower than tmax")
        return self


def make_epochs(raw: mne.io.BaseRaw, config: EpochConfig | None = None) -> mne.Epochs:
    """Create MNE epochs from a raw run's annotations.

    Args:
        raw: Annotated raw EEG recording.
        config: Optional epoching configuration.

    Returns:
        MNE epochs containing one training example per event.
    """

    epoch_config = config or EpochConfig()
    events, event_id = extract_events(raw)
    return mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=epoch_config.tmin,
        tmax=epoch_config.tmax,
        baseline=epoch_config.baseline,
        preload=epoch_config.preload,
        verbose=False,
    )


def make_epochs_by_run(
    runs: list[mne.io.BaseRaw],
    config: EpochConfig | None = None,
) -> list[mne.Epochs]:
    """Create epochs for several raw runs.

    Args:
        runs: Annotated raw EEG recordings.
        config: Optional epoching configuration.

    Returns:
        Epoch objects in the same order as ``runs``.
    """

    return [make_epochs(run, config) for run in runs]
