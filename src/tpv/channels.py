"""Per-experiment EEG channel selection.

Single-family motor tasks decode best from the sensorimotor strip: the
left/right and hands/feet contrasts live over the motor cortex, and restricting
CSP to those channels keeps its covariance estimate stable instead of diluting
the contrast across all 64 electrodes. Cross-task setups (execution vs imagery,
movement target family) keep the full montage because they rely on broader
cortical differences that the strip would discard.
"""

import mne
from tpv.loader import ExperimentType

# FC, C, and CP rows over the sensorimotor cortex. PhysioNet pads channel names
# to four characters with trailing dots, so the strings must match exactly.
# fmt: off
SENSORIMOTOR_CHANNELS: tuple[str, ...] = (
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.",
    "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..",
    "Cp5.", "Cp3.", "Cp1.", "Cpz.", "Cp2.", "Cp4.", "Cp6.",
)
# fmt: on

# Single-family motor experiments that benefit from the sensorimotor strip.
MOTOR_STRIP_EXPERIMENTS: frozenset[ExperimentType] = frozenset(
    {
        ExperimentType.EXECUTION_LEFT_RIGHT,
        ExperimentType.IMAGERY_LEFT_RIGHT,
        ExperimentType.EXECUTION_HANDS_FEET,
        ExperimentType.IMAGERY_HANDS_FEET,
    }
)


def channels_for_experiment(experiment: ExperimentType) -> list[str] | None:
    """Return the channel subset to keep for one experiment family.

    Args:
        experiment: Single PhysioNet experiment family.

    Returns:
        Sensorimotor channel names for single-family motor tasks, or ``None``
        to keep the full montage.
    """

    if experiment in MOTOR_STRIP_EXPERIMENTS:
        return list(SENSORIMOTOR_CHANNELS)
    return None


def pick_channels(
    runs: list[mne.io.BaseRaw], channels: list[str] | None
) -> list[mne.io.BaseRaw]:
    """Return runs restricted to a channel subset, or unchanged.

    Args:
        runs: Raw EEG runs.
        channels: Channel names to keep, or ``None`` for all channels.
d
    Returns:
        Channel-restricted copies, or the original runs when no subset is given.
    """

    if not channels:
        return runs
    return [run.copy().pick(channels) for run in runs]
