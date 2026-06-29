"""Load EEG Motor Movement/Imagery EDF runs and group them by experiment.

The PhysioNet dataset stores one folder per subject (``S001`` ... ``S109``) and
fourteen EDF runs per subject. Runs 1-2 are baselines; runs 3-14 are motor
execution or imagery tasks repeated three times per task family.
"""

from enum import Enum
from pathlib import Path
import mne
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tpv.config import data_path


class ExperimentType(Enum):
    """Task families used by the EEG Motor Movement/Imagery dataset."""

    BASELINE_EYES_OPEN = (1,)
    BASELINE_EYES_CLOSED = (2,)
    EXECUTION_LEFT_RIGHT = (3, 7, 11)
    IMAGERY_LEFT_RIGHT = (4, 8, 12)
    EXECUTION_HANDS_FEET = (5, 9, 13)
    IMAGERY_HANDS_FEET = (6, 10, 14)


Run = mne.io.BaseRaw
Experiments = dict[ExperimentType, list[Run]]


def empty_experiments() -> Experiments:
    """Return an empty run list for every known experiment type."""

    return {experiment_type: [] for experiment_type in ExperimentType}


def get_experiment_type(run_id: int) -> ExperimentType:
    """Return the experiment family containing a PhysioNet run id.

    Args:
        run_id: PhysioNet run number from 1 to 14.

    Returns:
        Experiment family for the requested run.

    Raises:
        ValueError: If ``run_id`` is outside the PhysioNet run layout.
    """

    for experiment_type in ExperimentType:
        if run_id in experiment_type.value:
            return experiment_type
    raise ValueError(f"Unknown run_id {run_id}; expected a value from 1 to 14")


class Subject(BaseModel):
    """All loaded EEG runs for one subject, grouped by experiment family.

    Args:
        id: Subject identifier with or without the leading ``S`` prefix.
        raw_dir: Directory containing subject folders.
        experiments: Optional preloaded runs, useful for tests and notebooks.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(description="Subject identifier formatted as S001")
    raw_dir: Path = Field(
        default=data_path, description="Directory containing raw EDF folders"
    )
    experiments: Experiments = Field(default_factory=empty_experiments)

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: str | int) -> str:
        """Normalize subject ids to PhysioNet's ``S001`` folder format.

        Args:
            value: Integer subject id, numeric string, or already-prefixed id.

        Returns:
            Subject id formatted as ``S001``.

        Raises:
            ValueError: If the identifier cannot represent a PhysioNet subject.
        """

        if isinstance(value, int):
            subject_number = value
        else:
            text = str(value).strip().upper()
            subject_number = int(text[1:] if text.startswith("S") else text)
        if not 1 <= subject_number <= 109:
            raise ValueError("subject id must be between 1 and 109")
        return f"S{subject_number:03d}"

    def model_post_init(self, __context: object) -> None:
        """Load all runs when no experiments were provided.

        Args:
            __context: Pydantic post-init context, unused.
        """

        if any(self.experiments.values()):
            return
        for run_id in range(1, 15):
            experiment_type = get_experiment_type(run_id)
            self.experiments[experiment_type].append(
                load_subject_run(self.id, run_id, self.raw_dir)
            )

    def runs(self, experiment_type: ExperimentType | None = None) -> list[Run]:
        """Return runs for one experiment type, or every loaded run.

        Args:
            experiment_type: Optional family filter.

        Returns:
            List of MNE raw runs in dataset order.
        """

        if experiment_type is not None:
            return self.experiments[experiment_type]
        return [run for runs in self.experiments.values() for run in runs]

    def describe(self) -> dict[str, int]:
        """Count loaded runs by experiment family.

        Returns:
            Mapping from experiment family name to number of loaded runs.
        """

        return {
            experiment_type.name: len(runs)
            for experiment_type, runs in self.experiments.items()
        }

def normalize_subject_id(subject_id: str | int) -> str:
    """Normalize a subject id to PhysioNet's ``S001`` folder format.

    Args:
        subject_id: Integer subject id, numeric string, or ``S``-prefixed id.

    Returns:
        Subject id formatted as ``S001``.

    Raises:
        ValueError: If the identifier cannot represent a PhysioNet subject.
    """

    if isinstance(subject_id, int):
        subject_number = subject_id
    else:
        text = str(subject_id).strip().upper()
        subject_number = int(text[1:] if text.startswith("S") else text)
    if not 1 <= subject_number <= 109:
        raise ValueError("subject id must be between 1 and 109")
    return f"S{subject_number:03d}"


def subject_run_path(
    subject_id: str | int, run_id: int, raw_dir: Path = data_path
) -> Path:
    """Build the EDF path for one subject/run pair.

    Args:
        subject_id: Integer subject id, numeric string, or ``S``-prefixed id.
        run_id: PhysioNet run number from 1 to 14.
        raw_dir: Directory containing subject folders.

    Returns:
        Path to the requested EDF file.

    Raises:
        ValueError: If ``run_id`` is invalid.
    """

    get_experiment_type(run_id)
    subject = normalize_subject_id(subject_id)
    return Path(raw_dir) / subject / f"{subject}R{run_id:02d}.edf"


def load_subject_run(
    subject_id: str | int, run_id: int, raw_dir: Path = data_path
) -> Run:
    """Load one EDF run with MNE.

    Args:
        subject_id: Integer subject id, numeric string, or ``S``-prefixed id.
        run_id: PhysioNet run number from 1 to 14.
        raw_dir: Directory containing subject folders.

    Returns:
        Preloaded MNE raw object.

    Raises:
        FileNotFoundError: If the EDF file is not available locally.
        ValueError: If ``run_id`` is invalid.
    """

    edf_path = subject_run_path(subject_id, run_id, raw_dir)
    if not edf_path.exists():
        raise FileNotFoundError(f"Missing EDF file: {edf_path}")
    return mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
