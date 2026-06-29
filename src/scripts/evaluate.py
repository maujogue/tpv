"""Evaluate mean EEG classification accuracy across subjects."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from tpv.channels import channels_for_experiment, pick_channels
from tpv.dataset import motion_epochs_to_arrays
from tpv.epoching import EpochConfig, make_epochs_by_run
from tpv.loader import ExperimentType, Subject
from tpv.pipeline import PipelineConfig, build_treatment_pipeline
from tpv.preprocessing import FilterConfig, filter_runs
from tpv.scoring import SplitConfig, evaluate_pipeline

EvaluationSetup = ExperimentType | str

EXECUTION_VS_IMAGERY = "execution_vs_imagery"
MOVEMENT_TARGET_FAMILY = "movement_target_family"
MOTION_EXPERIMENTS: tuple[EvaluationSetup, ...] = (
    ExperimentType.EXECUTION_LEFT_RIGHT,
    ExperimentType.IMAGERY_LEFT_RIGHT,
    ExperimentType.EXECUTION_HANDS_FEET,
    ExperimentType.IMAGERY_HANDS_FEET,
    EXECUTION_VS_IMAGERY,
    MOVEMENT_TARGET_FAMILY,
)


def setup_name(setup: EvaluationSetup) -> str:
    """Return the stable CLI/output name for one evaluation setup.

    Args:
        setup: Single PhysioNet experiment or combined evaluation setup.

    Returns:
        Lowercase setup name.
    """

    if isinstance(setup, ExperimentType):
        return setup.name.lower()
    return setup


def parse_experiment(value: str) -> EvaluationSetup:
    """Parse an experiment name for the evaluation CLI.

    Args:
        value: Case-insensitive ``ExperimentType`` member name.

    Returns:
        Matching experiment type.

    Raises:
        argparse.ArgumentTypeError: If the name is unknown.
    """

    normalized = value.lower()
    for setup in MOTION_EXPERIMENTS:
        if normalized == setup_name(setup):
            return setup
    choices = ", ".join(setup_name(experiment) for experiment in MOTION_EXPERIMENTS)
    raise argparse.ArgumentTypeError(f"expected one of: {choices}")


def build_parser() -> argparse.ArgumentParser:
    """Create the aggregate evaluation command parser.

    Returns:
        Configured argparse parser.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate TPV mean held-out accuracy across subjects"
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=list(range(1, 110)),
        help="Subject ids to evaluate; defaults to 1..109",
    )
    parser.add_argument(
        "--experiments",
        type=parse_experiment,
        nargs="+",
        default=list(MOTION_EXPERIMENTS),
        help="Motion experiment families to evaluate",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=None, help="Directory containing EDF data"
    )
    parser.add_argument(
        "--n-components", type=int, default=4, help="Number of CSP filters"
    )
    parser.add_argument(
        "--all-channels",
        action="store_true",
        help="Keep all 64 channels instead of the per-experiment sensorimotor strip",
    )
    parser.add_argument(
        "--l-freq", type=float, default=7.0, help="Lower band-pass cutoff"
    )
    parser.add_argument(
        "--h-freq", type=float, default=30.0, help="Upper band-pass cutoff"
    )
    parser.add_argument(
        "--tmin", type=float, default=0.5, help="Epoch start in seconds"
    )
    parser.add_argument("--tmax", type=float, default=2.5, help="Epoch end in seconds")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Held-out test set ratio"
    )
    parser.add_argument(
        "--validation-size", type=float, default=0.2, help="Validation set ratio"
    )
    parser.add_argument(
        "--cv-splits", type=int, default=5, help="Cross-validation fold count"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Reproducible split seed"
    )
    return parser


def arrays_for_setup(
    subject: Subject,
    setup: EvaluationSetup,
    filter_config: FilterConfig,
    epoch_config: EpochConfig,
    use_motor_strip: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build EEG chunks and labels for one evaluation setup.

    Single-family motor experiments are restricted to the sensorimotor strip;
    cross-task setups keep the full montage. ``use_motor_strip=False`` keeps the
    full montage everywhere for ablation.

    Args:
        subject: Loaded PhysioNet subject.
        setup: Single experiment or combined binary setup.
        filter_config: Band-pass filter configuration.
        epoch_config: Epoch extraction configuration.
        use_motor_strip: Apply per-experiment sensorimotor channel selection.

    Returns:
        Tuple of ``(X, y)`` arrays ready for sklearn evaluation.
    """

    if isinstance(setup, ExperimentType):
        channels = channels_for_experiment(setup) if use_motor_strip else None
        runs = pick_channels(subject.runs(setup), channels)
        filtered_runs = filter_runs(runs, filter_config)
        return motion_epochs_to_arrays(make_epochs_by_run(filtered_runs, epoch_config))

    if setup == EXECUTION_VS_IMAGERY:
        groups = (
            (
                1,
                (
                    ExperimentType.EXECUTION_LEFT_RIGHT,
                    ExperimentType.EXECUTION_HANDS_FEET,
                ),
            ),
            (
                2,
                (
                    ExperimentType.IMAGERY_LEFT_RIGHT,
                    ExperimentType.IMAGERY_HANDS_FEET,
                ),
            ),
        )
    elif setup == MOVEMENT_TARGET_FAMILY:
        groups = (
            (
                1,
                (
                    ExperimentType.EXECUTION_LEFT_RIGHT,
                    ExperimentType.IMAGERY_LEFT_RIGHT,
                ),
            ),
            (
                2,
                (
                    ExperimentType.EXECUTION_HANDS_FEET,
                    ExperimentType.IMAGERY_HANDS_FEET,
                ),
            ),
        )
    else:
        raise ValueError(f"unknown evaluation setup: {setup}")

    chunks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for label, experiments in groups:
        for experiment in experiments:
            filtered_runs = filter_runs(subject.runs(experiment), filter_config)
            epochs_by_run = make_epochs_by_run(filtered_runs, epoch_config)
            X, _ = motion_epochs_to_arrays(epochs_by_run)
            chunks.append(X)
            labels.append(np.full(X.shape[0], label, dtype=int))
    return np.concatenate(chunks), np.concatenate(labels)


def main(argv: list[str] | None = None) -> int:
    """Evaluate held-out test accuracy for each requested subject and experiment.

    Args:
        argv: Optional command line arguments.

    Returns:
        Process exit code.
    """

    args = build_parser().parse_args(argv)
    pipeline_config = PipelineConfig(n_components=args.n_components)
    split_config = SplitConfig(
        train_size=1.0 - args.validation_size - args.test_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        n_splits=args.cv_splits,
        random_state=args.random_state,
    )
    filter_config = FilterConfig(l_freq=args.l_freq, h_freq=args.h_freq)
    epoch_config = EpochConfig(tmin=args.tmin, tmax=args.tmax)

    accuracies_by_experiment: dict[EvaluationSetup, list[float]] = defaultdict(list)
    for subject_id in args.subjects:
        subject = (
            Subject(id=subject_id, raw_dir=args.raw_dir)
            if args.raw_dir
            else Subject(id=subject_id)
        )
        for experiment in args.experiments:
            X, y = arrays_for_setup(
                subject,
                experiment,
                filter_config,
                epoch_config,
                use_motor_strip=not args.all_channels,
            )
            pipeline = build_treatment_pipeline(pipeline_config)
            _, scores, _ = evaluate_pipeline(pipeline, X, y, split_config)
            accuracies_by_experiment[experiment].append(scores.test_accuracy)
            print(
                f"{setup_name(experiment)}: subject {subject_id:03d}: "
                f"accuracy = {scores.test_accuracy:.4f}"
            )

    all_accuracies = [
        accuracy
        for accuracies in accuracies_by_experiment.values()
        for accuracy in accuracies
    ]
    print(
        f"Mean accuracy of {len(args.experiments)} experiments for "
        f"{len(args.subjects)} subjects:"
    )
    for experiment in args.experiments:
        experiment_accuracies = accuracies_by_experiment[experiment]
        print(
            f"{setup_name(experiment)}: accuracy = "
            f"{float(np.mean(experiment_accuracies)):.4f}"
        )
    print(f"Mean accuracy: {float(np.mean(all_accuracies)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
