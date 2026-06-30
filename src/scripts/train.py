"""Train the EEG treatment pipeline and save it to disk."""

import argparse
import pickle
from pathlib import Path


from tpv.channels import channels_for_experiment, pick_channels
from tpv.dataset import motion_epochs_to_arrays
from tpv.epoching import EpochConfig, make_epochs_by_run
from tpv.loader import ExperimentType, Subject
from tpv.model import ModelBundle, ModelMetadata, model_filename_suffix
from tpv.pipeline import PipelineConfig, build_treatment_pipeline
from tpv.scoring import SplitConfig, evaluate_pipeline
from tpv.preprocessing import FilterConfig, filter_runs


def parse_experiment(value: str) -> ExperimentType:
    """Parse an experiment name for the training CLI.

    Args:
        value: Case-insensitive experiment enum name.

    Returns:
        Matching experiment type.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is unknown.
    """

    try:
        return ExperimentType[value.upper()]
    except KeyError as error:
        choices = ", ".join(experiment.name.lower() for experiment in ExperimentType)
        raise argparse.ArgumentTypeError(f"expected one of: {choices}") from error


def build_parser() -> argparse.ArgumentParser:
    """Create the training command parser.

    Returns:
        Configured argparse parser.
    """

    parser = argparse.ArgumentParser(description="Train a TPV sklearn EEG pipeline")
    parser.add_argument("subject", type=int, help="PhysioNet subject id, from 1 to 109")
    parser.add_argument(
        "--experiment",
        type=parse_experiment,
        default=ExperimentType.IMAGERY_LEFT_RIGHT,
        help="Experiment family to train on",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=None, help="Directory containing EDF data"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/models/tpv_pipeline.pkl")
    )
    parser.add_argument(
        "--n-components", type=int, default=4, help="Number of CSP filters"
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
        "--all-channels",
        action="store_true",
        help="Keep all 64 channels instead of the sensorimotor strip",
    )
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


def main(argv: list[str] | None = None) -> int:
    """Train and serialize the treatment pipeline.

    Args:
        argv: Optional command line arguments.

    Returns:
        Process exit code.
    """

    args = build_parser().parse_args(argv)
    subject = (
        Subject(id=args.subject, raw_dir=args.raw_dir)
        if args.raw_dir
        else Subject(id=args.subject)
    )
    channels = None if args.all_channels else channels_for_experiment(args.experiment)
    runs = pick_channels(subject.runs(args.experiment), channels)
    filtered_runs = filter_runs(
        runs, FilterConfig(l_freq=args.l_freq, h_freq=args.h_freq)
    )
    epochs_by_run = make_epochs_by_run(
        filtered_runs, EpochConfig(tmin=args.tmin, tmax=args.tmax)
    )

    X, y = motion_epochs_to_arrays(epochs_by_run)
    pipeline = build_treatment_pipeline(PipelineConfig(n_components=args.n_components))
    split_config = SplitConfig(
        train_size=1.0 - args.validation_size - args.test_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        n_splits=args.cv_splits,
        random_state=args.random_state,
    )
    fitted_pipeline, scores, predictions = evaluate_pipeline(
        pipeline, X, y, split_config
    )

    meta = ModelMetadata(
        experiment=args.experiment.name,
        channels=channels,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        tmin=args.tmin,
        tmax=args.tmax,
        subject=args.subject,
        n_components=args.n_components,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    bundle = ModelBundle(pipeline=fitted_pipeline, metadata=meta)

    output = args.output
    if output == Path("data/models/tpv_pipeline.pkl"):
        suffix = model_filename_suffix(meta)
        output = output.with_stem(output.stem + suffix)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as model_file:
        pickle.dump(bundle, model_file)

    rounded_scores = [round(score, 4) for score in scores.cross_val_scores]
    print(f"cross_val_score: {rounded_scores}")
    print(f"cross_val_score mean: {scores.cross_val_mean:.4f}")
    print(
        "split: "
        f"train={scores.n_train} validation={scores.n_validation} test={scores.n_test}"
    )
    print(f"validation accuracy: {scores.validation_accuracy:.4f}")
    for prediction in predictions:
        print(
            f"epoch {prediction.index:02d}: "
            f"[{prediction.prediction}] [{prediction.truth}] {prediction.equal}"
        )
    print(f"test accuracy: {scores.test_accuracy:.4f}")
    print(f"saved {output} with {len(y)} epochs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
