"""Run streamed predictions with a saved EEG treatment pipeline."""

import argparse
import pickle
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tpv.channels import pick_channels
from tpv.dataset import motion_epochs_to_arrays
from tpv.epoching import EpochConfig, make_epochs_by_run
from tpv.loader import ExperimentType, Subject
from tpv.model import ModelBundle
from tpv.preprocessing import FilterConfig, filter_runs
from tpv.stream import playback_epochs, predict_stream


def parse_experiment(value: str) -> ExperimentType:
    try:
        return ExperimentType[value.upper()]
    except KeyError as error:
        choices = ", ".join(experiment.name.lower() for experiment in ExperimentType)
        raise argparse.ArgumentTypeError(f"expected one of: {choices}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict EEG chunks from playback stream"
    )
    parser.add_argument("subject", type=int, help="PhysioNet subject id, from 1 to 109")
    parser.add_argument(
        "--model", type=Path, default=Path("data/models/tpv_pipeline.pkl")
    )
    parser.add_argument(
        "--experiment",
        type=parse_experiment,
        default=ExperimentType.IMAGERY_LEFT_RIGHT,
        help="Experiment family to predict",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=None, help="Directory containing EDF data"
    )
    parser.add_argument(
        "--interval", type=float, default=0.0, help="Playback delay between chunks"
    )
    parser.add_argument(
        "--deadline", type=float, default=2.0, help="Per-chunk latency limit"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model.exists():
        print(f"error: model file not found: {args.model}")
        print("hint: train a model first with `tpv-train`")
        return 1
    with args.model.open("rb") as model_file:
        obj = pickle.load(model_file)

    if not isinstance(obj, ModelBundle):
        print("error: model format not supported, please retrain with `tpv-train`")
        return 1
    meta = obj.metadata
    pipeline = obj.pipeline
    channels = meta.channels
    filter_config = FilterConfig(l_freq=meta.l_freq, h_freq=meta.h_freq)
    epoch_config = EpochConfig(tmin=meta.tmin, tmax=meta.tmax)

    subject = (
        Subject(id=args.subject, raw_dir=args.raw_dir)
        if args.raw_dir
        else Subject(id=args.subject)
    )
    runs = pick_channels(subject.runs(args.experiment), channels)
    filtered_runs = filter_runs(runs, filter_config)
    epochs_by_run = make_epochs_by_run(filtered_runs, epoch_config)
    chunks, true_labels = motion_epochs_to_arrays(epochs_by_run)

    _, chunks, _, true_labels = train_test_split(
        chunks,
        true_labels,
        test_size=meta.test_size,
        stratify=true_labels,
        random_state=meta.random_state,
    )
    print(f"Scoring on held-out test set ({len(true_labels)} epochs, {meta.test_size:.0%} split)")

    print(f"\n{'epoch nb:':<12} {'[prediction]':>12} {'[truth]':>8} {'equal?':>7} {'latency':>12}")
    predicted_labels = []
    for prediction, truth in zip(
        predict_stream(
            pipeline,
            playback_epochs(chunks, interval_seconds=args.interval),
            max_latency_seconds=args.deadline,
        ),
        true_labels,
    ):
        equal = prediction.label == truth
        print(
            f"epoch {prediction.index:02d}:"
            f"        [{prediction.label}]"
            f"       [{truth}]"
            f"  {'True' if equal else 'False':>5}"
            f"  {prediction.latency_seconds:.6f}s",
            flush=True,
        )
        predicted_labels.append(prediction.label)

    score = accuracy_score(true_labels[:len(predicted_labels)], predicted_labels)
    print(f"\nAccuracy: {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
