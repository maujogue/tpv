"""Run streamed predictions with a saved EEG treatment pipeline."""

import argparse
import pickle
from pathlib import Path

from tpv.channels import channels_for_experiment, pick_channels
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
    with args.model.open("rb") as model_file:
        obj = pickle.load(model_file)

    if isinstance(obj, ModelBundle):
        meta = obj.metadata
        pipeline = obj.pipeline
        channels = meta.channels
        filter_config = FilterConfig(l_freq=meta.l_freq, h_freq=meta.h_freq)
        epoch_config = EpochConfig(tmin=meta.tmin, tmax=meta.tmax)
    else:
        # Legacy: bare pipeline without metadata — use defaults.
        pipeline = obj
        channels = channels_for_experiment(args.experiment)
        filter_config = FilterConfig()
        epoch_config = EpochConfig()

    subject = (
        Subject(id=args.subject, raw_dir=args.raw_dir)
        if args.raw_dir
        else Subject(id=args.subject)
    )
    runs = pick_channels(subject.runs(args.experiment), channels)
    filtered_runs = filter_runs(runs, filter_config)
    epochs_by_run = make_epochs_by_run(filtered_runs, epoch_config)
    chunks, _ = motion_epochs_to_arrays(epochs_by_run)

    predictions = predict_stream(
        pipeline,
        playback_epochs(chunks, interval_seconds=args.interval),
        max_latency_seconds=args.deadline,
    )
    for prediction in predictions:
        print(
            f"chunk={prediction.index} label={prediction.label} "
            f"latency={prediction.latency_seconds:.6f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
