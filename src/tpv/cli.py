"""Command line tools for the first TPV preprocessing phase."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from tpv.channels import channels_for_experiment, pick_channels
from tpv.config import data_path
from tpv.dataset import motion_epochs_to_arrays
from tpv.epoching import EpochConfig, make_epochs_by_run
from tpv.loader import ExperimentType, Subject
from tpv.pipeline import CSPTransformer
from tpv.preprocessing import FilterConfig, filter_runs


def parse_experiment(value: str) -> ExperimentType:
    try:
        return ExperimentType[value.upper()]
    except KeyError as error:
        choices = ", ".join(experiment.name.lower() for experiment in ExperimentType)
        raise argparse.ArgumentTypeError(
            f"unknown experiment {value!r}; choices: {choices}"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore and preprocess PhysioNet EEG runs"
    )
    parser.add_argument("subject", type=int, help="PhysioNet subject id, from 1 to 109")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=data_path,
        help="Directory containing S001 ... S109 raw EDF folders",
    )
    parser.add_argument(
        "--experiment",
        type=parse_experiment,
        default=ExperimentType.IMAGERY_LEFT_RIGHT,
        help="Experiment family to preprocess",
    )
    parser.add_argument(
        "--l-freq", type=float, default=8.0, help="Lower filter cutoff in Hz"
    )
    parser.add_argument(
        "--h-freq", type=float, default=40.0, help="Upper filter cutoff in Hz"
    )
    parser.add_argument(
        "--tmin", type=float, default=-1.0, help="Epoch start in seconds"
    )
    parser.add_argument("--tmax", type=float, default=4.0, help="Epoch end in seconds")
    parser.add_argument(
        "--plot-run",
        type=int,
        default=0,
        help="Index of the run to plot within the experiment (0-based)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=4,
        help="Number of CSP filters for the class-separability plot; 0 to skip",
    )
    return parser


def plot_csp(X: np.ndarray, y: np.ndarray, n_components: int, title: str) -> None:
    classes = np.unique(y)
    csp = CSPTransformer(n_components=n_components).fit(X, y)
    features = csp.transform(X)

    raw_logvar = np.log(np.var(X, axis=2))
    spread = raw_logvar.std(axis=0)
    ch_x, ch_y = np.argsort(spread)[::-1][:2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = ["#2b6cb0", "#c53030"]
    labels = {classes[0]: "class A (T1)", classes[1]: "class B (T2)"}

    for color, cls in zip(colors, classes):
        m = y == cls
        axes[0].scatter(
            raw_logvar[m, ch_x], raw_logvar[m, ch_y],
            c=color, alpha=0.6, edgecolors="none", label=labels[cls],
        )
    axes[0].set_title(f"BEFORE CSP\nraw channel log-variance ({X.shape[1]} channels)")
    axes[0].set_xlabel(f"channel {ch_x} log-var")
    axes[0].set_ylabel(f"channel {ch_y} log-var")
    axes[0].legend()

    pair = features[:, [0, -1]]
    for color, cls in zip(colors, classes):
        m = y == cls
        axes[1].scatter(
            pair[m, 0], pair[m, 1],
            c=color, alpha=0.7, edgecolors="none", label=labels[cls],
        )

    lda = LinearDiscriminantAnalysis().fit(pair, y)
    train_acc = lda.score(pair, y)
    x_min, x_max = axes[1].get_xlim()
    y_min, y_max = axes[1].get_ylim()
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    )
    decision = lda.decision_function(np.c_[grid_x.ravel(), grid_y.ravel()])
    decision = decision.reshape(grid_x.shape)
    axes[1].contourf(grid_x, grid_y, decision > 0, alpha=0.08, levels=1, colors=["#2b6cb0", "#c53030"])
    axes[1].contour(grid_x, grid_y, decision, levels=[0], colors="black", linewidths=2, linestyles="--")
    axes[1].plot([], [], "k--", linewidth=2, label=f"LDA boundary ({train_acc:.0%} acc)")
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].set_title(f"AFTER CSP + LDA\nlog-variance features ({n_components} components)")
    axes[1].set_xlabel("CSP filter 1 (max variance class A)")
    axes[1].set_ylabel("CSP filter N (max variance class B)")
    axes[1].legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    subject = Subject(id=args.subject, raw_dir=args.raw_dir)
    print(f"subject {subject.id}")
    for experiment_name, count in subject.describe().items():
        print(f"{experiment_name.lower()}: {count} runs")

    runs = subject.runs(args.experiment)
    filtered_runs = filter_runs(
        runs, FilterConfig(l_freq=args.l_freq, h_freq=args.h_freq)
    )

    plot_index = args.plot_run
    experiment_label = args.experiment.name.lower()
    scalings = {"eeg": 4000e-7}
    runs[plot_index].plot(
        duration=20,
        n_channels=1,
        scalings=scalings,
        title=f"Subject {subject.id} raw - {experiment_label}",
    )
    filtered_runs[plot_index].plot(
        duration=20,
        n_channels=1,
        scalings=scalings,
        title=(
            f"Subject {subject.id} filtered "
            f"{args.l_freq}-{args.h_freq} Hz - {experiment_label}"
        ),
    )
    filtered_runs[plot_index].compute_psd(fmin=0, fmax=50, picks="eeg").plot(
        average=True,
        spatial_colors=True,
    )

    epochs = make_epochs_by_run(
        filtered_runs, EpochConfig(tmin=args.tmin, tmax=args.tmax)
    )
    n_epochs = sum(len(run_epochs) for run_epochs in epochs)
    print(
        f"preprocessed {len(filtered_runs)} runs from {args.experiment.name.lower()} "
        f"into {n_epochs} epochs"
    )

    if args.n_components > 0:
        channels = channels_for_experiment(args.experiment)
        channel_runs = pick_channels(filtered_runs, channels)
        channel_epochs = make_epochs_by_run(
            channel_runs, EpochConfig(tmin=args.tmin, tmax=args.tmax)
        )
        X, y = motion_epochs_to_arrays(channel_epochs)
        plot_csp(
            X, y, args.n_components,
            f"{subject.id} — {experiment_label} — {len(y)} epochs",
        )

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
