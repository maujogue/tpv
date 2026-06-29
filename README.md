# Total Perspective Vortex

EEG motor movement / motor imagery decoder for the 42 Total Perspective Vortex project.

The project loads PhysioNet EEG Motor Movement/Imagery EDF files, filters the EEG signal, cuts annotated events into epochs, extracts CSP features, trains an sklearn classifier, and streams predictions with a two-second latency budget.

## What is implemented

- Data loading for PhysioNet subject folders `S001` to `S109`.
- Experiment grouping for the four motor task families:
  - `execution_left_right`
  - `imagery_left_right`
  - `execution_hands_feet`
  - `imagery_hands_feet`
- Band-pass preprocessing, default `7-30 Hz`.
- Event extraction from MNE annotations.
- Epoch creation around events.
- Rest-event removal for binary motion classification.
- Custom CSP dimensionality reduction as an sklearn transformer.
- sklearn pipeline: `CSPTransformer` then `LinearDiscriminantAnalysis`.
- Train, predict, preprocessing, and aggregate evaluation commands.
- Full 109-subject scoring over six evaluation setups.

## Requirements

- Python `>=3.13`
- `uv`
- Local PhysioNet EEG Motor Movement/Imagery EDF data under `data/raw` by default.

Expected data layout:

```text
data/raw/
  S001/
    S001R01.edf
    S001R02.edf
    ...
    S001R14.edf
  S002/
  ...
  S109/
```

The project can also read from another raw data directory with `--raw-dir` or `--data-dir`, depending on the command.

## Install dependencies

```bash
uv sync
```

## Commands

### 1. Preprocess and visualize one subject

```bash
uv run tpv 1 --experiment execution_left_right
```

What it does:

1. Loads subject `S001`.
2. Prints available runs per experiment family.
3. Filters the selected experiment runs.
4. Opens MNE plots for the raw and filtered signal of the same run (the
   filtered trace should look cleaner) plus the filtered power spectral density.
5. Creates epochs and prints the epoch count.

Useful options:

```bash
uv run tpv 1 \
  --experiment imagery_hands_feet \
  --l-freq 7 \
  --h-freq 30 \
  --tmin -1 \
  --tmax 4
```

### 2. Train a model for one subject and experiment

```bash
uv run tpv-train 1 --experiment execution_left_right
```

Default output model:

```text
data/models/tpv_pipeline.pkl
```

The command prints:

- cross-validation scores
- cross-validation mean
- train / validation / test split sizes
- validation accuracy
- held-out test predictions
- held-out test accuracy
- saved model path

Useful options:

```bash
uv run tpv-train 1 \
  --experiment imagery_left_right \
  --output data/models/subject001_imagery_lr.pkl \
  --n-components 4 \
  --l-freq 7 \
  --h-freq 30 \
  --tmin 0 \
  --tmax 2 \
  --test-size 0.2 \
  --validation-size 0.2 \
  --cv-splits 5 \
  --random-state 42
```

### 3. Predict from a saved model as a stream

Train first, then run:

```bash
uv run tpv-predict 1 --experiment execution_left_right
```

The command loads `data/models/tpv_pipeline.pkl`, replays the selected subject's epochs one by one, and prints:

```text
chunk=0 label=3 latency=0.000606s
```

The default deadline is two seconds per chunk:

```bash
uv run tpv-predict 1 \
  --experiment execution_left_right \
  --model data/models/tpv_pipeline.pkl \
  --deadline 2.0 \
  --interval 0.0
```

### 4. Evaluate the whole subject requirement

```bash
uv run tpv-evaluate
```

Default behavior:

- subjects: `1..109`
- experiments:
  - `execution_left_right`
  - `imagery_left_right`
  - `execution_hands_feet`
  - `imagery_hands_feet`
  - `execution_vs_imagery`
  - `movement_target_family`

The correction asks for training over each subject, computing the mean by experiment type, then averaging the six means. This command does that.

Smoke-test one subject first:

```bash
uv run tpv-evaluate --subjects 1 --cv-splits 3
```

Run a subset:

```bash
uv run tpv-evaluate \
  --subjects 1 2 3 \
  --experiments execution_left_right imagery_left_right
```

Observed full-run result during readiness check:

```text
execution_left_right: accuracy = 0.6264
imagery_left_right: accuracy = 0.6376
execution_hands_feet: accuracy = 0.7850
imagery_hands_feet: accuracy = 0.6970
execution_vs_imagery: accuracy = 0.7398
movement_target_family: accuracy = 0.6983
Mean accuracy: 0.6974
```

The correction threshold is `>= 0.6000`, so this observed run passes the scoring requirement.

## Quality checks

Run all tests:

```bash
uv run pytest src/tests
```

Run linting:

```bash
uv run ruff check src
```

Run formatting check:

```bash
uv run ruff format --check src
```

Run type checking:

```bash
uv run ty check src
```

Or use the Makefile:

```bash
make
```

## Repository map

```text
src/tpv/config.py          local data path configuration
src/tpv/loader.py          subject/run loading and experiment grouping
src/tpv/events.py          MNE annotation to event-array conversion
src/tpv/preprocessing.py   band-pass filtering
src/tpv/epoching.py        event-centered epoch creation
src/tpv/dataset.py         MNE epochs to X/y arrays, rest removal
src/tpv/pipeline.py        CSP transformer and sklearn pipeline
src/tpv/scoring.py         cross-validation, validation, test scoring
src/tpv/stream.py          streamed epoch playback and prediction deadlines
src/tpv/cli.py             preprocessing / plotting command
src/scripts/train.py       train command
src/scripts/predict.py     predict command
src/scripts/evaluate.py    aggregate scoring command
src/tests/                 focused unit tests
```

More detailed documentation:

- [Project architecture](docs/architecture.md)
- [Evaluation study guide](docs/evaluation-guide.md)
