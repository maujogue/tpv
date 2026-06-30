all: lint format typing

lint:
	uv run ruff check --fix src

format:
	uv run ruff format src

typing:
	uv run ty check src

download-dataset:
	aws s3 sync --no-sign-request s3://physionet-open/eegmmidb/1.0.0/ data/raw/