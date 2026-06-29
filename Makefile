all: lint format typing

lint:
	uv run ruff check --fix src

format:
	uv run ruff format src

typing:
	uv run ty check src
