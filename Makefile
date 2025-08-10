.PHONY: lint test type run

all: lint type test

lint:
	uv run ruff format
	uv run ruff check --fix
	uv run ruff check

type:
	mypy .

test:
	uv run pytest

run:
	uv run python main.py
