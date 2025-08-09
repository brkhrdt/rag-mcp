.PHONY: lint test run

lint:
	uv run ruff format
	uv run ruff check --fix
	uv run ruff check

test:
	uv run pytest

run:
	uv run python main.py
