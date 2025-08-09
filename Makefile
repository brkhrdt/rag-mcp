.PHONY: lint test

lint:
	uv run ruff format
	uv run ruff check --fix
	uv run ruff check

test:
	uv run pytest
