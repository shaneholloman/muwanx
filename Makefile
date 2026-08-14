.PHONY: sync
sync:
	uv sync --all-extras

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: type
type:
	uv run ty check
	uv run pyright

.PHONY: check
check: format type

.PHONY: test
test:
	uv run pytest

.PHONY: test-all
test-all: check test

# `--directory docs` because zensical resolves zensical.toml from the cwd; it also
# resolves --with-requirements from there, hence the bare filename.
.PHONY: docs-build
docs-build:
	uv run --with-requirements requirements.txt --directory docs zensical build

.PHONY: docs-serve
docs-serve:
	uv run --with-requirements requirements.txt --directory docs zensical serve
