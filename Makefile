.PHONY: sync
sync:
	uv sync --all-extras

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

# Both checkers always run, and the target fails if either does: a failing first line
# used to stop the second from ever running.
.PHONY: type
type:
	@status=0; uv run ty check || status=1; uv run pyright || status=1; exit $$status

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
