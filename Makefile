.PHONY: install
install:
	@uv sync --all-extras
	@uv pip install .

# --- --- ---

.PHONY: lint
lint:
	@poetry run ruff check --fix
	@poetry run ruff format


# --- --- ---

.PHONY: test
test:
	@poetry run pytest