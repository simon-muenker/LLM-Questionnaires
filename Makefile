.PHONY: install
install:
	@poetry install

# --- --- ---

.PHONY: format
format:
	@poetry run ruff check --fix
	@poetry run ruff format
	@poetry run mypy