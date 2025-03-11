.PHONY: install
install:
	@poetry install

# --- --- ---

.PHONY: lint
lint:
	@poetry run ruff check --fix
	@poetry run ruff format
