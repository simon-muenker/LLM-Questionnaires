.PHONY: install
install:
	@poetry install

# --- --- ---

.PHONY: lint
lint:
	@poetry run ruff check --fix
	@poetry run ruff format


# --- --- ---

.PHONY: test
test:
	@poetry run pytest