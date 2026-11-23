.PHONY: install db-up db-down lint format test ask

install:
	uv sync

db-up:
	docker compose up -d

db-down:
	docker compose down

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

ask:
	uv run python -m stockdesk.ask --doc $(DOC) "$(Q)"
