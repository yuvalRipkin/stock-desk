.PHONY: install db-up db-down fetch ingest ask eval lint format test

install:
	uv sync

db-up:
	docker compose up -d --wait

db-down:
	docker compose down

fetch:
	uv run python scripts/fetch_data.py

ingest:
	uv run python -m stockdesk.ingestion.ingestion

ask:
ifndef DOC
	$(error usage: make ask DOC=AMD_2022_10K Q="What was total revenue in 2022?")
endif
ifndef Q
	$(error usage: make ask DOC=AMD_2022_10K Q="What was total revenue in 2022?")
endif
	uv run python -m stockdesk.ask --doc $(DOC) "$(Q)"

RUNS ?= 3

eval:
ifndef NAME
	$(error usage: make eval NAME=001_naive_1000 [RUNS=3])
endif
	uv run python -m stockdesk.evals.runner --name $(NAME) --runs $(RUNS)

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest
