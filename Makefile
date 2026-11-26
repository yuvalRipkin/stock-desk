# dev (the tuning filings), test (held out) or all
SPLIT ?= dev

.PHONY: install db-up db-down fetch ingest ask eval lint format test

install:
	uv sync
	uv run pre-commit install

db-up:
	docker compose up -d --wait

db-down:
	docker compose down

fetch:
	uv run python scripts/fetch_data.py --split $(SPLIT)

ingest:
	uv run python -m stockdesk.ingestion.ingestion --split $(SPLIT)

# ANSWER_MODE=agent answers with the LangGraph agent and prints its tool calls (make passes
# command-line variables to the recipe's environment, where the settings read it)
ask:
ifndef DOC
	$(error usage: make ask DOC=AMD_2022_10K Q="What was total revenue in 2022?" [ANSWER_MODE=agent])
endif
ifndef Q
	$(error usage: make ask DOC=AMD_2022_10K Q="What was total revenue in 2022?" [ANSWER_MODE=agent])
endif
	uv run python -m stockdesk.ask --doc $(DOC) "$(Q)"

RUNS ?= 3

eval:
ifndef NAME
	$(error usage: make eval NAME=001_naive_1000 [RUNS=3] [SPLIT=dev])
endif
	uv run python -m stockdesk.evals.runner --name $(NAME) --runs $(RUNS) --split $(SPLIT)

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest
