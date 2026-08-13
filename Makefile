SHELL := /bin/bash
.DEFAULT_GOAL := help

# Load .env so both docker compose and the CLI see the same values.
ifneq (,$(wildcard .env))
include .env
export
endif

UV ?= uv
COMPOSE ?= docker compose
RUN := $(UV) run

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

.PHONY: install
install: ## Create the venv and install the project with dev extras
	$(UV) sync --extra dev

.PHONY: env
env: ## Write a .env with local development defaults if none exists
	@test -f .env || (cp .env.example .env && echo "wrote .env from .env.example")
	@test -f .env && echo ".env present"

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

.PHONY: up
up: env ## Bring Postgres + Langfuse up and wait for both to be healthy
	$(COMPOSE) up -d --wait
	@echo "postgres: $$($(COMPOSE) ps --format '{{.Name}} {{.Status}}' postgres)"
	@echo "langfuse: $$($(COMPOSE) ps --format '{{.Name}} {{.Status}}' langfuse)"

.PHONY: down
down: ## Stop the stack, keeping the data volume
	$(COMPOSE) down

.PHONY: nuke
nuke: ## Stop the stack and delete the data volume (destructive)
	$(COMPOSE) down -v

.PHONY: logs
logs: ## Tail service logs
	$(COMPOSE) logs -f --tail=100

.PHONY: migrate
migrate: ## Apply pending SQL migrations
	$(RUN) cascade db migrate

.PHONY: seed
seed: ## Build and seal the scenario registry (M1)
	$(RUN) cascade ledger

# ---------------------------------------------------------------------------
# Quality gates -- all four must be green (spec §13)
# ---------------------------------------------------------------------------

.PHONY: fmt
fmt: ## Format with black and apply ruff autofixes
	$(RUN) black cascade tests
	$(RUN) ruff check --fix cascade tests

.PHONY: lint
lint: ## ruff + black --check
	$(RUN) ruff check cascade tests
	$(RUN) black --check cascade tests

.PHONY: typecheck
typecheck: ## mypy strict on cascade/
	$(RUN) mypy

.PHONY: test
test: ## Run the test suite (excludes tests needing live services)
	$(RUN) pytest -m "not integration and not network"

.PHONY: test-all
test-all: ## Run every test including integration (requires `make up`)
	$(RUN) pytest -m "not network"

.PHONY: ci
ci: lint typecheck test ## Everything CI runs

# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------

.PHONY: doctor
doctor: ## Verify toolchain, pinned stack and services
	$(RUN) cascade doctor

.PHONY: study
study: ## Run the full study end to end (M6+)
	$(RUN) cascade simulate

.PHONY: report
report: ## Write the report artifact (M7+)
	$(RUN) cascade report
