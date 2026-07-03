# Makefile for AI Code Reviewer
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   make test       — run the full test suite with pytest
#   make build      — build the Docker image locally
#   make up         — start all services (app + chromadb + prometheus + grafana)
#   make down       — stop and remove all containers (keeps volumes)
#   make logs       — tail logs from all running services
#   make clean      — remove containers AND volumes (destructive)
#   make lint       — run ruff linter on the app source
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: test build up down logs clean lint help

# Default target — show help
help:
	@echo "AI Code Reviewer — available make targets:"
	@echo ""
	@echo "  make test       Run the full pytest test suite"
	@echo "  make build      Build the Docker image (tag: ai-code-reviewer:local)"
	@echo "  make up         Start all docker-compose services"
	@echo "  make down       Stop all docker-compose services (keeps volumes)"
	@echo "  make logs       Tail logs from all running services"
	@echo "  make clean      Stop services AND remove volumes (WARNING: deletes data)"
	@echo "  make lint       Run ruff linter on app/ source"
	@echo ""

## ── Test ─────────────────────────────────────────────────────────────────────

test:
	@echo ">>> Running test suite..."
	pytest tests/ \
		--tb=short \
		--strict-markers \
		-v \
		-p no:warnings

## ── Docker build ─────────────────────────────────────────────────────────────

build:
	@echo ">>> Building Docker image..."
	docker build \
		--tag ai-code-reviewer:local \
		--file Dockerfile \
		.
	@echo ">>> Build complete: ai-code-reviewer:local"

## ── Compose lifecycle ────────────────────────────────────────────────────────

up:
	@echo ">>> Starting all services..."
	@if [ ! -f .env ]; then \
		echo "WARNING: .env not found — copying from .env.example"; \
		cp .env.example .env; \
	fi
	docker compose up --detach --build
	@echo ">>> Services started:"
	@echo "    App:        http://localhost:8000"
	@echo "    ChromaDB:   http://localhost:8001"
	@echo "    Prometheus: http://localhost:9090"
	@echo "    Grafana:    http://localhost:3000  (admin / changeme)"

down:
	@echo ">>> Stopping services..."
	docker compose down
	@echo ">>> Services stopped (volumes preserved)"

logs:
	docker compose logs --follow --tail=100

## ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	@echo ">>> WARNING: This will delete all volumes (ChromaDB data, Prometheus data, Grafana data)!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	docker compose down --volumes --remove-orphans
	@echo ">>> Clean complete"

## ── Linting ──────────────────────────────────────────────────────────────────

lint:
	@echo ">>> Running ruff linter..."
	ruff check app/ tests/
