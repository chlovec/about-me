.PHONY: help test test-project clean lint format check fix

# Default target when running just 'make'
help:
	@echo "Available commands:"
	@echo "  make test               Run all tests in the monorepo with coverage"
	@echo "  make test-project p=... Run tests for a single workspace package (e.g., make test-project p=search-core)"
	@echo "  test-search-core  		 Run tests for the search-core workspace"
	@echo "  make lint               Run ruff to check for lint errors and import sorting"
	@echo "  make format             Run ruff to check code formatting"
	@echo "  make check              Run both lint and format checks"
	@echo "  make fix                Automatically fix linting, sorting, and formatting issues"
	@echo "  make clean              Remove coverage and python cache artifacts"

# 1. Run all tests across the monorepo
test:
	@echo "🚀 Running all workspace tests with branch coverage..."
	uv run pytest --cov=packages --cov=services

# 2. Run tests for a specific workspace project using uv's native package flags
test-project:
	@if [ -z "$(p)" ]; then \
		echo "❌ Missing project argument."; \
		echo "Usage: make test-project p=<package-name>"; \
		exit 1; \
	fi
	@echo "🎯 Running workspace tests for package: $(p)..."
	uv run --package $(p) pytest --cov=packages --cov=services

test-search-core:
	@echo "🚀 Running search-core tests"
	uv run --package search-core pytest --cov=packages --cov=services

# 3. Linting, Formatting, and Sorting with Ruff
lint:
	@echo "🦊 Checking linting and import sorting with Ruff..."
	uv run ruff check .

format:
	@echo "🎨 Checking code formatting with Ruff..."
	uv run ruff format --check .

check: lint format

fix:
	@echo "🛠️ Fixing linting, import sorting, and formatting with Ruff..."
	uv run ruff check --fix .
	uv run ruff format .

# 4. Clean up coverage records and python cache
clean:
	@echo "🧹 Cleaning up test artifacts..."
	rm -rf .coverage .coverage.* htmlcov/ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +