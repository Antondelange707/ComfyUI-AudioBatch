# Makefile for ComfyUI-AudioBatch project

# Use the python from the current virtual environment
# This assumes you have activated your ComfyUI venv before running `make`
# If not, you might need to specify the path to the python executable
PYTHON := python

# Define test command with necessary pytest options
# --import-mode=append: Prevents pytest from prepending the test root to sys.path,
#                       solving many relative/absolute import conflicts.
# -v: Verbose output.
# -s: Show print statements (useful for debugging tests).
# --ignore=path/to/ignore: Example of ignoring a directory if needed.
PYTEST_CMD := pytest --import-mode=append -v -s

# --- Targets ---

.PHONY: help test test-one lint format check

help:
	@echo "Makefile for ComfyUI-AudioBatch"
	@echo ""
	@echo "Usage:"
	@echo "  make help           Show this help message."
	@echo "  make install        Install development dependencies from requirements-dev.txt."
	@echo "  make test           Run all tests in the 'tests/' directory."
	@echo "  make test-one       Run a specific test file. Usage: make test-one file=tests/test_audio_blend.py"
	@echo "  make lint           Run flake8 linter on the source code."
	@echo "  make format         Run black code formatter on the source code."
	@echo "  make check          Run all checks (lint, type check, tests)."
	@echo "  make type-check     Run mypy for static type checking."


# Target to install development dependencies
install:
	@echo "Installing development dependencies..."
	@$(PYTHON) -m pip install -r requirements-dev.txt

# Target to run all tests
test:
	@echo "Running all tests..."
	@$(PYTEST_CMD) tests/

# Target to run a single test file, specified with `make test-one file=...`
test-one:
	@if [ -z "$(file)" ]; then \
		echo "Error: Please specify a file to test. Usage: make test-one file=tests/your_test_file.py"; \
		exit 1; \
	fi
	@echo "Running test for file: $(file)"
	@$(PYTEST_CMD) $(file)

# Target to run the linter (e.g., flake8)
lint:
	@echo "Running linter (flake8)..."
	flake8 .

# Target to run mypy for type checking
type-check:
	@echo "Running static type checker (mypy)..."
	@$(PYTHON) -m mypy .

# Target to format the code (e.g., black)
format:
	@echo "Formatting code (black)..."
	@$(PYTHON) -m black .

# Target to run all checks together
check: lint type-check test
	@echo "All checks completed."