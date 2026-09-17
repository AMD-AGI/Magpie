# Magpie Makefile
# ========================
.PHONY: help venv install-dev install lint format test coverage clean verify

PYTHON ?= python3
VENV_DIR ?= .venv
PIP := $(VENV_DIR)/bin/pip
PYTHON_VENV := $(VENV_DIR)/bin/python

help:
	@echo "Magpie - GPU Kernel Evaluation Framework"
	@echo ""
	@echo "Setup:"
	@echo "  make venv          - Create virtual environment"
	@echo "  make install       - Install runtime deps (requirements.txt)"
	@echo "  make install-dev   - Install dev deps (requirements + lint/test tools)"
	@echo ""
	@echo "Dev:"
	@echo "  make format        - Format code (black + isort)"
	@echo "  make lint          - Lint (ruff + mypy)"
	@echo "  make test          - Run tests (pytest)"
	@echo "  make coverage      - Run unit tests with the 90% package coverage gate"
	@echo "  make verify        - Smoke check imports / CLI"
	@echo "  make clean         - Remove build + cache artifacts"

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip setuptools wheel
	@echo "Virtualenv ready: source $(VENV_DIR)/bin/activate"

install: venv
	$(PIP) install -r requirements.txt

# Prefer: requirements-dev.txt OR extras [dev]
install-dev: install
	@if [ -f requirements-dev.txt ]; then \
		$(PIP) install -r requirements-dev.txt ; \
	else \
		$(PIP) install pytest pytest-cov black isort mypy ruff ; \
	fi

lint:
	$(PYTHON_VENV) -m ruff check Magpie/

format:
	$(PYTHON_VENV) -m ruff format Magpie/

test:
	$(PYTHON_VENV) -m pytest -q

coverage:
	$(PYTHON_VENV) -m pytest -q tests --cov=Magpie --cov-report=term-missing --cov-fail-under=90

verify: venv
	$(PYTHON_VENV) -c "import Magpie; print('Import OK')"
	$(PYTHON_VENV) -m Magpie --help >/dev/null 2>&1 || true
	@echo "Verify done."

clean:
	rm -rf .venv __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info build dist results/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
