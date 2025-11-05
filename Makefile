# Makefile for NBA Player Predictions
# Run the entire pipeline with simple commands

# Virtual environment settings
VENV_NAME := venv
VENV_DIR := $(VENV_NAME)
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
PYTEST := $(VENV_DIR)/bin/pytest
BLACK := $(VENV_DIR)/bin/black
FLAKE8 := $(VENV_DIR)/bin/flake8
JUPYTER := $(VENV_DIR)/bin/jupyter
STREAMLIT := $(VENV_DIR)/bin/streamlit

# Detect OS for cross-platform compatibility
ifeq ($(OS),Windows_NT)
    PYTHON := $(VENV_DIR)/Scripts/python.exe
    PIP := $(VENV_DIR)/Scripts/pip.exe
    PYTEST := $(VENV_DIR)/Scripts/pytest.exe
    BLACK := $(VENV_DIR)/Scripts/black.exe
    FLAKE8 := $(VENV_DIR)/Scripts/flake8.exe
    JUPYTER := $(VENV_DIR)/Scripts/jupyter.exe
    STREAMLIT := $(VENV_DIR)/Scripts/streamlit.exe
endif

.PHONY: help venv install reinstall data features train evaluate visualize test clean all

# Default target
help:
	@echo "NBA Player Predictions - Available Commands:"
	@echo ""
	@echo "  make venv         Create virtual environment"
	@echo "  make install      Create venv + install dependencies"
	@echo "  make reinstall    Reinstall all dependencies (use after updating requirements.txt)"
	@echo "  make notebook     Launch Jupyter notebook"
	@echo "  make data         Collect NBA data from API (~15-20 min)"
	@echo "  make features     Engineer features from raw data"
	@echo "  make train        Train all models (baseline + ML)"
	@echo "  make evaluate     Generate evaluation metrics and reports"
	@echo "  make visualize    Create all plots and figures"
	@echo "  make test         Run all tests"
	@echo "  make lint         Run code quality checks"
	@echo "  make clean        Remove generated files"
	@echo "  make clean-all    Remove generated files + venv"
	@echo "  make all          Run entire pipeline (data → visualize)"
	@echo "  make app          Launch interactive dashboard"
	@echo ""
	@echo "Quick start:"
	@echo "  make install      # First time setup"
	@echo "  make notebook     # For data exploration"
	@echo "  make all          # Run full pipeline"
	@echo ""
	@echo "To activate venv manually:"
	@echo "  source venv/bin/activate    (macOS/Linux)"
	@echo "  venv\\Scripts\\activate        (Windows)"

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment already exists at $(VENV_DIR)"; \
	else \
		python3 -m venv $(VENV_DIR); \
		echo "✓ Virtual environment created"; \
	fi

# Install dependencies (creates venv first if needed)
install: venv
	@echo "Installing dependencies in virtual environment..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Verifying SSL certificates..."
	@$(PYTHON) -c "import certifi; print(f'SSL cert path: {certifi.where()}')" || \
		(echo "WARNING: SSL certificate verification failed" && exit 1)
	@echo "✓ SSL certificates OK"
	@echo ""
	@echo "✓ Installation complete!"
	@echo ""
	@echo "To activate the virtual environment:"
	@echo "  source venv/bin/activate    (macOS/Linux)"
	@echo "  venv\\Scripts\\activate        (Windows)"

# Reinstall dependencies (useful after updating requirements.txt)
reinstall: venv
	@echo "Reinstalling dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt
	@echo ""
	@echo "Verifying SSL certificates..."
	@$(PYTHON) -c "import certifi; print(f'SSL cert path: {certifi.where()}')" || \
		(echo "WARNING: SSL certificate verification failed" && exit 1)
	@echo "✓ SSL certificates OK"
	@echo ""
	@echo "✓ Dependencies reinstalled!"

# Launch Jupyter notebook
notebook: venv
	@echo "Launching Jupyter Notebook..."
	@echo "Navigate to notebooks/01_data_collection.ipynb"
	@echo ""
	$(JUPYTER) notebook

# Data collection
data: venv
	@echo "Collecting NBA data..."
	@echo "This may take 15-20 minutes due to API rate limiting..."
	$(PYTHON) -m src.collect
	@echo "✓ Data collection complete"

# Feature engineering
features: venv
	@echo "Engineering features..."
	$(PYTHON) -m src.features
	@echo "✓ Features created"

# Model training
train: venv
	@echo "Training models..."
	$(PYTHON) -m src.train
	@echo "✓ Models trained"

# Evaluation
evaluate: venv
	@echo "Evaluating models..."
	$(PYTHON) -m src.evaluate
	@echo "✓ Evaluation complete"

# Visualization
visualize: venv
	@echo "Creating visualizations..."
	$(PYTHON) -m src.visualize
	@echo "✓ Visualizations created"

# Run all tests
test: venv
	@echo "Running tests..."
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing
	@echo "✓ Tests complete"

# Lint code
lint: venv
	@echo "Running linters..."
	$(BLACK) --check src/ tests/
	$(FLAKE8) src/ tests/ --max-line-length=100
	@echo "✓ Linting complete"

# Format code
format: venv
	@echo "Formatting code..."
	$(BLACK) src/ tests/
	@echo "✓ Code formatted"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf data/cache/*
	rm -rf models/*
	rm -rf results/figures/*
	rm -rf results/metrics/*
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -f *.log
	@echo "✓ Cleaned"

# Clean everything including venv
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "✓ Everything cleaned"

# Run entire pipeline
all: data features train evaluate visualize
	@echo ""
	@echo "=========================================="
	@echo "Pipeline complete! 🎉"
	@echo "=========================================="
	@echo "Models saved to: models/"
	@echo "Metrics saved to: results/metrics/"
	@echo "Figures saved to: results/figures/"
	@echo ""
	@echo "View results:"
	@echo "  - Check results/metrics/evaluation.json"
	@echo "  - Open results/figures/*.png"
	@echo "  - Run 'make app' for interactive dashboard"

# Launch interactive dashboard
app: venv
	@echo "Launching dashboard at http://localhost:8501"
	$(STREAMLIT) run app.py

# Quick test (fast subset for CI)
test-quick: venv
	@echo "Running quick tests..."
	$(PYTEST) tests/unit/ -v
	@echo "✓ Quick tests complete"

# CI/CD target (used by GitHub Actions)
ci: install lint test-quick
	@echo "✓ CI checks passed"
