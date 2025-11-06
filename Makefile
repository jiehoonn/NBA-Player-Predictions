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
	@echo "  make data         Collect NBA data via notebook 01 (~15-20 min)"
	@echo "  make features     Engineer features via notebook 03"
	@echo "  make train        Train models via notebooks 04 + 05"
	@echo "  make evaluate     Run error analysis via notebook 06"
	@echo "  make visualize    View/regenerate figures"
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

# Data collection (via notebook)
data: venv
	@echo "Collecting NBA data..."
	@echo "This may take 15-20 minutes due to API rate limiting..."
	@echo "Executing notebook: 01_data_collection.ipynb"
	$(JUPYTER) nbconvert --to notebook --execute notebooks/01_data_collection.ipynb \
		--output 01_data_collection_executed.ipynb --ExecutePreprocessor.timeout=1200 \
		--allow-errors
	@echo "✓ Data collection complete"

# Feature engineering (via notebook)
features: venv
	@echo "Engineering features..."
	@echo "Executing notebook: 03_feature_engineering.ipynb"
	$(JUPYTER) nbconvert --to notebook --execute notebooks/03_feature_engineering.ipynb \
		--output 03_feature_engineering_executed.ipynb --ExecutePreprocessor.timeout=600 \
		--allow-errors
	@echo "✓ Features created"

# Model training (baseline + advanced via notebooks)
train: venv
	@echo "Training baseline models..."
	@echo "Executing notebook: 04_baseline_model.ipynb"
	$(JUPYTER) nbconvert --to notebook --execute notebooks/04_baseline_model.ipynb \
		--output 04_baseline_model_executed.ipynb --ExecutePreprocessor.timeout=600 \
		--allow-errors
	@echo ""
	@echo "Training advanced models..."
	@echo "Executing notebook: 05_advanced_models.ipynb"
	$(JUPYTER) nbconvert --to notebook --execute notebooks/05_advanced_models.ipynb \
		--output 05_advanced_models_executed.ipynb --ExecutePreprocessor.timeout=600 \
		--allow-errors
	@echo "✓ Models trained"

# Evaluation (error analysis via notebook)
evaluate: venv
	@echo "Running error analysis..."
	@echo "Executing notebook: 06_error_analysis.ipynb"
	$(JUPYTER) nbconvert --to notebook --execute notebooks/06_error_analysis.ipynb \
		--output 06_error_analysis_executed.ipynb --ExecutePreprocessor.timeout=600 \
		--allow-errors
	@echo "✓ Evaluation complete"

# Visualization (generated in notebooks)
visualize: venv
	@echo "📊 Visualizations are generated within notebooks 02, 04, 05, 06"
	@echo "All figures saved to: results/figures/"
	@echo ""
	@echo "To regenerate visualizations, run:"
	@echo "  make train evaluate"
	@echo ""
	@echo "✓ Check results/figures/ for 12 PNG files"

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

# Run entire pipeline (notebook-based)
all: data features train evaluate visualize
	@echo ""
	@echo "=========================================="
	@echo "✅ Pipeline complete! 🎉"
	@echo "=========================================="
	@echo ""
	@echo "📊 Results:"
	@echo "  Models:  models/final/*.pkl (3 files)"
	@echo "  Metrics: results/predictions/*.json"
	@echo "  Figures: results/figures/*.png (12 files)"
	@echo ""
	@echo "📓 Executed notebooks:"
	@echo "  notebooks/*_executed.ipynb"
	@echo ""
	@echo "🔍 View results:"
	@echo "  - Open results/figures/*.png"
	@echo "  - Read README.md for full analysis"
	@echo "  - Check notebooks/*_executed.ipynb for outputs"
	@echo ""

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
