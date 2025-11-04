# Makefile for NBA Player Predictions
# Run the entire pipeline with simple commands

.PHONY: help install data features train evaluate visualize test clean all

# Default target
help:
	@echo "NBA Player Predictions - Available Commands:"
	@echo ""
	@echo "  make install      Install all dependencies"
	@echo "  make data         Collect NBA data from API (~15-20 min)"
	@echo "  make features     Engineer features from raw data"
	@echo "  make train        Train all models (baseline + ML)"
	@echo "  make evaluate     Generate evaluation metrics and reports"
	@echo "  make visualize    Create all plots and figures"
	@echo "  make test         Run all tests"
	@echo "  make lint         Run code quality checks"
	@echo "  make clean        Remove generated files"
	@echo "  make all          Run entire pipeline (data → visualize)"
	@echo "  make app          Launch interactive dashboard"
	@echo ""
	@echo "Quick start: make install && make all"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Data collection
data:
	@echo "Collecting NBA data..."
	@echo "This may take 15-20 minutes due to API rate limiting..."
	python -m src.collect
	@echo "✓ Data collection complete"

# Feature engineering
features:
	@echo "Engineering features..."
	python -m src.features
	@echo "✓ Features created"

# Model training
train:
	@echo "Training models..."
	python -m src.train
	@echo "✓ Models trained"

# Evaluation
evaluate:
	@echo "Evaluating models..."
	python -m src.evaluate
	@echo "✓ Evaluation complete"

# Visualization
visualize:
	@echo "Creating visualizations..."
	python -m src.visualize
	@echo "✓ Visualizations created"

# Run all tests
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=term-missing
	@echo "✓ Tests complete"

# Lint code
lint:
	@echo "Running linters..."
	black --check src/ tests/
	flake8 src/ tests/ --max-line-length=100
	@echo "✓ Linting complete"

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/
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
app:
	@echo "Launching dashboard at http://localhost:8501"
	streamlit run app.py

# Quick test (fast subset for CI)
test-quick:
	@echo "Running quick tests..."
	pytest tests/unit/ -v
	@echo "✓ Quick tests complete"

# CI/CD target (used by GitHub Actions)
ci: install lint test-quick
	@echo "✓ CI checks passed"
