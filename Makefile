.PHONY: help data features models evaluate test lint clean all app

help:
	@echo "NBA Player Performance Prediction - Makefile Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  make data       - Collect all raw data (player logs, team context, schedules)"
	@echo "  make features   - Build feature dataset from raw data"
	@echo "  make models     - Train baseline and main models for all targets"
	@echo "  make evaluate   - Run evaluation and generate reports"
	@echo "  make test       - Run test suite with coverage"
	@echo "  make lint       - Run code quality checks (black, flake8)"
	@echo "  make clean      - Remove generated files and artifacts"
	@echo "  make all        - Run complete pipeline (data -> features -> models -> evaluate)"
	@echo "  make app        - Launch interactive dashboard (if implemented)"
	@echo ""

# Data collection
data:
	@echo "Collecting NBA data..."
	python src/data/collect_player_gamelogs.py
	python src/data/collect_team_context.py
	python src/data/collect_schedule.py
	@echo "Data collection complete!"

# Feature engineering
features: data
	@echo "Building features..."
	python src/features/build_features.py
	@echo "Feature engineering complete!"

# Model training
models: features
	@echo "Training models..."
	python src/models/train_baseline.py
	python src/models/train_models.py --target PTS
	python src/models/train_models.py --target REB
	python src/models/train_models.py --target AST
	@echo "Model training complete!"

# Evaluation
evaluate: models
	@echo "Running evaluation..."
	python src/models/evaluate.py
	@echo "Evaluation complete! Check reports/ for results."

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "Tests complete! Coverage report in htmlcov/index.html"

# Linting
lint:
	@echo "Running code quality checks..."
	black src/ tests/
	flake8 src/ tests/ --max-line-length=100
	@echo "Linting complete!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf data/raw/*.parquet
	rm -rf data/interim/*.parquet
	rm -rf data/processed/*.parquet
	rm -rf artifacts/models/*
	rm -rf artifacts/metrics/*
	rm -rf reports/figures/*
	rm -rf reports/cleaning/*
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	@echo "Cleanup complete!"

# Run complete pipeline
all: data features models evaluate
	@echo "Complete pipeline finished!"
	@echo "Results saved to reports/"
	@echo "Models saved to artifacts/models/"

# Launch dashboard (optional)
app:
	@echo "Launching interactive dashboard..."
	@echo "Dashboard will be available at http://127.0.0.1:8050"
	python src/app/dashboard.py
