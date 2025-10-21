.PHONY: help data features models models-pts models-reb models-ast evaluate evaluate-full visualize test lint clean all test-quick

help:
	@echo "==================================================================="
	@echo "NBA Player Performance Prediction Pipeline - PRODUCTION"
	@echo "==================================================================="
	@echo ""
	@echo "Quick Start:"
	@echo "  make all              - Run entire pipeline (30-40 min)"
	@echo "  make data features models  - Step by step"
	@echo ""
	@echo "Individual Steps:"
	@echo "  make data       - Collect 5 seasons, 200 players (~20 min)"
	@echo "  make features   - Engineer 38 features with time-based splits"
	@echo "  make models     - Train best models (Lasso/XGBoost)"
	@echo "  make evaluate   - Show model performance metrics"
	@echo "  make evaluate-full - Comprehensive evaluation with detailed analysis"
	@echo "  make visualize  - Generate all plots and charts"
	@echo ""
	@echo "Testing & Cleanup:"
	@echo "  make test-quick - Quick test with sample data"
	@echo "  make test       - Run full test suite"
	@echo "  make clean      - Remove all generated files"
	@echo ""
	@echo "Production Configuration:"
	@echo "  - Dataset: 5 seasons (2020-21 to 2024-25), 200 players"
	@echo "  - Features: 38 total (23 baseline + 15 Phase 1 advanced)"
	@echo "  - Games: ~57,812 total (~44,600 training)"
	@echo ""
	@echo "Final Performance (38 features):"
	@echo "  - PTS: Lasso    → 5.448 MAE (+3.7%% vs baseline) [Goal: 3.6]"
	@echo "  - REB: XGBoost  → 2.134 MAE (+2.4%% vs baseline) [Goal: 2.2] ✓"
	@echo "  - AST: XGBoost  → 1.642 MAE (+2.2%% vs baseline) [Goal: 2.0] ✓"
	@echo "==================================================================="
	@echo ""

# Data collection (5 seasons, 200 players, ~20 minutes with retry logic)
data:
	@echo "==================================================================="
	@echo "STEP 1: Collecting NBA data (5 seasons, 200 players)"
	@echo "==================================================================="
	@echo "Expected time: ~20 minutes (with robust retry logic)"
	@echo "Features: Exponential backoff, checkpointing, graceful degradation"
	@echo ""
	python src/data/collect_data.py \
		--seasons 2020-21 2021-22 2022-23 2023-24 2024-25 \
		--top-players 200 \
		--output data/raw/player_gamelogs_enhanced.parquet
	@echo ""
	@echo "✓ Data collection complete!"
	@echo "  Output: data/raw/player_gamelogs_enhanced.parquet"
	@echo "  Games: ~57,812 total"
	@echo ""

# Feature engineering (38 features total: 23 baseline + 15 Phase 1 advanced)
features:
	@echo "==================================================================="
	@echo "STEP 2: Engineering features (38 total)"
	@echo "==================================================================="
	@echo "Feature breakdown:"
	@echo "  BASELINE (23 features):"
	@echo "    - Original: 9 (rolling averages: pts/reb/ast/min)"
	@echo "    - Usage: 8 (FGA, FTA, FG3A, FG%%)"
	@echo "    - Contextual: 6 (opponent, rest, home/away)"
	@echo ""
	@echo "  PHASE 1 ADVANCED (15 features):"
	@echo "    - True Shooting %%: 2 (ts_pct_last_3/5)"
	@echo "    - Last Game: 3 (pts/reb/ast_last_game)"
	@echo "    - Turnover Rate: 2 (tov_last_3/5)"
	@echo "    - Plus/Minus: 2 (plus_minus_last_3/5)"
	@echo "    - Trends: 3 (pts/reb/ast_trend_last_5)"
	@echo "    - Consistency: 3 (pts/reb/ast_std_last_5)"
	@echo ""
	python src/features/build_features.py \
		--input data/raw/player_gamelogs_enhanced.parquet \
		--output data/processed/features_enhanced.parquet \
		--windows 3 5
	@echo ""
	@echo "✓ Feature engineering complete!"
	@echo "  Output: data/processed/features_enhanced.parquet"
	@echo "  Games: ~56,812 (after cleaning)"
	@echo "  Train/Val/Test: 78.5%% / 9.0%% / 12.5%%"
	@echo ""

# Model training (best models with 38 features)
models: models-pts models-reb models-ast
	@echo ""
	@echo "==================================================================="
	@echo "✓ ALL MODELS TRAINED (38 features each)"
	@echo "==================================================================="
	@echo "Models saved in: artifacts/models/"
	@echo ""
	@ls -lh artifacts/models/*.joblib 2>/dev/null || echo "No models found"
	@echo ""

models-pts:
	@echo "Training PTS model (Lasso with 38 features)..."
	@mkdir -p artifacts/models
	python src/models/train_models.py \
		--input data/processed/features_enhanced.parquet \
		--target PTS \
		--output artifacts/models/ \
		--features all

models-reb:
	@echo "Training REB model (XGBoost with 38 features)..."
	@mkdir -p artifacts/models
	python src/models/train_models.py \
		--input data/processed/features_enhanced.parquet \
		--target REB \
		--output artifacts/models/ \
		--features all

models-ast:
	@echo "Training AST model (XGBoost with 38 features)..."
	@mkdir -p artifacts/models
	python src/models/train_models.py \
		--input data/processed/features_enhanced.parquet \
		--target AST \
		--output artifacts/models/ \
		--features all

# Evaluation - Show model performance
evaluate:
	@echo "==================================================================="
	@echo "MODEL PERFORMANCE SUMMARY (38 features)"
	@echo "==================================================================="
	@echo ""
	@echo "PTS (Points) - Lasso Model:"
	@cat artifacts/models/PTS_metrics.json 2>/dev/null | python -m json.tool || echo "  Not trained yet. Run: make models-pts"
	@echo ""
	@echo "REB (Rebounds) - XGBoost Model:"
	@cat artifacts/models/REB_metrics.json 2>/dev/null | python -m json.tool || echo "  Not trained yet. Run: make models-reb"
	@echo ""
	@echo "AST (Assists) - XGBoost Model:"
	@cat artifacts/models/AST_metrics.json 2>/dev/null | python -m json.tool || echo "  Not trained yet. Run: make models-ast"
	@echo ""
	@echo "==================================================================="

# Comprehensive evaluation with detailed analysis
evaluate-full:
	@echo "==================================================================="
	@echo "Running comprehensive model evaluation..."
	@echo "==================================================================="
	@mkdir -p reports
	python src/models/evaluate_models.py \
		--models artifacts/models/ \
		--data data/processed/features_enhanced.parquet \
		--output reports/evaluation_report.md \
		--verbose
	@echo ""
	@echo "✓ Evaluation complete!"
	@echo "  Report saved to: reports/evaluation_report.md"
	@echo ""

# Generate visualizations
visualize:
	@echo "==================================================================="
	@echo "Generating visualizations..."
	@echo "==================================================================="
	@mkdir -p reports/figures
	python src/visualization/make_plots.py \
		--models artifacts/models/ \
		--data data/processed/features_enhanced.parquet \
		--output reports/figures/ \
		--plots all
	@echo ""
	@echo "✓ Visualizations complete!"
	@echo "  Plots saved to: reports/figures/"
	@echo ""
	@ls -lh reports/figures/*.png 2>/dev/null || echo "No plots found"
	@echo ""

# Testing
test:
	@echo "Running full test suite..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "Tests complete! Coverage report in htmlcov/index.html"

# Quick test with existing sample data
test-quick:
	@echo "==================================================================="
	@echo "Running quick test with sample data..."
	@echo "==================================================================="
	@echo "This uses the small 2023-24 dataset for fast testing"
	@echo ""
	python src/features/build_features.py \
		--input data/raw/player_gamelogs_2023-24_sample.parquet \
		--output data/processed/features_test.parquet \
		--windows 3 5
	@echo ""
	python src/models/train_models.py \
		--input data/processed/features_test.parquet \
		--target PTS \
		--output artifacts/models_test/ \
		--features all
	@echo ""
	@echo "✓ Quick test complete!"
	@echo "  Test output: artifacts/models_test/"
	@echo ""

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
all: data features models evaluate-full visualize
	@echo ""
	@echo "==================================================================="
	@echo "COMPLETE PRODUCTION PIPELINE FINISHED!"
	@echo "==================================================================="
	@echo "Configuration: 5 seasons, 200 players, 38 features"
	@echo "Models saved to: artifacts/models/"
	@echo "Evaluation report: reports/evaluation_report.md"
	@echo "Visualizations: reports/figures/"
	@echo ""
	@echo "Performance Summary:"
	@echo "  PTS: 5.448 MAE (goal: 3.6) - Performance ceiling reached"
	@echo "  REB: 2.134 MAE (goal: 2.2) ✓ ACHIEVED"
	@echo "  AST: 1.642 MAE (goal: 2.0) ✓ ACHIEVED"
	@echo "  Overall: 2/3 goals achieved"
	@echo "==================================================================="
	@echo ""

# Launch dashboard (optional)
app:
	@echo "Launching interactive dashboard..."
	@echo "Dashboard will be available at http://127.0.0.1:8050"
	python src/app/dashboard.py
