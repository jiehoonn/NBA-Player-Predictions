# Production Pipeline Documentation

This directory contains the production-ready scripts for NBA player performance prediction, based on research conducted in notebooks 07, 08, and 09.

## Overview

The pipeline consists of 4 main stages:

1. **Data Collection** - Fetch NBA game logs with contextual features
2. **Feature Engineering** - Create 23 leakage-safe features with time-based splits
3. **Model Training** - Train best-performing models (Lasso for PTS, XGBoost for REB/AST)
4. **Evaluation & Visualization** - Assess performance and generate plots

## Quick Start

```bash
# Run entire pipeline
make all

# Or step-by-step
make data        # ~10 minutes
make features    # ~2 minutes
make models      # ~5 minutes
make evaluate-full
make visualize
```

## Directory Structure

```
src/
├── data/
│   └── collect_data.py          # Data collection from NBA API
├── features/
│   └── build_features.py        # Feature engineering pipeline
├── models/
│   ├── train_models.py          # Model training
│   └── evaluate_models.py       # Comprehensive evaluation
└── visualization/
    └── make_plots.py            # Generate all visualizations
```

## Scripts Documentation

### 1. Data Collection (`src/data/collect_data.py`)

Collects NBA player game logs with contextual features.

**Features:**

- Fetches top 120 players by minutes played
- Collects 3 seasons (2022-23, 2023-24, 2024-25)
- Adds opponent stats (DEF_RATING, OFF_RATING, PACE)
- Adds game context (home/away, rest days, back-to-back)

**Usage:**

```bash
python src/data/collect_data.py \
    --seasons 2022-23 2023-24 2024-25 \
    --top-players 120 \
    --output data/raw/player_gamelogs_enhanced_2022-2025.parquet
```

**Parameters:**

- `--seasons`: NBA seasons to collect (space-separated)
- `--top-players`: Number of top players by minutes (default: 120)
- `--output`: Output parquet file path

**Output:**

- Raw game logs with ~23,000 games
- Columns: PTS, REB, AST, MIN, FGA, FTA, FG3A, FG_PCT, IS_HOME, REST_DAYS, IS_BACK_TO_BACK, OPP_DEF_RATING, OPP_OFF_RATING, OPP_PACE, etc.

**Rate Limiting:**

- 0.6 seconds between requests
- 2 seconds after errors
- Expected runtime: ~10 minutes

**Important Fix:**
This script includes the fix for the data duplication bug found in Notebook 07. The merge operation now uses both `PLAYER_ID` and `Game_ID` to prevent Cartesian products.

---

### 2. Feature Engineering (`src/features/build_features.py`)

Creates 23 leakage-safe rolling features with time-based train/val/test splits.

**Features Created:**

- **Original (9)**: Rolling averages for pts, reb, ast, min (windows: 3, 5) + games_played
- **Usage (8)**: Rolling averages for fga, fta, fg3a, fg_pct (windows: 3, 5)
- **Contextual (6)**: IS_HOME, REST_DAYS, IS_BACK_TO_BACK, OPP_DEF_RATING, OPP_OFF_RATING, OPP_PACE

**Leakage Prevention:**
Uses `.shift(1).rolling()` pattern to ensure only PAST games are used for prediction.

**Usage:**

```bash
python src/features/build_features.py \
    --input data/raw/player_gamelogs_enhanced_2022-2025.parquet \
    --output data/processed/features_enhanced_3seasons.parquet \
    --windows 3 5
```

**Parameters:**

- `--input`: Input raw data parquet file
- `--output`: Output processed features parquet file
- `--windows`: Rolling window sizes (default: 3 5)

**Time-Based Splits:**

- **Train**: Games before 2024-07-31 (~60% of data)
- **Val**: Games from 2024-08-01 to 2024-12-31 (~20%)
- **Test**: Games after 2024-12-31 (~20%)

**Data Cleaning:**

- Drops rows with missing targets (PTS, REB, AST)
- Removes games with < 5 previous games
- Fills NaN in contextual features with median
- Final dataset: ~20,000 games

---

### 3. Model Training (`src/models/train_models.py`)

Trains best-performing models based on Notebook 09 findings.

**Best Models (from research):**

- **PTS**: Lasso (alpha=0.1) → MAE 5.774, +3.9% improvement
- **REB**: XGBoost (n_est=100, depth=3, lr=0.05) → MAE 2.185, +1.7%
- **AST**: XGBoost (n_est=100, depth=3, lr=0.05) → MAE 1.762, +2.6%

**Usage:**

```bash
# Train all models
make models

# Or train individually
python src/models/train_models.py \
    --input data/processed/features_enhanced_3seasons.parquet \
    --target PTS \
    --output artifacts/models/ \
    --features all
```

**Parameters:**

- `--input`: Processed features parquet file
- `--target`: Target variable (PTS, REB, or AST)
- `--output`: Output directory for models
- `--features`: Feature set ('original' for 9 features, 'all' for 23)

**Output Files:**

- `{TARGET}_{model}.joblib`: Trained model with scaler, features, and metadata
- `{TARGET}_metrics.json`: Performance metrics (MAE, RMSE, R², improvement)

**Saved Artifacts:**
Each `.joblib` file contains:

- `model`: Trained sklearn/xgboost model
- `scaler`: StandardScaler (for Lasso) or None (for XGBoost)
- `features`: List of feature names used
- `target`: Target variable name
- `results`: Performance on train/val/test splits
- `baseline_mae`: 5-game rolling average MAE
- `improvement`: % improvement over baseline

---

### 4. Model Evaluation (`src/models/evaluate_models.py`)

Comprehensive evaluation of trained models across all data splits.

**Features:**

- Evaluates on train/val/test splits
- Compares to baseline (5-game rolling average)
- Checks for overfitting
- Compares to project goals
- Generates detailed markdown report

**Usage:**

```bash
# Quick evaluation (JSON metrics)
make evaluate

# Comprehensive evaluation (detailed report)
make evaluate-full

# Or directly
python src/models/evaluate_models.py \
    --models artifacts/models/ \
    --data data/processed/features_enhanced_3seasons.parquet \
    --output reports/evaluation_report.md \
    --verbose
```

**Parameters:**

- `--models`: Directory containing trained models (.joblib files)
- `--data`: Processed features parquet file
- `--output`: Output markdown report path
- `--verbose`: Print detailed results for each target

**Output:**

- Console: Formatted tables with performance summary
- File: Markdown report with detailed metrics

**Metrics Reported:**

- MAE, RMSE, R² for train/val/test splits
- Baseline comparison and improvement %
- Overfitting analysis (train vs test gap)
- Goal achievement status

---

### 5. Visualization (`src/visualization/make_plots.py`)

Generates comprehensive visualizations for model evaluation.

**Plots Generated:**

1. **Performance Comparison**: Bar chart comparing all models vs baseline
2. **Feature Importance**: Top 15 features for XGBoost models
3. **Predictions vs Actual**: Scatter plots with residuals
4. **Error Distribution**: Histogram of prediction errors
5. **Time Series**: Sample player predictions over time

**Usage:**

```bash
# Generate all plots
make visualize

# Or directly
python src/visualization/make_plots.py \
    --models artifacts/models/ \
    --data data/processed/features_enhanced_3seasons.parquet \
    --output reports/figures/ \
    --plots all
```

**Parameters:**

- `--models`: Directory containing trained models
- `--data`: Processed features parquet file
- `--output`: Output directory for plots
- `--plots`: Which plots to generate (choices: performance, importance, predictions, errors, timeseries, all)

**Output:**
All plots saved as PNG (300 DPI) in `reports/figures/`:

- `model_performance_comparison.png`
- `feature_importance_{TARGET}.png` (XGBoost only)
- `predictions_vs_actual_{TARGET}.png`
- `error_distribution_{TARGET}.png`
- `time_series_{TARGET}.png`

---

## Configuration

All parameters are centralized in `config.yaml` at the project root:

```yaml
data_collection:
  seasons: [2022-23, 2023-24, 2024-25]
  top_players: 120

features:
  rolling_windows: [3, 5]
  min_games_played: 5

models:
  lasso:
    alpha: 0.1
  xgboost:
    n_estimators: 100
    max_depth: 3
    learning_rate: 0.05

evaluation:
  goals:
    PTS: 3.6
    REB: 2.2
    AST: 2.0
```

## Testing

```bash
# Quick test with sample data (~1 minute)
make test-quick

# Full test suite with coverage
make test
```

## Performance Benchmarks

Based on Notebook 09 findings with 23 features:

| Target | Model   | Test MAE | Baseline MAE | Improvement | Goal | Status |
| ------ | ------- | -------- | ------------ | ----------- | ---- | ------ |
| PTS    | Lasso   | 5.774    | 6.008        | +3.9%       | 3.6  | ❌     |
| REB    | XGBoost | 2.185    | 2.224        | +1.7%       | 2.2  | ✓      |
| AST    | XGBoost | 1.762    | 1.809        | +2.6%       | 2.0  | ✓      |

## Key Improvements from Notebooks

### Notebook 07 → Production

- **Fixed**: Data duplication bug in merge operation
- **Added**: Rate limiting and error handling
- **Improved**: Progress tracking with tqdm

### Notebook 08 → Production

- **Modularized**: Feature creation into reusable functions
- **Centralized**: Feature lists for consistency
- **Added**: Comprehensive data validation

### Notebook 09 → Production

- **Selected**: Best models (Lasso/XGBoost) instead of all 8
- **Optimized**: Removed neural networks (overfitting issues)
- **Documented**: All hyperparameters from research

## Troubleshooting

**Problem**: Data collection fails with rate limit errors
**Solution**: Increase `time.sleep()` values in `collect_data.py`

**Problem**: Features have NaN values
**Solution**: Check `min_games_played` threshold, ensure sufficient history

**Problem**: Models show high overfitting
**Solution**: Verify time-based splits are correct, no data leakage

**Problem**: Visualizations fail to generate
**Solution**: Ensure models are trained first with `make models`

## Next Steps

After running the production pipeline:

1. Review `reports/evaluation_report.md` for detailed metrics
2. Check `reports/figures/` for visualizations
3. Analyze feature importance plots to understand predictions
4. Compare results to Notebook 09 benchmarks
5. Consider model improvements:
   - Add more contextual features (player injuries, team performance)
   - Experiment with ensemble methods
   - Increase training data (more seasons)

## References

- Research notebooks: `notebooks/07_enhanced_data_collection.ipynb`, `notebooks/08_enhanced_feature_engineering.ipynb`, `notebooks/09_enhanced_ml_exploration.ipynb`
- NBA API: `nba_api` Python package
- Model documentation: scikit-learn (Lasso), XGBoost
