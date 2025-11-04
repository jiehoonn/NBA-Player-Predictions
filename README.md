# NBA Player Performance Prediction

**Predicting NBA player per-game statistics (PTS, REB, AST) using machine learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎥 Presentation Video

[Coming Soon - December 2025]

---

## 🚀 Quick Start

**Get predictions in 3 commands:**

```bash
make install   # Install dependencies (~2 min)
make all       # Run full pipeline (~25 min)
make app       # Launch dashboard (optional)
```

That's it! Results will be in `results/`.

---

## 📊 Results

### Performance Summary

| Target | Our Model | Baseline | Improvement | Goal |
|--------|-----------|----------|-------------|------|
| PTS    | TBD       | TBD      | TBD%        | ≤4.50 |
| REB    | TBD       | TBD      | TBD%        | ≤2.00 |
| AST    | TBD       | TBD      | TBD%        | ≤1.50 |

*Baseline: 5-game rolling average*

---

## 📖 Project Description

This project implements a complete machine learning pipeline to predict NBA player performance:

- **Data Collection:** 5 seasons (2019-2024), 200+ players, ~60K games
- **Feature Engineering:** 40+ features (rolling averages, opponent stats, shot locations)
- **Models:** Ensemble (Lasso + XGBoost + LightGBM) with time-based validation
- **Evaluation:** Multiple slices (overall, by player, by context)

### Why This Project?

Practice the full data science lifecycle:
1. ✅ Data collection (NBA API)
2. ✅ Data cleaning (handle missing data, outliers)
3. ✅ Feature extraction (40+ engineered features)
4. ✅ Data visualization (15+ plots + interactive dashboard)
5. ✅ Model training (multiple algorithms, hyperparameter tuning)

---

## 🏗️ How to Build and Run

### Prerequisites

- Python 3.10 or higher
- 4GB RAM (8GB recommended)
- ~2GB disk space
- Internet connection (for data collection)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/NBA-Player-Predictions.git
cd NBA-Player-Predictions

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
make install
```

### Running the Pipeline

**Option 1: All at once (recommended for first run)**

```bash
make all
```

This runs:
1. `make data` - Collect NBA data (~15-20 min due to API rate limits)
2. `make features` - Engineer features (~2 min)
3. `make train` - Train models (~5 min)
4. `make evaluate` - Generate metrics (~1 min)
5. `make visualize` - Create plots (~1 min)

**Option 2: Step by step**

```bash
make data         # Collect data
make features     # Engineer features
make train        # Train models
make evaluate     # Generate metrics
make visualize    # Create visualizations
```

**Option 3: Interactive dashboard**

```bash
make app  # Opens http://localhost:8501
```

---

## 🧪 How to Test the Code

### Run All Tests

```bash
make test
```

This runs:
- Unit tests (feature engineering, model logic)
- Integration tests (pipeline compatibility)
- Coverage report (target: >80%)

### Run Specific Tests

```bash
# Unit tests only
pytest tests/unit/ -v

# Specific test file
pytest tests/test_features.py -v

# Specific test
pytest tests/test_features.py::test_leakage_prevention -v
```

### Linting

```bash
# Check code quality
make lint

# Auto-format code
make format
```

---

## 📁 Project Structure

```
NBA-Player-Predictions/
├── data/                      # Data files (gitignored)
│   ├── raw/                   # Raw NBA data
│   ├── processed/             # Engineered features
│   └── cache/                 # API response cache
├── src/                       # Production code
│   ├── collect.py             # Data collection
│   ├── clean.py               # Data cleaning
│   ├── features.py            # Feature engineering
│   ├── train.py               # Model training
│   ├── evaluate.py            # Evaluation
│   ├── visualize.py           # Plotting
│   └── utils.py               # Utilities
├── notebooks/                 # Jupyter notebooks (exploration)
│   └── exploration.ipynb
├── tests/                     # Test suite
│   ├── test_features.py
│   ├── test_models.py
│   └── test_pipeline.py
├── models/                    # Trained models (gitignored)
├── results/                   # Outputs
│   ├── figures/               # Plots
│   └── metrics/               # Performance metrics
├── config.yaml                # Configuration
├── Makefile                   # Build automation
├── requirements.txt           # Dependencies
├── app.py                     # Streamlit dashboard
└── README.md                  # This file
```

---

## 🔬 Methodology

### Data Collection

**Source:** NBA Stats API via `nba_api` library

**Data gathered:**
- Player game logs (PTS, REB, AST, shooting stats, minutes, etc.)
- Team stats (offensive/defensive ratings, pace)
- Shot location data (restricted area, paint, mid-range, 3PT%)
- Schedule data (rest days, home/away, back-to-backs)

**Time period:** 5 seasons (2019-2024), top 200 players by minutes

### Data Cleaning

1. Remove DNP (Did Not Play) games
2. Deduplicate by (player, game)
3. Handle missing values (time-aware imputation)
4. Remove outliers (winsorization at 99.9th percentile)

### Feature Engineering

**40+ leakage-safe features:**

**Rolling statistics (past performance):**
- 3/5/10-game averages for PTS, REB, AST, MIN, FG%, 3P%, FT%

**Contextual features:**
- Home/away indicator
- Rest days (0, 1, 2, 3+)
- Back-to-back games
- Opponent defensive rating
- Opponent pace

**Advanced features:**
- Shot location percentages (restricted area, paint, mid-range, 3PT)
- Efficiency metrics (True Shooting%, Effective FG%, Usage%)
- Matchup history (vs this opponent)
- Trend indicators (hot hand, recent form)

**Critical:** All features use `.shift(1).rolling()` pattern to prevent data leakage!

### Modeling

**Baseline:** 5-game rolling average

**Models:**

- **PTS:** Ensemble (Lasso + XGBoost + LightGBM)
  - Weighted average: 30% Lasso, 35% XGBoost, 35% LightGBM
- **REB:** XGBoost
- **AST:** XGBoost

**Validation:** Time-based splits (no shuffling!)
- Train: Before 2024-01-01
- Validation: 2024-01-01 to 2024-07-01
- Test: After 2024-07-01

### Evaluation

**Metrics:** MAE (primary), RMSE, R², MAPE

**Analysis slices:**
- Overall test set
- Top 20 players by minutes
- By position (G/F/C)
- By rest days (0/1/2/3+)
- Home vs away

---

## 📊 Visualizations

### Static Plots

- Distribution plots (PTS, REB, AST)
- Correlation heatmap
- Predicted vs Actual scatter plots
- Error distributions
- Feature importance charts
- Residual analysis

All saved to `results/figures/`

### Interactive Dashboard

Run `make app` to launch Streamlit dashboard with:
- Player selection
- Prediction display
- Feature contributions
- Historical trends
- Error analysis

---

## 🤝 Contributing

### Development Workflow

1. Create feature branch: `git checkout -b feature/name`
2. Make changes
3. Run tests: `make test`
4. Format code: `make format`
5. Commit and push
6. Create pull request

### Code Style

- Follow PEP 8
- Use `black` formatter (100 char line length)
- Add docstrings (Google style)
- Write tests for new features

---

## 💻 Environment Support

**Tested on:**
- macOS (M1/Intel)
- Ubuntu 20.04+
- Windows 10/11

**Python versions:** 3.10, 3.11, 3.12

---

## 📚 References

- [nba_api documentation](https://github.com/swar/nba_api)
- [NBA Stats API](https://www.nba.com/stats)
- Previous implementation: See `MidtermReport.md`
- Goals and benchmarks: See `GOALS.md`
- Technical decisions: See `CLAUDE.md`

---

## 📄 License

MIT License - see LICENSE file for details

---

**Last Updated:** November 2025
