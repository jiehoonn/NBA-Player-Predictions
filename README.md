# NBA Player Performance Prediction (PTS/REB/AST)

This repository contains a reproducible pipeline to forecast NBA players’ per-game **points**, **rebounds**, and **assists** using historical game logs and a suite of contextual features (opponent strength, rest, home/away, travel, etc.).  
The project is scoped to the **regular season** of the last three NBA seasons and uses the `nba_api` library as its primary data source.

One of the key pieces of feedback we received was to prioritise a **minimal viable pipeline** over an overly complex initial implementation.  
To address this, our plan now emphasises getting the core pipeline working end-to-end (data collection → cleaning → feature engineering → baseline and main predictive models → evaluation), and treating more advanced elements (e.g., SHAP explanations, opponent allowances by position, an interactive dashboard) as **optional extensions** if time permits.

---

## 1) Project Description

We aim to build an automated, end-to-end system that:

- Collects and snapshots NBA player game logs and context (team ratings, schedules, rest, travel distance).
- Engineers leakage-safe features using only information available **before** each game.
- Trains per-target regression models (PTS, REB, AST) with strict time-based validation.
- Benchmarks performance against naïve rolling-average baselines and reports results with confidence intervals.
- Provides reproducible code and interpretable insights.

### Minimal viable scope
The first milestone focuses on:
- Collecting data
- Implementing cleaning rules
- Generating a set of sensible features (rolling averages and a small number of context variables)
- Training baseline models and **one main predictive model per target**
- Evaluating on a time-forward split

**Complex features** (travel distance, opponent allowances by position, SHAP explanations, interactive dashboard) are clearly delineated as **optional extensions** to pursue if time allows.

### Scope details
- **Data window:** Regular season games from 2022-23, 2023-24, and 2024-25, frozen through **2025-04-15 23:59:59 UTC** (end of the 2024-25 regular season).
- **Targets:** Per-player per-game points (PTS), rebounds (REB), and assists (AST).
- **Reproducibility:** All steps are deterministic with pinned software versions and fixed random seeds. Data snapshots are saved with immutable file names containing the freeze date.

---

## 2) Clear Goals and Success Criteria

### Primary goal
Predict per-player per-game PTS, REB, and AST using contextual and historical features.

### Quantitative success criteria
On a held-out, time-forward test set (see **Test Plan**):
- Improve mean absolute error (MAE) over a 5-game rolling-average baseline by **≥ 12 %** for each target.
- Achieve target MAE thresholds:
  - **PTS:** ≤ 3.6
  - **REB:** ≤ 2.5
  - **AST:** ≤ 2.2
- For the **top 20 players by minutes played** in the test period, improve MAE by **≥ 8 %** versus the baseline for each target.

### Secondary goals (stretch)
These goals will be tackled **if time allows**:
- End-to-end determinism and reproducibility (frozen data, pinned environment, fixed seeds).
- Interpretable insights via global feature importance and local explanations using SHAP.
- An interactive dashboard for exploring predictions, residuals, and feature impacts.

---

## 3) Data Collection (What and How)

### Freeze window
- **Seasons:** 2022-23, 2023-24, 2024-25 (regular season only).
- **Data freeze date:** **2025-04-15 23:59:59 UTC**.

### Primary sources
- **NBA Stats API** via the `nba_api` Python library (preferred):
  - Player game logs (person_id, game_id, date, team_id, opponent_id, minutes, PTS/REB/AST, FGA/3PA/FTA, usage, starter flag).
  - Team context snapshots (offensive_rating, defensive_rating, pace; season-to-date and trailing windows).
  - Schedules (game dates, home/away, opponent).
- **Basketball-Reference** (backup/augmentation; used sparingly and in compliance with robots.txt):
  - Player game logs and team schedule pages.
  - Fills gaps if encountered with the primary API.
- **Injury/availability reports** are optional; we will only pursue them if a reliable source can be integrated easily.

### Acquisition approach
Implement Python collectors under `src/data/`:
- `collect_player_gamelogs.py`: write to `data/raw/player_gamelogs_{season}.parquet`
- `collect_team_context.py`: write to `data/raw/team_context_{season}.parquet`
- `collect_schedule.py`: write to `data/raw/schedules_{season}.parquet`

**Reference data:**
- `data/reference/arenas.csv`: arena lat/lon for travel distance (Haversine) and home arena mapping.
- `data/reference/team_id_map.csv`: mapping among NBA Stats IDs, Basketball-Reference IDs, and names.

**Controls:**
- Rate limit to **≤ 1 req/sec** with retry/backoff.
- Cache all HTTP responses in `data/cache/` for reproducibility.
- Log provenance and hashes; snapshot raw files with immutable filenames that include the freeze date.

---

## 4) Data Cleaning (Strict Rules)

Cleaning rules ensure data quality while preventing leakage.

**Identifier standardisation**
- Normalise player and team IDs to NBA Stats IDs (`person_id`, `team_id`) and include a crosswalk to Basketball-Reference IDs.

**Missing data**
- Exclude **DNP** (Did Not Play) games from training.
- Impute missing context features using time-aware methods (last observation carried forward capped at a **10-day** window); otherwise mark with binary `context_missing` flags and median imputation.

**Integrity checks**
- Deduplicate by `(person_id, game_id)`.
- Remove negative or nonsensical stats; **winsorise** outliers at the **99.9th percentile** per feature.

**Alignment and derived variables**
- Merge player-level rows with the **nearest preceding** team context snapshot (no future leakage).
- Compute **rest days** based on the previous game date per player.
- **Travel distance features (optional):** compute Haversine distance from the previous game’s location to the current arena using the arena mapping.  
  For the minimal pipeline, a simpler **home/away** flag and **rest days** will suffice.

**Logging**
- Save a cleaning report to `reports/cleaning/cleaning_log_{date}.json` with counts of dropped and imputed records.

---

## 5) Feature Engineering (Leakage-Safe)

All rolling/exponential features use only games **strictly before** the current game date for the same player.

**Recent form**
- Rolling averages: `last_3` and `last_5` for `pts`, `reb`, `ast`, `minutes`, `usage_pct`, `fga`, `fta`, `fg3a`, `tov`.
- Exponentially weighted mean (EWM) with **halflife = 3 games** for the same set.

**Opponent strength and tempo**
- Opponent `defensive_rating`, `offensive_rating`, and `pace` based on **trailing 10 games** and **season-to-date** snapshots.
- **Optional extension:** opponent allowances by position (average PTS/REB/AST allowed per position over trailing 10 games) if reliable positional data can be sourced.

**Context**
- `is_home` flag (binary)
- `rest_days` bucketed as **0, 1, 2, 3+**
- `back_to_back` flag and `three_in_four` flag
- Month of season and a **pre/post All-Star break** flag
- **Optional extension:** travel distance since the previous game; cumulative minutes in the last **7** and **14** days

**Stability and robustness**
- Clip extreme rolling values; add `insufficient_history` flags when fewer than *k* prior games exist.
- Persist the engineered dataset to `data/processed/features.parquet`; document columns in `docs/features.md`.

---

## 6) Modeling Plan

**Targets**  
Three independent regression tasks: **PTS**, **REB**, and **AST**.

**Baselines**
- Last-game value baseline
- 5-game rolling average baseline
- EWM baseline (halflife = 3)

**Models**
To keep the initial scope manageable, we will start with a **regularised linear model** and **one tree-ensemble** model, using a small grid of hyperparameters. If time permits, we can explore additional models.

- Regularised linear: **Ridge**, **Lasso** (with standardised numeric features)
- Tree ensembles: **RandomForestRegressor**, **XGBoostRegressor** (`xgboost` library)  
  *LightGBM may be explored as a stretch goal.*

**Feature handling**
- One-hot encode categorical flags; scale continuous features for linear models; tree models use raw scales.
- Feature selection via mutual information or model-based importance is optional and will be documented if applied.

**Hyperparameter search**
- Use **time-aware cross-validation** (rolling origin) on the training period.
- Small, explicit grids to control compute:
  - Ridge/Lasso: `alpha ∈ {0.1, 0.5, 1, 2, 5}`
  - RandomForest: `n_estimators ∈ {400, 800}`, `max_depth ∈ {8, 12, None}`, `min_samples_leaf ∈ {1, 3}`
  - XGBoost: `max_depth ∈ {4, 6, 8}`, `learning_rate ∈ {0.03, 0.06}`, `n_estimators ∈ {400, 800}`, `subsample ∈ {0.7, 1.0}`, `colsample_bytree ∈ {0.7, 1.0}`
- For XGBoost, use **early stopping** with a time-forward validation split.

**Artifacts**
- Save the best model per target plus any scaler/encoder to `artifacts/models/{target}_{model}.joblib`.
- Save full metrics and cross-validation results to `artifacts/metrics/{target}_metrics.json`.

**Interpretability (stretch)**
- **Global:** permutation importance and (for XGBoost) built-in feature importance metrics.
- **Local:** SHAP value summaries for a sample of games/players.  
  *This is optional and will only be pursued after the minimal pipeline is complete.*

---

## 7) Visualization Plan

**Exploratory plots (static)**  
Create static PNGs under `reports/figures/` to explore data and model behaviour:
- Distributions of points, rebounds, and assists
- Scatter plots of minutes vs. points/rebounds/assists
- Residuals vs. key features

**Interactive dashboard (stretch)**  
Implement a local interactive dashboard using **Plotly Dash** or **Streamlit** only after the core pipeline is complete. If developed, the app will support:
- Predicted vs. actual values for each target with drill-down by player and date
- Residual distributions and error heatmaps by opponent and rest buckets
- Feature importance charts (global) and local explanations per game

Launch locally with:

```bash
make app
```
(default: http://127.0.0.1:8050)

---

## 8) Test Plan (Strict, Time-Based)

### Data slicing
Use only regular season games, respecting the freeze date **2025-04-15**:
- **Train:** 2022-10-01 to 2024-12-31  
- **Validation:** 2025-01-01 to 2025-03-15  
- **Test (hold-out):** 2025-03-16 to 2025-04-15

### Evaluation metrics
- Report **MAE** (primary) and **RMSE** (secondary) for each target on the validation and test sets.
- Compare against baselines; report **absolute values** and **percentage improvement**.
- Compute **per-player metrics** for the top 20 players by minutes played in the test period.

### Uncertainty quantification
- Bootstrap the test set (**1,000 resamples**) to compute **95% confidence intervals** for MAE deltas versus baseline.

### Leakage checks
- Unit tests confirm that all rolling features use **only prior games**.
- Verify that no **post-game context** leaks (e.g., final ratings or season averages after the game).

### Continuous Integration (CI) tests
- On every commit/PR, run **unit tests**, **linting**, and a **small pipeline** on a two-week subsample.

### Acceptance criteria
- The **minimal pipeline** must meet the success criteria listed above across the test period and per-player slice.
- **Advanced features** are optional and will **not** block acceptance of the project.

---

## 9) Repository Layout

The repository is structured to support modular development and reproducibility:

- `src/`
  - `data/`: data acquisition scripts and ID mapping
  - `features/`: feature engineering pipeline
  - `models/`: training, hyperparameter search, evaluation
  - `app/`: dashboard code (**optional**)
  - `utils/`: common helpers (IO, logging, time splits)
- `data/`
  - `raw/`, `interim/`, `processed/`, `cache/`, `reference/`
- `artifacts/`
  - `models/`, `metrics/`, `shap/`
- `reports/`
  - `figures/`, `cleaning/`
- `docs/`
  - `data_dictionary.md`, `features.md`, `modeling.md`
- `tests/`
  - `unit/`, `integration/`
- `.github/workflows/`
  - `ci.yml` (continuous integration pipeline)
- `Makefile`, `pyproject.toml` (or `requirements.txt`), `README.md`

---

## 10) Prioritisation Plan

Given the **two-month** timeline, our strategy is to deliver the **core pipeline early**, ensuring data collection, cleaning, feature engineering, baseline models, and **one main model** are working end-to-end.

We will then iterate and, **if time permits**:
- Expand the feature set (e.g., travel distance, opponent positional allowances).
- Explore additional models.
- Implement interpretability tools.
- Build an interactive dashboard.

Any advanced component can be **dropped or postponed** without jeopardising the ability to meet the **core success criteria**.
