# NBA Player Performance Prediction (PTS/REB/AST)
Robust, reproducible pipeline to forecast NBA players’ per-game Points, Rebounds, and Assists using historical game logs and contextual features (opponent strength, rest, home/away, travel, etc.). The project spans data collection → cleaning → feature engineering → modeling → evaluation → visualization, with a strong emphasis on reproducibility, code quality, and interpretability.

## 1) Project Description
We will build a fully automated, end-to-end system that:
- Collects and snapshots NBA player game logs and context (team ratings, schedules, rest, travel distance).
- Engineers leakage-safe features (rolling windows, exponentially weighted averages, opponent strength).
- Trains per-target regression models (PTS, REB, AST) with strict time-based validation.
- Benchmarks against naïve rolling-average baselines and reports performance with confidence intervals.
- Serves interactive visualizations to inspect predictions, residuals, and feature importance.

Scope: Regular-season games only for the last 3 NBA seasons, frozen through the 2024–25 regular-season end.

## 2) Clear Goals and Success Criteria
Primary goal:
- Predict per-player per-game PTS, REB, and AST using contextual and historical features.

Quantitative success criteria on a held-out, time-forward test set (see Test Plan):
- Improve MAE over the 5-game rolling-average baseline by at least 12% for each target (PTS, REB, AST).
- Achieve MAE (test) thresholds:
  - PTS: ≤ 3.6
  - REB: ≤ 2.5
  - AST: ≤ 2.2
- Maintain per-player (top 20 by minutes played in test period) MAE improvement ≥ 8% vs baseline for each target.

Secondary goals:
- End-to-end determinism and reproducibility (frozen data, pinned environment, fixed seeds).
- Interpretable insights: global feature importance + local explanations for select games/players.
- Interactive dashboard to explore predictions, errors, and feature impacts.

## 3) Data Collection (What and How)
Freeze window:
- Seasons: 2022–23, 2023–24, 2024–25 (regular season only).
- Data freeze date: 2025-04-15 23:59:59 UTC (end of 2024–25 regular season).

Primary sources:
- NBA Stats API via the `nba_api` Python library (preferred; robust programmatic access).
  - Player game logs (person_id, game_id, date, team_id, opponent_id, minutes, PTS/REB/AST, FGA/3PA/FTA, usage, starter flag).
  - Team context (offensive_rating, defensive_rating, pace; season to date, and trailing windows).
  - Schedules (game dates, home/away, opponent).
- Basketball-Reference (as backup/augmentation; respect robots.txt & TOS):
  - Player game logs and team schedule pages.
  - Use HTTP caching and rate limiting to be a good citizen.

Optional source (if available and permitted):
- Injury/availability reports (game-time statuses). If not reliably available, we proceed without.

Acquisition approach:
- Implement Python collectors under `src/data/`:
  - `collect_player_gamelogs.py`: fetch and write to `data/raw/player_gamelogs_{season}.parquet`.
  - `collect_team_context.py`: fetch team ratings/pace snapshots and write to `data/raw/team_context_{season}.parquet`.
  - `collect_schedule.py`: fetch per-team schedules and write to `data/raw/schedules_{season}.parquet`.
- Reference data:
  - `data/reference/arenas.csv`: arena lat/lon for travel distance (Haversine) and home arena mapping.
  - `data/reference/team_id_map.csv`: mapping among NBA Stats IDs, Basketball-Reference IDs, and names.
- Controls:
  - Rate limit to ≤ 1 request per second with retry/backoff.
  - Cache all HTTP responses locally in `data/cache/`.
  - Log provenance and hashes; snapshot raw files with immutable filenames that include the freeze date.

Core schemas (selected columns; full dictionary in `docs/data_dictionary.md`):
- player_gamelogs:
  - keys: person_id, game_id
  - columns: game_date_utc, team_id, opponent_id, is_home, started, minutes, pts, reb, ast, fga, fta, fg3a, tov, usage_pct, plus_minus
- team_context:
  - keys: team_id, as_of_date_utc
  - columns: off_rating, def_rating, pace, wins, losses
- schedules:
  - keys: team_id, game_id
  - columns: game_date_utc, is_home, opponent_id

## 4) Data Cleaning (Strict Rules)
- Identifier standardization:
  - Player and team IDs normalized to NBA Stats IDs (person_id, team_id). Include crosswalk to Basketball-Reference IDs.
- Missing data:
  - Exclude DNP (Did Not Play) games from training.
  - Impute missing context features with time-aware methods (last observation carried forward, capped by 10-day window); otherwise mark with binary “context_missing” flags and median imputation.
- Integrity checks:
  - Deduplicate by (person_id, game_id).
  - Remove negative or nonsensical stats; cap outliers at the 99.9th percentile per feature (winsorization).
- Alignment:
  - Merge player-level rows with the nearest preceding team context snapshot (no future leakage).
  - Compute rest days based on previous game date per player.
  - Compute travel distance from the previous game’s location to current game’s arena via Haversine.
- Logging:
  - Save cleaning report to `reports/cleaning/cleaning_log_{date}.json` including counts of dropped/imputed records.

## 5) Feature Engineering (Leakage-Safe)
All rolling/exponential features only use games strictly before the current game_date_utc for the same player.

Recent form:
- Rolling averages: last_3 and last_5 for pts, reb, ast, minutes, usage_pct, fga, fta, fg3a, tov.
- EWM (halflife=3 games) for the same set.

Opponent strength and tempo:
- Opponent defensive_rating, offensive_rating, and pace (trailing 10 games and season-to-date).
- Opponent allowances: trailing-10 average pts/REB/AST allowed per position (if position available; else team-level).

Context:
- is_home (binary), rest_days (0/1/2/3+ buckets), back_to_back flag, 3_in_4_nights flag.
- Month of season and pre/post All-Star break flags.
- travel_km since previous game; cumulative minutes in last 7 and 14 days.

Stability and robustness:
- Clip extreme rolling values; add “insufficient_history” flags when fewer than k prior games exist.
- Persist engineered dataset to `data/processed/features.parquet`; columns documented in `docs/features.md`.

## 6) Modeling Plan
Targets:
- Separate regression models for PTS, REB, and AST (three independent models).

Baselines:
- Last-game value baseline.
- 5-game rolling average baseline.
- EWM baseline (halflife=3).

Models:
- Regularized linear: Ridge and Lasso (with standardized numeric features).
- Tree ensembles: RandomForestRegressor, XGBoostRegressor (xgboost).
- Optional: LightGBM if available; otherwise stick to XGBoost.

Feature handling:
- One-hot encode categorical flags; scale continuous features for linear models; tree models use raw scales.
- Feature selection via mutual information and model-based importance (optional, documented if applied).

Hyperparameter search:
- Time-aware cross-validation (rolling origin) on the training period.
- Small, explicit grids to control compute:
  - Ridge/Lasso: alpha in {0.1, 0.5, 1, 2, 5}.
  - RandomForest: n_estimators in {400, 800}, max_depth in {8, 12, None}, min_samples_leaf in {1, 3}.
  - XGBoost: max_depth in {4, 6, 8}, learning_rate in {0.03, 0.06}, n_estimators in {400, 800}, subsample in {0.7, 1.0}, colsample_bytree in {0.7, 1.0}.
- Early stopping for XGBoost with a time-forward validation split.

Artifacts:
- Save the best model per target plus scaler/encoder to `artifacts/models/{target}_{model}.joblib`.
- Save full metrics and CV results to `artifacts/metrics/{target}_metrics.json`.

Interpretability:
- Global: permutation importance + model-native importance (gain for XGBoost).
- Local: SHAP value summaries for a sample of games/players (documented and cached).

## 7) Visualization Plan
- Exploratory plots (static PNGs under `reports/figures/`):
  - Distributions of pts/reb/ast; scatter plots minutes vs pt/REB/AST; residuals vs features.
- Interactive dashboard (local app; Plotly Dash or Streamlit):
  - Predicted vs Actual for each target with drill-down by player and date.
  - Residual distributions and error heatmaps by opponent and rest buckets.
  - Feature importance charts (global) and local explanations per game.
- Launch locally with `make app` (serves at http://127.0.0.1:8050 by default).

## 8) Test Plan (Strict, Time-Based)
Data slicing (regular season only; respect freeze date 2025-04-15):
- Train: 2022-10-01 to 2024-12-31
- Validation: 2025-01-01 to 2025-03-15
- Test (hold-out): 2025-03-16 to 2025-04-15

Evaluation metrics:
- Report MAE (primary) and RMSE (secondary) for each target on validation and test.
- Compare against baselines; report absolute values and % improvement.
- Per-player metrics for top 20 players by test-period minutes played.

Uncertainty:
- Bootstrap test-set (1,000 resamples) to compute 95% CIs for MAE deltas vs baseline.

Leakage checks:
- Unit tests confirm all rolling features use only prior games.
- Ensure no post-game context leaks (e.g., final ratings after game).

CI tests:
- On every commit/PR, run unit tests, linting, and a small pipeline on a 2-week subsample.

Acceptance:
- Must meet the success criteria listed above across the test period and per-player slice.

## 9) Repository Layout (Planned)
- src/
  - data/: data acquisition scripts and ID mapping
  - features/: feature engineering pipeline
  - models/: training, hyperparameter search, evaluation
  - app/: dashboard code
  - utils/: common helpers (io, logging, time splits)
- data/
  - raw/, interim/, processed/, cache/, reference/
- artifacts/
  - models/, metrics/, shap/
- reports/
  - figures/, cleaning/
- docs/
  - data_dictionary.md, features.md, modeling.md
- tests/
  - unit/, integration/
- .github/workflows/
  - ci.yml
- Makefile, pyproject.toml (or requirements.txt), README.md
