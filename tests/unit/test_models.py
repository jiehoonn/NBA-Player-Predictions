"""
Unit tests for model training and selection.

These tests validate model selection logic, hyperparameters,
and training functionality.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from src.models.train_models import get_best_model, train_model, ORIGINAL_FEATURES


def test_model_selection_pts():
    """Test that PTS uses Lasso with correct hyperparameters."""
    model, model_name = get_best_model("PTS")

    assert model_name == "Lasso", "PTS should use Lasso"
    assert isinstance(model, Lasso), "Should return Lasso instance"

    # Verify hyperparameters from Notebook 09
    assert model.alpha == 0.1, "Lasso alpha should be 0.1"
    assert model.random_state == 42, "Random state should be 42"
    assert model.max_iter == 2000, "Max iter should be 2000"


def test_model_selection_reb():
    """Test that REB uses XGBoost with correct hyperparameters."""
    model, model_name = get_best_model("REB")

    assert model_name == "XGBoost", "REB should use XGBoost"
    assert isinstance(model, XGBRegressor), "Should return XGBRegressor instance"

    # Verify hyperparameters from Notebook 09
    assert model.n_estimators == 100, "n_estimators should be 100"
    assert model.max_depth == 3, "max_depth should be 3"
    assert model.learning_rate == 0.05, "learning_rate should be 0.05"
    assert model.random_state == 42, "Random state should be 42"


def test_model_selection_ast():
    """Test that AST uses XGBoost with correct hyperparameters."""
    model, model_name = get_best_model("AST")

    assert model_name == "XGBoost", "AST should use XGBoost"
    assert isinstance(model, XGBRegressor), "Should return XGBRegressor instance"

    # Verify same hyperparameters as REB
    assert model.n_estimators == 100
    assert model.max_depth == 3
    assert model.learning_rate == 0.05


def test_train_model_produces_predictions():
    """Test that train_model can actually train and predict."""
    # Create minimal synthetic training data
    np.random.seed(42)
    n_samples = 100

    # Create features (use first 3 from ORIGINAL_FEATURES for simplicity)
    features = ORIGINAL_FEATURES[:3]

    train_df = pd.DataFrame(
        {
            "pts_last_3": np.random.uniform(15, 25, n_samples),
            "pts_last_5": np.random.uniform(15, 25, n_samples),
            "reb_last_3": np.random.uniform(5, 10, n_samples),
            "PTS": np.random.uniform(18, 28, n_samples),  # Target
        }
    )

    val_df = train_df.copy()
    test_df = train_df.copy()

    # Train model
    result = train_model(train_df, val_df, test_df, "PTS", features)

    # Verify result structure
    assert "model" in result
    assert "model_name" in result
    assert "scaler" in result
    assert "features" in result
    assert "results" in result

    # Verify model can predict
    assert hasattr(result["model"], "predict")

    # Verify results contain metrics for all splits
    assert "train" in result["results"]
    assert "val" in result["results"]
    assert "test" in result["results"]

    # Verify metrics are reasonable (MAE should be positive and finite)
    assert result["results"]["train"]["mae"] > 0
    assert np.isfinite(result["results"]["train"]["mae"])
    assert result["results"]["test"]["r2"] <= 1.0


def test_lasso_uses_scaling():
    """Test that Lasso models use StandardScaler."""
    model, model_name = get_best_model("PTS")
    assert model_name == "Lasso"

    # Create minimal test data
    features = ORIGINAL_FEATURES[:3]
    train_df = pd.DataFrame(
        {
            "pts_last_3": [20, 22, 21],
            "pts_last_5": [19, 23, 20],
            "reb_last_3": [5, 6, 5],
            "PTS": [24, 26, 23],
        }
    )
    val_df = train_df.copy()
    test_df = train_df.copy()

    result = train_model(train_df, val_df, test_df, "PTS", features)

    # Lasso should have a scaler
    assert result["scaler"] is not None, "Lasso should use StandardScaler"
    assert hasattr(result["scaler"], "transform"), "Scaler should have transform method"


def test_xgboost_no_scaling():
    """Test that XGBoost models don't use scaling."""
    model, model_name = get_best_model("REB")
    assert model_name == "XGBoost"

    # Create minimal test data (need baseline column for train_model)
    features = ORIGINAL_FEATURES[:3]
    train_df = pd.DataFrame(
        {
            "pts_last_3": [20, 22, 21],
            "pts_last_5": [19, 23, 20],
            "reb_last_3": [5, 6, 5],
            "reb_last_5": [5, 6, 5],  # Baseline column
            "REB": [6, 7, 5],
        }
    )
    val_df = train_df.copy()
    test_df = train_df.copy()

    result = train_model(train_df, val_df, test_df, "REB", features)

    # XGBoost should NOT have a scaler
    assert result["scaler"] is None, "XGBoost should not use scaling"


def test_baseline_calculation():
    """Test that baseline is calculated using 5-game rolling average."""
    features = ORIGINAL_FEATURES[:3]

    # Create data where baseline is known
    train_df = pd.DataFrame(
        {
            "pts_last_3": [20] * 10,
            "pts_last_5": [25] * 10,  # Baseline prediction
            "reb_last_3": [5] * 10,
            "PTS": [24] * 10,  # Actual values
        }
    )
    val_df = train_df.copy()
    test_df = train_df.copy()

    result = train_model(train_df, val_df, test_df, "PTS", features)

    # Baseline MAE = |24 - 25| = 1.0
    expected_baseline_mae = 1.0
    assert abs(result["baseline_mae"] - expected_baseline_mae) < 0.01


def test_improvement_calculation():
    """Test that improvement percentage is calculated correctly."""
    features = ORIGINAL_FEATURES[:3]

    # Create data where we can control the outcome
    train_df = pd.DataFrame(
        {
            "pts_last_3": [20] * 20,
            "pts_last_5": [20] * 20,  # Baseline predicts 20
            "reb_last_3": [5] * 20,
            "PTS": [24] * 20,  # Actual is 24
        }
    )
    val_df = train_df.copy()
    test_df = train_df.copy()

    result = train_model(train_df, val_df, test_df, "PTS", features)

    # Baseline MAE = 4.0 (|24 - 20|)
    # If model gets MAE = 3.6, improvement = (4.0 - 3.6) / 4.0 * 100 = 10%
    assert result["baseline_mae"] == 4.0

    # Model should improve over baseline (improvement > 0)
    # Can't guarantee exact value, but should be better than always predicting 20
    assert result["improvement"] > 0, "Model should improve over baseline"


def test_feature_constants():
    """Test that feature constant lists are defined correctly."""
    from src.models.train_models import (
        ORIGINAL_FEATURES,
        USAGE_FEATURES,
        CONTEXTUAL_FEATURES,
        ALL_FEATURES,
        PHASE1_FEATURES,
    )

    # Check correct sizes
    assert len(ORIGINAL_FEATURES) == 9
    assert len(USAGE_FEATURES) == 8
    assert len(CONTEXTUAL_FEATURES) == 6
    # Now we have 38 total features (23 baseline + 15 Phase 1 advanced)
    assert len(ALL_FEATURES) == 38

    # Check specific features exist
    assert "pts_last_3" in ORIGINAL_FEATURES
    assert "fga_last_3" in USAGE_FEATURES
    assert "IS_HOME" in CONTEXTUAL_FEATURES

    # Check ALL_FEATURES contains baseline features
    combined = set(ORIGINAL_FEATURES + USAGE_FEATURES + CONTEXTUAL_FEATURES)
    assert combined.issubset(set(ALL_FEATURES)), "Baseline features should be in ALL_FEATURES"

    # Validate Phase 1 feature set
    assert len(PHASE1_FEATURES) == 15, "Phase 1 should have exactly 15 features"
    assert len(set(PHASE1_FEATURES)) == 15, "Phase 1 features should have no duplicates"
    assert set(PHASE1_FEATURES).issubset(set(ALL_FEATURES)), "Phase 1 features should be in ALL_FEATURES"

    # Verify Phase 1 features are disjoint from baseline features
    baseline_features = set(ORIGINAL_FEATURES + USAGE_FEATURES + CONTEXTUAL_FEATURES)
    assert set(PHASE1_FEATURES).isdisjoint(baseline_features), "Phase 1 features should not overlap with baseline"

    # Spot-check specific Phase 1 features
    assert "ts_pct_last_3" in PHASE1_FEATURES, "TS% should be in Phase 1"
    assert "pts_last_game" in PHASE1_FEATURES, "Last game stats should be in Phase 1"
    assert "plus_minus_last_3" in PHASE1_FEATURES, "Plus/minus should be in Phase 1"
    assert "pts_trend_last_5" in PHASE1_FEATURES, "Trends should be in Phase 1"
    assert "pts_std_last_5" in PHASE1_FEATURES, "Consistency metrics should be in Phase 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
