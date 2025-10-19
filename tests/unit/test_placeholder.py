"""
Placeholder tests - basic smoke tests for the production pipeline.

These tests verify that modules can be imported and basic functions exist.
Full integration tests should be run separately.
"""

import pytest


def test_imports():
    """Test that all production modules can be imported."""
    # Data collection
    from src.data import collect_data

    assert hasattr(collect_data, "get_top_players")
    assert hasattr(collect_data, "collect_player_gamelogs")
    assert hasattr(collect_data, "collect_team_stats")

    # Feature engineering
    from src.features import build_features

    assert hasattr(build_features, "add_rolling_features")
    assert hasattr(build_features, "create_train_val_test_splits")
    assert hasattr(build_features, "clean_data")

    # Model training
    from src.models import train_models

    assert hasattr(train_models, "get_best_model")
    assert hasattr(train_models, "train_model")
    assert hasattr(train_models, "save_model")

    # Model evaluation
    from src.models import evaluate_models

    assert hasattr(evaluate_models, "evaluate_model_on_split")
    assert hasattr(evaluate_models, "evaluate_all_models")


def test_feature_lists():
    """Test that feature lists are defined correctly."""
    from src.features.build_features import get_feature_lists

    features = get_feature_lists()

    assert "original" in features
    assert "usage" in features
    assert "contextual" in features
    assert "all" in features

    # Check counts
    assert len(features["original"]) == 9
    assert len(features["usage"]) == 8
    assert len(features["contextual"]) == 6
    assert len(features["all"]) == 23


def test_model_selection():
    """Test that model selection works correctly."""
    from src.models.train_models import get_best_model

    # Test PTS model (should be Lasso)
    model_pts, name_pts = get_best_model("PTS")
    assert name_pts == "Lasso"
    assert hasattr(model_pts, "fit")
    assert hasattr(model_pts, "predict")

    # Test REB model (should be XGBoost)
    model_reb, name_reb = get_best_model("REB")
    assert name_reb == "XGBoost"
    assert hasattr(model_reb, "fit")
    assert hasattr(model_reb, "predict")

    # Test AST model (should be XGBoost)
    model_ast, name_ast = get_best_model("AST")
    assert name_ast == "XGBoost"
    assert hasattr(model_ast, "fit")
    assert hasattr(model_ast, "predict")


def test_constants():
    """Test that important constants are defined."""
    from src.models import train_models

    # Check feature sets are defined
    assert len(train_models.ORIGINAL_FEATURES) == 9
    assert len(train_models.USAGE_FEATURES) == 8
    assert len(train_models.CONTEXTUAL_FEATURES) == 6
    assert len(train_models.ALL_FEATURES) == 23

    # Verify feature names
    assert "pts_last_3" in train_models.ORIGINAL_FEATURES
    assert "fga_last_3" in train_models.USAGE_FEATURES
    assert "IS_HOME" in train_models.CONTEXTUAL_FEATURES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
