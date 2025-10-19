"""
Integration tests for the NBA prediction pipeline.

These tests verify that the pipeline components work together correctly.
They use small sample datasets to avoid long API calls.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


def test_sample_data_exists():
    """Test that sample data file exists for quick testing (if available locally)."""
    sample_file = Path("data/raw/player_gamelogs_2023-24_sample.parquet")

    # Skip test if sample file doesn't exist (e.g., on CI)
    if not sample_file.exists():
        pytest.skip(
            f"Sample data not found at {sample_file} - skipping (expected on CI)"
        )

    # If we get here, file exists - verify it's readable
    assert sample_file.exists()
    df = pd.read_parquet(sample_file)
    assert len(df) > 0, "Sample data file is empty"


def test_feature_engineering_pipeline():
    """Test that feature engineering produces expected output structure."""
    from src.features.build_features import get_feature_lists

    features = get_feature_lists()

    # Verify all feature groups exist
    assert "original" in features
    assert "usage" in features
    assert "contextual" in features
    assert "all" in features

    # Verify feature counts
    assert len(features["original"]) == 9
    assert len(features["usage"]) == 8
    assert len(features["contextual"]) == 6
    assert len(features["all"]) == 23

    # Verify no duplicates
    assert len(set(features["all"])) == 23


def test_model_training_components():
    """Test that model training components are properly configured."""
    from src.models.train_models import get_best_model, ORIGINAL_FEATURES, ALL_FEATURES

    # Test that we can instantiate all models
    for target in ["PTS", "REB", "AST"]:
        model, model_name = get_best_model(target)
        assert model is not None
        assert model_name in ["Lasso", "XGBoost"]

        # Verify models have required methods
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    # Verify feature sets
    assert len(ORIGINAL_FEATURES) == 9
    assert len(ALL_FEATURES) == 23


def test_data_collection_functions():
    """Test that data collection functions are properly defined."""
    from src.data.collect_data import (
        get_top_players,
        collect_player_gamelogs,
        collect_team_stats,
        add_game_context,
        merge_opponent_stats,
    )

    # Verify all functions exist and are callable
    assert callable(get_top_players)
    assert callable(collect_player_gamelogs)
    assert callable(collect_team_stats)
    assert callable(add_game_context)
    assert callable(merge_opponent_stats)


def test_evaluation_functions():
    """Test that evaluation functions are properly defined."""
    from src.models.evaluate_models import (
        evaluate_model_on_split,
        evaluate_all_models,
        print_summary_table,
        compare_to_goals,
    )

    # Verify all functions exist and are callable
    assert callable(evaluate_model_on_split)
    assert callable(evaluate_all_models)
    assert callable(print_summary_table)
    assert callable(compare_to_goals)


def test_pipeline_compatibility():
    """Test that pipeline components are compatible with expected data structures."""
    # Create a minimal mock dataframe with expected structure
    mock_data = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1, 1, 1, 1],
            "GAME_DATE": pd.date_range("2024-01-01", periods=6),
            "PTS": [20, 25, 22, 23, 21, 24],
            "REB": [5, 6, 5, 7, 6, 5],
            "AST": [3, 4, 3, 5, 4, 3],
            "MIN": [30, 32, 31, 33, 30, 32],
            "FGA": [15, 18, 16, 17, 15, 16],
            "FTA": [5, 6, 5, 5, 4, 6],
            "FG3A": [3, 4, 3, 4, 3, 3],
            "FG_PCT": [0.45, 0.48, 0.44, 0.46, 0.47, 0.45],
        }
    )

    # Test that we can create rolling features
    from src.features.build_features import add_rolling_features

    result = add_rolling_features(mock_data, windows=[3, 5])

    # Verify rolling features were created
    assert "pts_last_3" in result.columns
    assert "pts_last_5" in result.columns
    assert "games_played" in result.columns

    # Verify data integrity
    assert len(result) == len(mock_data)
    assert result["PLAYER_ID"].equals(mock_data["PLAYER_ID"])


def test_config_file_exists():
    """Test that configuration file exists and is valid."""
    config_file = Path("config.yaml")
    assert config_file.exists(), "config.yaml not found"

    # Try to load it
    import yaml

    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Verify key sections exist
    assert "data_collection" in config
    assert "features" in config
    assert "models" in config
    assert "evaluation" in config


def test_makefile_exists():
    """Test that Makefile exists with expected targets."""
    makefile = Path("Makefile")
    assert makefile.exists(), "Makefile not found"

    content = makefile.read_text()

    # Verify key targets exist
    assert "data:" in content
    assert "features:" in content
    assert "models:" in content
    assert "evaluate:" in content
    assert "visualize:" in content
    assert "all:" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
