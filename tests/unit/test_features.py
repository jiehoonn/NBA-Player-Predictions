"""
Unit tests for feature engineering.

These tests validate the actual functionality of feature creation,
ensuring leakage prevention and correct calculations.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.build_features import (
    add_rolling_features,
    create_train_val_test_splits,
    clean_data,
    get_feature_lists,
)


def test_rolling_features_calculation():
    """Test that rolling averages are calculated correctly."""
    # Create test data with known values (including Phase 1 feature columns)
    test_data = pd.DataFrame(
        {
            "PLAYER_ID": [1] * 10,
            "GAME_DATE": pd.date_range("2024-01-01", periods=10),
            "PTS": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "REB": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            "AST": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "MIN": [30] * 10,
            "FGA": [10] * 10,
            "FTA": [5] * 10,
            "FG3A": [3] * 10,
            "FG_PCT": [0.5] * 10,
            "TOV": [2] * 10,  # Phase 1: Turnover data
            "PLUS_MINUS": [5] * 10,  # Phase 1: Plus/Minus data
        }
    )

    result = add_rolling_features(test_data, windows=[3])

    # Test leakage prevention: first game should have NaN for rolling features
    assert pd.isna(result.iloc[0]["pts_last_3"]), "First game should have NaN"

    # Test 3-game rolling average calculation
    # Game 3 (index 2): avg of games 0, 1 (10, 20) = 15
    assert result.iloc[2]["pts_last_3"] == 15.0, "3-game rolling avg incorrect"

    # Game 4 (index 3): avg of games 0, 1, 2 (10, 20, 30) = 20
    assert result.iloc[3]["pts_last_3"] == 20.0, "3-game rolling avg incorrect"

    # Game 5 (index 4): avg of games 1, 2, 3 (20, 30, 40) = 30
    assert result.iloc[4]["pts_last_3"] == 30.0, "3-game rolling avg incorrect"


def test_leakage_prevention():
    """Test that future games don't leak into past predictions."""
    test_data = pd.DataFrame(
        {
            "PLAYER_ID": [1] * 5,
            "GAME_DATE": pd.date_range("2024-01-01", periods=5),
            "PTS": [100, 10, 10, 10, 10],  # First game is outlier
            "REB": [5] * 5,
            "AST": [3] * 5,
            "MIN": [30] * 5,
            "FGA": [10] * 5,
            "FTA": [5] * 5,
            "FG3A": [3] * 5,
            "FG_PCT": [0.5] * 5,
            "TOV": [2] * 5,  # Phase 1
            "PLUS_MINUS": [5] * 5,  # Phase 1
        }
    )

    result = add_rolling_features(test_data, windows=[3])

    # Game 1 (index 1): should use only game 0 (100) = 100
    assert result.iloc[1]["pts_last_3"] == 100.0, "Should only use game 0"

    # Game 2 (index 2): should use games 0, 1 (100, 10) = 55
    assert result.iloc[2]["pts_last_3"] == 55.0, "Should use games 0, 1"

    # Game 3 (index 3): should use games 0, 1, 2 (100, 10, 10) = 40
    expected = (100 + 10 + 10) / 3
    assert (
        abs(result.iloc[3]["pts_last_3"] - expected) < 0.01
    ), "Should use games 0, 1, 2"


def test_time_based_splits():
    """Test that train/val/test splits are chronologically correct."""
    test_data = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(
                [
                    "2024-01-01",  # train
                    "2024-06-01",  # train
                    "2024-08-01",  # val (after train_end)
                    "2024-10-01",  # val
                    "2025-01-01",  # test (after val_end)
                    "2025-03-01",  # test
                ]
            )
        }
    )

    result = create_train_val_test_splits(
        test_data, train_end="2024-07-31", val_end="2024-12-31"
    )

    # Check splits are correct
    assert result.iloc[0]["SPLIT"] == "train"
    assert result.iloc[1]["SPLIT"] == "train"
    assert result.iloc[2]["SPLIT"] == "val"
    assert result.iloc[3]["SPLIT"] == "val"
    assert result.iloc[4]["SPLIT"] == "test"
    assert result.iloc[5]["SPLIT"] == "test"

    # Verify no time leakage
    train_dates = result[result["SPLIT"] == "train"]["GAME_DATE"]
    val_dates = result[result["SPLIT"] == "val"]["GAME_DATE"]
    test_dates = result[result["SPLIT"] == "test"]["GAME_DATE"]

    assert train_dates.max() < val_dates.min(), "Train must end before val starts"
    assert val_dates.max() < test_dates.min(), "Val must end before test starts"


def test_clean_data_removes_insufficient_history():
    """Test that games with insufficient history are removed."""
    test_data = pd.DataFrame(
        {
            "PLAYER_ID": [1] * 10,
            "PTS": [20] * 10,
            "REB": [5] * 10,
            "AST": [3] * 10,
            "games_played": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # First 5 should be removed
        }
    )

    result = clean_data(test_data)

    # Should only keep games where games_played >= 5
    assert len(result) == 5, "Should remove first 5 games"
    assert result["games_played"].min() >= 5, "All games should have >= 5 history"


def test_clean_data_handles_missing_targets():
    """Test that rows with missing target values are removed."""
    test_data = pd.DataFrame(
        {
            "PLAYER_ID": [1] * 5,
            "PTS": [20, np.nan, 20, 20, 20],
            "REB": [5, 5, np.nan, 5, 5],
            "AST": [3, 3, 3, np.nan, 3],
            "games_played": [5, 6, 7, 8, 9],
        }
    )

    result = clean_data(test_data)

    # Should remove rows with any missing targets
    assert len(result) == 2, "Should remove rows with NaN in PTS, REB, or AST"
    assert not result["PTS"].isna().any(), "No NaN in PTS"
    assert not result["REB"].isna().any(), "No NaN in REB"
    assert not result["AST"].isna().any(), "No NaN in AST"


def test_feature_list_completeness():
    """Test that all feature lists contain expected features."""
    features = get_feature_lists()

    # Check all groups exist
    assert "original" in features
    assert "usage" in features
    assert "contextual" in features
    assert "all" in features

    # Check correct counts
    assert len(features["original"]) == 9
    assert len(features["usage"]) == 8
    assert len(features["contextual"]) == 6
    # Verify baseline features (23)
    assert len(features["all"]) >= 23, "Should have at least baseline 23 features"

    # Verify baseline features are all included
    all_combined = (
        set(features["original"]) | set(features["usage"]) | set(features["contextual"])
    )
    assert all_combined.issubset(
        set(features["all"])
    ), "All baseline features should be included"

    # Verify Phase 1 features if present (15 advanced features)
    # Phase 1 features are optional - if present, verify completeness
    if "phase1" in features:
        assert (
            len(features["phase1"]) == 15
        ), "Phase 1 should have exactly 15 advanced features"
        assert set(features["phase1"]).issubset(
            set(features["all"])
        ), "Phase 1 features should be in 'all'"
        # If Phase 1 exists, total should be 38 (23 baseline + 15 Phase 1)
        assert len(features["all"]) == 38, "With Phase 1, should have 38 total features"

    # Verify no duplicates
    assert len(features["all"]) == len(set(features["all"])), "No duplicate features"


def test_rolling_features_multiple_players():
    """Test that rolling features are calculated per player."""
    test_data = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1, 2, 2, 2],
            "GAME_DATE": pd.date_range("2024-01-01", periods=6),
            "PTS": [10, 20, 30, 100, 200, 300],
            "REB": [5] * 6,
            "AST": [3] * 6,
            "MIN": [30] * 6,
            "FGA": [10] * 6,
            "FTA": [5] * 6,
            "FG3A": [3] * 6,
            "FG_PCT": [0.5] * 6,
            "TOV": [2] * 6,  # Phase 1
            "PLUS_MINUS": [5] * 6,  # Phase 1
        }
    )

    result = add_rolling_features(test_data, windows=[3])

    # Player 1's game 2 should use player 1's games 0, 1
    player1_game2 = result[(result["PLAYER_ID"] == 1)].iloc[2]
    assert player1_game2["pts_last_3"] == 15.0  # (10 + 20) / 2

    # Player 2's game 1 should use only player 2's game 0
    player2_game1 = result[(result["PLAYER_ID"] == 2)].iloc[1]
    assert player2_game1["pts_last_3"] == 100.0  # Only game 0 of player 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
