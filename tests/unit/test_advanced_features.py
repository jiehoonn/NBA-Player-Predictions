"""
Comprehensive tests for Phase 1 Advanced Features (15 features).

Tests validate mathematical correctness with known values, leakage prevention,
and edge case handling for all advanced features added in Phase 1.

Features tested:
- True Shooting % (2): ts_pct_last_3, ts_pct_last_5
- Last Game Performance (3): pts_last_game, reb_last_game, ast_last_game
- Turnover Rate (2): tov_last_3, tov_last_5
- Plus/Minus (2): plus_minus_last_3, plus_minus_last_5
- Performance Trends (3): pts_trend_last_5, reb_trend_last_5, ast_trend_last_5
- Consistency Metrics (3): pts_std_last_5, reb_std_last_5, ast_std_last_5
"""

import pandas as pd
import numpy as np
import pytest
from src.features.build_features import add_rolling_features


class TestTrueShootingPercentage:
    """Test True Shooting % calculation (TS% = PTS / (2 * (FGA + 0.44 * FTA)))"""

    def test_ts_pct_formula_correctness(self):
        """Test TS% formula with known values."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 6,
                "GAME_DATE": pd.date_range("2024-01-01", periods=6),
                "PTS": [20, 24, 18, 22, 26, 30],
                "FGA": [15, 18, 14, 16, 19, 22],
                "FTA": [4, 3, 2, 5, 4, 3],
                "FG3A": [3, 4, 2, 3, 5, 4],
                "FG_PCT": [0.45, 0.48, 0.44, 0.46, 0.47, 0.50],
                "REB": [8, 9, 7, 8, 10, 9],
                "AST": [5, 6, 4, 5, 7, 6],
                "MIN": [30, 32, 28, 31, 33, 34],
                "TOV": [2, 3, 1, 2, 3, 2],
                "PLUS_MINUS": [5, -3, 8, 2, -5, 10],
            }
        )

        result = add_rolling_features(test_data, windows=[3, 5])

        # Game 4: ts_pct_last_3 is the AVERAGE of last 3 games' TS% (leakage-safe)
        # Game 1 TS%: 20 / (2*(15+0.44*4) + 0.001) ≈ 0.5868
        # Game 2 TS%: 24 / (2*(18+0.44*3) + 0.001) ≈ 0.6053
        # Game 3 TS%: 18 / (2*(14+0.44*2) + 0.001) ≈ 0.6048
        # Average of these 3 values
        ts1 = 20 / (2 * (15 + 0.44 * 4) + 0.001)
        ts2 = 24 / (2 * (18 + 0.44 * 3) + 0.001)
        ts3 = 18 / (2 * (14 + 0.44 * 2) + 0.001)
        expected_ts_pct = (ts1 + ts2 + ts3) / 3
        actual_ts_pct = result.iloc[3]["ts_pct_last_3"]

        assert abs(actual_ts_pct - expected_ts_pct) < 0.001

    def test_ts_pct_division_by_zero_prevention(self):
        """Test TS% handles zero FGA and FTA gracefully."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 5,
                "GAME_DATE": pd.date_range("2024-01-01", periods=5),
                "PTS": [0, 0, 20, 24, 18],
                "FGA": [0, 0, 15, 18, 14],  # Zeros in first two games
                "FTA": [0, 0, 4, 3, 2],
                "FG3A": [0, 0, 3, 4, 2],
                "FG_PCT": [0.0, 0.0, 0.45, 0.48, 0.44],
                "REB": [8, 9, 7, 8, 10],
                "AST": [5, 6, 4, 5, 7],
                "MIN": [30, 32, 28, 31, 33],
                "TOV": [2, 3, 1, 2, 3],
                "PLUS_MINUS": [5, -3, 8, 2, -5],
            }
        )

        result = add_rolling_features(test_data, windows=[3, 5])

        # Should not raise division by zero error
        # Check that ts_pct exists and is not NaN (or is handled gracefully)
        assert "ts_pct_last_3" in result.columns
        # First few games will be near 0 due to 0.001 denominator offset
        assert result.iloc[2]["ts_pct_last_3"] >= 0

    def test_ts_pct_leakage_prevention(self):
        """Test TS% doesn't use current game data."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 5,
                "GAME_DATE": pd.date_range("2024-01-01", periods=5),
                "PTS": [100, 10, 10, 10, 10],  # Outlier in first game
                "FGA": [50, 8, 8, 8, 8],
                "FTA": [20, 2, 2, 2, 2],
                "FG3A": [10, 1, 1, 1, 1],
                "FG_PCT": [0.50, 0.40, 0.40, 0.40, 0.40],
                "REB": [8, 9, 7, 8, 10],
                "AST": [5, 6, 4, 5, 7],
                "MIN": [30, 32, 28, 31, 33],
                "TOV": [2, 3, 1, 2, 3],
                "PLUS_MINUS": [5, -3, 8, 2, -5],
            }
        )

        result = add_rolling_features(test_data, windows=[3])

        # Game 4 should use games [1, 2, 3] = [100pts, 10pts, 10pts]
        # Should NOT include game 4's 10 points
        game4_ts = result.iloc[3]["ts_pct_last_3"]

        # Calculate expected TS% for games [100, 10, 10]
        # TS% for game 1: 100 / (2 * (50 + 0.44*20)) = 100 / 117.6 = 0.8503
        # TS% for game 2: 10 / (2 * (8 + 0.44*2)) = 10 / 17.76 = 0.5631
        # TS% for game 3: 10 / (2 * (8 + 0.44*2)) = 10 / 17.76 = 0.5631
        # Average: (0.8503 + 0.5631 + 0.5631) / 3 = 0.6588
        expected_avg = (
            100 / (2 * (50 + 0.44 * 20))
            + 10 / (2 * (8 + 0.44 * 2))
            + 10 / (2 * (8 + 0.44 * 2))
        ) / 3

        assert abs(game4_ts - expected_avg) < 0.001


class TestLastGamePerformance:
    """Test last game performance features (pts_last_game, reb_last_game, ast_last_game)"""

    def test_last_game_pts_correctness(self):
        """Test pts_last_game is exactly the previous game's points."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 6,
                "GAME_DATE": pd.date_range("2024-01-01", periods=6),
                "PTS": [20, 25, 18, 22, 28, 30],
                "FGA": [15, 18, 14, 16, 19, 22],
                "FTA": [4, 3, 2, 5, 4, 3],
                "FG3A": [3, 4, 2, 3, 5, 4],
                "FG_PCT": [0.45, 0.48, 0.44, 0.46, 0.47, 0.50],
                "REB": [8, 9, 7, 10, 11, 9],
                "AST": [5, 6, 4, 7, 8, 6],
                "MIN": [30, 32, 28, 31, 33, 34],
                "TOV": [2, 3, 1, 2, 3, 2],
                "PLUS_MINUS": [5, -3, 8, 2, -5, 10],
            }
        )

        result = add_rolling_features(test_data, windows=[3, 5])

        # Game 4 should have pts_last_game = 18 (game 3's points)
        assert result.iloc[3]["pts_last_game"] == 18

        # Game 5 should have pts_last_game = 22 (game 4's points)
        assert result.iloc[4]["pts_last_game"] == 22

    def test_last_game_reb_ast_correctness(self):
        """Test reb_last_game and ast_last_game are correct."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 5,
                "GAME_DATE": pd.date_range("2024-01-01", periods=5),
                "PTS": [20, 25, 18, 22, 28],
                "FGA": [15, 18, 14, 16, 19],
                "FTA": [4, 3, 2, 5, 4],
                "FG3A": [3, 4, 2, 3, 5],
                "FG_PCT": [0.45, 0.48, 0.44, 0.46, 0.47],
                "REB": [8, 12, 7, 10, 11],  # Varied rebounds
                "AST": [5, 9, 4, 7, 8],  # Varied assists
                "MIN": [30, 32, 28, 31, 33],
                "TOV": [2, 3, 1, 2, 3],
                "PLUS_MINUS": [5, -3, 8, 2, -5],
            }
        )

        result = add_rolling_features(test_data, windows=[3, 5])

        # Game 3 should have reb_last_game = 12, ast_last_game = 9
        assert result.iloc[2]["reb_last_game"] == 12
        assert result.iloc[2]["ast_last_game"] == 9

    def test_last_game_first_game_is_nan(self):
        """Test first game has NaN for last_game features (no previous game)."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 5,
                "GAME_DATE": pd.date_range("2024-01-01", periods=5),
                "PTS": [20, 25, 18, 22, 28],
                "FGA": [15, 18, 14, 16, 19],
                "FTA": [4, 3, 2, 5, 4],
                "FG3A": [3, 4, 2, 3, 5],
                "FG_PCT": [0.45, 0.48, 0.44, 0.46, 0.47],
                "REB": [8, 12, 7, 10, 11],
                "AST": [5, 9, 4, 7, 8],
                "MIN": [30, 32, 28, 31, 33],
                "TOV": [2, 3, 1, 2, 3],
                "PLUS_MINUS": [5, -3, 8, 2, -5],
            }
        )

        result = add_rolling_features(test_data, windows=[3, 5])

        # First game should have NaN for last_game features
        assert pd.isna(result.iloc[0]["pts_last_game"])
        assert pd.isna(result.iloc[0]["reb_last_game"])
        assert pd.isna(result.iloc[0]["ast_last_game"])


class TestTurnoverRate:
    """Test turnover rolling average features (tov_last_3, tov_last_5)"""

    def test_tov_rolling_average_calculation(self):
        """Test TOV rolling average with known values."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 7,
                "GAME_DATE": pd.date_range("2024-01-01", periods=7),
                "PTS": [20] * 7,
                "FGA": [15] * 7,
                "FTA": [4] * 7,
                "FG3A": [3] * 7,
                "FG_PCT": [0.45] * 7,
                "REB": [8] * 7,
                "AST": [5] * 7,
                "MIN": [30] * 7,
                "TOV": [2, 4, 3, 1, 5, 2, 3],  # Known turnover sequence
                "PLUS_MINUS": [5] * 7,
            }
        )

        result = add_rolling_features(test_data, windows=[3, 5])

        # Game 4: tov_last_3 should be avg of [2, 4, 3] = 3.0
        assert result.iloc[3]["tov_last_3"] == 3.0

        # Game 6: tov_last_5 should be avg of [2, 4, 3, 1, 5] = 3.0
        assert result.iloc[5]["tov_last_5"] == 3.0

    def test_tov_leakage_prevention(self):
        """Test TOV rolling average doesn't include current game."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 5,
                "GAME_DATE": pd.date_range("2024-01-01", periods=5),
                "PTS": [20] * 5,
                "FGA": [15] * 5,
                "FTA": [4] * 5,
                "FG3A": [3] * 5,
                "FG_PCT": [0.45] * 5,
                "REB": [8] * 5,
                "AST": [5] * 5,
                "MIN": [30] * 5,
                "TOV": [100, 1, 1, 1, 1],  # Outlier in first game
                "PLUS_MINUS": [5] * 5,
            }
        )

        result = add_rolling_features(test_data, windows=[3])

        # Game 4 should use [100, 1, 1], NOT including game 4's 1
        # Average: (100 + 1 + 1) / 3 = 34.0
        assert result.iloc[3]["tov_last_3"] == pytest.approx(34.0, abs=0.01)


class TestPlusMinusFeatures:
    """Test Plus/Minus rolling average features"""

    def test_plus_minus_rolling_average(self):
        """Test Plus/Minus rolling average with known values."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 7,
                "GAME_DATE": pd.date_range("2024-01-01", periods=7),
                "PTS": [20] * 7,
                "FGA": [15] * 7,
                "FTA": [4] * 7,
                "FG3A": [3] * 7,
                "FG_PCT": [0.45] * 7,
                "REB": [8] * 7,
                "AST": [5] * 7,
                "MIN": [30] * 7,
                "TOV": [2] * 7,
                "PLUS_MINUS": [10, -5, 8, 3, -2, 12, 5],  # Mixed +/- values
            }
        )

        result = add_rolling_features(test_data, windows=[3, 5])

        # Game 4: plus_minus_last_3 = avg of [10, -5, 8] = 4.333...
        expected = (10 + (-5) + 8) / 3
        assert result.iloc[3]["plus_minus_last_3"] == pytest.approx(expected, abs=0.01)

        # Game 6: plus_minus_last_5 = avg of [10, -5, 8, 3, -2] = 2.8
        expected5 = (10 + (-5) + 8 + 3 + (-2)) / 5
        assert result.iloc[5]["plus_minus_last_5"] == pytest.approx(expected5, abs=0.01)

    def test_plus_minus_negative_values_handling(self):
        """Test Plus/Minus correctly handles negative values."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 5,
                "GAME_DATE": pd.date_range("2024-01-01", periods=5),
                "PTS": [20] * 5,
                "FGA": [15] * 5,
                "FTA": [4] * 5,
                "FG3A": [3] * 5,
                "FG_PCT": [0.45] * 5,
                "REB": [8] * 5,
                "AST": [5] * 5,
                "MIN": [30] * 5,
                "TOV": [2] * 5,
                "PLUS_MINUS": [-10, -15, -5, -20, -8],  # All negative
            }
        )

        result = add_rolling_features(test_data, windows=[3])

        # Game 4: avg of [-10, -15, -5] = -10.0
        assert result.iloc[3]["plus_minus_last_3"] == -10.0


class TestPerformanceTrends:
    """Test performance trend calculation (linear regression slope)"""

    def test_upward_trend_detection(self):
        """Test that upward trend is detected correctly."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 8,
                "GAME_DATE": pd.date_range("2024-01-01", periods=8),
                "PTS": [10, 12, 14, 16, 18, 20, 22, 24],  # Perfect upward trend +2
                "FGA": [15] * 8,
                "FTA": [4] * 8,
                "FG3A": [3] * 8,
                "FG_PCT": [0.45] * 8,
                "REB": [5, 6, 7, 8, 9, 10, 11, 12],  # Upward trend +1
                "AST": [3, 4, 5, 6, 7, 8, 9, 10],  # Upward trend +1
                "MIN": [30] * 8,
                "TOV": [2] * 8,
                "PLUS_MINUS": [5] * 8,
            }
        )

        result = add_rolling_features(test_data, windows=[3, 5])

        # Game 6: pts_trend_last_5 should be positive (slope ≈ 2.0)
        # Using games [10, 12, 14, 16, 18]
        assert result.iloc[5]["pts_trend_last_5"] > 1.5  # Slope should be ~2.0

    def test_downward_trend_detection(self):
        """Test that downward trend is detected correctly."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 8,
                "GAME_DATE": pd.date_range("2024-01-01", periods=8),
                "PTS": [30, 28, 26, 24, 22, 20, 18, 16],  # Downward trend -2
                "FGA": [15] * 8,
                "FTA": [4] * 8,
                "FG3A": [3] * 8,
                "FG_PCT": [0.45] * 8,
                "REB": [8] * 8,
                "AST": [5] * 8,
                "MIN": [30] * 8,
                "TOV": [2] * 8,
                "PLUS_MINUS": [5] * 8,
            }
        )

        result = add_rolling_features(test_data, windows=[3, 5])

        # Game 6: pts_trend_last_5 should be negative (slope ≈ -2.0)
        assert result.iloc[5]["pts_trend_last_5"] < -1.5

    def test_flat_trend_zero_slope(self):
        """Test that flat performance results in near-zero slope."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 8,
                "GAME_DATE": pd.date_range("2024-01-01", periods=8),
                "PTS": [20, 20, 20, 20, 20, 20, 20, 20],  # Flat
                "FGA": [15] * 8,
                "FTA": [4] * 8,
                "FG3A": [3] * 8,
                "FG_PCT": [0.45] * 8,
                "REB": [8] * 8,
                "AST": [5] * 8,
                "MIN": [30] * 8,
                "TOV": [2] * 8,
                "PLUS_MINUS": [5] * 8,
            }
        )

        result = add_rolling_features(test_data, windows=[3, 5])

        # Game 6: pts_trend_last_5 should be ≈ 0
        assert abs(result.iloc[5]["pts_trend_last_5"]) < 0.1

    def test_trend_with_insufficient_data(self):
        """Test trend calculation handles min_periods correctly."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 3,
                "GAME_DATE": pd.date_range("2024-01-01", periods=3),
                "PTS": [20, 22, 24],
                "FGA": [15, 16, 17],
                "FTA": [4, 3, 5],
                "FG3A": [3, 3, 3],
                "FG_PCT": [0.45, 0.45, 0.45],
                "REB": [8, 9, 10],
                "AST": [5, 6, 7],
                "MIN": [30, 32, 31],
                "TOV": [2, 3, 1],
                "PLUS_MINUS": [5, -3, 8],
            }
        )

        result = add_rolling_features(test_data, windows=[3, 5])

        # First few games should handle min_periods gracefully (not crash)
        assert "pts_trend_last_5" in result.columns


class TestConsistencyMetrics:
    """Test consistency metrics (standard deviation)"""

    def test_std_calculation_correctness(self):
        """Test standard deviation is calculated correctly."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 8,
                "GAME_DATE": pd.date_range("2024-01-01", periods=8),
                "PTS": [20, 20, 20, 20, 20, 30, 10, 20],  # Varied in games 6-7
                "FGA": [15] * 8,
                "FTA": [4] * 8,
                "FG3A": [3] * 8,
                "FG_PCT": [0.45] * 8,
                "REB": [8, 8, 8, 8, 8, 15, 3, 8],  # Varied
                "AST": [5, 5, 5, 5, 5, 10, 2, 5],  # Varied
                "MIN": [30] * 8,
                "TOV": [2] * 8,
                "PLUS_MINUS": [5] * 8,
            }
        )

        result = add_rolling_features(test_data, windows=[3, 5])

        # Game 6: pts_std_last_5 for [20, 20, 20, 20, 20] should be 0
        assert result.iloc[5]["pts_std_last_5"] == pytest.approx(0.0, abs=0.01)

        # Game 7: pts_std_last_5 for [20, 20, 20, 20, 30]
        # pandas .std() uses ddof=1 (sample std): sqrt(((20-22)²*4 + (30-22)²)/4) = sqrt(20) ≈ 4.472
        assert result.iloc[6]["pts_std_last_5"] == pytest.approx(4.472, abs=0.1)

    def test_std_high_variance_detection(self):
        """Test std detects high variance correctly."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 8,
                "GAME_DATE": pd.date_range("2024-01-01", periods=8),
                "PTS": [10, 30, 5, 35, 15, 25, 8, 32],  # High variance
                "FGA": [15] * 8,
                "FTA": [4] * 8,
                "FG3A": [3] * 8,
                "FG_PCT": [0.45] * 8,
                "REB": [8] * 8,
                "AST": [5] * 8,
                "MIN": [30] * 8,
                "TOV": [2] * 8,
                "PLUS_MINUS": [5] * 8,
            }
        )

        result = add_rolling_features(test_data, windows=[3, 5])

        # High variance should result in high std
        # Game 6: pts_std_last_5 for [10, 30, 5, 35, 15] should be > 10
        assert result.iloc[5]["pts_std_last_5"] > 10

    def test_std_leakage_prevention(self):
        """Test std doesn't include current game."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 8,
                "GAME_DATE": pd.date_range("2024-01-01", periods=8),
                "PTS": [20, 20, 20, 20, 20, 20, 100, 20],  # Outlier at game 7
                "FGA": [15] * 8,
                "FTA": [4] * 8,
                "FG3A": [3] * 8,
                "FG_PCT": [0.45] * 8,
                "REB": [8] * 8,
                "AST": [5] * 8,
                "MIN": [30] * 8,
                "TOV": [2] * 8,
                "PLUS_MINUS": [5] * 8,
            }
        )

        result = add_rolling_features(test_data, windows=[3, 5])

        # Game 7 (outlier at 100): pts_std_last_5 should use [20,20,20,20,20] = 0
        # NOT including game 7's 100
        assert result.iloc[6]["pts_std_last_5"] == pytest.approx(0.0, abs=0.01)

        # Game 8: pts_std_last_5 should use [20,20,20,20,100] > 0
        assert result.iloc[7]["pts_std_last_5"] > 30  # Should be ~32


class TestMultiplePlayersIsolation:
    """Test that features are calculated per-player (no cross-contamination)"""

    def test_multiple_players_no_contamination(self):
        """Test features are calculated separately for each player."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1, 1, 1, 2, 2, 2],
                "GAME_DATE": pd.date_range("2024-01-01", periods=6),
                "PTS": [20, 22, 24, 100, 102, 104],  # Player 2 has much higher PTS
                "FGA": [15, 16, 17, 50, 51, 52],
                "FTA": [4, 3, 5, 10, 11, 12],
                "FG3A": [3, 3, 3, 3, 3, 3],
                "FG_PCT": [0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
                "REB": [8, 9, 10, 20, 21, 22],
                "AST": [5, 6, 7, 15, 16, 17],
                "MIN": [30, 32, 31, 40, 41, 42],
                "TOV": [2, 3, 1, 5, 6, 4],
                "PLUS_MINUS": [5, -3, 8, 20, 18, 22],
            }
        )

        result = add_rolling_features(test_data, windows=[3])

        # Player 1, game 3: pts_last_game should be 22 (NOT 102 from player 2)
        player1_game3 = result[result["PLAYER_ID"] == 1].iloc[2]
        assert player1_game3["pts_last_game"] == 22

        # Player 2, game 1 (index 3): pts_last_game should be NaN (first game)
        player2_game1 = result[result["PLAYER_ID"] == 2].iloc[0]
        assert pd.isna(player2_game1["pts_last_game"])


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_all_nan_values(self):
        """Test handling of all NaN values in a column."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1] * 5,
                "GAME_DATE": pd.date_range("2024-01-01", periods=5),
                "PTS": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "FGA": [15, 16, 17, 18, 19],
                "FTA": [4, 3, 5, 2, 4],
                "FG3A": [3, 3, 3, 3, 3],
                "FG_PCT": [0.45, 0.45, 0.45, 0.45, 0.45],
                "REB": [8, 9, 10, 7, 8],
                "AST": [5, 6, 7, 4, 5],
                "MIN": [30, 32, 31, 29, 33],
                "TOV": [2, 3, 1, 2, 3],
                "PLUS_MINUS": [5, -3, 8, 2, -5],
            }
        )

        # Should not crash
        result = add_rolling_features(test_data, windows=[3, 5])
        assert "pts_last_3" in result.columns

    def test_single_game_player(self):
        """Test player with only one game."""
        test_data = pd.DataFrame(
            {
                "PLAYER_ID": [1],
                "GAME_DATE": pd.date_range("2024-01-01", periods=1),
                "PTS": [20],
                "FGA": [15],
                "FTA": [4],
                "FG3A": [3],
                "FG_PCT": [0.45],
                "REB": [8],
                "AST": [5],
                "MIN": [30],
                "TOV": [2],
                "PLUS_MINUS": [5],
            }
        )

        # Should not crash
        result = add_rolling_features(test_data, windows=[3, 5])
        assert len(result) == 1
        # First game features should be NaN or use min_periods=1
        assert pd.isna(result.iloc[0]["pts_last_game"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
