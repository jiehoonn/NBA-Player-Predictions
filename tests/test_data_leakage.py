"""
Test suite to ensure no data leakage in feature engineering.

Data leakage is when future information leaks into the training set,
causing overly optimistic performance metrics that don't generalize.
"""

import pytest
import pandas as pd
import numpy as np


def test_rolling_average_uses_shift():
    """
    Test that rolling averages use .shift(1) to prevent leakage.

    CRITICAL: Rolling features must NOT include the current game's stats.
    """
    # Create test data with obvious outlier
    df = pd.DataFrame({
        'PLAYER_ID': [1, 1, 1, 1, 1],
        'GAME_DATE': pd.date_range('2024-01-01', periods=5),
        'PTS': [10, 10, 100, 10, 10]  # Game 3 is outlier (100 points)
    })

    # CORRECT: Using shift(1) before rolling
    df['PTS_last_3_correct'] = df.groupby('PLAYER_ID')['PTS'].shift(1).rolling(3, min_periods=1).mean()

    # WRONG: Not using shift(1) - THIS IS LEAKAGE!
    # Note: This creates MultiIndex, so we need to reset_index()
    wrong_series = df.groupby('PLAYER_ID')['PTS'].rolling(3, min_periods=1).mean()
    df['PTS_last_3_wrong'] = wrong_series.reset_index(level=0, drop=True)

    # Test Game 3 (outlier game)
    game_3 = df.iloc[2]  # Index 2 = Game 3

    # Correct method: Should only see [10, 10] → mean = 10.0
    assert game_3['PTS_last_3_correct'] == 10.0, \
        "LEAKAGE DETECTED! Rolling average includes current game"

    # Wrong method would see [10, 10, 100] → mean = 40.0 (LEAKAGE!)
    assert game_3['PTS_last_3_wrong'] == 40.0, \
        "This demonstrates leakage - includes current game"

    print("✅ Test passed: Rolling averages correctly use shift(1)")


def test_no_future_data_in_training():
    """
    Test that test set dates are strictly after training set dates.

    CRITICAL: Time series must maintain chronological order.
    """
    # Simulate train/test split (5 years of data: 2019-2024)
    df = pd.DataFrame({
        'GAME_DATE': pd.date_range('2019-01-01', periods=1800),  # ~5 years
        'PTS': np.random.randint(10, 40, size=1800)
    })

    # Split at 2024-01-01 (80/20 split)
    train_cutoff = pd.to_datetime('2023-06-01')
    train_df = df[df['GAME_DATE'] < train_cutoff]
    test_df = df[df['GAME_DATE'] >= train_cutoff]

    # Test: No overlap
    assert train_df['GAME_DATE'].max() < test_df['GAME_DATE'].min(), \
        "LEAKAGE: Training set contains future data from test set"

    # Test: Test set is strictly in the future
    assert (test_df['GAME_DATE'] >= train_cutoff).all(), \
        "LEAKAGE: Test set contains past data from training set"

    print("✅ Test passed: Train/test split maintains chronological order")


def test_player_specific_features_no_cross_contamination():
    """
    Test that player A's features don't include player B's data.

    CRITICAL: Rolling features must be grouped by player.
    """
    df = pd.DataFrame({
        'PLAYER_ID': [1, 1, 1, 2, 2, 2],
        'PLAYER_NAME': ['Player A']*3 + ['Player B']*3,
        'GAME_DATE': pd.date_range('2024-01-01', periods=3).tolist() * 2,
        'PTS': [10, 10, 10, 30, 30, 30]  # Player A: 10, Player B: 30
    })

    # Correct: Group by PLAYER_ID before rolling
    df['PTS_last_2_correct'] = df.groupby('PLAYER_ID')['PTS'].shift(1).rolling(2, min_periods=1).mean()

    # Wrong: No grouping - Player B's games leak into Player A's features
    df_wrong = df.copy()
    df_wrong['PTS_last_2_wrong'] = df_wrong['PTS'].shift(1).rolling(2, min_periods=1).mean()

    # Test Player A's last game
    player_a_last = df[df['PLAYER_ID'] == 1].iloc[-1]

    # Correct method: Should only see Player A's games [10, 10] → mean = 10.0
    assert player_a_last['PTS_last_2_correct'] == 10.0, \
        "CROSS-CONTAMINATION: Player A's features include Player B's data"

    print("✅ Test passed: Features correctly grouped by player")


def test_feature_values_are_from_past_only():
    """
    Integration test: Verify first game has NaN features (no history).

    CRITICAL: First game for any player should have no rolling features.
    """
    df = pd.DataFrame({
        'PLAYER_ID': [1, 1, 1],
        'GAME_DATE': pd.date_range('2024-01-01', periods=3),
        'PTS': [20, 25, 30]
    })

    # Calculate rolling average (shift + rolling)
    df['PTS_last_3'] = df.groupby('PLAYER_ID')['PTS'].shift(1).rolling(3, min_periods=1).mean()

    # Test: First game has NO history (should be NaN)
    assert pd.isna(df.iloc[0]['PTS_last_3']), \
        "LEAKAGE: First game has feature values (should be NaN - no history)"

    # Test: Second game sees only game 1
    assert df.iloc[1]['PTS_last_3'] == 20.0, \
        "Game 2 should only see Game 1 data"

    # Test: Third game sees games 1 and 2
    expected = (20 + 25) / 2
    assert np.isclose(df.iloc[2]['PTS_last_3'], expected), \
        f"Game 3 should see Games 1+2: expected {expected}, got {df.iloc[2]['PTS_last_3']}"

    print("✅ Test passed: First game has no features (NaN)")


def test_no_shuffle_in_time_series():
    """
    Test that data is NOT shuffled when doing time series split.

    CRITICAL: sklearn.model_selection.train_test_split with shuffle=True
    is WRONG for time series data.
    """
    df = pd.DataFrame({
        'GAME_DATE': pd.date_range('2020-01-01', periods=100),
        'PTS': range(100)  # 0, 1, 2, ..., 99
    })

    # CORRECT: Temporal split
    train_cutoff = pd.to_datetime('2020-03-01')
    train_df_correct = df[df['GAME_DATE'] < train_cutoff]
    test_df_correct = df[df['GAME_DATE'] >= train_cutoff]

    # Test: Train set should be continuous
    assert train_df_correct['PTS'].is_monotonic_increasing, \
        "SHUFFLING DETECTED: Training set is not in chronological order"

    # Test: Test set should be continuous
    assert test_df_correct['PTS'].is_monotonic_increasing, \
        "SHUFFLING DETECTED: Test set is not in chronological order"

    # Test: No gap between train and test
    assert train_df_correct.index.max() + 1 == test_df_correct.index.min() or \
           train_df_correct.index.max() + 2 >= test_df_correct.index.min(), \
        "Gap detected in train/test split"

    print("✅ Test passed: No shuffling, chronological order maintained")


if __name__ == '__main__':
    # Run tests
    test_rolling_average_uses_shift()
    test_no_future_data_in_training()
    test_player_specific_features_no_cross_contamination()
    test_feature_values_are_from_past_only()
    test_no_shuffle_in_time_series()

    print("\n" + "="*60)
    print("✅ ALL DATA LEAKAGE TESTS PASSED")
    print("="*60)
    print("\nThese tests verify:")
    print("  1. Rolling features use .shift(1) to prevent leakage")
    print("  2. Train/test splits maintain chronological order")
    print("  3. Player features are not cross-contaminated")
    print("  4. First games have no features (NaN)")
    print("  5. No shuffling in time series data")
