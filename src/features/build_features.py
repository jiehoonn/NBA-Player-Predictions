#!/usr/bin/env python3
"""
Feature Engineering Script - Create leakage-safe rolling features

Based on notebook 08 findings:
- 23 total features: 9 rolling + 8 usage + 6 contextual
- Leakage-safe: .shift(1).rolling() pattern
- Time-based train/val/test splits

Usage:
    python src/features/build_features.py --input data/raw/player_gamelogs_enhanced.parquet --output data/processed/features_enhanced.parquet
"""

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def add_rolling_features(df, windows=[3, 5]):
    """
    Add leakage-safe rolling average features.

    Uses .shift(1).rolling() pattern to ensure only PAST games are used.
    """
    print(f"Adding rolling features (windows: {windows})...")

    features = []

    for player_id in tqdm(df['PLAYER_ID'].unique(), desc="Players"):
        player_df = df[df['PLAYER_ID'] == player_id].copy()
        player_df = player_df.sort_values('GAME_DATE')

        for window in windows:
            # Performance rolling averages (LEAKAGE-SAFE)
            player_df[f'pts_last_{window}'] = player_df['PTS'].shift(1).rolling(window, min_periods=1).mean()
            player_df[f'reb_last_{window}'] = player_df['REB'].shift(1).rolling(window, min_periods=1).mean()
            player_df[f'ast_last_{window}'] = player_df['AST'].shift(1).rolling(window, min_periods=1).mean()
            player_df[f'min_last_{window}'] = player_df['MIN'].shift(1).rolling(window, min_periods=1).mean()

            # Usage rolling averages
            player_df[f'fga_last_{window}'] = player_df['FGA'].shift(1).rolling(window, min_periods=1).mean()
            player_df[f'fta_last_{window}'] = player_df['FTA'].shift(1).rolling(window, min_periods=1).mean()
            player_df[f'fg3a_last_{window}'] = player_df['FG3A'].shift(1).rolling(window, min_periods=1).mean()

            # Efficiency rolling averages (handle NaN in FG_PCT)
            fg_pct_rolling = player_df['FG_PCT'].shift(1).rolling(window, min_periods=1).mean()
            player_df[f'fg_pct_last_{window}'] = fg_pct_rolling.fillna(player_df['FG_PCT'].mean())

        # Games played counter
        player_df['games_played'] = range(len(player_df))

        features.append(player_df)

    print(f"✓ Rolling features added")
    return pd.concat(features, ignore_index=True)


def create_train_val_test_splits(df, train_end='2024-07-31', val_end='2024-12-31'):
    """Create time-based train/validation/test splits."""
    print("\nCreating time-based splits...")

    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)

    df['SPLIT'] = 'train'
    df.loc[df['GAME_DATE'] > train_end, 'SPLIT'] = 'val'
    df.loc[df['GAME_DATE'] > val_end, 'SPLIT'] = 'test'

    train_count = (df['SPLIT'] == 'train').sum()
    val_count = (df['SPLIT'] == 'val').sum()
    test_count = (df['SPLIT'] == 'test').sum()

    print(f"  Train: {train_count:,} games ({(train_count/len(df)*100):.1f}%)")
    print(f"  Val:   {val_count:,} games ({(val_count/len(df)*100):.1f}%)")
    print(f"  Test:  {test_count:,} games ({(test_count/len(df)*100):.1f}%)")

    return df


def clean_data(df):
    """Clean and prepare data for modeling."""
    print("\nCleaning data...")

    initial_count = len(df)

    # Drop rows with missing targets
    df = df.dropna(subset=['PTS', 'REB', 'AST'])

    # Drop rows with insufficient history (< 5 games)
    df = df[df['games_played'] >= 5]

    # Fill NaN in contextual features with median
    contextual_features = ['REST_DAYS', 'OPP_DEF_RATING', 'OPP_OFF_RATING', 'OPP_PACE']
    for feat in contextual_features:
        if feat in df.columns and df[feat].isna().sum() > 0:
            median_val = df[feat].median()
            df[feat] = df[feat].fillna(median_val)
            print(f"  Filled {feat} NaN with median: {median_val:.2f}")

    # Fill IS_HOME and IS_BACK_TO_BACK with 0 if NaN
    if 'IS_HOME' in df.columns:
        df['IS_HOME'] = df['IS_HOME'].fillna(0).astype(int)
    if 'IS_BACK_TO_BACK' in df.columns:
        df['IS_BACK_TO_BACK'] = df['IS_BACK_TO_BACK'].fillna(0).astype(int)

    final_count = len(df)
    print(f"  Removed {initial_count - final_count:,} rows")
    print(f"  Final dataset: {final_count:,} games")

    return df


def get_feature_lists():
    """Return the three feature sets used in the project."""
    original_features = [
        'pts_last_3', 'pts_last_5',
        'reb_last_3', 'reb_last_5',
        'ast_last_3', 'ast_last_5',
        'min_last_3', 'min_last_5',
        'games_played'
    ]

    usage_features = [
        'fga_last_3', 'fga_last_5',
        'fta_last_3', 'fta_last_5',
        'fg3a_last_3', 'fg3a_last_5',
        'fg_pct_last_3', 'fg_pct_last_5'
    ]

    contextual_features = [
        'IS_HOME',
        'REST_DAYS',
        'IS_BACK_TO_BACK',
        'OPP_DEF_RATING',
        'OPP_OFF_RATING',
        'OPP_PACE'
    ]

    all_features = original_features + usage_features + contextual_features

    return {
        'original': original_features,
        'usage': usage_features,
        'contextual': contextual_features,
        'all': all_features
    }


def main():
    parser = argparse.ArgumentParser(description='Build features for NBA player prediction')
    parser.add_argument('--input', required=True, help='Input parquet file (raw data)')
    parser.add_argument('--output', required=True, help='Output parquet file (processed features)')
    parser.add_argument('--windows', nargs='+', type=int, default=[3, 5], help='Rolling window sizes')

    args = parser.parse_args()

    print("="*70)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*70)
    print(f"\nInput: {args.input}")
    print(f"Output: {args.output}")
    print(f"Windows: {args.windows}")

    # Load data
    print(f"\nLoading data...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df):,} games")
    print(f"  Players: {df['PLAYER_ID'].nunique()}")
    print(f"  Date range: {df['GAME_DATE'].min().date()} to {df['GAME_DATE'].max().date()}")

    # Add rolling features
    df_enhanced = add_rolling_features(df, windows=args.windows)

    # Clean data
    df_clean = clean_data(df_enhanced)

    # Create splits
    df_final = create_train_val_test_splits(df_clean)

    # Verify features
    feature_lists = get_feature_lists()
    all_features = feature_lists['all']

    missing_features = [f for f in all_features if f not in df_final.columns]
    if missing_features:
        print(f"\n⚠ Warning: Missing features: {missing_features}")
    else:
        print(f"\n✓ All {len(all_features)} features present")

    # Save
    df_final.to_parquet(args.output)

    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*70)
    print(f"  Output: {args.output}")
    print(f"  Records: {len(df_final):,}")
    print(f"  Features: {len(all_features)}")
    print(f"    - Original (rolling): {len(feature_lists['original'])}")
    print(f"    - Usage: {len(feature_lists['usage'])}")
    print(f"    - Contextual: {len(feature_lists['contextual'])}")
    print("\n  Splits:")
    print(df_final['SPLIT'].value_counts().to_string())
    print("="*70)


if __name__ == '__main__':
    main()
