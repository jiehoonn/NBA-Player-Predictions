#!/usr/bin/env python3
"""
Experiment: Improve PTS Prediction
Goal: Reduce MAE from 5.45 to closer to 3.6

Approaches:
1. Try XGBoost instead of Lasso
2. Add feature interactions
3. Try ensemble methods
4. Hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import json

def add_interaction_features(df):
    """Add interaction features that might help PTS prediction."""
    df = df.copy()

    # Home advantage interactions
    df['home_x_opp_def'] = df['IS_HOME'] * df['OPP_DEF_RATING']
    df['home_x_pts_avg'] = df['IS_HOME'] * df['pts_last_5']

    # Rest/fatigue interactions
    df['rest_x_pts_avg'] = df['REST_DAYS'] * df['pts_last_5']
    df['b2b_x_pts_avg'] = df['IS_BACK_TO_BACK'] * df['pts_last_5']
    df['rest_x_min_avg'] = df['REST_DAYS'] * df['min_last_5']

    # Opponent strength interactions
    df['opp_def_x_fga'] = df['OPP_DEF_RATING'] * df['fga_last_5']
    df['opp_pace_x_min'] = df['OPP_PACE'] * df['min_last_5']

    # Shot efficiency interactions
    df['fg_pct_x_fga'] = df['fg_pct_last_5'] * df['fga_last_5']
    df['fg3_pct_ratio'] = df['fg3a_last_5'] / (df['fga_last_5'] + 1)  # +1 to avoid division by zero

    # Usage interactions
    df['total_attempts'] = df['fga_last_5'] + df['fta_last_5']
    df['usage_x_min'] = df['total_attempts'] * df['min_last_5']

    return df

def calculate_baseline(df, target='PTS'):
    """Calculate baseline MAE using 5-game rolling average."""
    baseline_col = f'{target.lower()}_last_5'
    mask = df[baseline_col].notna() & df[target].notna()
    return mean_absolute_error(df[mask][target], df[mask][baseline_col])

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, scaler=None):
    """Train and evaluate a model."""
    # Scale if needed
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        X_test_scaled = X_test

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)

    # Evaluate
    results = {
        'train_mae': mean_absolute_error(y_train, train_pred),
        'val_mae': mean_absolute_error(y_val, val_pred),
        'test_mae': mean_absolute_error(y_test, test_pred)
    }

    return results, model

def main():
    print("="*70)
    print("EXPERIMENT: Improving PTS Prediction")
    print("="*70)
    print("\nCurrent PTS MAE: 5.454")
    print("Goal: 3.6")
    print("Gap: -1.854 points per prediction")
    print()

    # Load data
    print("Loading data...")
    df = pd.read_parquet('data/processed/features_enhanced_5seasons_200players.parquet')

    # Original features
    original_features = [
        'pts_last_3', 'pts_last_5', 'reb_last_3', 'reb_last_5',
        'ast_last_3', 'ast_last_5', 'min_last_3', 'min_last_5',
        'games_played', 'fga_last_3', 'fga_last_5', 'fta_last_3',
        'fta_last_5', 'fg3a_last_3', 'fg3a_last_5', 'fg_pct_last_3',
        'fg_pct_last_5', 'IS_HOME', 'REST_DAYS', 'IS_BACK_TO_BACK',
        'OPP_DEF_RATING', 'OPP_OFF_RATING', 'OPP_PACE'
    ]

    # Split data
    train_df = df[df['SPLIT'] == 'train'].copy()
    val_df = df[df['SPLIT'] == 'val'].copy()
    test_df = df[df['SPLIT'] == 'test'].copy()

    # Calculate baseline
    baseline_mae = calculate_baseline(test_df, 'PTS')
    print(f"Baseline MAE (5-game avg): {baseline_mae:.3f}\n")

    results = []

    # ========================================
    # Experiment 1: Original Lasso (baseline)
    # ========================================
    print("-" * 70)
    print("Experiment 1: Lasso (Current Baseline)")
    print("-" * 70)

    X_train = train_df[original_features]
    y_train = train_df['PTS']
    X_val = val_df[original_features]
    y_val = val_df['PTS']
    X_test = test_df[original_features]
    y_test = test_df['PTS']

    lasso = Lasso(alpha=0.1, max_iter=2000, random_state=42)
    lasso_results, _ = evaluate_model(lasso, X_train, y_train, X_val, y_val, X_test, y_test, StandardScaler())

    improvement = ((baseline_mae - lasso_results['test_mae']) / baseline_mae) * 100
    print(f"  Test MAE: {lasso_results['test_mae']:.3f}")
    print(f"  Improvement over baseline: +{improvement:.1f}%\n")

    results.append({
        'experiment': 'EXP1_Lasso',
        'name': 'Lasso (baseline)',
        'features': len(original_features),
        **lasso_results,
        'improvement_pct': improvement
    })

    # ========================================
    # Experiment 2: XGBoost instead of Lasso
    # ========================================
    print("-" * 70)
    print("Experiment 2: XGBoost (non-linear model)")
    print("-" * 70)

    xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0)
    xgb_results, _ = evaluate_model(xgb, X_train, y_train, X_val, y_val, X_test, y_test, scaler=None)

    improvement = ((baseline_mae - xgb_results['test_mae']) / baseline_mae) * 100
    print(f"  Test MAE: {xgb_results['test_mae']:.3f}")
    print(f"  Improvement over baseline: +{improvement:.1f}%")
    print(f"  vs Lasso: {xgb_results['test_mae'] - lasso_results['test_mae']:+.3f} points\n")

    results.append({
        'experiment': 'EXP2_XGBoost',
        'name': 'XGBoost',
        'features': len(original_features),
        **xgb_results,
        'improvement_pct': improvement
    })

    # ========================================
    # Experiment 3: XGBoost with deeper trees
    # ========================================
    print("-" * 70)
    print("Experiment 3: XGBoost (deeper trees, max_depth=5)")
    print("-" * 70)

    xgb_deep = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0)
    xgb_deep_results, _ = evaluate_model(xgb_deep, X_train, y_train, X_val, y_val, X_test, y_test, scaler=None)

    improvement = ((baseline_mae - xgb_deep_results['test_mae']) / baseline_mae) * 100
    print(f"  Test MAE: {xgb_deep_results['test_mae']:.3f}")
    print(f"  Improvement over baseline: +{improvement:.1f}%")
    print(f"  vs Lasso: {xgb_deep_results['test_mae'] - lasso_results['test_mae']:+.3f} points\n")

    results.append({
        'experiment': 'EXP3_XGBoost_Deep',
        'name': 'XGBoost (depth=5)',
        'features': len(original_features),
        **xgb_deep_results,
        'improvement_pct': improvement
    })

    # ========================================
    # Experiment 4: Add interaction features
    # ========================================
    print("-" * 70)
    print("Experiment 4: XGBoost + Interaction Features")
    print("-" * 70)

    # Add interactions
    train_interact = add_interaction_features(train_df)
    val_interact = add_interaction_features(val_df)
    test_interact = add_interaction_features(test_df)

    # Get new feature columns (only numeric ones, EXCLUDING current game stats to prevent leakage)
    current_game_stats = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
                         'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST',
                         'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS', 'VIDEO_AVAILABLE']
    exclude_cols = (original_features + current_game_stats +
                   ['SPLIT', 'PTS', 'PLAYER_ID', 'GAME_DATE', 'SEASON', 'SEASON_ID',
                    'Game_ID', 'Player_ID', 'MATCHUP', 'WL', 'OPP_ABBREV', 'OPP_TEAM_NAME',
                    'OPP_TEAM_ID', 'OPP_GP', 'OPP_W', 'OPP_L'])
    interaction_cols = [col for col in train_interact.columns
                       if col not in exclude_cols and train_interact[col].dtype in ['int64', 'float64', 'bool']]
    all_features = original_features + interaction_cols

    print(f"  Added {len(interaction_cols)} interaction features")
    print(f"  Total features: {len(all_features)}")

    X_train_int = train_interact[all_features]
    X_val_int = val_interact[all_features]
    X_test_int = test_interact[all_features]

    xgb_int = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0)
    xgb_int_results, xgb_int_model = evaluate_model(xgb_int, X_train_int, y_train, X_val_int, y_val, X_test_int, y_test, scaler=None)

    improvement = ((baseline_mae - xgb_int_results['test_mae']) / baseline_mae) * 100
    print(f"  Test MAE: {xgb_int_results['test_mae']:.3f}")
    print(f"  Improvement over baseline: +{improvement:.1f}%")
    print(f"  vs Lasso: {xgb_int_results['test_mae'] - lasso_results['test_mae']:+.3f} points\n")

    results.append({
        'experiment': 'EXP4_XGBoost_Interact',
        'name': 'XGBoost + Interactions',
        'features': len(all_features),
        **xgb_int_results,
        'improvement_pct': improvement
    })

    # ========================================
    # Experiment 5: Ensemble (Lasso + XGBoost)
    # ========================================
    print("-" * 70)
    print("Experiment 5: Ensemble (Lasso + XGBoost weighted average)")
    print("-" * 70)

    # Train both models
    lasso_ens = Lasso(alpha=0.1, max_iter=2000, random_state=42)
    scaler_ens = StandardScaler()
    X_train_scaled = scaler_ens.fit_transform(X_train)
    X_val_scaled = scaler_ens.transform(X_val)
    X_test_scaled = scaler_ens.transform(X_test)
    lasso_ens.fit(X_train_scaled, y_train)

    xgb_ens = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0)
    xgb_ens.fit(X_train, y_train)

    # Make predictions
    lasso_test_pred = lasso_ens.predict(X_test_scaled)
    xgb_test_pred = xgb_ens.predict(X_test)

    # Weighted average (optimize weight on validation set)
    best_weight = 0.5
    best_val_mae = float('inf')

    lasso_val_pred = lasso_ens.predict(X_val_scaled)
    xgb_val_pred = xgb_ens.predict(X_val)

    for weight in np.arange(0, 1.05, 0.05):
        ensemble_val_pred = weight * lasso_val_pred + (1 - weight) * xgb_val_pred
        val_mae = mean_absolute_error(y_val, ensemble_val_pred)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_weight = weight

    print(f"  Optimal weight: {best_weight:.2f} (Lasso) + {1-best_weight:.2f} (XGBoost)")

    # Final ensemble prediction
    ensemble_test_pred = best_weight * lasso_test_pred + (1 - best_weight) * xgb_test_pred
    ensemble_test_mae = mean_absolute_error(y_test, ensemble_test_pred)

    improvement = ((baseline_mae - ensemble_test_mae) / baseline_mae) * 100
    print(f"  Test MAE: {ensemble_test_mae:.3f}")
    print(f"  Improvement over baseline: +{improvement:.1f}%")
    print(f"  vs Lasso: {ensemble_test_mae - lasso_results['test_mae']:+.3f} points\n")

    results.append({
        'experiment': 'EXP5_Ensemble',
        'name': f'Ensemble ({best_weight:.0%} Lasso)',
        'features': len(original_features),
        'train_mae': np.nan,  # Not computed
        'val_mae': best_val_mae,
        'test_mae': ensemble_test_mae,
        'improvement_pct': improvement
    })

    # ========================================
    # Summary
    # ========================================
    print("=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_mae')

    print(f"\nBest Model: {results_df.iloc[0]['name']}")
    print(f"Test MAE: {results_df.iloc[0]['test_mae']:.3f}")
    print(f"Improvement over Lasso baseline: {results_df.iloc[0]['test_mae'] - lasso_results['test_mae']:+.3f} points")
    print(f"Progress toward goal: {5.454 - results_df.iloc[0]['test_mae']:.3f} points gained")
    print(f"Remaining gap to goal (3.6): {results_df.iloc[0]['test_mae'] - 3.6:.3f} points")
    print()

    print("\nAll Results (sorted by Test MAE):")
    print(results_df[['name', 'features', 'test_mae', 'improvement_pct']].to_string(index=False))

    # Save results
    results_df.to_csv('results/experiments/pts_improvement_results.csv', index=False)
    print(f"\n✓ Results saved to: results/experiments/pts_improvement_results.csv")

    # Show top 3 feature importances if XGBoost + Interactions won
    if results_df.iloc[0]['experiment'] == 'EXP4_XGBoost_Interact':
        print("\nTop 15 Most Important Features:")
        importances = pd.DataFrame({
            'feature': all_features,
            'importance': xgb_int_model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        print(importances.to_string(index=False))

if __name__ == '__main__':
    main()
