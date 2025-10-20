#!/usr/bin/env python3
"""
Train models with Phase 1 Enhanced Features (38 total)

Compares performance of:
- Baseline: 23 features (original + usage + contextual)
- Phase 1: 38 features (baseline + 15 advanced features)

Expected improvement: -0.28 points in PTS MAE (5.454 → 5.17)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import json

# Baseline 23 features (from previous experiments)
ORIGINAL_FEATURES = [
    "pts_last_3", "pts_last_5", "reb_last_3", "reb_last_5",
    "ast_last_3", "ast_last_5", "min_last_3", "min_last_5",
    "games_played"
]

USAGE_FEATURES = [
    "fga_last_3", "fga_last_5", "fta_last_3", "fta_last_5",
    "fg3a_last_3", "fg3a_last_5", "fg_pct_last_3", "fg_pct_last_5"
]

CONTEXTUAL_FEATURES = [
    "IS_HOME", "REST_DAYS", "IS_BACK_TO_BACK",
    "OPP_DEF_RATING", "OPP_OFF_RATING", "OPP_PACE"
]

BASELINE_FEATURES = ORIGINAL_FEATURES + USAGE_FEATURES + CONTEXTUAL_FEATURES

# Phase 1 Advanced Features (15 new)
PHASE1_FEATURES = [
    # True Shooting % (2)
    "ts_pct_last_3", "ts_pct_last_5",

    # Last game performance (3)
    "pts_last_game", "reb_last_game", "ast_last_game",

    # Turnover rate (2)
    "tov_last_3", "tov_last_5",

    # Plus/Minus (2)
    "plus_minus_last_3", "plus_minus_last_5",

    # Performance trends (3)
    "pts_trend_last_5", "reb_trend_last_5", "ast_trend_last_5",

    # Scoring consistency (3)
    "pts_std_last_5", "reb_std_last_5", "ast_std_last_5"
]

ENHANCED_FEATURES = BASELINE_FEATURES + PHASE1_FEATURES


def train_and_evaluate(train_df, val_df, test_df, target, features, model_type='auto'):
    """Train model and return performance metrics."""

    # Prepare data
    X_train = train_df[features]
    y_train = train_df[target]
    X_val = val_df[features]
    y_val = val_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Select model (based on previous best results)
    if model_type == 'auto':
        if target == 'PTS':
            model = Lasso(alpha=0.1, random_state=42, max_iter=2000)
            model_name = 'Lasso'
            use_scaling = True
        else:  # REB or AST
            model = XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                random_state=42, n_jobs=-1, verbosity=0
            )
            model_name = 'XGBoost'
            use_scaling = False

    # Scale if needed
    if use_scaling:
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train)
        X_val_proc = scaler.transform(X_val)
        X_test_proc = scaler.transform(X_test)
    else:
        X_train_proc = X_train.values
        X_val_proc = X_val.values
        X_test_proc = X_test.values

    # Train
    model.fit(X_train_proc, y_train)

    # Predict
    train_pred = model.predict(X_train_proc)
    val_pred = model.predict(X_val_proc)
    test_pred = model.predict(X_test_proc)

    # Calculate metrics
    results = {
        'train_mae': mean_absolute_error(y_train, train_pred),
        'val_mae': mean_absolute_error(y_val, val_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'train_r2': r2_score(y_train, train_pred),
        'val_r2': r2_score(y_val, val_pred),
        'test_r2': r2_score(y_test, test_pred)
    }

    # Baseline (5-game rolling average)
    baseline_pred = test_df[f'{target.lower()}_last_5'].values
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    results['baseline_mae'] = baseline_mae
    results['improvement_pct'] = ((baseline_mae - results['test_mae']) / baseline_mae) * 100

    return results, model, model_name


def main():
    print("="*70)
    print("PHASE 1 ENHANCED FEATURES EXPERIMENT")
    print("="*70)
    print(f"\nBaseline features: {len(BASELINE_FEATURES)}")
    print(f"Phase 1 features: {len(PHASE1_FEATURES)}")
    print(f"Enhanced features: {len(ENHANCED_FEATURES)}")
    print()

    # Load data
    print("Loading data...")
    df = pd.read_parquet('data/processed/features_enhanced_5seasons_200players_phase1.parquet')

    train_df = df[df['SPLIT'] == 'train'].copy()
    val_df = df[df['SPLIT'] == 'val'].copy()
    test_df = df[df['SPLIT'] == 'test'].copy()

    print(f"  Train: {len(train_df):,} games")
    print(f"  Val:   {len(val_df):,} games")
    print(f"  Test:  {len(test_df):,} games")
    print()

    # Track all results
    all_results = []

    # Train models for each target
    for target in ['PTS', 'REB', 'AST']:
        print("-"*70)
        print(f"{target} PREDICTION")
        print("-"*70)

        # Baseline (23 features)
        print(f"\n1. Baseline Model ({len(BASELINE_FEATURES)} features)")
        baseline_results, baseline_model, baseline_model_name = train_and_evaluate(
            train_df, val_df, test_df, target, BASELINE_FEATURES
        )

        print(f"   Model: {baseline_model_name}")
        print(f"   Test MAE: {baseline_results['test_mae']:.3f}")
        print(f"   Baseline MAE: {baseline_results['baseline_mae']:.3f}")
        print(f"   Improvement: +{baseline_results['improvement_pct']:.1f}%")

        # Enhanced (38 features)
        print(f"\n2. Phase 1 Enhanced Model ({len(ENHANCED_FEATURES)} features)")
        enhanced_results, enhanced_model, enhanced_model_name = train_and_evaluate(
            train_df, val_df, test_df, target, ENHANCED_FEATURES
        )

        print(f"   Model: {enhanced_model_name}")
        print(f"   Test MAE: {enhanced_results['test_mae']:.3f}")
        print(f"   Baseline MAE: {enhanced_results['baseline_mae']:.3f}")
        print(f"   Improvement: +{enhanced_results['improvement_pct']:.1f}%")

        # Compare
        improvement = baseline_results['test_mae'] - enhanced_results['test_mae']
        print(f"\n3. Comparison")
        print(f"   Baseline (23 features):  {baseline_results['test_mae']:.3f} MAE")
        print(f"   Enhanced (38 features):  {enhanced_results['test_mae']:.3f} MAE")
        print(f"   Improvement: {improvement:+.3f} points ({(improvement/baseline_results['test_mae']*100):+.1f}%)")
        print()

        # Store results
        all_results.append({
            'target': target,
            'baseline_features': len(BASELINE_FEATURES),
            'baseline_mae': baseline_results['test_mae'],
            'baseline_r2': baseline_results['test_r2'],
            'enhanced_features': len(ENHANCED_FEATURES),
            'enhanced_mae': enhanced_results['test_mae'],
            'enhanced_r2': enhanced_results['test_r2'],
            'improvement_points': improvement,
            'improvement_pct': (improvement / baseline_results['test_mae']) * 100
        })

    # Summary
    print("="*70)
    print("SUMMARY: Phase 1 Feature Impact")
    print("="*70)

    results_df = pd.DataFrame(all_results)

    print("\n" + results_df.to_string(index=False))

    # Overall assessment
    print("\n" + "-"*70)
    print("OVERALL ASSESSMENT")
    print("-"*70)

    avg_improvement = results_df['improvement_points'].mean()
    pts_improvement = results_df[results_df['target'] == 'PTS']['improvement_points'].values[0]

    print(f"\nAverage improvement: {avg_improvement:+.3f} points per target")
    print(f"PTS improvement: {pts_improvement:+.3f} points")

    pts_current = results_df[results_df['target'] == 'PTS']['enhanced_mae'].values[0]
    pts_goal = 3.6

    print(f"\nPTS Goal Progress:")
    print(f"  Previous (23 features): 5.454 MAE")
    print(f"  Current (38 features):  {pts_current:.3f} MAE")
    print(f"  Goal:                   {pts_goal} MAE")
    print(f"  Remaining gap:          {pts_current - pts_goal:.3f} points")

    # Save results
    results_df.to_csv('results/experiments/phase1_comparison.csv', index=False)
    print(f"\n✓ Results saved to: results/experiments/phase1_comparison.csv")

    # Feature importance for PTS (if XGBoost was used)
    # Note: We use Lasso for PTS, so we'll show Lasso coefficients instead
    print("\n" + "-"*70)
    print("TOP 15 MOST IMPORTANT FEATURES (PTS)")
    print("-"*70)

    # Get feature importances (for Lasso, use absolute coefficients)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[ENHANCED_FEATURES])
    lasso_model = Lasso(alpha=0.1, random_state=42, max_iter=2000)
    lasso_model.fit(X_train_scaled, train_df['PTS'])

    feature_importance = pd.DataFrame({
        'feature': ENHANCED_FEATURES,
        'importance': np.abs(lasso_model.coef_)
    }).sort_values('importance', ascending=False).head(15)

    print("\n" + feature_importance.to_string(index=False))

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
