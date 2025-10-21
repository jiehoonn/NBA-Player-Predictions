#!/usr/bin/env python3
"""
Model Training Script - Train best-performing models (PRODUCTION)

Production Configuration (5 seasons, 200 players, 38 features):
- PTS: Lasso (MAE 5.448, +3.7% vs baseline)
- REB: XGBoost (MAE 2.134, +2.4% vs baseline) ✓ Goal achieved
- AST: XGBoost (MAE 1.642, +2.2% vs baseline) ✓ Goal achieved

Features: 38 total (23 baseline + 15 Phase 1 advanced)

Usage:
    python src/models/train_models.py --input data/processed/features_enhanced.parquet --target PTS --output artifacts/models/
"""

import argparse
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# Feature sets (from notebook 09)
ORIGINAL_FEATURES = [
    "pts_last_3",
    "pts_last_5",
    "reb_last_3",
    "reb_last_5",
    "ast_last_3",
    "ast_last_5",
    "min_last_3",
    "min_last_5",
    "games_played",
]

USAGE_FEATURES = [
    "fga_last_3",
    "fga_last_5",
    "fta_last_3",
    "fta_last_5",
    "fg3a_last_3",
    "fg3a_last_5",
    "fg_pct_last_3",
    "fg_pct_last_5",
]

CONTEXTUAL_FEATURES = [
    "IS_HOME",
    "REST_DAYS",
    "IS_BACK_TO_BACK",
    "OPP_DEF_RATING",
    "OPP_OFF_RATING",
    "OPP_PACE",
]

# Phase 1 Advanced Features (15 total)
PHASE1_FEATURES = [
    # True Shooting %
    "ts_pct_last_3",
    "ts_pct_last_5",
    # Last game performance
    "pts_last_game",
    "reb_last_game",
    "ast_last_game",
    # Turnover rate
    "tov_last_3",
    "tov_last_5",
    # Plus/Minus
    "plus_minus_last_3",
    "plus_minus_last_5",
    # Performance trends
    "pts_trend_last_5",
    "reb_trend_last_5",
    "ast_trend_last_5",
    # Scoring consistency
    "pts_std_last_5",
    "reb_std_last_5",
    "ast_std_last_5",
]

# All features (38 total: 23 baseline + 15 Phase 1 advanced)
ALL_FEATURES = ORIGINAL_FEATURES + USAGE_FEATURES + CONTEXTUAL_FEATURES + PHASE1_FEATURES


def get_git_commit():
    """
    Get current git commit hash for reproducibility tracking.

    Returns:
        str: Git commit hash (40-character SHA-1) or None if not in git repo
    """
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return git_commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not in a git repo or git not installed
        return None


def get_best_model(target, use_all_features=True):
    """
    Get the best model for each target based on notebook 09 results.

    Args:
        target: 'PTS', 'REB', or 'AST'
        use_all_features: If True, use all 23 features; if False, use original 9

    Returns:
        model object, model_name
    """
    if target == "PTS":
        # Lasso best for PTS
        model = Lasso(alpha=0.1, random_state=42, max_iter=2000)
        model_name = "Lasso"
    else:
        # XGBoost best for REB and AST
        model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model_name = "XGBoost"

    return model, model_name


def train_model(train_df, val_df, test_df, target, features):
    """
    Train model and evaluate on all splits.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        target: Target variable ('PTS', 'REB', 'AST')
        features: List of feature names to use

    Returns:
        Dictionary with model, scaler, and metrics
    """
    print(f"\nTraining {target} model...")
    print(f"  Features: {len(features)}")
    print(f"  Train samples: {len(train_df):,}")
    print(f"  Val samples: {len(val_df):,}")
    print(f"  Test samples: {len(test_df):,}")

    # Prepare data
    X_train = train_df[features]
    y_train = train_df[target]
    X_val = val_df[features]
    y_val = val_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Get best model
    model, model_name = get_best_model(target)

    # Scale features if needed (Lasso needs scaling, XGBoost doesn't)
    scaler = None
    if model_name == "Lasso":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        X_train_proc, X_val_proc, X_test_proc = (
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
        )
    else:
        X_train_proc, X_val_proc, X_test_proc = (
            X_train.values,
            X_val.values,
            X_test.values,
        )

    # Train
    print(f"  Training {model_name}...")
    model.fit(X_train_proc, y_train)

    # Evaluate on all splits
    results = {}

    for split_name, X_split, y_split in [
        ("train", X_train_proc, y_train),
        ("val", X_val_proc, y_val),
        ("test", X_test_proc, y_test),
    ]:
        y_pred = model.predict(X_split)

        mae = mean_absolute_error(y_split, y_pred)
        rmse = np.sqrt(mean_squared_error(y_split, y_pred))
        r2 = r2_score(y_split, y_pred)

        results[split_name] = {"mae": mae, "rmse": rmse, "r2": r2}

    # Compute baseline (5-game rolling average)
    baseline_pred_test = test_df[f"{target.lower()}_last_5"].values
    baseline_mae = mean_absolute_error(y_test, baseline_pred_test)

    improvement = (baseline_mae - results["test"]["mae"]) / baseline_mae * 100

    # Print results
    print(f"\n  Results:")
    print(f"    Train MAE: {results['train']['mae']:.3f}")
    print(f"    Val MAE:   {results['val']['mae']:.3f}")
    print(f"    Test MAE:  {results['test']['mae']:.3f}")
    print(f"    Baseline:  {baseline_mae:.3f}")
    print(f"    Improvement: {improvement:+.1f}%")

    return {
        "model": model,
        "model_name": model_name,
        "scaler": scaler,
        "features": features,
        "results": results,
        "baseline_mae": baseline_mae,
        "improvement": improvement,
    }


def save_model(model_info, target, output_dir):
    """Save model, scaler, and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_info["model_name"]
    filename = f"{target}_{model_name.lower().replace(' ', '_')}.joblib"
    filepath = output_dir / filename

    # Save everything
    joblib.dump(
        {
            "model": model_info["model"],
            "scaler": model_info["scaler"],
            "features": model_info["features"],
            "target": target,
            "model_name": model_name,
            "results": model_info["results"],
            "baseline_mae": model_info["baseline_mae"],
            "improvement": model_info["improvement"],
        },
        filepath,
    )

    print(f"\n✓ Model saved: {filepath}")

    # Save metrics separately
    metrics_file = output_dir / f"{target}_metrics.json"

    # Capture git commit for reproducibility
    git_commit = get_git_commit()

    metrics = {
        "target": target,
        "model_name": model_name,
        "num_features": len(model_info["features"]),
        "train_mae": model_info["results"]["train"]["mae"],
        "val_mae": model_info["results"]["val"]["mae"],
        "test_mae": model_info["results"]["test"]["mae"],
        "baseline_mae": model_info["baseline_mae"],
        "improvement_pct": model_info["improvement"],
    }

    # Add git commit to metadata if available
    if git_commit:
        if "_metadata" not in metrics:
            metrics["_metadata"] = {}
        metrics["_metadata"]["git_commit"] = git_commit
        print(f"✓ Git commit tracked: {git_commit[:8]}")

    import json

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Metrics saved: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Train best-performing models for NBA predictions"
    )
    parser.add_argument("--input", required=True, help="Input features parquet file")
    parser.add_argument(
        "--target", required=True, choices=["PTS", "REB", "AST"], help="Target variable"
    )
    parser.add_argument(
        "--output", default="artifacts/models/", help="Output directory"
    )
    parser.add_argument(
        "--features",
        choices=["original", "all"],
        default="all",
        help="Feature set: original (9) or all (38: 23 baseline + 15 Phase 1 advanced)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(f"TRAINING {args.target} PREDICTION MODEL")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input: {args.input}")
    print(f"  Target: {args.target}")
    print(f"  Features: {args.features}")
    print(f"  Output: {args.output}")

    # Load data
    print(f"\nLoading data...")
    df = pd.read_parquet(args.input)

    train = df[df["SPLIT"] == "train"].copy()
    val = df[df["SPLIT"] == "val"].copy()
    test = df[df["SPLIT"] == "test"].copy()

    print(f"  Train: {len(train):,} games")
    print(f"  Val: {len(val):,} games")
    print(f"  Test: {len(test):,} games")

    # Select features
    features = ORIGINAL_FEATURES if args.features == "original" else ALL_FEATURES

    # Train model
    model_info = train_model(train, val, test, args.target, features)

    # Save model
    save_model(model_info, args.target, args.output)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
