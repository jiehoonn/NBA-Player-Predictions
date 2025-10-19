#!/usr/bin/env python3
"""
Model Evaluation Script - Comprehensive evaluation of trained models

Provides detailed metrics, comparison to baselines, and performance analysis
across all data splits (train/val/test).

Usage:
    python src/models/evaluate_models.py --models artifacts/models/ --data data/processed/features_enhanced_3seasons.parquet
"""

import argparse
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from tabulate import tabulate


def evaluate_model_on_split(model, scaler, features, df_split, target):
    """
    Evaluate model on a specific data split.

    Args:
        model: Trained model
        scaler: Fitted scaler (or None)
        features: List of feature names
        df_split: DataFrame for this split
        target: Target variable name

    Returns:
        Dictionary with metrics
    """
    X = df_split[features]
    y_actual = df_split[target]

    # Scale if needed
    if scaler is not None:
        X_proc = scaler.transform(X)
    else:
        X_proc = X.values

    # Predict
    y_pred = model.predict(X_proc)

    # Compute metrics
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)

    # Compute baseline (5-game rolling average)
    baseline_col = f"{target.lower()}_last_5"
    if baseline_col in df_split.columns:
        baseline_pred = df_split[baseline_col].values
        baseline_mae = mean_absolute_error(y_actual, baseline_pred)
        improvement = (baseline_mae - mae) / baseline_mae * 100
    else:
        baseline_mae = None
        improvement = None

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "baseline_mae": baseline_mae,
        "improvement_pct": improvement,
        "n_samples": len(df_split),
    }


def evaluate_all_models(models_dir, data_path):
    """
    Evaluate all trained models on all data splits.

    Returns:
        DataFrame with comprehensive results
    """
    print("\nEvaluating all models...")

    models_dir = Path(models_dir)
    model_files = list(models_dir.glob("*_*.joblib"))

    if not model_files:
        print("  ❌ No model files found!")
        return None

    # Load data
    print(f"  Loading data: {data_path}")
    df = pd.read_parquet(data_path)

    train_df = df[df["SPLIT"] == "train"]
    val_df = df[df["SPLIT"] == "val"]
    test_df = df[df["SPLIT"] == "test"]

    print(
        f"  Data splits: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}"
    )

    # Evaluate each model
    results = []

    for model_path in model_files:
        print(f"\n  Evaluating: {model_path.name}")

        # Load model
        model_data = joblib.load(model_path)
        model = model_data["model"]
        scaler = model_data["scaler"]
        features = model_data["features"]
        target = model_data["target"]
        model_name = model_data["model_name"]

        # Evaluate on all splits
        for split_name, df_split in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            if len(df_split) == 0:
                continue

            metrics = evaluate_model_on_split(model, scaler, features, df_split, target)

            results.append(
                {"target": target, "model": model_name, "split": split_name, **metrics}
            )

    results_df = pd.DataFrame(results)
    print("\n  ✓ Evaluation complete")

    return results_df


def print_summary_table(results_df):
    """Print formatted summary table."""
    print("\n" + "=" * 90)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 90)

    # Test set results
    test_results = results_df[results_df["split"] == "test"].copy()

    if len(test_results) == 0:
        print("No test results found!")
        return

    # Format table
    table_data = []
    for _, row in test_results.iterrows():
        table_data.append(
            [
                row["target"],
                row["model"],
                f"{row['mae']:.3f}",
                f"{row['rmse']:.3f}",
                f"{row['r2']:.3f}",
                f"{row['baseline_mae']:.3f}" if row["baseline_mae"] else "N/A",
                f"{row['improvement_pct']:+.1f}%" if row["improvement_pct"] else "N/A",
            ]
        )

    headers = ["Target", "Model", "MAE", "RMSE", "R²", "Baseline MAE", "Improvement"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def print_detailed_splits(results_df, target):
    """Print detailed results for a specific target across all splits."""
    print(f"\n{'='*70}")
    print(f"DETAILED RESULTS: {target}")
    print("=" * 70)

    target_results = results_df[results_df["target"] == target].copy()

    table_data = []
    for _, row in target_results.iterrows():
        table_data.append(
            [
                row["split"].upper(),
                f"{row['n_samples']:,}",
                f"{row['mae']:.3f}",
                f"{row['rmse']:.3f}",
                f"{row['r2']:.3f}",
            ]
        )

    headers = ["Split", "N", "MAE", "RMSE", "R²"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Check for overfitting
    train_mae = target_results[target_results["split"] == "train"]["mae"].values[0]
    test_mae = target_results[target_results["split"] == "test"]["mae"].values[0]
    overfit_pct = (test_mae - train_mae) / train_mae * 100

    print(f"\nOverfitting check: {overfit_pct:.1f}% MAE increase from train to test")

    if overfit_pct < 5:
        print("  ✓ No overfitting detected")
    elif overfit_pct < 15:
        print("  ⚠ Slight overfitting")
    else:
        print("  ❌ Significant overfitting detected")


def compare_to_goals(results_df):
    """Compare results to project goals."""
    print("\n" + "=" * 70)
    print("COMPARISON TO PROJECT GOALS")
    print("=" * 70)

    goals = {"PTS": 3.6, "REB": 2.2, "AST": 2.0}

    test_results = results_df[results_df["split"] == "test"]

    table_data = []
    for target in ["PTS", "REB", "AST"]:
        row = test_results[test_results["target"] == target]
        if len(row) > 0:
            mae = row["mae"].values[0]
            goal = goals[target]
            achieved = mae <= goal
            status = "✓ ACHIEVED" if achieved else "✗ NOT YET"

            table_data.append(
                [target, f"{mae:.3f}", f"{goal:.3f}", f"{mae - goal:+.3f}", status]
            )

    headers = ["Target", "Actual MAE", "Goal MAE", "Difference", "Status"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def save_report(results_df, output_path):
    """Save detailed evaluation report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving evaluation report to {output_path}...")

    with open(output_path, "w") as f:
        f.write("# NBA Player Performance Prediction - Model Evaluation Report\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary table
        f.write("## Test Set Performance\n\n")
        test_results = results_df[results_df["split"] == "test"]
        f.write(test_results.to_markdown(index=False))
        f.write("\n\n")

        # Detailed results per target
        for target in ["PTS", "REB", "AST"]:
            f.write(f"## {target} - All Splits\n\n")
            target_results = results_df[results_df["target"] == target]
            f.write(target_results.to_markdown(index=False))
            f.write("\n\n")

    print(f"  ✓ Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained NBA prediction models"
    )
    parser.add_argument(
        "--models",
        default="artifacts/models/",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--data",
        default="data/processed/features_enhanced_3seasons.parquet",
        help="Processed features file",
    )
    parser.add_argument(
        "--output",
        default="reports/evaluation_report.md",
        help="Output path for evaluation report",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed results for each target"
    )

    args = parser.parse_args()

    print("=" * 90)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 90)
    print(f"\nModels: {args.models}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")

    # Evaluate all models
    results_df = evaluate_all_models(args.models, args.data)

    if results_df is None or len(results_df) == 0:
        print("\n❌ No results to display")
        return

    # Print summary
    print_summary_table(results_df)

    # Compare to goals
    compare_to_goals(results_df)

    # Print detailed results if verbose
    if args.verbose:
        for target in results_df["target"].unique():
            print_detailed_splits(results_df, target)

    # Save report
    save_report(results_df, args.output)

    print("\n" + "=" * 90)
    print("EVALUATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
