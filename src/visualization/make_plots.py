#!/usr/bin/env python3
"""
Visualization Script - Generate plots for model evaluation and feature analysis

Based on notebook 09 findings, creates:
- Model performance comparison charts
- Feature importance plots (XGBoost)
- Prediction vs Actual scatter plots
- Residual analysis
- Time series predictions

Usage:
    python src/visualization/make_plots.py --models artifacts/models/ --data data/processed/features_enhanced_3seasons.parquet --output reports/figures/
"""

import argparse
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


def load_model_and_data(model_path, data_path):
    """Load trained model and test data."""
    print(f"Loading model: {model_path}")
    model_data = joblib.load(model_path)

    print(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)

    return model_data, df


def plot_performance_comparison(metrics_dir, output_dir):
    """
    Create bar chart comparing model performance across all targets.

    Shows MAE, baseline, and improvement for PTS, REB, AST.
    """
    print("\nCreating performance comparison plot...")

    metrics_dir = Path(metrics_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics for all targets
    targets = ["PTS", "REB", "AST"]
    data = []

    for target in targets:
        metrics_file = metrics_dir / f"{target}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                data.append(
                    {
                        "Target": target,
                        "Model MAE": metrics["test_mae"],
                        "Baseline MAE": metrics["baseline_mae"],
                        "Improvement (%)": metrics["improvement_pct"],
                    }
                )

    if not data:
        print("  No metrics found!")
        return

    df_metrics = pd.DataFrame(data)

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: MAE comparison
    x = np.arange(len(targets))
    width = 0.35

    ax1.bar(
        x - width / 2, df_metrics["Model MAE"], width, label="Model", color="steelblue"
    )
    ax1.bar(
        x + width / 2,
        df_metrics["Baseline MAE"],
        width,
        label="Baseline (5-game avg)",
        color="lightcoral",
    )

    ax1.set_xlabel("Target Variable")
    ax1.set_ylabel("Mean Absolute Error")
    ax1.set_title("Model Performance vs Baseline")
    ax1.set_xticks(x)
    ax1.set_xticklabels(targets)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Improvement percentage
    colors = ["green" if imp > 0 else "red" for imp in df_metrics["Improvement (%)"]]
    ax2.barh(
        df_metrics["Target"], df_metrics["Improvement (%)"], color=colors, alpha=0.7
    )
    ax2.set_xlabel("Improvement over Baseline (%)")
    ax2.set_title("Model Improvement")
    ax2.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "model_performance_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def plot_feature_importance(model_path, output_dir, top_n=15):
    """
    Plot feature importance for XGBoost models.
    """
    print(f"\nCreating feature importance plot for {model_path.stem}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_data = joblib.load(model_path)

    # Only works for XGBoost
    if model_data["model_name"] != "XGBoost":
        print(f"  Skipping (not XGBoost)")
        return

    model = model_data["model"]
    features = model_data["features"]

    # Get feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame(
        {"feature": features, "importance": importance}
    ).sort_values("importance", ascending=False)

    # Plot top N features
    top_features = feature_importance.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features["importance"], color="steelblue")
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Feature Importance")
    plt.title(
        f'Top {top_n} Features - {model_data["target"]} ({model_data["model_name"]})'
    )
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / f'feature_importance_{model_data["target"]}.png'
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def plot_predictions_vs_actual(model_path, data_path, output_dir):
    """
    Create scatter plot of predictions vs actual values for test set.
    """
    print(f"\nCreating predictions vs actual plot for {model_path.stem}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    model_data = joblib.load(model_path)
    df = pd.read_parquet(data_path)

    # Get test set
    test_df = df[df["SPLIT"] == "test"].copy()

    target = model_data["target"]
    features = model_data["features"]
    model = model_data["model"]
    scaler = model_data["scaler"]

    # Prepare features
    X_test = test_df[features]
    y_test = test_df[target]

    # Scale if needed
    if scaler is not None:
        X_test_proc = scaler.transform(X_test)
    else:
        X_test_proc = X_test.values

    # Predict
    y_pred = model.predict(X_test_proc)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax1.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction"
    )

    ax1.set_xlabel(f"Actual {target}")
    ax1.set_ylabel(f"Predicted {target}")
    ax1.set_title(
        f'{target} Predictions vs Actual ({model_data["model_name"]})\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}'
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Residual plot
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.3, s=10, color="steelblue")
    ax2.axhline(y=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel(f"Predicted {target}")
    ax2.set_ylabel("Residuals")
    ax2.set_title(f"Residual Plot - {target}")
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f"predictions_vs_actual_{target}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def plot_error_distribution(model_path, data_path, output_dir):
    """
    Plot distribution of prediction errors.
    """
    print(f"\nCreating error distribution plot for {model_path.stem}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    model_data = joblib.load(model_path)
    df = pd.read_parquet(data_path)

    # Get test set
    test_df = df[df["SPLIT"] == "test"].copy()

    target = model_data["target"]
    features = model_data["features"]
    model = model_data["model"]
    scaler = model_data["scaler"]

    # Prepare features
    X_test = test_df[features]
    y_test = test_df[target]

    # Scale if needed
    if scaler is not None:
        X_test_proc = scaler.transform(X_test)
    else:
        X_test_proc = X_test.values

    # Predict
    y_pred = model.predict(X_test_proc)
    errors = y_test - y_pred

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    plt.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero Error")
    plt.axvline(
        x=errors.mean(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Mean Error: {errors.mean():.2f}",
    )
    plt.xlabel("Prediction Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title(
        f'Error Distribution - {target} ({model_data["model_name"]})\nMAE: {abs(errors).mean():.2f}, Std: {errors.std():.2f}'
    )
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / f"error_distribution_{target}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def plot_time_series_predictions(
    model_path, data_path, output_dir, player_name=None, n_games=50
):
    """
    Plot time series of predictions vs actual for a sample player.
    """
    print(f"\nCreating time series plot for {model_path.stem}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    model_data = joblib.load(model_path)
    df = pd.read_parquet(data_path)

    # Get test set
    test_df = df[df["SPLIT"] == "test"].copy()

    # Select a player with many games
    if player_name:
        player_df = test_df[test_df["PLAYER_NAME"] == player_name].copy()
    else:
        # Pick player with most test games
        player_counts = test_df["PLAYER_ID"].value_counts()
        top_player_id = player_counts.index[0]
        player_df = test_df[test_df["PLAYER_ID"] == top_player_id].copy()

    if len(player_df) == 0:
        print("  No data found for player")
        return

    # Take last n_games
    player_df = player_df.sort_values("GAME_DATE").tail(n_games)

    target = model_data["target"]
    features = model_data["features"]
    model = model_data["model"]
    scaler = model_data["scaler"]

    # Prepare features
    X = player_df[features]
    y_actual = player_df[target]

    # Scale if needed
    if scaler is not None:
        X_proc = scaler.transform(X)
    else:
        X_proc = X.values

    # Predict
    y_pred = model.predict(X_proc)

    # Plot
    plt.figure(figsize=(14, 6))
    game_numbers = range(len(y_actual))

    plt.plot(
        game_numbers,
        y_actual.values,
        "o-",
        label="Actual",
        linewidth=2,
        markersize=6,
        color="steelblue",
    )
    plt.plot(
        game_numbers,
        y_pred,
        "s--",
        label="Predicted",
        linewidth=2,
        markersize=6,
        color="coral",
    )

    player_name_str = (
        player_df.iloc[0]["PLAYER_NAME"]
        if "PLAYER_NAME" in player_df.columns
        else "Unknown"
    )
    mae = mean_absolute_error(y_actual, y_pred)

    plt.xlabel("Game Number")
    plt.ylabel(target)
    plt.title(f"{target} Predictions - {player_name_str}\nMAE: {mae:.2f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / f"time_series_{target}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for model evaluation"
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
        "--output", default="reports/figures/", help="Output directory for plots"
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=[
            "performance",
            "importance",
            "predictions",
            "errors",
            "timeseries",
            "all",
        ],
        default=["all"],
        help="Which plots to generate",
    )

    args = parser.parse_args()

    models_dir = Path(args.models)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MODEL VISUALIZATION PIPELINE")
    print("=" * 70)
    print(f"\nModels: {args.models}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Plots: {', '.join(args.plots)}")

    plots_to_generate = args.plots
    if "all" in plots_to_generate:
        plots_to_generate = [
            "performance",
            "importance",
            "predictions",
            "errors",
            "timeseries",
        ]

    # 1. Performance comparison (uses metrics files)
    if "performance" in plots_to_generate:
        plot_performance_comparison(models_dir, output_dir)

    # Find all model files
    model_files = list(models_dir.glob("*_*.joblib"))

    if not model_files:
        print("\n❌ No model files found!")
        return

    print(f"\nFound {len(model_files)} models:")
    for mf in model_files:
        print(f"  - {mf.name}")

    # Generate plots for each model
    for model_path in model_files:

        if "importance" in plots_to_generate:
            plot_feature_importance(model_path, output_dir)

        if "predictions" in plots_to_generate:
            plot_predictions_vs_actual(model_path, args.data, output_dir)

        if "errors" in plots_to_generate:
            plot_error_distribution(model_path, args.data, output_dir)

        if "timeseries" in plots_to_generate:
            plot_time_series_predictions(model_path, args.data, output_dir)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"  Plots saved to: {output_dir}")
    print(f"  Total plots: {len(list(output_dir.glob('*.png')))}")
    print("=" * 70)


if __name__ == "__main__":
    main()
