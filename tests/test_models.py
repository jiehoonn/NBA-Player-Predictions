"""
Test suite for model validation and predictions.

Tests ensure models produce reasonable predictions and are properly saved/loaded.

NOTE: These tests require trained models and processed data, which are gitignored.
They will be skipped in CI/CD (GitHub Actions) but run locally after `make all`.
"""

import pytest
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Check if required files exist (for conditional test skipping)
MODELS_DIR = Path('models/final')
DATA_DIR = Path('data/processed')

models_exist = (
    (MODELS_DIR / 'best_model_pts.pkl').exists() and
    (MODELS_DIR / 'best_model_reb.pkl').exists() and
    (MODELS_DIR / 'best_model_ast.pkl').exists()
)

data_exists = (DATA_DIR / 'reduced_feature_names.json').exists()
metadata_exists = (MODELS_DIR / 'best_models_metadata.json').exists()
baseline_exists = Path('results/predictions/baseline_linear_results.json').exists()

skip_if_no_models = pytest.mark.skipif(
    not models_exist,
    reason="Models not found (gitignored). Run 'make all' to generate."
)

skip_if_no_data = pytest.mark.skipif(
    not data_exists,
    reason="Processed data not found (gitignored). Run 'make all' to generate."
)

skip_if_no_metadata = pytest.mark.skipif(
    not metadata_exists,
    reason="Model metadata not found (gitignored). Run 'make all' to generate."
)

skip_if_no_baseline = pytest.mark.skipif(
    not baseline_exists,
    reason="Baseline results not found (gitignored). Run 'make all' to generate."
)


@skip_if_no_models
def test_models_exist():
    """Test that final models are saved and loadable."""
    models_dir = Path('models/final')

    # Check files exist
    assert (models_dir / 'best_model_pts.pkl').exists(), "PTS model not found"
    assert (models_dir / 'best_model_reb.pkl').exists(), "REB model not found"
    assert (models_dir / 'best_model_ast.pkl').exists(), "AST model not found"

    # Try loading
    with open(models_dir / 'best_model_pts.pkl', 'rb') as f:
        model_pts = pickle.load(f)
        assert model_pts is not None, "PTS model failed to load"

    with open(models_dir / 'best_model_reb.pkl', 'rb') as f:
        model_reb = pickle.load(f)
        assert model_reb is not None, "REB model failed to load"

    with open(models_dir / 'best_model_ast.pkl', 'rb') as f:
        model_ast = pickle.load(f)
        assert model_ast is not None, "AST model failed to load"

    print("✅ Test passed: All 3 models exist and are loadable")


@skip_if_no_models
def test_predictions_in_reasonable_range():
    """Test that model predictions are within realistic NBA stat ranges."""
    models_dir = Path('models/final')

    # Load a model
    with open(models_dir / 'best_model_pts.pkl', 'rb') as f:
        model_pts = pickle.load(f)

    # Create dummy features (65 features, realistic values)
    np.random.seed(42)
    X_dummy = np.random.randn(10, 65)  # 10 samples, 65 features

    # Make predictions
    if isinstance(model_pts, dict) and model_pts.get('type') == 'ensemble':
        # Ensemble model
        w1, w2, w3 = model_pts['weights']
        predictions = (
            w1 * model_pts['lasso'].predict(X_dummy) +
            w2 * model_pts['xgboost'].predict(X_dummy) +
            w3 * model_pts['lightgbm'].predict(X_dummy)
        )
    else:
        predictions = model_pts.predict(X_dummy)

    # Test: Predictions should be in realistic range
    # NBA players score between 0-80 points (Kobe's 81 is extreme outlier)
    assert np.all(predictions >= -10), \
        f"Unrealistic prediction: {predictions.min():.1f} < -10 PTS (negative is impossible)"

    assert np.all(predictions <= 100), \
        f"Unrealistic prediction: {predictions.max():.1f} > 100 PTS (extremely rare)"

    # Test: Mean should be reasonable (NBA average ~13 PTS per game)
    mean_pred = predictions.mean()
    assert 0 < mean_pred < 50, \
        f"Unrealistic mean prediction: {mean_pred:.1f} PTS"

    print(f"✅ Test passed: Predictions in range [-10, 100], mean={mean_pred:.1f} PTS")


@skip_if_no_models
@skip_if_no_data
def test_feature_count_consistency():
    """Test that models expect 65 features."""
    models_dir = Path('models/final')

    # Load feature names
    with open(Path('data/processed/reduced_feature_names.json'), 'r') as f:
        import json
        meta = json.load(f)
        feature_names = meta['feature_names']

    expected_count = len(feature_names)

    # Load models and check
    with open(models_dir / 'best_model_pts.pkl', 'rb') as f:
        model_pts = pickle.load(f)

    # Create test data with correct number of features
    X_test = np.random.randn(1, expected_count)

    # Test: Model should accept this input
    try:
        if isinstance(model_pts, dict) and model_pts.get('type') == 'ensemble':
            _ = model_pts['lasso'].predict(X_test)
        else:
            _ = model_pts.predict(X_test)
        success = True
    except ValueError as e:
        success = False
        error_msg = str(e)

    assert success, f"Model expects different number of features: {error_msg}"

    print(f"✅ Test passed: Models correctly expect {expected_count} features")


@skip_if_no_metadata
def test_model_metadata_exists():
    """Test that model metadata JSON exists and is valid."""
    metadata_path = Path('models/final/best_models_metadata.json')

    assert metadata_path.exists(), "Model metadata not found"

    # Load and validate
    with open(metadata_path, 'r') as f:
        import json
        metadata = json.load(f)

    # Check required fields
    assert 'best_models' in metadata, "Missing 'best_models' key"
    assert 'PTS' in metadata['best_models'], "Missing PTS model metadata"
    assert 'REB' in metadata['best_models'], "Missing REB model metadata"
    assert 'AST' in metadata['best_models'], "Missing AST model metadata"

    # Check performance metrics
    for target in ['PTS', 'REB', 'AST']:
        model_info = metadata['best_models'][target]
        assert 'test_mae' in model_info, f"Missing test_mae for {target}"
        assert 'model_name' in model_info, f"Missing model_name for {target}"

        # Test: MAE should be positive
        mae = model_info['test_mae']
        assert mae > 0, f"Invalid MAE for {target}: {mae}"

    print("✅ Test passed: Model metadata is valid")


@skip_if_no_metadata
@skip_if_no_baseline
def test_baseline_vs_model_improvement():
    """Test that models beat baseline (5-game rolling average)."""
    metadata_path = Path('models/final/best_models_metadata.json')

    with open(metadata_path, 'r') as f:
        import json
        metadata = json.load(f)

    # Load baseline results
    baseline_path = Path('results/predictions/baseline_linear_results.json')
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)

    # Test: Each model should beat baseline
    for target in ['PTS', 'REB', 'AST']:
        model_mae = metadata['best_models'][target]['test_mae']
        baseline_mae = baseline['baselines'][target]

        improvement_pct = (baseline_mae - model_mae) / baseline_mae * 100

        assert model_mae < baseline_mae, \
            f"{target}: Model ({model_mae:.3f}) worse than baseline ({baseline_mae:.3f})"

        assert improvement_pct > 0, \
            f"{target}: No improvement over baseline"

        print(f"✅ {target}: Model MAE = {model_mae:.3f}, Baseline = {baseline_mae:.3f}, "
              f"Improvement = {improvement_pct:.1f}%")

    print("✅ Test passed: All models beat baseline")


if __name__ == '__main__':
    # Run tests
    test_models_exist()
    test_predictions_in_reasonable_range()
    test_feature_count_consistency()
    test_model_metadata_exists()
    test_baseline_vs_model_improvement()

    print("\n" + "="*60)
    print("✅ ALL MODEL TESTS PASSED")
    print("="*60)
    print("\nThese tests verify:")
    print("  1. All 3 models exist and are loadable")
    print("  2. Predictions are in realistic ranges")
    print("  3. Models expect 65 features")
    print("  4. Metadata JSON is valid")
    print("  5. Models beat baseline performance")
