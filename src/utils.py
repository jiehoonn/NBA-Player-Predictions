"""
Utility functions for NBA player prediction pipeline.

Common helpers for configuration, logging, data I/O, and time-based splits.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Configure logging for the pipeline.

    Args:
        config: Configuration dictionary

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_time_splits(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    date_col: str = 'GAME_DATE'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create time-based train/validation/test splits.

    CRITICAL: Never shuffle! This is time series data.

    Args:
        df: DataFrame with game data
        train_end: End date for training (YYYY-MM-DD)
        val_end: End date for validation (YYYY-MM-DD)
        date_col: Name of date column

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Create splits
    train = df[df[date_col] < train_end].copy()
    val = df[(df[date_col] >= train_end) & (df[date_col] < val_end)].copy()
    test = df[df[date_col] >= val_end].copy()

    return train, val, test


def save_parquet(df: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame to parquet file.

    Args:
        df: DataFrame to save
        filepath: Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)


def load_parquet(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame from parquet file.

    Args:
        filepath: Input file path

    Returns:
        Loaded DataFrame
    """
    return pd.read_parquet(filepath)


if __name__ == "__main__":
    # Test utilities
    config = load_config()
    print("✓ Config loaded successfully")
    print(f"  Seasons: {config['data']['seasons']}")
    print(f"  Random seed: {config['models']['random_seed']}")
