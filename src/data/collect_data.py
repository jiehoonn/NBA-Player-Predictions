#!/usr/bin/env python3
"""
Data Collection Script - Collect 3 seasons of NBA player data with contextual features

Based on notebook 07 findings:
- Collect 3 seasons (2022-23, 2023-24, 2024-25)
- Get top 120 players by minutes played
- Add opponent team stats (DEF_RATING, OFF_RATING, PACE)
- Add game context (home/away, rest days, back-to-back)

Usage:
    python src/data/collect_data.py --seasons 2022-23 2023-24 2024-25 --output data/raw/player_gamelogs_enhanced.parquet
"""

import argparse
import time
import os
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from nba_api.stats.endpoints import (
    playergamelog,
    leaguedashteamstats,
    leaguedashplayerstats,
)
from nba_api.stats.static import players, teams


def retry_with_exponential_backoff(
    func, max_retries=5, base_delay=2.0, max_delay=60.0, *args, **kwargs
):
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        *args, **kwargs: Arguments to pass to func

    Returns:
        Result of func if successful, None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  ✗ Failed after {max_retries} attempts: {e}")
                return None

            delay = min(base_delay * (2 ** attempt), max_delay)
            print(f"  ⚠ Attempt {attempt + 1} failed: {e}")
            print(f"  ↻ Retrying in {delay:.1f}s...")
            time.sleep(delay)

    return None


def get_top_players(season="2023-24", top_n=120):
    """Get top N players by total minutes played in a season."""
    print(f"Fetching top {top_n} players from {season} season...")

    try:
        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="Totals",
        )

        stats_df = player_stats.get_data_frames()[0]
        time.sleep(1.5)

        top_players = stats_df.nlargest(top_n, "MIN")[
            ["PLAYER_ID", "PLAYER_NAME", "MIN", "GP"]
        ]

        print(f"✓ Found {len(top_players)} players")
        print(
            f"  Minutes range: {top_players['MIN'].min():.0f} - {top_players['MIN'].max():.0f}"
        )

        return top_players["PLAYER_ID"].tolist()

    except Exception as e:
        print(f"Error fetching player stats: {e}")
        return []


def collect_player_gamelogs(player_ids, seasons, checkpoint_dir="data/interim"):
    """
    Collect game logs for all players across multiple seasons with retry logic and checkpointing.

    Args:
        player_ids: List of player IDs
        seasons: List of season strings
        checkpoint_dir: Directory to save intermediate checkpoints

    Returns:
        DataFrame with all game logs
    """
    print(
        f"\nCollecting game logs for {len(player_ids)} players across {len(seasons)} seasons..."
    )

    # Create checkpoint directory if needed
    os.makedirs(checkpoint_dir, exist_ok=True)

    all_gamelogs = []
    failed_requests = []

    def fetch_player_gamelog(player_id, season):
        """Helper to fetch gamelog for a single player-season."""
        gamelog = playergamelog.PlayerGameLog(
            player_id=str(player_id),
            season=season,
            season_type_all_star="Regular Season",
        )
        df = gamelog.get_data_frames()[0]

        if len(df) > 0:
            df["PLAYER_ID"] = player_id
            df["SEASON"] = season
            return df
        return None

    for season in seasons:
        print(f"\n{'='*60}")
        print(f"Season: {season}")
        print("=" * 60)

        # Check for existing checkpoint
        checkpoint_file = os.path.join(
            checkpoint_dir, f"gamelogs_checkpoint_{season}.parquet"
        )

        if os.path.exists(checkpoint_file):
            print(f"  ↻ Loading checkpoint from {checkpoint_file}")
            season_df = pd.read_parquet(checkpoint_file)
            all_gamelogs.append(season_df)
            print(f"  ✓ Loaded {len(season_df)} games from checkpoint")
            continue

        # Collect data for this season
        season_gamelogs = []

        for player_id in tqdm(player_ids, desc=f"{season}"):
            # Retry with exponential backoff
            result = retry_with_exponential_backoff(
                fetch_player_gamelog,
                max_retries=3,  # 3 retries for individual players
                base_delay=2.0,
                player_id=player_id,
                season=season,
            )

            if result is not None:
                season_gamelogs.append(result)
            else:
                failed_requests.append(
                    {
                        "player_id": player_id,
                        "season": season,
                        "error": "Failed after retries",
                    }
                )

            time.sleep(0.8)  # Increased from 0.6 to 0.8 seconds

        # Save checkpoint for this season
        if len(season_gamelogs) > 0:
            season_df = pd.concat(season_gamelogs, ignore_index=True)
            season_df.to_parquet(checkpoint_file)
            all_gamelogs.append(season_df)
            print(f"\n  ✓ Checkpoint saved: {checkpoint_file}")

    print(f"\n✓ Collected {len(all_gamelogs)} player-seasons")
    if failed_requests:
        print(f"⚠ {len(failed_requests)} failed requests")

    # Handle empty list BEFORE concat
    if len(all_gamelogs) == 0:
        raise RuntimeError(
            "❌ No game logs collected! All player requests failed. Check NBA API status."
        )

    return pd.concat(all_gamelogs, ignore_index=True)


def collect_team_stats(seasons):
    """
    Collect team statistics for opponent context with retry logic.

    Args:
        seasons: List of season strings (e.g., ['2022-23', '2023-24'])

    Returns:
        DataFrame with team stats, or empty DataFrame if all attempts fail
    """
    print(f"\nCollecting team statistics for {len(seasons)} seasons...")

    all_team_stats = []

    def fetch_season_team_stats(season):
        """Helper to fetch team stats for a single season."""
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
        )

        df = team_stats.get_data_frames()[0]
        df["SEASON"] = season

        df = df[
            [
                "TEAM_ID",
                "TEAM_NAME",
                "SEASON",
                "GP",
                "W",
                "L",
                "OFF_RATING",
                "DEF_RATING",
                "NET_RATING",
                "PACE",
            ]
        ]

        return df

    for season in seasons:
        print(f"  Fetching {season} team stats...")

        # Retry with exponential backoff
        result = retry_with_exponential_backoff(
            fetch_season_team_stats,
            max_retries=5,
            base_delay=3.0,  # Longer initial delay for team stats
            season=season,
        )

        if result is not None:
            all_team_stats.append(result)
            print(f"  ✓ Success for {season}")
            time.sleep(2.0)  # Increased delay between seasons
        else:
            print(f"  ✗ Failed to collect {season} team stats after all retries")

    # Handle empty list BEFORE concat
    if len(all_team_stats) == 0:
        print("⚠ WARNING: No team stats collected. Returning empty DataFrame.")
        print("  Opponent stats will be missing from final dataset.")
        return pd.DataFrame(
            columns=[
                "TEAM_ID",
                "TEAM_NAME",
                "SEASON",
                "GP",
                "W",
                "L",
                "OFF_RATING",
                "DEF_RATING",
                "NET_RATING",
                "PACE",
            ]
        )

    print(f"✓ Collected team stats for {len(all_team_stats)}/{len(seasons)} seasons")
    return pd.concat(all_team_stats, ignore_index=True)


def add_game_context(df):
    """Add home/away, opponent info, rest days, and back-to-back flags."""
    print("\nAdding game context features...")

    # Extract opponent and location
    def extract_opponent_and_location(matchup):
        if " vs. " in matchup:
            return matchup.split(" vs. ")[1], True
        elif " @ " in matchup:
            return matchup.split(" @ ")[1], False
        return None, None

    df[["OPP_ABBREV", "IS_HOME"]] = df["MATCHUP"].apply(
        lambda x: pd.Series(extract_opponent_and_location(x))
    )

    # Map opponent abbreviation to team ID
    all_teams = teams.get_teams()
    team_abbrev_to_id = {t["abbreviation"]: t["id"] for t in all_teams}
    df["OPP_TEAM_ID"] = df["OPP_ABBREV"].map(team_abbrev_to_id)

    # Compute rest days per player
    rest_features = []

    for player_id in tqdm(df["PLAYER_ID"].unique(), desc="Computing rest days"):
        player_df = df[df["PLAYER_ID"] == player_id].copy()
        player_df = player_df.sort_values("GAME_DATE")

        player_df["PREV_GAME_DATE"] = player_df["GAME_DATE"].shift(1)
        player_df["REST_DAYS"] = (
            player_df["GAME_DATE"] - player_df["PREV_GAME_DATE"]
        ).dt.days - 1
        player_df.loc[player_df.index[0], "REST_DAYS"] = np.nan

        player_df["IS_BACK_TO_BACK"] = (player_df["REST_DAYS"] == 0).astype(int)

        # FIX: Include PLAYER_ID to avoid duplication
        rest_features.append(
            player_df[["PLAYER_ID", "Game_ID", "REST_DAYS", "IS_BACK_TO_BACK"]]
        )

    rest_features_df = pd.concat(rest_features, ignore_index=True)

    # FIX: Merge on BOTH PLAYER_ID and Game_ID
    df = df.merge(rest_features_df, on=["PLAYER_ID", "Game_ID"], how="left")

    print(f"✓ Game context added")
    print(f"  Home games: {df['IS_HOME'].sum():,} ({df['IS_HOME'].mean()*100:.1f}%)")
    print(f"  Back-to-back games: {df['IS_BACK_TO_BACK'].sum():,}")

    return df


def merge_opponent_stats(gamelogs_df, team_stats_df):
    """Merge opponent team statistics to game logs."""
    print("\nMerging opponent statistics...")

    opponent_stats = team_stats_df.copy()
    opponent_stats = opponent_stats.rename(
        columns={
            "TEAM_ID": "OPP_TEAM_ID",
            "TEAM_NAME": "OPP_TEAM_NAME",
            "GP": "OPP_GP",
            "W": "OPP_W",
            "L": "OPP_L",
            "OFF_RATING": "OPP_OFF_RATING",
            "DEF_RATING": "OPP_DEF_RATING",
            "NET_RATING": "OPP_NET_RATING",
            "PACE": "OPP_PACE",
        }
    )

    enhanced_df = gamelogs_df.merge(
        opponent_stats, on=["OPP_TEAM_ID", "SEASON"], how="left"
    )

    coverage = enhanced_df["OPP_DEF_RATING"].notna().mean() * 100
    print(f"✓ Opponent stats merged ({coverage:.1f}% coverage)")

    return enhanced_df


def main():
    parser = argparse.ArgumentParser(
        description="Collect NBA player game data with contextual features"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2022-23", "2023-24", "2024-25"],
        help="NBA seasons to collect",
    )
    parser.add_argument(
        "--top-players", type=int, default=120, help="Number of top players to collect"
    )
    parser.add_argument(
        "--output",
        default="data/raw/player_gamelogs_enhanced_2022-2025.parquet",
        help="Output file path",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NBA DATA COLLECTION - Enhanced Pipeline")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Seasons: {', '.join(args.seasons)}")
    print(f"  Top players: {args.top_players}")
    print(f"  Output: {args.output}")

    start_time = datetime.now()

    # Step 1: Get top players
    player_ids = get_top_players(season="2023-24", top_n=args.top_players)

    if not player_ids:
        print("❌ Failed to get player IDs. Exiting.")
        return

    # Step 2: Collect game logs (with checkpointing)
    try:
        gamelogs_df = collect_player_gamelogs(player_ids, args.seasons)

        # Convert GAME_DATE to datetime (handle both string and datetime types from checkpoints)
        # Use format='mixed' to handle different date formats from API vs checkpoints
        gamelogs_df["GAME_DATE"] = pd.to_datetime(
            gamelogs_df["GAME_DATE"], format="mixed", errors="coerce"
        )

        gamelogs_df = gamelogs_df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(
            drop=True
        )

        print(f"\n✓ Collected {len(gamelogs_df):,} games")

    except RuntimeError as e:
        print(f"\n❌ FATAL ERROR: {e}")
        print("Exiting...")
        return

    # Step 3: Collect team stats (gracefully handle failures)
    team_stats_df = collect_team_stats(args.seasons)

    # Step 4: Add game context
    gamelogs_df = add_game_context(gamelogs_df)

    # Step 5: Merge opponent stats (gracefully handle empty team stats)
    if len(team_stats_df) > 0:
        enhanced_df = merge_opponent_stats(gamelogs_df, team_stats_df)
    else:
        print("\n⚠ WARNING: Skipping opponent stats merge (no team data available)")
        print("  Dataset will have missing opponent features")
        # Add placeholder columns for opponent stats
        enhanced_df = gamelogs_df.copy()
        for col in [
            "OPP_TEAM_NAME",
            "OPP_GP",
            "OPP_W",
            "OPP_L",
            "OPP_OFF_RATING",
            "OPP_DEF_RATING",
            "OPP_NET_RATING",
            "OPP_PACE",
        ]:
            enhanced_df[col] = np.nan

    # Step 6: Save (ALWAYS save what we collected)
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        enhanced_df.to_parquet(args.output)
        print(f"\n✓ Data saved successfully: {args.output}")

    except Exception as e:
        print(f"\n❌ ERROR saving file: {e}")
        # Try to save to a backup location
        backup_file = "data/interim/backup_gamelogs.parquet"
        try:
            os.makedirs("data/interim", exist_ok=True)
            enhanced_df.to_parquet(backup_file)
            print(f"  ✓ Saved backup to: {backup_file}")
        except Exception as backup_error:
            print(f"  ✗ Backup also failed: {backup_error}")
            return

    elapsed = (datetime.now() - start_time).total_seconds() / 60

    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"  File: {args.output}")
    print(f"  Records: {len(enhanced_df):,}")
    print(f"  Players: {enhanced_df['PLAYER_ID'].nunique()}")
    print(
        f"  Date range: {enhanced_df['GAME_DATE'].min().date()} to {enhanced_df['GAME_DATE'].max().date()}"
    )
    print(f"  Time elapsed: {elapsed:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
