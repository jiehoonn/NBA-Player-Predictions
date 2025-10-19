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
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, leaguedashplayerstats
from nba_api.stats.static import players, teams


def get_top_players(season='2023-24', top_n=120):
    """Get top N players by total minutes played in a season."""
    print(f"Fetching top {top_n} players from {season} season...")

    try:
        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star='Regular Season',
            per_mode_detailed='Totals'
        )

        stats_df = player_stats.get_data_frames()[0]
        time.sleep(1.5)

        top_players = stats_df.nlargest(top_n, 'MIN')[['PLAYER_ID', 'PLAYER_NAME', 'MIN', 'GP']]

        print(f"✓ Found {len(top_players)} players")
        print(f"  Minutes range: {top_players['MIN'].min():.0f} - {top_players['MIN'].max():.0f}")

        return top_players['PLAYER_ID'].tolist()

    except Exception as e:
        print(f"Error fetching player stats: {e}")
        return []


def collect_player_gamelogs(player_ids, seasons):
    """Collect game logs for all players across multiple seasons."""
    print(f"\nCollecting game logs for {len(player_ids)} players across {len(seasons)} seasons...")

    all_gamelogs = []
    failed_requests = []

    for season in seasons:
        print(f"\n{'='*60}")
        print(f"Season: {season}")
        print('='*60)

        for player_id in tqdm(player_ids, desc=f"{season}"):
            try:
                gamelog = playergamelog.PlayerGameLog(
                    player_id=str(player_id),
                    season=season,
                    season_type_all_star='Regular Season'
                )
                df = gamelog.get_data_frames()[0]

                if len(df) > 0:
                    df['PLAYER_ID'] = player_id
                    df['SEASON'] = season
                    all_gamelogs.append(df)

                time.sleep(0.6)  # Rate limit

            except Exception as e:
                failed_requests.append({'player_id': player_id, 'season': season, 'error': str(e)})
                time.sleep(2)
                continue

    print(f"\n✓ Collected {len(all_gamelogs)} player-seasons")
    if failed_requests:
        print(f"⚠ {len(failed_requests)} failed requests")

    return pd.concat(all_gamelogs, ignore_index=True)


def collect_team_stats(seasons):
    """Collect team statistics for opponent context."""
    print(f"\nCollecting team statistics for {len(seasons)} seasons...")

    all_team_stats = []

    for season in seasons:
        try:
            print(f"  Fetching {season} team stats...")

            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Advanced'
            )

            df = team_stats.get_data_frames()[0]
            df['SEASON'] = season

            df = df[['TEAM_ID', 'TEAM_NAME', 'SEASON', 'GP', 'W', 'L',
                     'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE']]

            all_team_stats.append(df)
            time.sleep(1.5)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    print(f"✓ Collected team stats for {len(all_team_stats)} seasons")
    return pd.concat(all_team_stats, ignore_index=True)


def add_game_context(df):
    """Add home/away, opponent info, rest days, and back-to-back flags."""
    print("\nAdding game context features...")

    # Extract opponent and location
    def extract_opponent_and_location(matchup):
        if ' vs. ' in matchup:
            return matchup.split(' vs. ')[1], True
        elif ' @ ' in matchup:
            return matchup.split(' @ ')[1], False
        return None, None

    df[['OPP_ABBREV', 'IS_HOME']] = df['MATCHUP'].apply(
        lambda x: pd.Series(extract_opponent_and_location(x))
    )

    # Map opponent abbreviation to team ID
    all_teams = teams.get_teams()
    team_abbrev_to_id = {t['abbreviation']: t['id'] for t in all_teams}
    df['OPP_TEAM_ID'] = df['OPP_ABBREV'].map(team_abbrev_to_id)

    # Compute rest days per player
    rest_features = []

    for player_id in tqdm(df['PLAYER_ID'].unique(), desc="Computing rest days"):
        player_df = df[df['PLAYER_ID'] == player_id].copy()
        player_df = player_df.sort_values('GAME_DATE')

        player_df['PREV_GAME_DATE'] = player_df['GAME_DATE'].shift(1)
        player_df['REST_DAYS'] = (player_df['GAME_DATE'] - player_df['PREV_GAME_DATE']).dt.days - 1
        player_df.loc[player_df.index[0], 'REST_DAYS'] = np.nan

        player_df['IS_BACK_TO_BACK'] = (player_df['REST_DAYS'] == 0).astype(int)

        # FIX: Include PLAYER_ID to avoid duplication
        rest_features.append(player_df[['PLAYER_ID', 'Game_ID', 'REST_DAYS', 'IS_BACK_TO_BACK']])

    rest_features_df = pd.concat(rest_features, ignore_index=True)

    # FIX: Merge on BOTH PLAYER_ID and Game_ID
    df = df.merge(rest_features_df, on=['PLAYER_ID', 'Game_ID'], how='left')

    print(f"✓ Game context added")
    print(f"  Home games: {df['IS_HOME'].sum():,} ({df['IS_HOME'].mean()*100:.1f}%)")
    print(f"  Back-to-back games: {df['IS_BACK_TO_BACK'].sum():,}")

    return df


def merge_opponent_stats(gamelogs_df, team_stats_df):
    """Merge opponent team statistics to game logs."""
    print("\nMerging opponent statistics...")

    opponent_stats = team_stats_df.copy()
    opponent_stats = opponent_stats.rename(columns={
        'TEAM_ID': 'OPP_TEAM_ID',
        'TEAM_NAME': 'OPP_TEAM_NAME',
        'GP': 'OPP_GP',
        'W': 'OPP_W',
        'L': 'OPP_L',
        'OFF_RATING': 'OPP_OFF_RATING',
        'DEF_RATING': 'OPP_DEF_RATING',
        'NET_RATING': 'OPP_NET_RATING',
        'PACE': 'OPP_PACE'
    })

    enhanced_df = gamelogs_df.merge(
        opponent_stats,
        on=['OPP_TEAM_ID', 'SEASON'],
        how='left'
    )

    coverage = enhanced_df['OPP_DEF_RATING'].notna().mean() * 100
    print(f"✓ Opponent stats merged ({coverage:.1f}% coverage)")

    return enhanced_df


def main():
    parser = argparse.ArgumentParser(description='Collect NBA player game data with contextual features')
    parser.add_argument('--seasons', nargs='+', default=['2022-23', '2023-24', '2024-25'],
                       help='NBA seasons to collect')
    parser.add_argument('--top-players', type=int, default=120,
                       help='Number of top players to collect')
    parser.add_argument('--output', default='data/raw/player_gamelogs_enhanced_2022-2025.parquet',
                       help='Output file path')

    args = parser.parse_args()

    print("="*70)
    print("NBA DATA COLLECTION - Enhanced Pipeline")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Seasons: {', '.join(args.seasons)}")
    print(f"  Top players: {args.top_players}")
    print(f"  Output: {args.output}")

    start_time = datetime.now()

    # Step 1: Get top players
    player_ids = get_top_players(season='2023-24', top_n=args.top_players)

    if not player_ids:
        print("❌ Failed to get player IDs. Exiting.")
        return

    # Step 2: Collect game logs
    gamelogs_df = collect_player_gamelogs(player_ids, args.seasons)
    gamelogs_df['GAME_DATE'] = pd.to_datetime(gamelogs_df['GAME_DATE'])
    gamelogs_df = gamelogs_df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

    print(f"\n✓ Collected {len(gamelogs_df):,} games")

    # Step 3: Collect team stats
    team_stats_df = collect_team_stats(args.seasons)

    # Step 4: Add game context
    gamelogs_df = add_game_context(gamelogs_df)

    # Step 5: Merge opponent stats
    enhanced_df = merge_opponent_stats(gamelogs_df, team_stats_df)

    # Step 6: Save
    enhanced_df.to_parquet(args.output)

    elapsed = (datetime.now() - start_time).total_seconds() / 60

    print("\n" + "="*70)
    print("COLLECTION COMPLETE")
    print("="*70)
    print(f"  File: {args.output}")
    print(f"  Records: {len(enhanced_df):,}")
    print(f"  Players: {enhanced_df['PLAYER_ID'].nunique()}")
    print(f"  Date range: {enhanced_df['GAME_DATE'].min().date()} to {enhanced_df['GAME_DATE'].max().date()}")
    print(f"  Time elapsed: {elapsed:.1f} minutes")
    print("="*70)


if __name__ == '__main__':
    main()
