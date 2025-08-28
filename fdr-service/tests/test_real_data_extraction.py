# fdr-service/test_extraction.py
"""
Simple test script to debug data extraction issues
"""

import sys
import json
from pathlib import Path



def test_extraction():
    """Test data extraction with real data"""

    try:
        from data.extractor import DataExtractor

        # Path to your data file
        data_path = Path("data-service/data/fbref_data.json")

        if not data_path.exists():
            print(f"Data file not found: {data_path}")
            print("Please update the path in this script")
            return

        print(f"Loading data from: {data_path}")
        extractor = DataExtractor(data_path)

        # Load and inspect raw data structure
        raw_data = extractor.load_data()

        print(f"\nAvailable leagues: {list(raw_data.keys())}")

        # Test extraction for first league
        first_league = list(raw_data.keys())[0]
        print(f"\nTesting extraction for: {first_league}")

        # Get a sample team to inspect structure
        teams_data = raw_data[first_league].get('teams', {})
        if teams_data:
            sample_team_name = list(teams_data.keys())[0]
            sample_team = teams_data[sample_team_name]

            print(f"\nSample team ({sample_team_name}) structure:")
            for key, value in sample_team.items():
                if isinstance(value, dict):
                    print(f"  {key}: {list(value.keys())}")
                elif isinstance(value, list):
                    print(f"  {key}: list with {len(value)} items")
                else:
                    print(f"  {key}: {type(value)}")

            # Check home_away_stats specifically
            if 'home_away_stats' in sample_team:
                ha_stats = sample_team['home_away_stats']
                print(f"\nHome/Away stats structure:")
                for location, stats in ha_stats.items():
                    print(f"  {location}: {list(stats.keys())}")

        # Now try the extraction
        print(f"\n{'=' * 50}")
        print("RUNNING EXTRACTION...")
        print('=' * 50)

        fixtures_df, team_stats_df, player_aggregates_df = extractor.extract_league_data(first_league)

        print(f"\nExtraction Results:")
        print(f"  Fixtures: {len(fixtures_df)} rows")
        print(f"  Team Stats: {len(team_stats_df)} rows")
        print(f"  Player Aggregates: {len(player_aggregates_df)} rows")

        if not team_stats_df.empty:
            print(f"\nTeam Stats columns ({len(team_stats_df.columns)}):")
            home_away_cols = [col for col in team_stats_df.columns if 'home_' in col or 'away_' in col]
            print(f"  Home/Away columns: {home_away_cols}")

            overall_cols = [col for col in team_stats_df.columns if 'overall_' in col]
            print(f"  Overall columns: {overall_cols[:10]}...")  # First 10

        print(f"\nSUCCESS: Data extraction completed without errors!")

    except Exception as e:
        print(f"ERROR during extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_extraction()