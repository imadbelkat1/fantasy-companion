import os
import sys

import pytest
import pandas as pd
import json
from pathlib import Path
import tempfile

try:
    from data.extractor import DataExtractor
except ImportError:
    try:
        from src.data.extractor import DataExtractor
    except ImportError:
        import sys
        from pathlib import Path
        # Last resort: add paths manually
        test_dir = Path(__file__).parent
        project_root = test_dir.parent
        sys.path.insert(0, str(project_root))
        sys.path.insert(0, str(project_root / "src"))
        from data.extractor import DataExtractor




class TestDataExtractor:
    """Test data extraction functionality"""

    def setup_method(self):
        """Setup test data"""
        self.test_data = {
            "epl": {
                "teams": {
                    "Test Team": {
                        "fixtures": [
                            {
                                "gameweek": 1,
                                "date": "2023-08-12",
                                "home_team": "Test Team",
                                "away_team": "Other Team",
                                "home_xg": 2.0,
                                "away_xg": 1.0,
                                "score": "2-1"
                            }
                        ],
                        "overall_stats": {
                            "games_played": 1,
                            "wins": 1,
                            "draws": 0,
                            "losses": 0,
                            "points": 3,
                            "goals_for": 2,
                            "goals_against": 1,
                            "xg_for": 2.0,
                            "xg_against": 1.0
                        }
                    }
                },
                "players": {
                    "players": {
                        "Test Player": {
                            "team": "Test Team",
                            "position": "FW",
                            "goals": 1,
                            "assists": 0,
                            "xg_per90": 0.8
                        }
                    },
                    "goalkeepers": {
                        "Test Keeper": {
                            "team": "Test Team",
                            "saves": 3,
                            "save_pct": 75.0
                        }
                    }
                }
            }
        }

    def test_extract_fixtures(self):
        """Test fixture extraction"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            temp_path = Path(f.name)

        try:
            extractor = DataExtractor(temp_path)
            fixtures_df, team_stats_df, player_aggregates_df = extractor.extract_league_data("epl")

            assert not fixtures_df.empty
            assert len(fixtures_df) == 1
            assert fixtures_df.iloc[0]['home_team'] == 'Test Team'
            assert fixtures_df.iloc[0]['home_xg'] == 2.0
            assert fixtures_df.iloc[0]['away_xg'] == 1.0

        finally:
            temp_path.unlink()

    def test_parse_score(self):
        """Test score parsing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            temp_path = Path(f.name)

        try:
            extractor = DataExtractor(temp_path)

            # Test valid score
            home_goals, away_goals = extractor._parse_score("2-1")
            assert home_goals == 2
            assert away_goals == 1

            # Test invalid score
            home_goals, away_goals = extractor._parse_score("invalid")
            assert home_goals is None
            assert away_goals is None

        finally:
            temp_path.unlink()

    def test_empty_league(self):
        """Test handling of empty league data"""
        empty_data = {"epl": {}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(empty_data, f)
            temp_path = Path(f.name)

        try:
            extractor = DataExtractor(temp_path)
            fixtures_df, team_stats_df, player_aggregates_df = extractor.extract_league_data("epl")

            assert fixtures_df.empty
            assert team_stats_df.empty
            assert player_aggregates_df.empty

        finally:
            temp_path.unlink()
