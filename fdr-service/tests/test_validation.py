import pytest
import pandas as pd
import numpy as np

from src.utils.validation import DataValidator


class TestDataValidator:
    """Test data validation functionality"""

    def setup_method(self):
        """Setup test validator"""
        self.validator = DataValidator()

    def test_valid_data_validation(self):
        """Test validation of valid data"""
        fixtures_df = pd.DataFrame([
            {
                'fixture_id': 'test_1',
                'home_team': 'Team A',
                'away_team': 'Team B',
                'date': '2023-08-12',
                'home_xg': 1.5,
                'away_xg': 1.2,
                'home_goals': 1,
                'away_goals': 1
            }
        ])

        team_stats_df = pd.DataFrame([
            {
                'team': 'Team A',
                'league': 'epl',
                'overall_games_played': 10,
                'overall_points': 15,
                'overall_xg_for': 12.0,
                'overall_xg_against': 10.0
            },
            {
                'team': 'Team B',
                'league': 'epl',
                'overall_games_played': 10,
                'overall_points': 18,
                'overall_xg_for': 15.0,
                'overall_xg_against': 8.0
            }
        ])

        player_aggregates_df = pd.DataFrame([
            {
                'team': 'Team A',
                'league': 'epl',
                'keeper_save_pct': 72.0
            },
            {
                'team': 'Team B',
                'league': 'epl',
                'keeper_save_pct': 78.0
            }
        ])

        result = self.validator.validate_league_data(
            fixtures_df, team_stats_df, player_aggregates_df, 'epl'
        )

        assert result['is_valid']
        assert len(result['errors']) == 0
        assert result['league'] == 'epl'
        assert result['stats']['num_fixtures'] == 1
        assert result['stats']['num_teams'] == 2

    def test_invalid_fixture_data(self):
        """Test validation with invalid fixture data"""
        # Missing required columns
        bad_fixtures_df = pd.DataFrame([
            {
                'home_team': 'Team A',
                # Missing fixture_id, away_team, date
                'home_xg': 1.5
            }
        ])

        team_stats_df = pd.DataFrame([
            {'team': 'Team A', 'league': 'epl'}
        ])

        player_aggregates_df = pd.DataFrame([
            {'team': 'Team A', 'league': 'epl'}
        ])

        result = self.validator.validate_league_data(
            bad_fixtures_df, team_stats_df, player_aggregates_df, 'epl'
        )

        assert not result['is_valid']
        assert len(result['errors']) > 0
        assert any('Missing required fixture columns' in error for error in result['errors'])

    def test_unrealistic_values_warning(self):
        """Test warnings for unrealistic values"""
        fixtures_df = pd.DataFrame([
            {
                'fixture_id': 'test_1',
                'home_team': 'Team A',
                'away_team': 'Team B',
                'date': '2023-08-12',
                'home_xg': 15.0,  # Unrealistically high
                'away_xg': -1.0  # Impossible negative
            }
        ])

        team_stats_df = pd.DataFrame([
            {'team': 'Team A', 'league': 'epl'},
            {'team': 'Team B', 'league': 'epl'}
        ])

        player_aggregates_df = pd.DataFrame()

        result = self.validator.validate_league_data(
            fixtures_df, team_stats_df, player_aggregates_df, 'epl'
        )

        # Should still be valid (warnings, not errors)
        assert result['is_valid']
        assert len(result['warnings']) > 0
        assert any('unrealistic' in warning.lower() for warning in result['warnings'])

    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        empty_df = pd.DataFrame()

        result = self.validator.validate_league_data(
            empty_df, empty_df, empty_df, 'epl'
        )

        assert not result['is_valid']
        assert 'No fixture data found' in result['errors']
        assert 'No team stats data found' in result['errors']