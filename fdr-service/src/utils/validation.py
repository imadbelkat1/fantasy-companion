import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data quality and integrity"""

    def validate_league_data(self,
                             fixtures_df: pd.DataFrame,
                             team_stats_df: pd.DataFrame,
                             player_aggregates_df: pd.DataFrame,
                             league: str) -> Dict[str, Any]:
        """Validate data for a league"""

        errors = []
        warnings = []

        # Validate fixtures
        fixture_validation = self._validate_fixtures(fixtures_df)
        errors.extend(fixture_validation['errors'])
        warnings.extend(fixture_validation['warnings'])

        # Validate team stats
        team_validation = self._validate_team_stats(team_stats_df)
        errors.extend(team_validation['errors'])
        warnings.extend(team_validation['warnings'])

        # Validate player aggregates
        player_validation = self._validate_player_aggregates(player_aggregates_df)
        errors.extend(player_validation['errors'])
        warnings.extend(player_validation['warnings'])

        # Cross-validation between datasets
        cross_validation = self._cross_validate_data(fixtures_df, team_stats_df, player_aggregates_df)
        errors.extend(cross_validation['errors'])
        warnings.extend(cross_validation['warnings'])

        is_valid = len(errors) == 0

        result = {
            'league': league,
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'stats': {
                'num_fixtures': len(fixtures_df),
                'num_teams': len(team_stats_df),
                'num_player_aggregates': len(player_aggregates_df),
                'fixtures_with_scores': len(
                    fixtures_df.dropna(subset=['home_goals', 'away_goals'])) if not fixtures_df.empty else 0
            }
        }

        if warnings:
            logger.warning(f"{league} validation warnings: {warnings}")

        if errors:
            logger.error(f"{league} validation errors: {errors}")
        else:
            logger.info(f"{league} data validation passed")

        return result

    def _validate_fixtures(self, fixtures_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate fixture data"""
        errors = []
        warnings = []

        if fixtures_df.empty:
            errors.append("No fixture data found")
            return {'errors': errors, 'warnings': warnings}

        # Required columns
        required_cols = ['fixture_id', 'home_team', 'away_team', 'date']
        missing_cols = [col for col in required_cols if col not in fixtures_df.columns]
        if missing_cols:
            errors.append(f"Missing required fixture columns: {missing_cols}")

        # Check for duplicate fixtures
        if 'fixture_id' in fixtures_df.columns:
            duplicates = fixtures_df['fixture_id'].duplicated().sum()
            if duplicates > 0:
                warnings.append(f"Found {duplicates} duplicate fixtures")

        # Check for missing xG data
        if 'home_xg' in fixtures_df.columns and 'away_xg' in fixtures_df.columns:
            missing_xg = fixtures_df[['home_xg', 'away_xg']].isnull().all(axis=1).sum()
            if missing_xg > 0:
                warnings.append(f"{missing_xg} fixtures missing xG data")

        # Check for unrealistic xG values
        for col in ['home_xg', 'away_xg']:
            if col in fixtures_df.columns:
                unrealistic = ((fixtures_df[col] < 0) | (fixtures_df[col] > 10)).sum()
                if unrealistic > 0:
                    warnings.append(f"{unrealistic} fixtures have unrealistic {col} values")

        return {'errors': errors, 'warnings': warnings}

    def _validate_team_stats(self, team_stats_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate team statistics"""
        errors = []
        warnings = []

        if team_stats_df.empty:
            errors.append("No team stats data found")
            return {'errors': errors, 'warnings': warnings}

        # Required columns
        required_cols = ['team', 'league']
        missing_cols = [col for col in required_cols if col not in team_stats_df.columns]
        if missing_cols:
            errors.append(f"Missing required team stats columns: {missing_cols}")

        # Check for duplicate teams
        if 'team' in team_stats_df.columns:
            duplicates = team_stats_df['team'].duplicated().sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate teams")

        # Check for unrealistic values
        numeric_cols = team_stats_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if 'xg' in col.lower():
                # xG values should be reasonable
                unrealistic = ((team_stats_df[col] < 0) | (team_stats_df[col] > 200)).sum()
                if unrealistic > 0:
                    warnings.append(f"{unrealistic} teams have unrealistic {col} values")

            elif 'points' in col.lower():
                # Points should be reasonable for a season
                unrealistic = ((team_stats_df[col] < 0) | (
                            team_stats_df[col] > 114)).sum()  # Max possible in 38-game season
                if unrealistic > 0:
                    warnings.append(f"{unrealistic} teams have unrealistic {col} values")

        return {'errors': errors, 'warnings': warnings}

    def _validate_player_aggregates(self, player_aggregates_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate player aggregate data"""
        errors = []
        warnings = []

        if player_aggregates_df.empty:
            warnings.append("No player aggregate data found")
            return {'errors': errors, 'warnings': warnings}

        # Required columns
        required_cols = ['team', 'league']
        missing_cols = [col for col in required_cols if col not in player_aggregates_df.columns]
        if missing_cols:
            errors.append(f"Missing required player aggregate columns: {missing_cols}")

        # Check for unrealistic values
        if 'keeper_save_pct' in player_aggregates_df.columns:
            unrealistic_saves = ((player_aggregates_df['keeper_save_pct'] < 0) |
                                 (player_aggregates_df['keeper_save_pct'] > 100)).sum()
            if unrealistic_saves > 0:
                warnings.append(f"{unrealistic_saves} teams have unrealistic keeper save percentages")

        return {'errors': errors, 'warnings': warnings}

    def _cross_validate_data(self, fixtures_df: pd.DataFrame, team_stats_df: pd.DataFrame,
                             player_aggregates_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Cross-validate consistency between datasets"""
        errors = []
        warnings = []

        if fixtures_df.empty or team_stats_df.empty:
            return {'errors': errors, 'warnings': warnings}

        # Check team consistency between fixtures and team stats
        if 'home_team' in fixtures_df.columns and 'away_team' in fixtures_df.columns and 'team' in team_stats_df.columns:
            fixture_teams = set(fixtures_df['home_team'].unique()) | set(fixtures_df['away_team'].unique())
            stats_teams = set(team_stats_df['team'].unique())

            missing_from_stats = fixture_teams - stats_teams
            if missing_from_stats:
                warnings.append(f"Teams in fixtures but not in stats: {list(missing_from_stats)[:5]}")  # Show first 5

            missing_from_fixtures = stats_teams - fixture_teams
            if missing_from_fixtures:
                warnings.append(f"Teams in stats but not in fixtures: {list(missing_from_fixtures)[:5]}")

        # Check team consistency between stats and player aggregates
        if not player_aggregates_df.empty and 'team' in team_stats_df.columns and 'team' in player_aggregates_df.columns:
            stats_teams = set(team_stats_df['team'].unique())
            player_teams = set(player_aggregates_df['team'].unique())

            missing_player_data = stats_teams - player_teams
            if missing_player_data:
                warnings.append(f"Teams missing player data: {len(missing_player_data)} teams")

        return {'errors': errors, 'warnings': warnings}