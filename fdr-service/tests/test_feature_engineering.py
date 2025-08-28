import pytest
import pandas as pd
import numpy as np

from src.data.preprocessor import FeatureEngine


class TestFeatureEngine:
    """Test feature engineering functionality"""

    def setup_method(self):
        """Setup test data"""
        self.feature_engine = FeatureEngine()

        self.fixtures_df = pd.DataFrame([
            {
                'fixture_id': 'test_1',
                'league': 'epl',
                'date': '2023-08-12',
                'gameweek': 1,
                'home_team': 'Team A',
                'away_team': 'Team B',
                'home_xg': 2.0,
                'away_xg': 1.0,
                'home_goals': 2,
                'away_goals': 1
            }
        ])

        self.team_stats_df = pd.DataFrame([
            {
                'team': 'Team A',
                'league': 'epl',
                'overall_games_played': 10,
                'overall_xg_for': 15.0,
                'overall_xg_against': 8.0,
                'overall_points': 20,
                'last_5_encoded': 2.0,
                'xg_for_per_game': 1.5,
                'xg_against_per_game': 0.8,
                'net_xg_per_game': 0.7
            },
            {
                'team': 'Team B',
                'league': 'epl',
                'overall_games_played': 10,
                'overall_xg_for': 12.0,
                'overall_xg_against': 12.0,
                'overall_points': 15,
                'last_5_encoded': 1.5,
                'xg_for_per_game': 1.2,
                'xg_against_per_game': 1.2,
                'net_xg_per_game': 0.0
            }
        ])

        self.player_aggregates_df = pd.DataFrame([
            {
                'team': 'Team A',
                'league': 'epl',
                'sum_goals': 15,
                'sum_assists': 8,
                'keeper_save_pct': 75.0,
                'top_player_xgxa_per90': 1.2
            },
            {
                'team': 'Team B',
                'league': 'epl',
                'sum_goals': 12,
                'sum_assists': 6,
                'keeper_save_pct': 70.0,
                'top_player_xgxa_per90': 1.0
            }
        ])

    def test_create_match_features(self):
        """Test match feature creation"""
        match_features = self.feature_engine.create_match_features(
            self.fixtures_df, self.team_stats_df, self.player_aggregates_df
        )

        assert not match_features.empty
        assert len(match_features) == 1

        # Check required columns exist
        required_cols = ['fixture_id', 'home_team', 'away_team', 'is_home']
        for col in required_cols:
            assert col in match_features.columns

        # Check relative features
        assert 'net_xg_differential' in match_features.columns
        assert 'home_attack_vs_away_defense' in match_features.columns

        # Check target variables
        assert 'target_goal_diff' in match_features.columns
        assert match_features.iloc[0]['target_goal_diff'] == 1  # 2-1 = 1

    def test_prepare_training_data(self):
        """Test training data preparation"""
        match_features = self.feature_engine.create_match_features(
            self.fixtures_df, self.team_stats_df, self.player_aggregates_df
        )

        X_train, y_train = self.feature_engine.prepare_training_data(match_features, 'epl')

        assert not X_train.empty
        assert not y_train.empty
        assert len(X_train) == len(y_train)

        # Check that features are scaled (should have mean ~0, std ~1 for single sample)
        # Note: With single sample, scaling will result in zeros
        assert X_train.shape[1] > 0  # Has features

        # Check target columns
        target_cols = ['target_goal_diff', 'target_home_win', 'target_draw', 'target_away_win']
        for col in target_cols:
            if col in y_train.columns:
                assert not y_train[col].isnull().all()

    def test_get_relative_features(self):
        """Test relative feature calculation"""
        home_stats = self.team_stats_df[self.team_stats_df['team'] == 'Team A'].iloc[0]
        away_stats = self.team_stats_df[self.team_stats_df['team'] == 'Team B'].iloc[0]
        home_players = self.player_aggregates_df[self.player_aggregates_df['team'] == 'Team A'].iloc[0]
        away_players = self.player_aggregates_df[self.player_aggregates_df['team'] == 'Team B'].iloc[0]

        relative_features = self.feature_engine._get_relative_features(
            home_stats, away_stats, home_players, away_players
        )

        # Check key relative features exist
        assert 'home_attack_vs_away_defense' in relative_features
        assert 'net_xg_differential' in relative_features
        assert 'form_differential' in relative_features

        # Check calculations make sense
        assert relative_features['net_xg_differential'] == 0.7 - 0.0  # Team A - Team B
        assert relative_features['form_differential'] == 2.0 - 1.5  # Team A - Team B