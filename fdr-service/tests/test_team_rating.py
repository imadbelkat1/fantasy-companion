import pytest
import pandas as pd
import numpy as np

from src.models.team_rater import TeamRater
from src.core.config import Config


class TestTeamRater:
    """Test team rating model"""

    def setup_method(self):
        """Setup test model"""
        config = Config()
        self.model = TeamRater('test_league', config.MODEL_CONFIG)

        # Create dummy training data
        self.X_train = pd.DataFrame({
            'home_net_xg_per_game': [0.5, -0.2, 0.8, 0.1],
            'away_net_xg_per_game': [-0.3, 0.4, -0.5, 0.2],
            'form_differential': [0.5, -0.5, 1.0, 0.0],
            'is_home': [1, 1, 1, 1]
        })

        self.y_train = pd.DataFrame({
            'target_goal_diff': [1.0, -0.5, 2.0, 0.5],
            'target_home_win': [1, 0, 1, 1],
            'target_draw': [0, 0, 0, 0],
            'target_away_win': [0, 1, 0, 0]
        })

    def test_model_fitting(self):
        """Test model can be fitted"""
        self.model.fit(self.X_train, self.y_train)

        assert self.model.is_fitted
        assert self.model.goal_diff_model is not None
        assert self.model.outcome_classifier is not None

    def test_predictions(self):
        """Test model predictions"""
        self.model.fit(self.X_train, self.y_train)

        # Test goal difference prediction
        goal_diff_pred = self.model.predict(self.X_train)
        assert len(goal_diff_pred) == len(self.X_train)
        assert isinstance(goal_diff_pred, np.ndarray)

        # Test outcome probabilities
        outcome_probs = self.model.predict_outcome_probabilities(self.X_train)
        assert outcome_probs.shape == (len(self.X_train), 3)

        # Probabilities should sum to ~1
        prob_sums = np.sum(outcome_probs, axis=1)
        assert np.allclose(prob_sums, 1.0, atol=0.1)

    def test_feature_importance(self):
        """Test feature importance extraction"""
        self.model.fit(self.X_train, self.y_train)

        importance = self.model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) > 0

        # Check all features are included
        for feature in self.X_train.columns:
            assert feature in importance

    def test_convert_to_team_ratings(self):
        """Test conversion to team ratings"""
        self.model.fit(self.X_train, self.y_train)

        # Create team features (similar to X_train but with team names)
        X_teams = self.X_train.copy()
        X_teams['team'] = ['Team A', 'Team B', 'Team C', 'Team D']

        ratings_df = self.model.convert_to_team_ratings(X_teams)

        assert not ratings_df.empty
        assert len(ratings_df) == len(X_teams)

        # Check required columns
        required_cols = ['team', 'league', 'rating', 'attack_rating', 'defense_rating', 'form_rating']
        for col in required_cols:
            assert col in ratings_df.columns

        # Check rating bounds
        assert (ratings_df['rating'] >= 0).all()
        assert (ratings_df['rating'] <= 100).all()

    def test_empty_data_handling(self):
        """Test handling of empty training data"""
        empty_X = pd.DataFrame()
        empty_y = pd.DataFrame()

        # Should not raise error
        self.model.fit(empty_X, empty_y)

        # Model should not be fitted
        assert not self.model.is_fitted