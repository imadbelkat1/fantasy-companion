# fdr-service/src/models/fixture_predictor.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from .team_rater import TeamRater
import logging

logger = logging.getLogger(__name__)


class FixturePredictor:
    """Predict fixture difficulty and outcomes using trained team rating model"""

    def __init__(self, team_rater: TeamRater):
        self.team_rater = team_rater
        self.league = team_rater.league

    def predict_fixture_difficulties(self, fixtures_df: pd.DataFrame, match_features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict fixture difficulties for all fixtures"""

        if fixtures_df.empty or match_features_df.empty:
            logger.warning(f"Empty fixture data for {self.league}")
            return pd.DataFrame()

        predictions = []

        for _, fixture in fixtures_df.iterrows():
            fixture_id = fixture['fixture_id']

            # Find corresponding match features
            match_features = match_features_df[match_features_df['fixture_id'] == fixture_id]

            if match_features.empty:
                logger.warning(f"No features found for fixture {fixture_id}")
                continue

            match_features = match_features.iloc[0]

            # Prepare feature row for prediction (remove non-feature columns)
            feature_cols = [col for col in match_features.index
                            if col not in ['fixture_id', 'league', 'date', 'home_team', 'away_team']
                            and not col.startswith('target_')]

            X_fixture = match_features[feature_cols].values.reshape(1, -1)
            X_fixture_df = pd.DataFrame(X_fixture, columns=feature_cols)

            # Get predictions
            goal_diff_pred = self.team_rater.predict(X_fixture_df)[0]
            outcome_probs = self.team_rater.predict_outcome_probabilities(X_fixture_df)[0]

            # Convert to fixture difficulty (1-5 scale)
            home_difficulty = self._convert_to_difficulty_score(goal_diff_pred, outcome_probs, perspective='home')
            away_difficulty = self._convert_to_difficulty_score(-goal_diff_pred, outcome_probs[::-1],
                                                                perspective='away')

            prediction = {
                'fixture_id': fixture_id,
                'league': self.league,
                'date': fixture.get('date'),
                'gameweek': fixture.get('gameweek'),
                'home_team': fixture.get('home_team'),
                'away_team': fixture.get('away_team'),
                'predicted_goal_diff_home': goal_diff_pred,
                'win_prob_home': outcome_probs[0] if len(outcome_probs) > 0 else 0.33,
                'draw_prob': outcome_probs[1] if len(outcome_probs) > 1 else 0.33,
                'win_prob_away': outcome_probs[2] if len(outcome_probs) > 2 else 0.33,
                'fixture_difficulty_home': home_difficulty,
                'fixture_difficulty_away': away_difficulty,
                'home_xg_predicted': max(0, 1.3 + goal_diff_pred * 0.5),  # Heuristic conversion
                'away_xg_predicted': max(0, 1.3 - goal_diff_pred * 0.5)
            }

            predictions.append(prediction)

        predictions_df = pd.DataFrame(predictions)

        # Add explanations
        if not predictions_df.empty:
            predictions_df = self._add_fixture_explanations(predictions_df, match_features_df)

        logger.info(f"Predicted difficulties for {len(predictions_df)} {self.league} fixtures")

        return predictions_df

    def _convert_to_difficulty_score(self, goal_diff: float, outcome_probs: np.ndarray, perspective: str) -> float:
        """Convert goal difference and probabilities to 1-5 difficulty score"""

        # For home team: positive goal_diff = easier, negative = harder
        # For away team: it's the opposite

        win_prob = outcome_probs[0] if len(outcome_probs) > 0 else 0.33

        if perspective == 'home':
            # Home team perspective
            if win_prob > 0.7:
                difficulty = 1.0 + (1.0 - win_prob) * 2  # Very likely to win = easy (1-1.6)
            elif win_prob > 0.5:
                difficulty = 2.0 + (0.7 - win_prob) * 5  # Likely to win = moderate-easy (2-3)
            elif win_prob > 0.3:
                difficulty = 3.0 + (0.5 - win_prob) * 5  # Even match = moderate (3-4)
            else:
                difficulty = 4.0 + (0.3 - win_prob) * 3.33  # Unlikely to win = hard (4-5)
        else:
            # Away team perspective (use away win probability)
            away_win_prob = outcome_probs[2] if len(outcome_probs) > 2 else 0.33

            if away_win_prob > 0.7:
                difficulty = 1.0 + (1.0 - away_win_prob) * 2
            elif away_win_prob > 0.5:
                difficulty = 2.0 + (0.7 - away_win_prob) * 5
            elif away_win_prob > 0.3:
                difficulty = 3.0 + (0.5 - away_win_prob) * 5
            else:
                difficulty = 4.0 + (0.3 - away_win_prob) * 3.33

        return np.clip(difficulty, 1.0, 5.0)

    def _add_fixture_explanations(self, predictions_df: pd.DataFrame, match_features_df: pd.DataFrame) -> pd.DataFrame:
        """Add explanations for fixture predictions [Inference]"""

        feature_importance = self.team_rater.get_feature_importance()
        top_features = list(feature_importance.keys())[:10]  # Top 10 features

        explanations = []

        for _, prediction in predictions_df.iterrows():
            fixture_id = prediction['fixture_id']

            # Get match features for this fixture
            match_features = match_features_df[match_features_df['fixture_id'] == fixture_id]

            if match_features.empty:
                explanations.append([])
                continue

            match_features = match_features.iloc[0]

            # Create explanations based on top features [Inference]
            fixture_explanations = []

            for feature in top_features[:3]:  # Top 3 explanations
                if feature in match_features.index:
                    feature_value = match_features[feature]
                    importance = feature_importance[feature]

                    # Simple impact calculation [Inference]
                    impact = importance * abs(feature_value) / 100  # Normalized impact

                    explanation = {
                        'feature': feature,
                        'impact': float(impact),
                        'feature_value': float(feature_value),
                        'description': self._get_feature_description(feature, feature_value)
                    }

                    fixture_explanations.append(explanation)

            explanations.append(fixture_explanations)

        predictions_df['explainers'] = explanations

        return predictions_df

    def _get_feature_description(self, feature: str, value: float) -> str:
        """Generate human-readable feature descriptions [Inference]"""

        descriptions = {
            'net_xg_differential': f"Net expected goals advantage: {value:.2f}",
            'home_attack_vs_away_defense': f"Home attack vs away defense strength ratio: {value:.2f}",
            'away_attack_vs_home_defense': f"Away attack vs home defense strength ratio: {value:.2f}",
            'form_differential': f"Recent form difference: {value:.2f}",
            'points_per_game_differential': f"Points per game difference: {value:.2f}",
            'goalkeeper_quality_diff': f"Goalkeeper quality difference: {value:.1f}%",
            'top_player_quality_diff': f"Star player quality difference: {value:.2f}",
            'home_net_xg_per_game': f"Home team expected goal difference per game: {value:.2f}",
            'away_net_xg_per_game': f"Away team expected goal difference per game: {value:.2f}",
            'home_last_5_form': f"Home team recent form: {value:.2f}/3.0",
            'away_last_5_form': f"Away team recent form: {value:.2f}/3.0",
            'is_home': "Home advantage factor" if value > 0 else "No home advantage"
        }

        return descriptions.get(feature, f"{feature}: {value:.2f}")
