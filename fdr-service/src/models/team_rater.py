import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from .base import BaseModel
import logging

logger = logging.getLogger(__name__)


class TeamRater(BaseModel):
    """Team rating model using regularized regression"""

    def __init__(self, league: str, config: Dict[str, Any]):
        super().__init__(league, config)
        self.goal_diff_model = None
        self.outcome_classifier = None
        self.rating_scaler_params = {}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Fit team rating models"""

        if X.empty or y.empty:
            logger.warning(f"Empty training data for {self.league}")
            return

        self.feature_names = X.columns.tolist()

        # Fit goal difference regression model
        if 'target_goal_diff' in y.columns:
            logger.info(f"Training {self.league} goal difference model...")

            # Choose model based on data size
            if len(X) < 100:
                # Small dataset: use Ridge regression
                self.goal_diff_model = Ridge(
                    alpha=self.config.get('ridge_alpha', 1.0),
                    random_state=42
                )
            else:
                # Larger dataset: use LightGBM
                lgb_params = self.config.get('lgb_params', {})
                lgb_params.update({
                    'random_state': 42,
                    'n_estimators': min(100, len(X) // 2)  # Conservative for small data
                })
                self.goal_diff_model = lgb.LGBMRegressor(**lgb_params)

            self.goal_diff_model.fit(X, y['target_goal_diff'])

        # Fit outcome classification model
        outcome_cols = ['target_home_win', 'target_draw', 'target_away_win']
        available_outcome_cols = [col for col in outcome_cols if col in y.columns]

        if available_outcome_cols:
            logger.info(f"Training {self.league} outcome classification model...")

            if len(available_outcome_cols) == 3:
                # Multi-class classification
                y_outcome = np.argmax(y[available_outcome_cols].values, axis=1)

                if len(X) < 100:
                    self.outcome_classifier = LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        random_state=42,
                        multi_class='ovr'
                    )
                else:
                    lgb_params = self.config.get('lgb_params', {}).copy()
                    lgb_params.update({
                        'objective': 'multiclass',
                        'num_class': 3,
                        'metric': 'multi_logloss',
                        'random_state': 42,
                        'n_estimators': min(100, len(X) // 2)
                    })
                    self.outcome_classifier = lgb.LGBMClassifier(**lgb_params)

                self.outcome_classifier.fit(X, y_outcome)

            elif 'target_home_win' in available_outcome_cols:
                # Binary classification (home win vs not)
                if len(X) < 100:
                    self.outcome_classifier = LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        random_state=42
                    )
                else:
                    lgb_params = self.config.get('lgb_params', {}).copy()
                    lgb_params.update({
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        'random_state': 42,
                        'n_estimators': min(100, len(X) // 2)
                    })
                    self.outcome_classifier = lgb.LGBMClassifier(**lgb_params)

                self.outcome_classifier.fit(X, y['target_home_win'])

        self.is_fitted = True
        logger.info(f"Successfully fitted {self.league} team rating models")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict goal difference (primary prediction)"""
        if not self.is_fitted or self.goal_diff_model is None:
            logger.warning("Goal difference model not fitted")
            return np.zeros(len(X))

        return self.goal_diff_model.predict(X)

    def predict_outcome_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match outcome probabilities"""
        if not self.is_fitted or self.outcome_classifier is None:
            logger.warning("Outcome classifier not fitted")
            # Return neutral probabilities
            return np.full((len(X), 3), 1 / 3)

        if hasattr(self.outcome_classifier, 'predict_proba'):
            probs = self.outcome_classifier.predict_proba(X)

            # Ensure 3 classes (home_win, draw, away_win)
            if probs.shape[1] == 2:
                # Binary classifier: expand to 3 classes
                home_win_prob = probs[:, 1]
                not_home_win_prob = probs[:, 0]

                # Assume equal split between draw and away win for not_home_win
                draw_prob = not_home_win_prob * 0.3  # Draws less likely
                away_win_prob = not_home_win_prob * 0.7

                probs = np.column_stack([home_win_prob, draw_prob, away_win_prob])

            return probs
        else:
            return np.full((len(X), 3), 1 / 3)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the goal difference model"""
        if not self.is_fitted or self.goal_diff_model is None:
            return {}

        importance_dict = {}

        if hasattr(self.goal_diff_model, 'feature_importances_'):
            # Tree-based models (LightGBM, RandomForest)
            importance_scores = self.goal_diff_model.feature_importances_
        elif hasattr(self.goal_diff_model, 'coef_'):
            # Linear models (Ridge, Lasso)
            importance_scores = np.abs(self.goal_diff_model.coef_)
        else:
            logger.warning("Cannot extract feature importance")
            return {}

        for feature, importance in zip(self.feature_names, importance_scores):
            importance_dict[feature] = float(importance)

        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return importance_dict

    def convert_to_team_ratings(self, X_teams: pd.DataFrame) -> pd.DataFrame:
        """Convert model predictions to 0-100 team ratings"""

        if not self.is_fitted:
            logger.warning("Model not fitted yet")
            return pd.DataFrame()

        # Store team names and leagues before dropping them
        team_info = X_teams[
            ['team', 'league']].copy() if 'team' in X_teams.columns and 'league' in X_teams.columns else pd.DataFrame()

        # Create feature-only DataFrame for predictions (use exact same features as training)
        X_features = X_teams.copy()

        # Remove non-feature columns
        non_feature_cols = ['team', 'league']
        X_features = X_features.drop(columns=[col for col in non_feature_cols if col in X_features.columns])

        # Ensure we have exactly the same features as training
        if hasattr(self, 'feature_names') and self.feature_names:
            # Reorder/filter columns to match training exactly
            missing_features = set(self.feature_names) - set(X_features.columns)
            if missing_features:
                logger.warning(f"Missing features for prediction: {missing_features}")
                for feature in missing_features:
                    X_features[feature] = 0.0

            # Keep only the training features in the same order
            X_features = X_features[self.feature_names]

        logger.info(f"Predicting with {X_features.shape[1]} features for {len(X_features)} teams")

        # Get predictions for each team vs league average
        goal_diff_preds = self.predict(X_features)
        outcome_probs = self.predict_outcome_probabilities(X_features)

        # Convert to z-scores for rating normalization
        pred_mean = np.mean(goal_diff_preds)
        pred_std = max(np.std(goal_diff_preds), 0.01)  # Avoid division by zero
        z_scores = (goal_diff_preds - pred_mean) / pred_std

        # Map z-scores to 0-100 ratings
        rating_config = self.config.get('rating_config', {})
        base_rating = rating_config.get('base_rating', 50)
        scale_factor = rating_config.get('scale_factor', 12)

        raw_ratings = base_rating + scale_factor * z_scores
        ratings = np.clip(raw_ratings,
                          rating_config.get('min_rating', 0),
                          rating_config.get('max_rating', 100))

        # Store scaling parameters
        self.rating_scaler_params = {
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'base_rating': base_rating,
            'scale_factor': scale_factor
        }

        # Create results DataFrame
        results = []

        for idx, (_, row) in enumerate(X_teams.iterrows()):
            team_name = row.get('team', f'Team_{idx}')
            league_name = row.get('league', self.league)

            # Sub-ratings based on specific features
            attack_rating = self._calculate_attack_rating(row, ratings[idx])
            defense_rating = self._calculate_defense_rating(row, ratings[idx])
            form_rating = self._calculate_form_rating(row)

            # Apply shrinkage toward prior season if available
            prior_anchor = row.get('home_prior_points_per_game', 1.0) * 100 / 3  # Convert to 0-100 scale
            shrinkage_lambda = rating_config.get('shrinkage_lambda', 0.3)
            final_rating = (1 - shrinkage_lambda) * ratings[idx] + shrinkage_lambda * prior_anchor

            result = {
                'team': team_name,
                'league': league_name,
                'rating': final_rating,
                'attack_rating': attack_rating,
                'defense_rating': defense_rating,
                'form_rating': form_rating,
                'raw_model_rating': ratings[idx],
                'prior_anchor': prior_anchor,
                'expected_goal_diff_vs_avg': goal_diff_preds[idx],
                'win_prob_vs_avg': outcome_probs[idx][0] if len(outcome_probs[idx]) > 0 else 0.33
            }

            results.append(result)

        return pd.DataFrame(results)

    def _calculate_attack_rating(self, team_features: pd.Series, overall_rating: float) -> float:
        """Calculate attack-specific rating"""
        # Use attacking features to adjust from overall rating
        attack_features = [col for col in team_features.index if
                           'xg_for' in col or 'goals_per90' in col or 'npxg' in col]

        if attack_features:
            attack_values = [team_features.get(feat, 0) for feat in attack_features]
            attack_score = np.mean(attack_values) if attack_values else 1.0

            # Adjust relative to league average (assuming 1.0 is average)
            attack_multiplier = min(max(attack_score / 1.0, 0.7), 1.3)
            return np.clip(overall_rating * attack_multiplier, 0, 100)

        return overall_rating

    def _calculate_defense_rating(self, team_features: pd.Series, overall_rating: float) -> float:
        """Calculate defense-specific rating (lower xG against = higher rating)"""
        defense_features = [col for col in team_features.index if 'xg_against' in col or 'save_pct' in col]

        if defense_features:
            # For defensive metrics, lower is often better (except save%)
            xg_against_features = [col for col in defense_features if 'xg_against' in col]
            save_pct_features = [col for col in defense_features if 'save_pct' in col]

            defense_score = 1.0
            if xg_against_features:
                xg_against_avg = np.mean([team_features.get(feat, 1.0) for feat in xg_against_features])
                defense_score *= (1.0 / max(xg_against_avg, 0.1))  # Invert: lower xGA = better

            if save_pct_features:
                save_pct_avg = np.mean([team_features.get(feat, 70.0) for feat in save_pct_features])
                defense_score *= (save_pct_avg / 70.0)  # Higher save% = better

            defense_multiplier = min(max(defense_score, 0.7), 1.3)
            return np.clip(overall_rating * defense_multiplier, 0, 100)

        return overall_rating

    def _calculate_form_rating(self, team_features: pd.Series) -> float:
        """Calculate form-specific rating"""
        form_features = [col for col in team_features.index if 'last_5' in col or 'form' in col]

        if form_features:
            form_values = [team_features.get(feat, 1.5) for feat in form_features]
            form_score = np.mean(form_values)  # 1.5 is neutral, 3.0 is excellent, 0 is terrible

            # Convert to 0-100 scale
            return np.clip((form_score / 3.0) * 100, 0, 100)

        return 50.0  # Neutral form

    def estimate_uncertainty(self, X_teams: pd.DataFrame, n_bootstrap: int = 100) -> pd.DataFrame:
        """Estimate rating uncertainty using bootstrap [Inference]"""

        if not self.is_fitted:
            logger.warning("Model not fitted for uncertainty estimation")
            return pd.DataFrame()

        base_ratings = self.convert_to_team_ratings(X_teams)

        # Simple uncertainty based on data size and model confidence
        # [Inference] This is a heuristic approximation, not guaranteed bounds
        n_samples = len(X_teams)
        base_uncertainty = max(5.0, 50.0 / np.sqrt(max(n_samples, 1)))  # Larger uncertainty for smaller datasets

        uncertainty_results = []

        for _, row in base_ratings.iterrows():
            rating = row['rating']

            # Add model-specific uncertainty factors [Inference]
            model_uncertainty = base_uncertainty

            # Higher uncertainty for extreme ratings [Inference]
            if rating < 20 or rating > 80:
                model_uncertainty *= 1.5

            # Prior season confidence factor [Inference]
            games_played_estimate = 10  # Default assumption if not available
            confidence_factor = min(np.sqrt(games_played_estimate / 10), 1.0)
            adjusted_uncertainty = model_uncertainty / confidence_factor

            uncertainty_results.append({
                'team': row['team'],
                'rating_low': max(0, rating - adjusted_uncertainty),
                'rating_high': min(100, rating + adjusted_uncertainty),
                'uncertainty': adjusted_uncertainty
            })

        uncertainty_df = pd.DataFrame(uncertainty_results)

        # Merge with base ratings
        final_results = base_ratings.merge(uncertainty_df, on='team', how='left')

        return final_results