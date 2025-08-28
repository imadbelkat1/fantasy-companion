import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ExplainerEngine:
    """Generate explanations for model predictions"""

    def add_team_explanations(self, team_ratings_df: pd.DataFrame, model,
                              team_features_df: pd.DataFrame) -> pd.DataFrame:
        """Add explanations to team ratings [Inference]"""

        if team_ratings_df.empty:
            return team_ratings_df

        # Get feature importance from model
        feature_importance = model.get_feature_importance()

        explanations = []

        for _, team_rating in team_ratings_df.iterrows():
            team_name = team_rating['team']

            # Get team features
            team_features = team_features_df[team_features_df['team'] == team_name]

            if team_features.empty:
                explanations.append([])
                continue

            team_features = team_features.iloc[0]

            # Create explanations based on top features [Inference]
            team_explanations = self._create_team_explanations(team_features, feature_importance, team_rating)

            explanations.append(team_explanations)

        team_ratings_df['explainers'] = explanations
        team_ratings_df['confidence_level'] = self._calculate_confidence_level(team_ratings_df)
        team_ratings_df['plain_language_summary'] = self._create_plain_language_summary(team_ratings_df)

        return team_ratings_df

    def _create_team_explanations(self, team_features: pd.Series, feature_importance: Dict[str, float],
                                  team_rating: pd.Series) -> List[Dict[str, Any]]:
        """Create top 3 explanations for a team's rating [Inference]"""

        explanations = []
        top_features = list(feature_importance.keys())[:10]  # Top 10 features to consider

        feature_impacts = []

        for feature in top_features:
            if feature in team_features.index:
                feature_value = team_features[feature]
                importance = feature_importance[feature]

                # Calculate impact based on feature value and importance [Inference]
                # This is a heuristic approximation
                normalized_value = self._normalize_feature_value(feature, feature_value)
                impact = importance * abs(normalized_value)

                feature_impacts.append({
                    'feature': feature,
                    'impact': impact,
                    'feature_value': feature_value,
                    'normalized_value': normalized_value,
                    'description': self._get_feature_explanation(feature, feature_value, normalized_value)
                })

        # Sort by impact and take top 3
        feature_impacts.sort(key=lambda x: x['impact'], reverse=True)

        for impact_info in feature_impacts[:3]:
            explanations.append({
                'feature': impact_info['feature'],
                'impact': float(impact_info['impact']),
                'description': impact_info['description']
            })

        return explanations

    def _normalize_feature_value(self, feature: str, value: float) -> float:
        """Normalize feature value to understand its impact [Inference]"""

        # Heuristic normalization based on feature type [Inference]
        if 'xg' in feature.lower():
            return (value - 1.0) / max(abs(value - 1.0), 0.1)  # 1.0 as baseline
        elif 'per_game' in feature.lower():
            if 'points' in feature.lower():
                return (value - 1.0) / max(abs(value - 1.0), 0.1)  # 1.0 points per game as baseline
            else:
                return (value - 1.0) / max(abs(value - 1.0), 0.1)
        elif 'form' in feature.lower():
            return (value - 1.5) / 1.5  # 1.5 as neutral form
        elif 'pct' in feature.lower():
            return (value - 50.0) / 50.0  # 50% as baseline for percentages
        elif 'differential' in feature.lower():
            return value / max(abs(value), 0.1)  # Already a difference
        else:
            return value / max(abs(value), 0.1)  # Generic normalization

    def _get_feature_explanation(self, feature: str, value: float, normalized_value: float) -> str:
        """Generate human-readable explanation for a feature [Inference]"""

        # Determine if the feature is positive or negative for the team
        is_positive = normalized_value > 0.1
        is_negative = normalized_value < -0.1

        explanations = {
            'home_net_xg_per_game': {
                'positive': f"Strong expected goal difference (+{value:.2f} per game)",
                'negative': f"Poor expected goal difference ({value:.2f} per game)",
                'neutral': f"Average expected goal difference ({value:.2f} per game)"
            },
            'home_attack_vs_away_defense': {
                'positive': f"Excellent attacking strength relative to opponents ({value:.2f})",
                'negative': f"Weak attacking strength relative to opponents ({value:.2f})",
                'neutral': f"Average attacking performance vs opponents ({value:.2f})"
            },
            'home_last_5_form': {
                'positive': f"Excellent recent form ({value:.1f}/3.0 points per game)",
                'negative': f"Poor recent form ({value:.1f}/3.0 points per game)",
                'neutral': f"Average recent form ({value:.1f}/3.0 points per game)"
            },
            'home_keeper_save_pct': {
                'positive': f"Strong goalkeeping ({value:.1f}% save rate)",
                'negative': f"Weak goalkeeping ({value:.1f}% save rate)",
                'neutral': f"Average goalkeeping ({value:.1f}% save rate)"
            },
            'home_prior_points_per_game': {
                'positive': f"Strong historical performance ({value:.2f} points per game last season)",
                'negative': f"Weak historical performance ({value:.2f} points per game last season)",
                'neutral': f"Average historical performance ({value:.2f} points per game last season)"
            }
        }

        if feature in explanations:
            if is_positive:
                return explanations[feature]['positive']
            elif is_negative:
                return explanations[feature]['negative']
            else:
                return explanations[feature]['neutral']
        else:
            # Generic explanation
            clean_feature = feature.replace('home_', '').replace('away_', '').replace('_', ' ')
            if is_positive:
                return f"Above average {clean_feature}: {value:.2f}"
            elif is_negative:
                return f"Below average {clean_feature}: {value:.2f}"
            else:
                return f"Average {clean_feature}: {value:.2f}"

    def _calculate_confidence_level(self, team_ratings_df: pd.DataFrame) -> List[str]:
        """Calculate confidence level for each team rating [Inference]"""

        confidence_levels = []

        for _, team_rating in team_ratings_df.iterrows():
            uncertainty = team_rating.get('uncertainty', 10.0)

            if uncertainty < 5.0:
                confidence = "High"
            elif uncertainty < 10.0:
                confidence = "Medium"
            else:
                confidence = "Low"

            confidence_levels.append(confidence)

        return confidence_levels

    def _create_plain_language_summary(self, team_ratings_df: pd.DataFrame) -> List[str]:
        """Create plain language summaries for each team [Inference]"""

        summaries = []

        for _, team_rating in team_ratings_df.iterrows():
            team_name = team_rating['team']
            rating = team_rating['rating']
            confidence = team_rating.get('confidence_level', 'Medium')
            explainers = team_rating.get('explainers', [])

            # Rating interpretation
            if rating >= 80:
                rating_desc = "excellent"
            elif rating >= 70:
                rating_desc = "very good"
            elif rating >= 60:
                rating_desc = "good"
            elif rating >= 40:
                rating_desc = "average"
            else:
                rating_desc = "poor"

            # Top factors
            top_factors = []
            if explainers:
                for exp in explainers[:2]:  # Top 2 factors
                    factor_desc = exp.get('description', exp.get('feature', 'unknown factor'))
                    if exp.get('impact', 0) > 0.1:
                        top_factors.append(factor_desc.lower())

            factor_text = " and ".join(top_factors) if top_factors else "overall team performance"

            summary = f"{team_name} has an {rating_desc} rating of {rating:.1f}/100, driven primarily by {factor_text}. Confidence: {confidence} (based on available match data and model uncertainty)."

            summaries.append(summary)

        return summaries