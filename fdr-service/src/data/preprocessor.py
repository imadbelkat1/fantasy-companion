# fdr-service/src/data/preprocessor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Feature engineering for team ratings and fixture predictions"""

    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.feature_columns = {}

    def create_match_features(self,
                              fixtures_df: pd.DataFrame,
                              team_stats_df: pd.DataFrame,
                              player_aggregates_df: pd.DataFrame) -> pd.DataFrame:
        """Create match-level features by combining home and away team stats"""

        if fixtures_df.empty or team_stats_df.empty:
            logger.warning("Empty fixtures or team stats data")
            return pd.DataFrame()

        match_features = []

        for _, fixture in fixtures_df.iterrows():
            home_team = fixture['home_team']
            away_team = fixture['away_team']

            # Get team stats
            home_stats = team_stats_df[team_stats_df['team'] == home_team]
            away_stats = team_stats_df[team_stats_df['team'] == away_team]

            if home_stats.empty or away_stats.empty:
                continue

            home_stats = home_stats.iloc[0]
            away_stats = away_stats.iloc[0]

            # Get player aggregates
            home_players = player_aggregates_df[player_aggregates_df['team'] == home_team]
            away_players = player_aggregates_df[player_aggregates_df['team'] == away_team]

            home_players = home_players.iloc[0] if not home_players.empty else pd.Series()
            away_players = away_players.iloc[0] if not away_players.empty else pd.Series()

            # Create feature row
            feature_row = self._create_match_feature_row(
                fixture, home_stats, away_stats, home_players, away_players
            )

            match_features.append(feature_row)

        return pd.DataFrame(match_features)

    def _create_match_feature_row(self,
                                  fixture: pd.Series,
                                  home_stats: pd.Series,
                                  away_stats: pd.Series,
                                  home_players: pd.Series,
                                  away_players: pd.Series) -> Dict:
        """Create feature row for a single match"""

        features = {
            # Fixture info
            'fixture_id': fixture.get('fixture_id'),
            'league': fixture.get('league'),
            'date': fixture.get('date'),
            'gameweek': fixture.get('gameweek'),
            'home_team': fixture.get('home_team'),
            'away_team': fixture.get('away_team'),
            'is_home': 1  # This row represents home team perspective
        }

        # Home team features (prefixed with home_)
        features.update(self._get_team_features(home_stats, home_players, 'home'))

        # Away team features (prefixed with away_)
        features.update(self._get_team_features(away_stats, away_players, 'away'))

        # Relative features (home vs away)
        features.update(self._get_relative_features(home_stats, away_stats, home_players, away_players))

        # Target variables (if available)
        if fixture.get('home_goals') is not None and fixture.get('away_goals') is not None:
            features['target_goal_diff'] = fixture['home_goals'] - fixture['away_goals']
            features['target_home_win'] = 1 if fixture['home_goals'] > fixture['away_goals'] else 0
            features['target_draw'] = 1 if fixture['home_goals'] == fixture['away_goals'] else 0
            features['target_away_win'] = 1 if fixture['home_goals'] < fixture['away_goals'] else 0
        elif fixture.get('home_xg') is not None and fixture.get('away_xg') is not None:
            # Use xG as weak supervision target
            features['target_goal_diff'] = fixture['home_xg'] - fixture['away_xg']
            features['target_home_win'] = 1 if fixture['home_xg'] > fixture['away_xg'] else 0
            features['target_draw'] = 0  # xG rarely exactly equal
            features['target_away_win'] = 1 if fixture['home_xg'] < fixture['away_xg'] else 0

        # Recorded xG for this fixture
        features['fixture_home_xg'] = fixture.get('home_xg', 0)
        features['fixture_away_xg'] = fixture.get('away_xg', 0)
        features['fixture_xg_diff'] = features['fixture_home_xg'] - features['fixture_away_xg']

        return features

    def _create_league_average_opponent(self, team_stats_df: pd.DataFrame, player_aggregates_df: pd.DataFrame) -> Dict:
        """Create league average opponent for rating purposes"""

        # Calculate league averages for team stats
        numeric_cols = team_stats_df.select_dtypes(include=[np.number]).columns
        avg_team_stats = team_stats_df[numeric_cols].mean()

        # Calculate league averages for player stats
        if not player_aggregates_df.empty:
            numeric_player_cols = player_aggregates_df.select_dtypes(include=[np.number]).columns
            avg_player_stats = player_aggregates_df[numeric_player_cols].mean()
        else:
            avg_player_stats = pd.Series()

        return {
            'team_stats': avg_team_stats,
            'player_stats': avg_player_stats
        }

    def _get_team_features(self, team_stats: pd.Series, player_stats: pd.Series, prefix: str) -> Dict:
        """Extract standardized features for a team with given prefix"""
        features = {}

        # Core attacking features
        features[f'{prefix}_xg_for_per_game'] = team_stats.get('xg_for_per_game', 1.0)
        features[f'{prefix}_goals_per90'] = team_stats.get('attack_goals_per90', 1.0)
        features[f'{prefix}_npxg_per90'] = team_stats.get('attack_npxg_per90', 1.0)
        features[f'{prefix}_xa_per90'] = team_stats.get('attack_xa_per90', 0.5)

        # Core defensive features
        features[f'{prefix}_xg_against_per_game'] = team_stats.get('xg_against_per_game', 1.0)
        features[f'{prefix}_gk_save_pct'] = team_stats.get('defense_gk_save_pct', 70.0)
        features[f'{prefix}_clean_sheets_pct'] = team_stats.get('overall_clean_sheets', 0) / max(
            team_stats.get('overall_games_played', 1), 1)

        # Form and momentum
        features[f'{prefix}_last_5_form'] = team_stats.get('last_5_encoded', 1.5)
        features[f'{prefix}_points_per_game'] = team_stats.get('points_per_game', 1.0)

        # Net performance
        features[f'{prefix}_net_xg_per_game'] = team_stats.get('net_xg_per_game', 0.0)

        # Home/Away splits based on context
        if prefix == 'home':
            features[f'{prefix}_home_points_per_game'] = team_stats.get('home_points_per_game', 1.0)
            features[f'{prefix}_home_xg_for_per_game'] = team_stats.get('home_xg_for_per_game', 1.0)
            features[f'{prefix}_home_xg_against_per_game'] = team_stats.get('home_xg_against_per_game', 1.0)
        else:
            features[f'{prefix}_away_points_per_game'] = team_stats.get('away_points_per_game', 1.0)
            features[f'{prefix}_away_xg_for_per_game'] = team_stats.get('away_xg_for_per_game', 1.0)
            features[f'{prefix}_away_xg_against_per_game'] = team_stats.get('away_xg_against_per_game', 1.0)

        # Advanced stats
        features[f'{prefix}_possession'] = team_stats.get('attack_possession', 50.0)
        features[f'{prefix}_tackles_per90'] = team_stats.get('defense_tackles_per90', 15.0)
        features[f'{prefix}_interceptions_per90'] = team_stats.get('defense_interceptions_per90', 8.0)

        # Prior season anchor
        features[f'{prefix}_prior_points_per_game'] = team_stats.get('prior_points_per_game', 1.0)
        features[f'{prefix}_prior_xg_diff_per_game'] = team_stats.get('prior_xg_diff_per_game', 0.0)

        # Player aggregates
        features[f'{prefix}_top_player_xgxa'] = player_stats.get('top_player_xgxa_per90', 0.5)
        features[f'{prefix}_keeper_save_pct'] = player_stats.get('keeper_save_pct', 70.0)
        features[f'{prefix}_mean_player_xg'] = player_stats.get('mean_xg_per90', 0.3)

        return features

    def _get_relative_features(self,
                               home_stats: pd.Series,
                               away_stats: pd.Series,
                               home_players: pd.Series,
                               away_players: pd.Series) -> Dict:
        """Create relative features comparing home vs away team"""

        features = {}

        # Attack vs Defense matchups
        home_attack = home_stats.get('xg_for_per_game', 1.0)
        away_defense = away_stats.get('xg_against_per_game', 1.0)
        features['home_attack_vs_away_defense'] = home_attack / max(away_defense, 0.1)

        away_attack = away_stats.get('xg_for_per_game', 1.0)
        home_defense = home_stats.get('xg_against_per_game', 1.0)
        features['away_attack_vs_home_defense'] = away_attack / max(home_defense, 0.1)

        # Net performance differential
        home_net_xg = home_stats.get('net_xg_per_game', 0.0)
        away_net_xg = away_stats.get('net_xg_per_game', 0.0)
        features['net_xg_differential'] = home_net_xg - away_net_xg

        # Form differential
        home_form = home_stats.get('last_5_encoded', 1.5)
        away_form = away_stats.get('last_5_encoded', 1.5)
        features['form_differential'] = home_form - away_form

        # Points per game differential
        home_ppg = home_stats.get('points_per_game', 1.0)
        away_ppg = away_stats.get('points_per_game', 1.0)
        features['points_per_game_differential'] = home_ppg - away_ppg

        # Goalkeeper matchup
        home_gk_save = home_players.get('keeper_save_pct', 70.0)
        away_gk_save = away_players.get('keeper_save_pct', 70.0)
        features['goalkeeper_quality_diff'] = home_gk_save - away_gk_save

        # Player quality differential
        home_top_player = home_players.get('top_player_xgxa_per90', 0.5)
        away_top_player = away_players.get('top_player_xgxa_per90', 0.5)
        features['top_player_quality_diff'] = home_top_player - away_top_player

        return features

    def prepare_training_data(self, match_features_df: pd.DataFrame, league: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training data with proper scaling and imputation"""

        if match_features_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Separate features and targets
        exclude_cols = ['fixture_id', 'league', 'date', 'home_team', 'away_team'] + [col for col in
                                                                                     match_features_df.columns if
                                                                                     col.startswith('target_')]
        feature_cols = [col for col in match_features_df.columns if col not in exclude_cols]

        target_cols = [col for col in match_features_df.columns if col.startswith('target_')]

        X = match_features_df[feature_cols].copy()
        y = match_features_df[target_cols].copy() if target_cols else pd.DataFrame()

        # Ensure all feature columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                logger.warning(f"Converting non-numeric column {col} to numeric")
                X[col] = pd.to_numeric(X[col], errors='coerce')

        # Handle missing values
        if league not in self.imputers:
            self.imputers[league] = SimpleImputer(strategy='median')
            X_imputed = self.imputers[league].fit_transform(X)
        else:
            X_imputed = self.imputers[league].transform(X)

        X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

        # Scale features
        if league not in self.scalers:
            self.scalers[league] = StandardScaler()
            X_scaled = self.scalers[league].fit_transform(X_imputed)
        else:
            X_scaled = self.scalers[league].transform(X_imputed)

        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Store feature columns for this league
        self.feature_columns[league] = X.columns.tolist()

        logger.info(f"Prepared {league} training data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")

        return X_scaled, y

    def get_feature_names(self, league: str) -> List[str]:
        """Get feature column names for a league"""
        return self.feature_columns.get(league, [])

    def transform_team_features_for_rating(self, team_stats_df: pd.DataFrame, player_aggregates_df: pd.DataFrame,
                                           league: str) -> pd.DataFrame:
        """Transform team stats into model features for rating prediction - ensures same feature set as training"""

        if league not in self.feature_columns:
            logger.error(f"No feature columns stored for {league}. Train model first.")
            return pd.DataFrame()

        expected_features = self.feature_columns[league]
        logger.info(f"Expected {len(expected_features)} features for {league} team rating prediction")

        team_features = []

        for _, team_stats in team_stats_df.iterrows():
            team_name = team_stats['team']
            league_name = team_stats.get('league', league)

            # Get player stats
            player_stats = player_aggregates_df[player_aggregates_df['team'] == team_name]
            player_stats = player_stats.iloc[0] if not player_stats.empty else pd.Series()

            # Create baseline opponent (league average)
            league_avg_opponent = self._create_league_average_opponent(team_stats_df, player_aggregates_df)

            # Create feature row as if this team is playing at home vs league average
            features = self._get_team_features(team_stats, player_stats, 'home')
            features.update(
                self._get_team_features(league_avg_opponent['team_stats'], league_avg_opponent['player_stats'], 'away'))
            features.update(self._get_relative_features(team_stats, league_avg_opponent['team_stats'], player_stats,
                                                        league_avg_opponent['player_stats']))

            # Add team identifier and metadata (keep for later use)
            features['team'] = team_name
            features['league'] = league_name
            features['is_home'] = 1

            team_features.append(features)

        team_features_df = pd.DataFrame(team_features)

        # Ensure we have exactly the same features as training data
        feature_only_df = pd.DataFrame(index=team_features_df.index)

        for feature in expected_features:
            if feature in team_features_df.columns:
                feature_only_df[feature] = pd.to_numeric(team_features_df[feature], errors='coerce').fillna(0)
            else:
                logger.warning(f"Missing feature {feature} for team rating prediction, using default 0")
                feature_only_df[feature] = 0.0

        # Add team metadata back
        feature_only_df['team'] = team_features_df['team']
        feature_only_df['league'] = team_features_df['league']

        logger.info(
            f"Created team features with {len(feature_only_df.columns) - 2} numeric features (plus team/league)")

        return feature_only_df

    def preprocess_team_features(self, team_features_df: pd.DataFrame, league: str) -> pd.DataFrame:
        """Apply same preprocessing (scaling/imputation) to team features as training data"""

        if team_features_df.empty or league not in self.scalers:
            logger.warning(f"Cannot preprocess team features - no scaler available for {league}")
            return team_features_df

        # Separate team metadata from features
        team_metadata = team_features_df[
            ['team', 'league']].copy() if 'team' in team_features_df.columns else pd.DataFrame()

        # Get feature columns only (exclude team/league)
        feature_cols = [col for col in team_features_df.columns if col not in ['team', 'league']]
        X_features = team_features_df[feature_cols].copy()

        # Apply the same imputation and scaling as training data
        try:
            # Impute missing values
            X_imputed = self.imputers[league].transform(X_features)
            X_imputed = pd.DataFrame(X_imputed, columns=X_features.columns, index=X_features.index)

            # Scale features
            X_scaled = self.scalers[league].transform(X_imputed)
            X_scaled = pd.DataFrame(X_scaled, columns=X_features.columns, index=X_features.index)

            # Add team metadata back
            if not team_metadata.empty:
                X_scaled = pd.concat([X_scaled, team_metadata], axis=1)

            logger.info(
                f"Preprocessed team features for {league}: {X_scaled.shape[0]} teams, {len(feature_cols)} features")

            return X_scaled

        except Exception as e:
            logger.error(f"Error preprocessing team features: {e}")
            return team_features_df