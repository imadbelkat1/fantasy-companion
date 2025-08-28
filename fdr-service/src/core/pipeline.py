import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime

from ..data import DataExtractor, FeatureEngine
from ..models import TeamRater, FixturePredictor
from ..utils.validation import DataValidator
from ..utils.explainer import ExplainerEngine
from .config import Config

logger = logging.getLogger(__name__)


class FDRPipeline:
    """Main pipeline for FDR service"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.extractor = None
        self.feature_engine = None
        self.models = {}  # League -> TeamRater
        self.predictors = {}  # League -> FixturePredictor
        self.validator = DataValidator()
        self.explainer = ExplainerEngine()

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete FDR pipeline"""

        logger.info("Starting FDR pipeline...")

        # Create output directories
        self.config.create_directories()

        results = {
            'timestamp': datetime.now().isoformat(),
            'leagues_processed': [],
            'team_ratings': {},
            'fixture_difficulties': {},
            'model_summaries': {},
            'errors': []
        }

        try:
            # Initialize components
            self.extractor = DataExtractor(self.config.DATA_PATH)
            self.feature_engine = FeatureEngine()

            # Load raw data
            raw_data = self.extractor.load_data()

            # Process each league independently
            for league in self.config.LEAGUES:
                logger.info(f"Processing league: {league}")

                try:
                    league_results = self._process_league(league)

                    results['leagues_processed'].append(league)
                    results['team_ratings'][league] = league_results['team_ratings']
                    results['fixture_difficulties'][league] = league_results['fixture_difficulties']
                    results['model_summaries'][league] = league_results['model_summary']

                    logger.info(f"Successfully processed {league}")

                except Exception as e:
                    error_msg = f"Error processing {league}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)

            # Save results to files
            self._save_results(results)

            logger.info("FDR pipeline completed successfully")

        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

        return results

    def _process_league(self, league: str) -> Dict[str, Any]:
        """Process a single league"""

        # Extract data
        fixtures_df, team_stats_df, player_aggregates_df = self.extractor.extract_league_data(league)

        # Validate data
        validation_results = self.validator.validate_league_data(fixtures_df, team_stats_df, player_aggregates_df,
                                                                 league)

        if not validation_results['is_valid']:
            raise ValueError(f"Data validation failed for {league}: {validation_results['errors']}")

        # Create match features
        match_features_df = self.feature_engine.create_match_features(fixtures_df, team_stats_df, player_aggregates_df)

        # Prepare training data
        X_train, y_train = self.feature_engine.prepare_training_data(match_features_df, league)

        # Train model
        model = TeamRater(league, self.config.MODEL_CONFIG)

        if not X_train.empty and not y_train.empty:
            # Time-based train/validation split
            split_idx = int(len(X_train) * 0.8)
            X_train_split = X_train.iloc[:split_idx]
            y_train_split = y_train.iloc[:split_idx]
            X_val_split = X_train.iloc[split_idx:]
            y_val_split = y_train.iloc[split_idx:]

            # Fit model
            model.fit(X_train_split, y_train_split)

            # Validate model
            if not X_val_split.empty and not y_val_split.empty:
                model.validate_model(X_train_split, y_train_split, X_val_split, y_val_split, 'target_goal_diff')
        else:
            logger.warning(f"No training data available for {league}")

        self.models[league] = model

        # Create team rating features that match training features exactly
        team_features_df = self.feature_engine.transform_team_features_for_rating(team_stats_df, player_aggregates_df,
                                                                                  league)

        # Apply same preprocessing as training data
        team_features_processed = self.feature_engine.preprocess_team_features(team_features_df, league)

        # Generate team ratings with uncertainty
        team_ratings_df = model.estimate_uncertainty(team_features_processed)

        # Add explanations
        team_ratings_with_explanations = self.explainer.add_team_explanations(team_ratings_df, model, team_features_df)

        # Create fixture predictor
        predictor = FixturePredictor(model)
        self.predictors[league] = predictor

        # Predict fixture difficulties
        fixture_difficulties_df = predictor.predict_fixture_difficulties(fixtures_df, match_features_df)

        return {
            'team_ratings': team_ratings_with_explanations.to_dict('records'),
            'fixture_difficulties': fixture_difficulties_df.to_dict('records'),
            'model_summary': model.get_model_summary(),
            'validation_results': validation_results
        }

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to output files"""

        output_dir = self.config.OUTPUT_DIR

        # Save team ratings by league
        for league, ratings in results['team_ratings'].items():
            ratings_file = output_dir / f"{league}_team_ratings.json"
            with open(ratings_file, 'w') as f:
                json.dump(ratings, f, indent=2)
            logger.info(f"Saved {league} team ratings to {ratings_file}")

        # Save fixture difficulties by league
        for league, difficulties in results['fixture_difficulties'].items():
            difficulties_file = output_dir / f"{league}_fixture_difficulties.json"
            with open(difficulties_file, 'w') as f:
                json.dump(difficulties, f, indent=2)
            logger.info(f"Saved {league} fixture difficulties to {difficulties_file}")

        # Save combined summary
        summary_file = output_dir / "pipeline_summary.json"
        summary = {
            'timestamp': results['timestamp'],
            'leagues_processed': results['leagues_processed'],
            'model_summaries': results['model_summaries'],
            'errors': results['errors']
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved pipeline summary to {summary_file}")

    def get_team_rating(self, league: str, team: str) -> Dict[str, Any]:
        """Get rating for a specific team"""

        if league not in self.models:
            return {'error': f'League {league} not found'}

        # Load saved ratings if not in memory
        ratings_file = self.config.OUTPUT_DIR / f"{league}_team_ratings.json"

        if ratings_file.exists():
            with open(ratings_file, 'r') as f:
                ratings = json.load(f)

            team_rating = next((r for r in ratings if r['team'] == team), None)

            if team_rating:
                return team_rating
            else:
                return {'error': f'Team {team} not found in {league}'}
        else:
            return {'error': f'Ratings file for {league} not found'}

    def get_fixture_difficulty(self, league: str, fixture_id: str = None, home_team: str = None,
                               away_team: str = None) -> Dict[str, Any]:
        """Get fixture difficulty by ID or team names"""

        if league not in self.predictors:
            return {'error': f'League {league} not found'}

        # Load saved difficulties if not in memory
        difficulties_file = self.config.OUTPUT_DIR / f"{league}_fixture_difficulties.json"

        if difficulties_file.exists():
            with open(difficulties_file, 'r') as f:
                difficulties = json.load(f)

            if fixture_id:
                fixture_difficulty = next((d for d in difficulties if d['fixture_id'] == fixture_id), None)
            elif home_team and away_team:
                fixture_difficulty = next((d for d in difficulties
                                           if d['home_team'] == home_team and d['away_team'] == away_team), None)
            else:
                return {'error': 'Must provide either fixture_id or both home_team and away_team'}

            if fixture_difficulty:
                return fixture_difficulty
            else:
                return {'error': 'Fixture not found'}
        else:
            return {'error': f'Difficulties file for {league} not found'}