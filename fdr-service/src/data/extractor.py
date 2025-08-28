import json
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from hashlib import md5

logger = logging.getLogger(__name__)


class DataExtractor:
    """Extract and canonicalize data from fbref JSON"""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.raw_data = None

    def load_data(self) -> Dict:
        """Load raw JSON data"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            logger.info(f"Loaded data from {self.data_path}")
            return self.raw_data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def extract_league_data(self, league: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Extract canonical DataFrames for a specific league

        Returns:
            fixtures_df: Fixture-level data
            team_stats_df: Team-level aggregated stats
            player_aggregates_df: Player aggregates per team
        """
        if self.raw_data is None:
            self.load_data()

        league_data = self.raw_data.get(league, {})

        if not league_data:
            logger.warning(f"No data found for league: {league}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Extract fixtures
        fixtures_df = self._extract_fixtures(league_data, league)

        # Extract team stats
        team_stats_df = self._extract_team_stats(league_data, league)

        # Extract player aggregates
        player_aggregates_df = self._extract_player_aggregates(league_data, league)

        logger.info(f"Extracted {league} data: {len(fixtures_df)} fixtures, "
                    f"{len(team_stats_df)} teams, {len(player_aggregates_df)} player aggregates")

        return fixtures_df, team_stats_df, player_aggregates_df

    def _extract_fixtures(self, league_data: Dict, league: str) -> pd.DataFrame:
        """Extract fixture-level data"""
        fixtures = []
        seen_fixture_ids = set()

        teams_data = league_data.get('teams', {})

        for team_name, team_data in teams_data.items():
            team_fixtures = team_data.get('fixtures', [])

            for fixture in team_fixtures:
                # Create unique fixture ID
                fixture_key = f"{fixture.get('date')}_{fixture.get('home_team')}_{fixture.get('away_team')}_{fixture.get('gameweek')}"
                fixture_id = md5(fixture_key.encode()).hexdigest()[:12]

                # Skip duplicates (same fixture appears in both teams' data)
                if fixture_id in seen_fixture_ids:
                    continue
                seen_fixture_ids.add(fixture_id)

                # Parse score if available
                score = fixture.get('score', '')
                home_goals, away_goals = self._parse_score(score)

                fixture_row = {
                    'fixture_id': fixture_id,
                    'league': league,
                    'date': fixture.get('date'),
                    'gameweek': fixture.get('gameweek'),
                    'dayofweek': fixture.get('dayofweek'),
                    'start_time': fixture.get('start_time'),
                    'home_team': fixture.get('home_team'),
                    'away_team': fixture.get('away_team'),
                    'home_xg': fixture.get('home_xg'),
                    'away_xg': fixture.get('away_xg'),
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'score': score
                }
                fixtures.append(fixture_row)

        return pd.DataFrame(fixtures)

    def _extract_team_stats(self, league_data: Dict, league: str) -> pd.DataFrame:
        """Extract team-level statistics"""
        teams = []
        teams_data = league_data.get('teams', {})

        for team_name, team_data in teams_data.items():
            team_row = {'team': team_name, 'league': league}

            # Overall stats
            overall_stats = team_data.get('overall_stats', {})
            for key, value in overall_stats.items():
                if key == 'last_5':
                    # Encode last 5 results
                    team_row['last_5_encoded'] = self._encode_last_5(value)
                else:
                    team_row[f'overall_{key}'] = value

            # Home/Away stats
            home_away = team_data.get('home_away_stats', {})
            for location in ['home', 'away']:
                location_stats = home_away.get(location, {})
                for key, value in location_stats.items():
                    if key == 'last_5':
                        team_row[f'{location}_last_5_encoded'] = self._encode_last_5(value)
                    else:
                        team_row[f'{location}_{key}'] = value

            # Advanced stats
            advanced_stats = team_data.get('advanced_stats', {})
            for category in ['attack', 'defense']:
                category_stats = advanced_stats.get(category, {})
                for key, value in category_stats.items():
                    team_row[f'{category}_{key}'] = value

            # Historical stats (prior season anchor)
            history = team_data.get('history', {})
            prior_stats = history.get('overall_stats_24-25', {})
            for key, value in prior_stats.items():
                if key != 'last_5':  # Skip last_5 for prior season
                    team_row[f'prior_{key}'] = value

            teams.append(team_row)

        df = pd.DataFrame(teams)

        # Add derived features
        if not df.empty:
            df = self._add_derived_team_features(df)

        return df

    def _extract_player_aggregates(self, league_data: Dict, league: str) -> pd.DataFrame:
        """Extract player aggregates per team"""
        team_aggregates = []
        players_data = league_data.get('players', {})

        # Process outfield players
        outfield_players = players_data.get('players', {})
        player_by_team = {}

        for player_name, player_data in outfield_players.items():
            team = player_data.get('team')
            if team not in player_by_team:
                player_by_team[team] = []
            player_by_team[team].append(player_data)

        # Process goalkeepers
        goalkeepers = players_data.get('goalkeepers', {})
        gk_by_team = {}

        for gk_name, gk_data in goalkeepers.items():
            team = gk_data.get('team')
            gk_by_team[team] = gk_data  # Assume one main GK per team

        # Create team aggregates
        all_teams = set(list(player_by_team.keys()) + list(gk_by_team.keys()))

        for team in all_teams:
            team_agg = {
                'team': team,
                'league': league
            }

            # Outfield player aggregates
            team_players = player_by_team.get(team, [])
            if team_players:
                team_agg['sum_goals'] = sum(p.get('goals', 0) for p in team_players)
                team_agg['sum_assists'] = sum(p.get('assists', 0) for p in team_players)
                team_agg['mean_xg_per90'] = sum(p.get('xg_per90', 0) for p in team_players) / len(team_players)
                team_agg['mean_xa_per90'] = sum(p.get('xa_per90', 0) for p in team_players) / len(team_players)

                # Top player metrics
                xg_xa_values = [(p.get('xg_per90', 0) + p.get('xa_per90', 0)) for p in team_players]
                team_agg['top_player_xgxa_per90'] = max(xg_xa_values) if xg_xa_values else 0
            else:
                # Fill with zeros if no player data
                for key in ['sum_goals', 'sum_assists', 'mean_xg_per90', 'mean_xa_per90', 'top_player_xgxa_per90']:
                    team_agg[key] = 0

            # Goalkeeper aggregates
            gk_data = gk_by_team.get(team, {})
            team_agg['keeper_save_pct'] = gk_data.get('save_pct', 50.0)  # Default 50% if missing
            team_agg['keeper_clean_sheet_pct'] = gk_data.get('clean_sheets_pct', 0.0)
            team_agg['keeper_goals_against_per90'] = gk_data.get('goals_against_per90', 1.5)

            team_aggregates.append(team_agg)

        return pd.DataFrame(team_aggregates)

    def _parse_score(self, score: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse score string into home_goals, away_goals"""
        if not score or '-' not in score:
            return None, None

        try:
            parts = score.split('-')
            if len(parts) == 2:
                home_goals = int(parts[0].strip())
                away_goals = int(parts[1].strip())
                return home_goals, away_goals
        except (ValueError, IndexError):
            pass

        return None, None

    def _encode_last_5(self, last_5_list) -> float:
        """Encode last 5 results as W=3, D=1, L=0, return mean"""
        if not last_5_list:
            return 1.5  # Neutral default

        encoding = {'W': 3, 'D': 1, 'L': 0}
        encoded_values = [encoding.get(result, 1) for result in last_5_list]
        return sum(encoded_values) / len(encoded_values)

    def _add_derived_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to team stats"""
        # Avoid division by zero
        df['overall_games_played'] = df['overall_games_played'].fillna(1).replace(0, 1)

        # Per-game metrics
        df['xg_for_per_game'] = df['overall_xg_for'] / df['overall_games_played']
        df['xg_against_per_game'] = df['overall_xg_against'] / df['overall_games_played']
        df['net_xg_per_game'] = df['xg_for_per_game'] - df['xg_against_per_game']
        df['points_per_game'] = df['overall_points'] / df['overall_games_played']

        # Home/Away per-game metrics
        for location in ['home', 'away']:
            games_col = f'{location}_games_played'
            df[games_col] = df[games_col].fillna(1).replace(0, 1)
            df[f'{location}_points_per_game'] = df[f'{location}_points'] / df[games_col]
            df[f'{location}_xg_for_per_game'] = df[f'{location}_xg_for'] / df[games_col]
            df[f'{location}_xg_against_per_game'] = df[f'{location}_xg_against'] / df[games_col]

        # Goal difference
        df['overall_goal_diff'] = df['overall_goals_for'] - df['overall_goals_against']

        # Prior season anchor (if available)
        if 'prior_games_played' in df.columns:
            df['prior_games_played'] = df['prior_games_played'].fillna(38).replace(0, 38)
            df['prior_points_per_game'] = df['prior_points'] / df['prior_games_played']
            df['prior_xg_diff_per_game'] = (df['prior_xg_for'] - df['prior_xg_against']) / df['prior_games_played']

        return df