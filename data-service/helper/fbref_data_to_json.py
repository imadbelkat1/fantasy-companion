import json
from datetime import datetime
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional
import os

class FBRefConverter:
    def __init__(self, structure_file: str = None):
        """Initialize with column mappings from structure file"""

        # Team name standardization mappings
        self.TEAM_NAME_MAPPINGS = {
            "Manchester": "Man",
            "Tottenham": "Spurs",
            "Nott'ham": "Nott'm",
            "Newcastle Utd": "Newcastle",
            "Leeds United": "Leeds",
        }
        if structure_file is None:
            # Try to find the structure file in common locations
            possible_paths = [
                'fbref_data_structure.json',
                '../fbref_data_structure.json',
                '../../fbref_data_structure.json',
                'data-service/helper/fbref_data_structure.json',
                '../helper/fbref_data_structure.json'
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    structure_file = path
                    break

            if structure_file is None:
                raise FileNotFoundError("Could not find fbref_data_structure.json. Please provide the correct path.")

        with open(structure_file, 'r') as f:
            self.structure = json.load(f)

    def standardize_team_name(self, team_name: str) -> str:
        """Standardize team names according to mappings"""
        if not team_name:
            return team_name

        standardized_name = team_name
        for original, replacement in self.TEAM_NAME_MAPPINGS.items():
            if original in standardized_name:
                standardized_name = standardized_name.replace(original, replacement)

        return standardized_name

    def extract_cell_text(self, cell) -> str:
        """Extract clean text from a table cell, handling nested elements"""
        cell_text = cell.get_text(strip=True)

        # Handle special cases for FBRef data
        if not cell_text or cell_text in ['', '—', '-']:
            link = cell.find('a')
            if link:
                cell_text = link.get_text(strip=True)

        # Clean up common FBRef formatting
        if '–' in cell_text:
            cell_text = cell_text.replace(' – ', '-').replace('–', '-')

        return ' '.join(cell_text.split())

    def convert_cell_value(self, text: str) -> Any:
        """Convert cell text to appropriate data type"""
        if not text or text in ['', '—', '-', 'N/A']:
            return None

        try:
            cleaned_text = text.replace(',', '').replace('%', '')

            if '.' not in cleaned_text:
                return int(cleaned_text)
            else:
                return float(cleaned_text)
        except ValueError:
            return text

    def parse_html_table(self, html_content: str, table_id: str, expected_columns: List[str] = None) -> List[Dict]:
        """Extract table data by ID and return as list of dictionaries"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table', {'id': table_id})

            if not table:
                print(f"Warning: Table {table_id} not found")
                return []

            tbody = table.find('tbody')
            if not tbody:
                print(f"Warning: No tbody found in table {table_id}")
                return []

            rows = tbody.find_all('tr')
            if not rows:
                print(f"Warning: No data rows found in table {table_id}")
                return []

            data = []

            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) == 0:
                    continue

                row_data = {}

                # Use data-stat attributes to map columns correctly (FBRef standard)
                for cell in cells:
                    data_stat = cell.get('data-stat', '')
                    if data_stat:
                        cell_text = self.extract_cell_text(cell)
                        cell_value = self.convert_cell_value(cell_text)
                        row_data[data_stat] = cell_value

                # Fallback: positional mapping if data-stat didn't work and expected_columns provided
                if not row_data and expected_columns and len(cells) >= len(expected_columns):
                    for i, cell in enumerate(cells[:len(expected_columns)]):
                        column_name = expected_columns[i]
                        cell_text = self.extract_cell_text(cell)
                        cell_value = self.convert_cell_value(cell_text)
                        row_data[column_name] = cell_value

                # Only add row if it has meaningful data
                if any(str(v).strip() for v in row_data.values() if v is not None):
                    data.append(row_data)

            print(f"Extracted {len(data)} rows from table '{table_id}'")
            return data

        except Exception as e:
            print(f"Error extracting table '{table_id}': {e}")
            return []

    def get_table_id(self, table_type: str, league: str, season: str = '2025-2026') -> Optional[str]:
        """Map table type to actual table ID based on league and season"""

        # Season-specific table IDs
        if 'fixtures' in table_type:
            league_code = '9' if league == 'epl' else '12'  # laliga = 12
            return f"sched_{season}_{league_code}_1"

        # Results tables have league-specific IDs
        if table_type == 'results_overall':
            league_code = '91' if league == 'epl' else '121'  # laliga = 121
            return f"results{season}{league_code}_overall"

        if table_type == 'results_home_away':
            league_code = '91' if league == 'epl' else '121'
            return f"results{season}{league_code}_home_away"

        # For squad stats tables, use the table_type directly
        # These typically have consistent IDs across leagues
        return table_type

    def convert_fixtures(self, fixtures_data: List[Dict], league: str) -> Dict:
        """Convert fixtures data to team-based structure"""
        teams = {}

        for match in fixtures_data:
            home_team = self.standardize_team_name(match.get('home_team', ''))
            away_team = self.standardize_team_name(match.get('away_team', ''))

            # Parse score
            score = match.get('score', '0-0') or '0-0'
            home_goals = away_goals = 0
            if score and '-' in score and score != '0-0':
                try:
                    home_goals, away_goals = map(int, score.split('-'))
                except ValueError:
                    pass

            # Create fixture object
            fixture = {
                'gameweek': self.safe_int(match.get('gameweek', 0)),
                'dayofweek': match.get('dayofweek', ''),
                'date': match.get('date', ''),
                'start_time': match.get('start_time', ''),
                'home_team': home_team,
                'home_xg': self.safe_float(match.get('home_xg', 0)),
                'score': score,
                'away_xg': self.safe_float(match.get('away_xg', 0)),
                'away_team': away_team
            }

            # Add to home team
            if home_team and home_team not in teams:
                teams[home_team] = {'fixtures': []}
            if home_team:
                teams[home_team]['fixtures'].append({**fixture, 'is_home': True})

            # Add to away team
            if away_team and away_team not in teams:
                teams[away_team] = {'fixtures': []}
            if away_team:
                teams[away_team]['fixtures'].append({**fixture, 'is_home': False})

        return teams

    def safe_int(self, value, default=0):
        """Safely convert value to int"""
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def safe_float(self, value, default=0.0):
        """Safely convert value to float"""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def extract_team_stats_from_html(self, html_content: str, league: str, season: str = '2025-2026') -> Dict:
        """Extract team statistics from teams_stats HTML file"""
        team_stats_data = {}

        # Extract results overall
        results_overall_id = self.get_table_id('results_overall', league, season)
        results_overall = self.parse_html_table(html_content, results_overall_id)
        team_stats_data['results_overall'] = results_overall

        # Extract results home/away
        results_home_away_id = self.get_table_id('results_home_away', league, season)
        results_home_away = self.parse_html_table(html_content, results_home_away_id)
        team_stats_data['results_home_away'] = results_home_away

        # Extract squad stats for advanced stats
        squad_standard = self.parse_html_table(html_content, 'stats_squads_standard_for')
        team_stats_data['squad_standard'] = squad_standard

        squad_keeper = self.parse_html_table(html_content, 'stats_squads_keeper_for')
        team_stats_data['squad_keeper'] = squad_keeper

        squad_defense = self.parse_html_table(html_content, 'stats_squads_defense_for')
        team_stats_data['squad_defense'] = squad_defense

        return team_stats_data

    def find_team_in_stats(self, team_name: str, stats_list: List[Dict]) -> Dict:
        """Find a team's data in a stats table"""
        standardized_team = self.standardize_team_name(team_name)
        for team_data in stats_list:
            if self.standardize_team_name(team_data.get('team', '')) == standardized_team:
                return team_data
        return {}

    def build_team_stats_from_data(self, team_name: str, team_stats_data: Dict, team_fixtures: List[Dict] = None) -> Dict:
        """Build team stats structure from extracted HTML data"""
        # Find team data in each table
        overall_data = self.find_team_in_stats(team_name, team_stats_data.get('results_overall', []))
        home_away_data = self.find_team_in_stats(team_name, team_stats_data.get('results_home_away', []))
        standard_data = self.find_team_in_stats(team_name, team_stats_data.get('squad_standard', []))
        keeper_data = self.find_team_in_stats(team_name, team_stats_data.get('squad_keeper', []))
        defense_data = self.find_team_in_stats(team_name, team_stats_data.get('squad_defense', []))

        # Build overall stats from results table
        overall_stats = {
            'rank': self.safe_int(overall_data.get('rank', 0)),
            'games_played': self.safe_int(overall_data.get('games', 0)),
            'wins': self.safe_int(overall_data.get('wins', 0)),
            'draws': self.safe_int(overall_data.get('ties', 0)),
            'losses': self.safe_int(overall_data.get('losses', 0)),
            'points': self.safe_int(overall_data.get('points', 0)),
            'goals_for': self.safe_int(overall_data.get('goals_for', 0)),
            'goals_against': self.safe_int(overall_data.get('goals_against', 0)),
            'xg_for': self.safe_float(overall_data.get('xg_for', 0)),
            'xg_against': self.safe_float(overall_data.get('xg_against', 0)),
            'clean_sheets': self.safe_float(keeper_data.get('gk_clean_sheets', 0)),
            'last_5': list(overall_data.get('last_5', '')) if overall_data.get('last_5') else []
        }

        home_clean_sheets, away_clean_sheets, home_last_5, away_last_5 = self.calculate_home_away_details(team_name, team_fixtures or [])

        # Build home/away stats
        home_away_stats = {
            'home': {
                'games_played': self.safe_int(home_away_data.get('home_games', 0)),
                'wins': self.safe_int(home_away_data.get('home_wins', 0)),
                'draws': self.safe_int(home_away_data.get('home_ties', 0)),
                'losses': self.safe_int(home_away_data.get('home_losses', 0)),
                'points': self.safe_int(home_away_data.get('home_points', 0)),
                'goals_for': self.safe_int(home_away_data.get('home_goals_for', 0)),
                'goals_against': self.safe_int(home_away_data.get('home_goals_against', 0)),
                'xg_for': self.safe_float(home_away_data.get('home_xg_for', 0)),
                'xg_against': self.safe_float(home_away_data.get('home_xg_against', 0)),
                'clean_sheets': home_clean_sheets,
                'last_5': home_last_5
            },
            'away': {
                'games_played': self.safe_int(home_away_data.get('away_games', 0)),
                'wins': self.safe_int(home_away_data.get('away_wins', 0)),
                'draws': self.safe_int(home_away_data.get('away_ties', 0)),
                'losses': self.safe_int(home_away_data.get('away_losses', 0)),
                'points': self.safe_int(home_away_data.get('away_points', 0)),
                'goals_for': self.safe_int(home_away_data.get('away_goals_for', 0)),
                'goals_against': self.safe_int(home_away_data.get('away_goals_against', 0)),
                'xg_for': self.safe_float(home_away_data.get('away_xg_for', 0)),
                'xg_against': self.safe_float(home_away_data.get('away_xg_against', 0)),
                'clean_sheets': away_clean_sheets,
                'last_5': away_last_5
            }
        }

        # Build advanced stats from squad tables
        advanced_stats = {
            'attack': {
                'possession': self.safe_float(standard_data.get('possession', 0)),
                'goals_per90': self.safe_float(standard_data.get('goals_per90', 0)),
                'assists_per90': self.safe_float(standard_data.get('assists_per90', 0)),
                'goals_involvement_per90': self.safe_float(standard_data.get('goals_assists_per90', 0)),
                'npxg_per90': self.safe_float(standard_data.get('npxg_per90', 0)),
                'xa_per90': self.safe_float(standard_data.get('xg_assist_per90', 0)),
                'xg_involvement_per90': self.safe_float(standard_data.get('xg_xg_assist_per90', 0))
            },
            'defense': {
                'tackles_per90': self.safe_float(defense_data.get('tackles', 0)) / max(self.safe_float(defense_data.get('minutes_90s', 1)), 1),
                'interceptions_per90': self.safe_float(defense_data.get('interceptions', 0)) / max(self.safe_float(defense_data.get('minutes_90s', 1)), 1),
                'blocks_per90': self.safe_float(defense_data.get('blocks', 0)) / max(self.safe_float(defense_data.get('minutes_90s', 1)), 1),
                'clearances_per90': self.safe_float(defense_data.get('clearances', 0)) / max(self.safe_float(defense_data.get('minutes_90s', 1)), 1),
                'errors_leading_to_goal': self.safe_int(defense_data.get('errors', 0)),
                'gk_saves_per90': self.safe_float(keeper_data.get('gk_saves', 0)) / max(self.safe_float(keeper_data.get('minutes_90s', 1)), 1),
                'gk_save_pct': self.safe_float(keeper_data.get('gk_save_pct', 0))
            }
        }

        return {
            'overall_stats': overall_stats,
            'home_away_stats': home_away_stats,
            'advanced_stats': advanced_stats
        }

    def calculate_home_away_details(self, team_name: str, team_fixtures: List[Dict]) -> tuple:
        """Calculate home/away clean sheets and last 5 from fixtures data"""
        if not team_fixtures:
            return 0, 0, [], []

        today = datetime.now().date()

        # Filter fixtures that have been played (date <= today) and have valid scores
        played_fixtures = []
        for fixture in team_fixtures:
            date_str = fixture.get('date', '')
            score = fixture.get('score', '0-0') or '0-0'

            # Skip fixtures without valid scores or dates
            if '-' not in score or not date_str:
                continue

            try:
                # Parse fixture date
                fixture_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                # Only include fixtures that have been played
                if fixture_date <= today:
                    played_fixtures.append(fixture)
            except ValueError:
                # Skip fixtures with invalid date format
                continue

        # Sort by date (most recent first)
        played_fixtures.sort(key=lambda x: x.get('date', ''), reverse=True)

        home_clean_sheets = 0
        away_clean_sheets = 0
        home_results = []
        away_results = []

        for fixture in played_fixtures:
            is_home = fixture.get('is_home', False)
            score = fixture.get('score', '0-0') or '0-0'

            # Skip fixtures without valid scores
            if '-' not in score:
                continue

            try:
                home_goals, away_goals = map(int, score.split('-'))
            except ValueError:
                continue

            # Determine goals for/against and result
            if is_home:
                goals_for = home_goals
                goals_against = away_goals
            else:
                goals_for = away_goals
                goals_against = home_goals

            # Determine result
            if goals_for > goals_against:
                result = 'W'
            elif goals_for == goals_against:
                result = 'D'
            else:
                result = 'L'

            # Count clean sheets and collect results
            if is_home:
                if goals_against == 0:
                    home_clean_sheets += 1
                home_results.append(result)
            else:
                if goals_against == 0:
                    away_clean_sheets += 1
                away_results.append(result)

        # Get last 5 results for home and away (most recent 5 matches)
        home_last_5 = home_results[:5] if len(home_results) >= 5 else home_results
        away_last_5 = away_results[:5] if len(away_results) >= 5 else away_results

        return home_clean_sheets, away_clean_sheets, home_last_5, away_last_5

    def build_historical_stats_from_data(self, team_name: str, historical_team_stats_data: Dict) -> Dict:
        """Build historical stats structure from extracted 24-25 season HTML data"""
        # Find team data in historical results table
        overall_data = self.find_team_in_stats(team_name, historical_team_stats_data.get('results_overall', []))
        keeper_data = self.find_team_in_stats(team_name, historical_team_stats_data.get('squad_keeper', []))

        # Build historical overall stats from results table
        historical_stats = {
            'games_played': self.safe_int(overall_data.get('games', 0)),
            'wins': self.safe_int(overall_data.get('wins', 0)),
            'draws': self.safe_int(overall_data.get('ties', 0)),
            'losses': self.safe_int(overall_data.get('losses', 0)),
            'points': self.safe_int(overall_data.get('points', 0)),
            'goals_for': self.safe_int(overall_data.get('goals_for', 0)),
            'goals_against': self.safe_int(overall_data.get('goals_against', 0)),
            'xg_for': self.safe_float(overall_data.get('xg_for', 0)),
            'xg_against': self.safe_float(overall_data.get('xg_against', 0)),
            'clean_sheets': self.safe_int(keeper_data.get('gk_clean_sheets', 0)),
            'last_5': list(overall_data.get('last_5', '')) if overall_data.get('last_5') else []
        }

        return historical_stats

    def process_players(self, standard_data: List[Dict], keeper_data: List[Dict],
                        defense_data: List[Dict]) -> Dict:
        """Process player data into players and goalkeepers categories"""
        players = {}
        goalkeepers = {}

        # Process standard stats for outfield players
        for player in standard_data:
            if player.get('position') == 'GK':
                continue

            name = player.get('player', '')
            if not name:
                continue

            players[name] = {
                'team': self.standardize_team_name(player.get('team', '')),
                'position': player.get('position', ''),
                'games': self.safe_int(player.get('games', 0)),
                'games_starts': self.safe_int(player.get('games_starts', 0)),
                'minutes': self.safe_int(player.get('minutes', 0)),
                'minutes_per90': self.safe_float(player.get('minutes_90s', 0)),
                'goals': self.safe_int(player.get('goals', 0)),
                'goals_per90': self.safe_float(player.get('goals_per90', 0)),
                'assists': self.safe_int(player.get('assists', 0)),
                'assists_per90': self.safe_float(player.get('assists_per90', 0)),
                'xg_per90': self.safe_float(player.get('xg_per90', 0)),
                'xa_per90': self.safe_float(player.get('xg_assist_per90', 0)),
                'npxg_per90': self.safe_float(player.get('npxg_per90', 0)),
                'npxg_involvement_per90': self.safe_float(player.get('npxg_xg_assist_per90', 0))
            }

        # Add defensive stats
        for defender in defense_data:
            name = defender.get('player', '')
            if name in players:
                minutes_90s = self.safe_float(defender.get('minutes_90s', 1))
                if minutes_90s == 0:
                    minutes_90s = 1  # Avoid division by zero

                players[name].update({
                    'tackles': self.safe_int(defender.get('tackles', 0)),
                    'tackles_per90': self.safe_float(defender.get('tackles', 0)) / minutes_90s,
                    'interceptions': self.safe_int(defender.get('interceptions', 0)),
                    'interceptions_per90': self.safe_float(defender.get('interceptions', 0)) / minutes_90s,
                    'blocks': self.safe_int(defender.get('blocks', 0)),
                    'blocks_per90': self.safe_float(defender.get('blocks', 0)) / minutes_90s,
                    'clearances': self.safe_int(defender.get('clearances', 0)),
                    'clearances_per90': self.safe_float(defender.get('clearances', 0)) / minutes_90s
                })

        # Process goalkeepers
        for keeper in keeper_data:
            name = keeper.get('player', '')
            if not name:
                continue

            goalkeepers[name] = {
                'team': self.standardize_team_name(keeper.get('team', '')),
                'position': keeper.get('position', 'GK'),
                'games': self.safe_int(keeper.get('gk_games', 0)),
                'games_starts': self.safe_int(keeper.get('gk_games_starts', 0)),
                'minutes': self.safe_int(keeper.get('gk_minutes', 0)),
                'minutes_per90': self.safe_float(keeper.get('minutes_90s', 0)),
                'goals_against': self.safe_int(keeper.get('gk_goals_against', 0)),
                'goals_against_per90': self.safe_float(keeper.get('gk_goals_against_per90', 0)),
                'shots_on_target_against': self.safe_int(keeper.get('gk_shots_on_target_against', 0)),
                'saves': self.safe_int(keeper.get('gk_saves', 0)),
                'save_pct': self.safe_float(keeper.get('gk_save_pct', 0)),
                'clean_sheets': self.safe_int(keeper.get('gk_clean_sheets', 0)),
                'clean_sheets_pct': self.safe_float(keeper.get('gk_clean_sheets_pct', 0)),
                'pens_att': self.safe_int(keeper.get('gk_pens_att', 0)),
                'pens_allowed': self.safe_int(keeper.get('gk_pens_allowed', 0)),
                'pens_saved': self.safe_int(keeper.get('gk_pens_saved', 0)),
                'pens_missed': self.safe_int(keeper.get('gk_pens_missed', 0)),
                'pens_save_pct': self.safe_float(keeper.get('gk_pens_save_pct', 0))
            }

        return {'players': players, 'goalkeepers': goalkeepers}

    def convert(self, html_files: Dict[str, str], league: str, season: str = '2025-2026') -> Dict:
        """Main conversion function"""
        result = {league: {'teams': {}, 'players': {}}}

        # Extract team stats from teams_stats file first
        team_stats_data = {}
        if 'teams_stats' in html_files:
            team_stats_data = self.extract_team_stats_from_html(html_files['teams_stats'], league, season)

        # Process current season fixtures
        if 'fixtures' in html_files:
            table_id = self.get_table_id('fixtures', league, season)
            expected_columns = self.structure.get('teams', {}).get(f'fixtures_{league}', [])
            fixtures_data = self.parse_html_table(html_files['fixtures'], table_id, expected_columns)

            if fixtures_data:
                teams_with_fixtures = self.convert_fixtures(fixtures_data, league)

                # Build stats for each team using extracted data instead of calculating
                for team_name, team_data in teams_with_fixtures.items():
                    if team_stats_data:
                        team_stats = self.build_team_stats_from_data(team_name, team_stats_data, team_data['fixtures'])
                    else:
                        # Fallback to empty stats if no team stats file
                        team_stats = {
                            'overall_stats': {},
                            'home_away_stats': {'home': {}, 'away': {}},
                            'advanced_stats': {'attack': {}, 'defense': {}}
                        }

                    result[league]['teams'][team_name] = {
                        'fixtures': team_data['fixtures'],
                        **team_stats
                    }

        # Process historical data if available
        if 'fixtures_24' in html_files:
            table_id = self.get_table_id('fixtures', league, '2024-2025')
            expected_columns = self.structure.get('teams', {}).get(f'fixtures_{league}', [])
            historical_fixtures = self.parse_html_table(html_files['fixtures_24'], table_id, expected_columns)

            # Extract historical team stats if available
            historical_team_stats_data = {}
            if 'teams_stats_24' in html_files:
                historical_team_stats_data = self.extract_team_stats_from_html(html_files['teams_stats_24'], league, '2024-2025')

            for team_name in result[league]['teams']:
                # Filter historical fixtures for this team
                team_historical = [f for f in historical_fixtures
                                   if self.standardize_team_name(f.get('home_team', '')) == team_name or
                                   self.standardize_team_name(f.get('away_team', '')) == team_name]

                if 'history' not in result[league]['teams'][team_name]:
                    result[league]['teams'][team_name]['history'] = {}

                result[league]['teams'][team_name]['history']['fixtures_24-25'] = team_historical

                # Get actual historical stats from teams_stats_24 file
                if historical_team_stats_data:
                    historical_stats = self.build_historical_stats_from_data(team_name, historical_team_stats_data)
                else:
                    # Fallback to placeholder if no historical stats file
                    historical_stats = {
                        'games_played': 38, 'wins': 27, 'draws': 7, 'losses': 2,
                        'points': 88, 'goals_for': 72, 'goals_against': 20
                    }

                result[league]['teams'][team_name]['history']['overall_stats_24-25'] = historical_stats

        # Process players
        player_data = {}

        # Map file keys to structure keys and table IDs
        file_to_structure_map = {
            'standard_stats': ('stats_standard', 'stats_standard'),
            'keeper_stats': ('stats_keeper', 'stats_keeper'),
            'defense_stats': ('stats_defense', 'stats_defense')
        }

        for file_key, (structure_key, table_id) in file_to_structure_map.items():
            if file_key in html_files:
                expected_columns = self.structure.get('players', {}).get(structure_key, [])
                data = self.parse_html_table(html_files[file_key], table_id, expected_columns)
                player_data[file_key] = data

        if player_data:
            result[league]['players'] = self.process_players(
                player_data.get('standard_stats', []),
                player_data.get('keeper_stats', []),
                player_data.get('defense_stats', [])
            )

        return result

def load_html_files(base_path: str, league: str) -> Dict[str, str]:
    """Load HTML files for a specific league"""
    league_folder = 'epl' if league == 'epl' else 'la-liga'
    files = {}

    file_mappings = {
        'fixtures': f'{base_path}/{league_folder}/teams_fixtures',
        'teams_stats': f'{base_path}/{league_folder}/teams_stats',
        'standard_stats': f'{base_path}/{league_folder}/players_standard_stats',
        'keeper_stats': f'{base_path}/{league_folder}/players_goalkeeping_stats',
        'defense_stats': f'{base_path}/{league_folder}/players_defensive_stats',
        'fixtures_24': f'{base_path}/season24-25/{league_folder}/teams_fixtures_24',
        'teams_stats_24': f'{base_path}/season24-25/{league_folder}/teams_stats_24'
    }

    for key, filepath in file_mappings.items():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                files[key] = f.read()
            print(f"Loaded {filepath}")
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    return files

def main():
    # Try to find and use the correct structure file path
    structure_file_path = None
    possible_structure_paths = [
        'fbref_data_structure.json',
        '../fbref_data_structure.json',
        '../../fbref_data_structure.json',
        '../../../fbref_data_structure.json'
    ]

    for path in possible_structure_paths:
        if os.path.exists(path):
            structure_file_path = path
            break

    converter = FBRefConverter(structure_file_path)
    base_path = '../fbref/fbref-data'  # Adjusted relative path

    # Process EPL
    print("Loading EPL data...")
    epl_files = load_html_files(base_path, 'epl')
    epl_data = converter.convert(epl_files, 'epl', '2025-2026')

    # Process La Liga
    print("\nLoading La Liga data...")
    laliga_files = load_html_files(base_path, 'laliga')
    laliga_data = converter.convert(laliga_files, 'laliga', '2025-2026')

    # Combine both leagues
    combined_data = {**epl_data, **laliga_data}

    # Save result
    with open('../data/fbref_data.json', 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f"\nData saved to output_data_structure.json")
    print(f"EPL teams: {len(epl_data.get('epl', {}).get('teams', {}))}")
    print(f"La Liga teams: {len(laliga_data.get('laliga', {}).get('teams', {}))}")

if __name__ == "__main__":
    main()