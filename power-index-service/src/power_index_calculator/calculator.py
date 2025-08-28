"""
Power Performance Index (PPI) calculator with triple metrics (PPI, OFF, DEF)
"""
from typing import Dict, Any, Optional, Tuple
from .models import TeamMetrics, LeagueConstants, LeagueResults
from .data_loader import validate_team_stats


def calculate_xgd_per_game(xg_for: float, xg_against: float, games_played: int) -> float:
    """
    Calculate Expected Goal Difference per game

    Args:
        xg_for: Expected goals for
        xg_against: Expected goals against
        games_played: Number of games played

    Returns:
        float: xGD per game
    """
    if games_played <= 0:
        return 0.0
    return (xg_for - xg_against) / games_played


def calculate_ppi(xgd_per_game: float, c_value: float) -> float:
    """
    Calculate Power Performance Index using the formula:
    PPI = ((xGD_per_game + C) / (2 × C)) × 100

    Args:
        xgd_per_game: Expected Goal Difference per game
        c_value: League scaling constant

    Returns:
        float: PPI value (0-100)
    """
    if c_value <= 0:
        return 50.0  # Default to average if invalid constant

    ppi = ((xgd_per_game + c_value) / (2 * c_value)) * 100

    # Clamp between 0 and 100
    return max(0.0, min(100.0, ppi))


def calculate_off_metric(xg_for: float, games_played: int, c_value: float) -> float:
    """
    Calculate Offensive Power Index
    OFF = ((xG_for_per_game + C_off) / (2 × C_off)) × 100

    Args:
        xg_for: Expected goals for
        games_played: Number of games played
        c_value: Offensive scaling constant

    Returns:
        float: OFF value (0-100)
    """
    if games_played <= 0 or c_value <= 0:
        return 50.0

    xg_for_per_game = xg_for / games_played
    off = ((xg_for_per_game + c_value) / (2 * c_value)) * 100

    # Clamp between 0 and 100
    return max(0.0, min(100.0, off))


def calculate_def_metric(xg_against: float, games_played: int, c_value: float) -> float:
    """
    Calculate Defensive Power Index (inverted - lower xG_against = higher DEF)
    DEF = ((C_def - xG_against_per_game + C_def) / (2 × C_def)) × 100

    Args:
        xg_against: Expected goals against
        games_played: Number of games played
        c_value: Defensive scaling constant

    Returns:
        float: DEF value (0-100, where 100 = best defense)
    """
    if games_played <= 0 or c_value <= 0:
        return 50.0

    xg_against_per_game = xg_against / games_played

    # Inverted formula: Better defense (lower xG_against) = higher DEF score
    def_score = ((2 * c_value - xg_against_per_game) / (2 * c_value)) * 100

    # Clamp between 0 and 100
    return max(0.0, min(100.0, def_score))


def extract_home_away_from_fixtures(fixtures: list, team_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract home and away statistics from fixtures data

    Args:
        fixtures: List of fixture data
        team_name: Name of the team to extract stats for

    Returns:
        Dict with 'home' and 'away' stats in standard format
    """
    home_stats = {'games_played': 0, 'xg_for': 0.0, 'xg_against': 0.0}
    away_stats = {'games_played': 0, 'xg_for': 0.0, 'xg_against': 0.0}

    for fixture in fixtures:
        try:
            home_team = fixture.get('home_team', '')
            away_team = fixture.get('away_team', '')
            home_xg = fixture.get('home_xg', 0.0)
            away_xg = fixture.get('away_xg', 0.0)

            # Skip if no xG data
            if home_xg is None or away_xg is None:
                continue

            if home_team == team_name:
                # Team played at home
                home_stats['games_played'] += 1
                home_stats['xg_for'] += home_xg
                home_stats['xg_against'] += away_xg

            elif away_team == team_name:
                # Team played away
                away_stats['games_played'] += 1
                away_stats['xg_for'] += away_xg
                away_stats['xg_against'] += home_xg

        except (KeyError, TypeError, ValueError):
            # Skip invalid fixture data
            continue

    return {'home': home_stats, 'away': away_stats}


def get_historical_stats_for_team(team_data: Dict[str, Any], team_name: str) -> Optional[Dict[str, Any]]:
    """
    Extract historical season stats for a specific team, including derived home/away from fixtures

    Args:
        team_data: Team's data including history
        team_name: Name of the team

    Returns:
        Dict with historical overall_stats and derived home/away stats or None
    """
    try:
        if 'history' not in team_data:
            return None

        history = team_data['history']

        # Get overall historical stats
        if 'overall_stats_24-25' not in history:
            return None

        historical_overall = history['overall_stats_24-25']

        # Try to derive home/away from fixtures
        historical_home = None
        historical_away = None

        if 'fixtures_24-25' in history:
            fixtures = history['fixtures_24-25']
            if isinstance(fixtures, list):
                home_away_derived = extract_home_away_from_fixtures(fixtures, team_name)
                historical_home = home_away_derived['home'] if home_away_derived['home']['games_played'] > 0 else None
                historical_away = home_away_derived['away'] if home_away_derived['away']['games_played'] > 0 else None

        return {
            'overall_stats': historical_overall,
            'home_stats': historical_home,
            'away_stats': historical_away
        }

    except (KeyError, TypeError):
        pass

    return None


def calculate_league_constants(teams_data: Dict[str, Any], use_historical: bool = False) -> LeagueConstants:
    """
    Calculate league-specific C values for PPI (xGD), OFF (xG_for), and DEF (xG_against)

    Args:
        teams_data: Dictionary of team data for a league
        use_historical: Whether to use historical data for constants

    Returns:
        LeagueConstants: C values for PPI, OFF, DEF across overall/home/away
    """
    # PPI constants (xGD-based)
    overall_xgds = []
    home_xgds = []
    away_xgds = []

    # OFF constants (xG_for-based)
    overall_xg_fors = []
    home_xg_fors = []
    away_xg_fors = []

    # DEF constants (xG_against-based)
    overall_xg_againsts = []
    home_xg_againsts = []
    away_xg_againsts = []

    for team_name, team_data in teams_data.items():
        try:
            # Choose data source based on use_historical flag
            if use_historical:
                historical_data = get_historical_stats_for_team(team_data, team_name)
                if not historical_data:
                    continue

                # Overall historical stats
                stats = historical_data['overall_stats']
                if validate_team_stats(stats):
                    xgd = calculate_xgd_per_game(stats['xg_for'], stats['xg_against'], stats['games_played'])
                    xg_for_per_game = stats['xg_for'] / stats['games_played']
                    xg_against_per_game = stats['xg_against'] / stats['games_played']

                    overall_xgds.append(abs(xgd))
                    overall_xg_fors.append(xg_for_per_game)
                    overall_xg_againsts.append(xg_against_per_game)

                # Historical home stats (derived from fixtures)
                if historical_data['home_stats'] and validate_team_stats(historical_data['home_stats']):
                    home_stats = historical_data['home_stats']
                    home_xgd = calculate_xgd_per_game(home_stats['xg_for'], home_stats['xg_against'], home_stats['games_played'])
                    home_xg_for_per_game = home_stats['xg_for'] / home_stats['games_played']
                    home_xg_against_per_game = home_stats['xg_against'] / home_stats['games_played']

                    home_xgds.append(abs(home_xgd))
                    home_xg_fors.append(home_xg_for_per_game)
                    home_xg_againsts.append(home_xg_against_per_game)
                else:
                    # Fallback to overall if no home data
                    if validate_team_stats(stats):
                        home_xgds.append(abs(xgd))
                        home_xg_fors.append(xg_for_per_game)
                        home_xg_againsts.append(xg_against_per_game)

                # Historical away stats (derived from fixtures)
                if historical_data['away_stats'] and validate_team_stats(historical_data['away_stats']):
                    away_stats = historical_data['away_stats']
                    away_xgd = calculate_xgd_per_game(away_stats['xg_for'], away_stats['xg_against'], away_stats['games_played'])
                    away_xg_for_per_game = away_stats['xg_for'] / away_stats['games_played']
                    away_xg_against_per_game = away_stats['xg_against'] / away_stats['games_played']

                    away_xgds.append(abs(away_xgd))
                    away_xg_fors.append(away_xg_for_per_game)
                    away_xg_againsts.append(away_xg_against_per_game)
                else:
                    # Fallback to overall if no away data
                    if validate_team_stats(stats):
                        away_xgds.append(abs(xgd))
                        away_xg_fors.append(xg_for_per_game)
                        away_xg_againsts.append(xg_against_per_game)

            else:
                # Current season data (original logic)
                # Overall stats
                if 'overall_stats' in team_data:
                    stats = team_data['overall_stats']
                    if validate_team_stats(stats):
                        xgd = calculate_xgd_per_game(stats['xg_for'], stats['xg_against'], stats['games_played'])
                        xg_for_per_game = stats['xg_for'] / stats['games_played']
                        xg_against_per_game = stats['xg_against'] / stats['games_played']

                        overall_xgds.append(abs(xgd))
                        overall_xg_fors.append(xg_for_per_game)
                        overall_xg_againsts.append(xg_against_per_game)

                # Home and away stats
                if 'home_away_stats' in team_data:
                    home_away = team_data['home_away_stats']

                    # Home stats
                    if 'home' in home_away and validate_team_stats(home_away['home']):
                        home_stats = home_away['home']
                        home_xgd = calculate_xgd_per_game(home_stats['xg_for'], home_stats['xg_against'], home_stats['games_played'])
                        home_xg_for_per_game = home_stats['xg_for'] / home_stats['games_played']
                        home_xg_against_per_game = home_stats['xg_against'] / home_stats['games_played']

                        home_xgds.append(abs(home_xgd))
                        home_xg_fors.append(home_xg_for_per_game)
                        home_xg_againsts.append(home_xg_against_per_game)

                    # Away stats
                    if 'away' in home_away and validate_team_stats(home_away['away']):
                        away_stats = home_away['away']
                        away_xgd = calculate_xgd_per_game(away_stats['xg_for'], away_stats['xg_against'], away_stats['games_played'])
                        away_xg_for_per_game = away_stats['xg_for'] / away_stats['games_played']
                        away_xg_against_per_game = away_stats['xg_against'] / away_stats['games_played']

                        away_xgds.append(abs(away_xgd))
                        away_xg_fors.append(away_xg_for_per_game)
                        away_xg_againsts.append(away_xg_against_per_game)

        except (KeyError, TypeError, ZeroDivisionError):
            # Skip invalid team data
            continue

    # Calculate PPI constants (xGD-based)
    overall_c = sum(overall_xgds) / len(overall_xgds) if overall_xgds else 2.0
    home_c = sum(home_xgds) / len(home_xgds) if home_xgds else 2.0
    away_c = sum(away_xgds) / len(away_xgds) if away_xgds else 2.0

    # Calculate OFF constants (xG_for-based)
    overall_c_off = sum(overall_xg_fors) / len(overall_xg_fors) if overall_xg_fors else 1.5
    home_c_off = sum(home_xg_fors) / len(home_xg_fors) if home_xg_fors else 1.5
    away_c_off = sum(away_xg_fors) / len(away_xg_fors) if away_xg_fors else 1.5

    # Calculate DEF constants (xG_against-based)
    overall_c_def = sum(overall_xg_againsts) / len(overall_xg_againsts) if overall_xg_againsts else 1.5
    home_c_def = sum(home_xg_againsts) / len(home_xg_againsts) if home_xg_againsts else 1.5
    away_c_def = sum(away_xg_againsts) / len(away_xg_againsts) if away_xg_againsts else 1.5

    # Ensure minimum values to avoid division issues
    overall_c = max(0.3, overall_c)
    home_c = max(0.3, home_c)
    away_c = max(0.3, away_c)

    overall_c_off = max(0.3, overall_c_off)
    home_c_off = max(0.3, home_c_off)
    away_c_off = max(0.3, away_c_off)

    overall_c_def = max(0.3, overall_c_def)
    home_c_def = max(0.3, home_c_def)
    away_c_def = max(0.3, away_c_def)

    return LeagueConstants(
        # PPI constants
        overall_c=round(overall_c, 3),
        home_c=round(home_c, 3),
        away_c=round(away_c, 3),
        # OFF constants
        overall_c_off=round(overall_c_off, 3),
        home_c_off=round(home_c_off, 3),
        away_c_off=round(away_c_off, 3),
        # DEF constants
        overall_c_def=round(overall_c_def, 3),
        home_c_def=round(home_c_def, 3),
        away_c_def=round(away_c_def, 3)
    )


def is_early_season(teams_data: Dict[str, Any], threshold: int = 10) -> bool:
    """
    Check if current season has insufficient games for reliable constants

    Args:
        teams_data: Dictionary of team data
        threshold: Minimum games needed for reliable current season constants

    Returns:
        bool: True if early season (use historical constants)
    """
    total_games = 0
    team_count = 0

    for team_data in teams_data.values():
        if 'overall_stats' in team_data:
            stats = team_data['overall_stats']
            if validate_team_stats(stats):
                total_games += stats['games_played']
                team_count += 1

    if team_count == 0:
        return True

    avg_games = total_games / team_count
    return avg_games < threshold


def calculate_team_metrics(team_data: Dict[str, Any], constants: LeagueConstants) -> TeamMetrics:
    """
    Calculate PPI, OFF, and DEF values for a single team

    Args:
        team_data: Team's statistical data
        constants: League constants for scaling

    Returns:
        TeamMetrics: PPI, OFF, DEF values for overall, home, away
    """
    team_metrics = TeamMetrics()

    try:
        # Overall metrics
        if 'overall_stats' in team_data:
            stats = team_data['overall_stats']
            if validate_team_stats(stats):
                # PPI (overall strength)
                xgd = calculate_xgd_per_game(stats['xg_for'], stats['xg_against'], stats['games_played'])
                team_metrics.overall_ppi = round(calculate_ppi(xgd, constants.overall_c), 1)

                # OFF (attacking strength)
                team_metrics.overall_off = round(calculate_off_metric(stats['xg_for'], stats['games_played'], constants.overall_c_off), 1)

                # DEF (defensive strength)
                team_metrics.overall_def = round(calculate_def_metric(stats['xg_against'], stats['games_played'], constants.overall_c_def), 1)

        # Home and Away metrics
        if 'home_away_stats' in team_data:
            home_away = team_data['home_away_stats']

            # Home metrics
            if 'home' in home_away:
                home_stats = home_away['home']
                if validate_team_stats(home_stats):
                    home_xgd = calculate_xgd_per_game(home_stats['xg_for'], home_stats['xg_against'], home_stats['games_played'])
                    team_metrics.home_ppi = round(calculate_ppi(home_xgd, constants.home_c), 1)
                    team_metrics.home_off = round(calculate_off_metric(home_stats['xg_for'], home_stats['games_played'], constants.home_c_off), 1)
                    team_metrics.home_def = round(calculate_def_metric(home_stats['xg_against'], home_stats['games_played'], constants.home_c_def), 1)

            # Away metrics
            if 'away' in home_away:
                away_stats = home_away['away']
                if validate_team_stats(away_stats):
                    away_xgd = calculate_xgd_per_game(away_stats['xg_for'], away_stats['xg_against'], away_stats['games_played'])
                    team_metrics.away_ppi = round(calculate_ppi(away_xgd, constants.away_c), 1)
                    team_metrics.away_off = round(calculate_off_metric(away_stats['xg_for'], away_stats['games_played'], constants.away_c_off), 1)
                    team_metrics.away_def = round(calculate_def_metric(away_stats['xg_against'], away_stats['games_played'], constants.away_c_def), 1)

    except (KeyError, TypeError, ZeroDivisionError):
        # Return partial results if some calculations fail
        pass

    return team_metrics


def blend_team_metrics(current_metrics: TeamMetrics, historical_data: Dict[str, Any],
                      constants: LeagueConstants, blend_weight: float) -> TeamMetrics:
    """
    Blend current and historical metrics (PPI, OFF, DEF) based on season progress

    Args:
        current_metrics: Current season metrics
        historical_data: Historical team data with derived home/away stats
        constants: League constants for calculation
        blend_weight: Weight for historical data (0.0 = all current, 1.0 = all historical)

    Returns:
        TeamMetrics: Blended PPI, OFF, DEF values
    """
    if blend_weight <= 0 or not historical_data:
        return current_metrics

    try:
        # Calculate historical metrics with proper home/away data
        historical_team_data = {'overall_stats': historical_data['overall_stats']}

        # Add home/away stats if available
        if historical_data['home_stats'] or historical_data['away_stats']:
            historical_team_data['home_away_stats'] = {
                'home': historical_data['home_stats'],
                'away': historical_data['away_stats']
            }

        historical_metrics = calculate_team_metrics(historical_team_data, constants)

        # Blend the values
        blended_metrics = TeamMetrics()

        # Helper function to blend individual metrics
        def blend_metric(current_val, historical_val):
            if current_val is not None and historical_val is not None:
                return round((1 - blend_weight) * current_val + blend_weight * historical_val, 1)
            elif historical_val is not None and current_val is None:
                return round(historical_val * (1 - blend_weight * 0.5), 1)
            else:
                return current_val

        # Overall metrics blending
        blended_metrics.overall_ppi = blend_metric(current_metrics.overall_ppi, historical_metrics.overall_ppi)
        blended_metrics.overall_off = blend_metric(current_metrics.overall_off, historical_metrics.overall_off)
        blended_metrics.overall_def = blend_metric(current_metrics.overall_def, historical_metrics.overall_def)

        # Home metrics blending
        blended_metrics.home_ppi = blend_metric(current_metrics.home_ppi, historical_metrics.home_ppi)
        blended_metrics.home_off = blend_metric(current_metrics.home_off, historical_metrics.home_off)
        blended_metrics.home_def = blend_metric(current_metrics.home_def, historical_metrics.home_def)

        # Away metrics blending
        blended_metrics.away_ppi = blend_metric(current_metrics.away_ppi, historical_metrics.away_ppi)
        blended_metrics.away_off = blend_metric(current_metrics.away_off, historical_metrics.away_off)
        blended_metrics.away_def = blend_metric(current_metrics.away_def, historical_metrics.away_def)

        return blended_metrics

    except (KeyError, TypeError):
        return current_metrics


def calculate_league_ppi(teams_data: Dict[str, Any]) -> LeagueResults:
    """
    Calculate triple metrics (PPI, OFF, DEF) for all teams in a league with historical blending

    Args:
        teams_data: Dictionary of team data for the league

    Returns:
        LeagueResults: Complete league results with constants and team metrics
    """
    # Check if it's early season
    early_season = is_early_season(teams_data)

    # Calculate league constants (use historical for early season)
    constants = calculate_league_constants(teams_data, use_historical=early_season)

    # Determine blend weight based on season progress
    blend_weight = 0.0
    avg_games = 0
    if early_season:
        # Get average games played for blend weight calculation
        total_games = 0
        team_count = 0
        for team_data in teams_data.values():
            if 'overall_stats' in team_data and validate_team_stats(team_data['overall_stats']):
                total_games += team_data['overall_stats']['games_played']
                team_count += 1

        avg_games = total_games / team_count if team_count > 0 else 0

        # Blend weight: more historical weight early in season
        # Games 1-5: 80% historical, Games 6-10: 60% historical, etc.
        blend_weight = max(0.0, min(0.8, 0.8 - (avg_games - 1) * 0.1))

    # Calculate triple metrics for each team
    teams_metrics = {}
    for team_name, team_data in teams_data.items():
        # Calculate current season metrics
        current_metrics = calculate_team_metrics(team_data, constants)

        # If early season, blend with historical data
        if blend_weight > 0:
            historical_data = get_historical_stats_for_team(team_data, team_name)
            if historical_data:
                blended_metrics = blend_team_metrics(current_metrics, historical_data, constants, blend_weight)
                teams_metrics[team_name] = blended_metrics
            else:
                teams_metrics[team_name] = current_metrics
        else:
            teams_metrics[team_name] = current_metrics

    # Create calculation metadata
    calc_info = {
        "early_season": early_season,
        "avg_games_played": round(avg_games, 1),
        "historical_blend_weight": round(blend_weight, 2),
        "constants_source": "historical_24-25_with_derived_home_away" if early_season else "current_season",
        "approach": "triple_metrics_with_fixture_derived_stats" if blend_weight > 0 else "current_triple_metrics_only",
        "metrics": ["PPI (overall strength)", "OFF (attacking power)", "DEF (defensive power)"]
    }

    return LeagueResults(
        league_constants=constants,
        teams=teams_metrics,
        calculation_info=calc_info
    )


def calculate_all_leagues_ppi(leagues_data: Dict[str, Dict[str, Any]]) -> Dict[str, LeagueResults]:
    """
    Calculate triple metrics for all leagues independently

    Args:
        leagues_data: Dictionary with league names as keys and team data as values

    Returns:
        Dict: League results for each league
    """
    results = {}

    for league_name, teams_data in leagues_data.items():
        if teams_data:  # Only process leagues with team data
            results[league_name] = calculate_league_ppi(teams_data)

    return results