"""
Data loader for fbref_data.json
"""
import json
import os
from typing import Dict, Any


def load_fbref_data() -> Dict[str, Any]:
    """
    Load data from fbref_data.json

    Returns:
        Dict containing the loaded JSON data

    Raises:
        FileNotFoundError: If the data file cannot be found
        json.JSONDecodeError: If the JSON is invalid
    """
    # Get the current directory (power-index-service)
    current_file = os.path.abspath(__file__)
    power_index_service_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    fantasy_companion_dir = os.path.dirname(power_index_service_dir)
    data_path = os.path.join(fantasy_companion_dir, "data-service", "data", "fbref_data.json")

    try:
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data file at: {data_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in data file: {e}", "", 0)


def validate_team_stats(stats: Dict[str, Any]) -> bool:
    """
    Validate that team stats contain required fields for PPI calculation

    Args:
        stats: Team statistics dictionary

    Returns:
        bool: True if stats are valid, False otherwise
    """
    required_fields = ['games_played', 'xg_for', 'xg_against']

    # Check if stats is a dictionary
    if not isinstance(stats, dict):
        return False

    # Check if all required fields exist and are numeric
    for field in required_fields:
        if field not in stats:
            return False
        if not isinstance(stats[field], (int, float)):
            return False
        if stats['games_played'] <= 0:
            return False

    return True


def extract_league_data(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract and validate league data from loaded JSON

    Args:
        data: Complete loaded JSON data

    Returns:
        Dict with league names as keys and team data as values
    """
    leagues = {}

    for league_name, league_data in data.items():
        if not isinstance(league_data, dict) or 'teams' not in league_data:
            continue

        leagues[league_name] = league_data['teams']

    return leagues