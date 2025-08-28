"""
Pytest configuration and fixtures
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import json
import sys
import os
from pathlib import Path

# Add project root and src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

# Insert at the beginning to ensure our modules are found first
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

print(f"Added to Python path:")
print(f"  - {project_root}")
print(f"  - {src_path}")

@pytest.fixture
def sample_fbref_data():
    """Sample fbref data structure for testing"""
    return {
        "epl": {
            "teams": {
                "Arsenal": {
                    "fixtures": [
                        {
                            "gameweek": 1,
                            "dayofweek": "Saturday",
                            "date": "2023-08-12",
                            "start_time": "15:00",
                            "is_home": True,
                            "home_team": "Arsenal",
                            "home_xg": 2.1,
                            "score": "2-0",
                            "away_xg": 0.8,
                            "away_team": "Nottm Forest"
                        }
                    ],
                    "overall_stats": {
                        "games_played": 2,
                        "wins": 1,
                        "draws": 1,
                        "losses": 0,
                        "points": 4,
                        "last_5": ["W", "D"],
                        "goals_for": 3,
                        "goals_against": 1,
                        "xg_for": 3.9,
                        "xg_against": 2.0,
                        "clean_sheets": 1
                    },
                    "advanced_stats": {
                        "attack": {
                            "possession": 58.5,
                            "goals_per90": 1.35,
                            "assists_per90": 0.9,
                            "npxg_per90": 1.75,
                            "xa_per90": 0.85
                        },
                        "defense": {
                            "tackles_per90": 16.2,
                            "interceptions_per90": 9.1,
                            "blocks_per90": 4.5,
                            "clearances_per90": 18.3,
                            "gk_saves_per90": 2.5,
                            "gk_save_pct": 78.0
                        }
                    },
                    "history": {
                        "overall_stats_24-25": {
                            "games_played": 38,
                            "wins": 24,
                            "draws": 8,
                            "losses": 6,
                            "points": 80,
                            "goals_for": 68,
                            "goals_against": 35,
                            "xg_for": 72.5,
                            "xg_against": 38.2
                        }
                    }
                }
            },
            "players": {
                "players": {
                    "Bukayo Saka": {
                        "team": "Arsenal",
                        "position": "RW",
                        "games": 2,
                        "games_starts": 2,
                        "minutes": 180,
                        "goals": 1,
                        "assists": 1,
                        "xg_per90": 0.6,
                        "xa_per90": 0.8,
                        "npxg_per90": 0.55
                    }
                },
                "goalkeepers": {
                    "Aaron Ramsdale": {
                        "team": "Arsenal",
                        "position": "GK",
                        "games": 2,
                        "games_starts": 2,
                        "minutes": 180,
                        "goals_against": 1,
                        "saves": 6,
                        "save_pct": 85.7,
                        "clean_sheets": 1
                    }
                }
            }
        }
    }

@pytest.fixture
def temp_data_file(sample_fbref_data):
    """Create temporary data file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_fbref_data, f)
        yield Path(f.name)

    # Cleanup
    Path(f.name).unlink(missing_ok=True)