from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import json
from pathlib import Path

from ..models.schemas import FixtureDifficulty, ErrorResponse
from ...src.core.config import Config

router = APIRouter()
config = Config()


def _load_fixture_difficulties(league: str) -> List[dict]:
    """Load fixture difficulties from file"""
    difficulties_file = config.OUTPUT_DIR / f"{league}_fixture_difficulties.json"

    if not difficulties_file.exists():
        raise HTTPException(status_code=404, detail=f"Fixture difficulties not found for league: {league}")

    try:
        with open(difficulties_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading difficulties: {str(e)}")


@router.get("/{league}/fixtures/difficulties", response_model=List[FixtureDifficulty])
async def get_league_fixture_difficulties(
        league: str,
        team: Optional[str] = Query(None, description="Filter by team (home or away)"),
        gameweek: Optional[int] = Query(None, ge=1, le=38, description="Filter by gameweek"),
        min_difficulty: Optional[float] = Query(None, ge=1, le=5, description="Minimum difficulty filter"),
        max_difficulty: Optional[float] = Query(None, ge=1, le=5, description="Maximum difficulty filter")
):
    """Get fixture difficulties for a specific league"""

    if league not in config.LEAGUES:
        raise HTTPException(status_code=400, detail=f"Unsupported league. Supported: {config.LEAGUES}")

    difficulties = _load_fixture_difficulties(league)

    # Apply filters
    if team:
        difficulties = [d for d in difficulties
                        if d.get('home_team', '').lower() == team.lower() or
                        d.get('away_team', '').lower() == team.lower()]

    if gameweek is not None:
        difficulties = [d for d in difficulties if d.get('gameweek') == gameweek]

    if min_difficulty is not None:
        difficulties = [d for d in difficulties
                        if d.get('fixture_difficulty_home', 1) >= min_difficulty or
                        d.get('fixture_difficulty_away', 1) >= min_difficulty]

    if max_difficulty is not None:
        difficulties = [d for d in difficulties
                        if d.get('fixture_difficulty_home', 5) <= max_difficulty or
                        d.get('fixture_difficulty_away', 5) <= max_difficulty]

    return difficulties


@router.get("/{league}/fixtures/{fixture_id}/difficulty", response_model=FixtureDifficulty)
async def get_fixture_difficulty(league: str, fixture_id: str):
    """Get difficulty for a specific fixture"""

    if league not in config.LEAGUES:
        raise HTTPException(status_code=400, detail=f"Unsupported league. Supported: {config.LEAGUES}")

    difficulties = _load_fixture_difficulties(league)

    fixture_difficulty = next((d for d in difficulties if d.get('fixture_id') == fixture_id), None)

    if not fixture_difficulty:
        raise HTTPException(status_code=404, detail=f"Fixture {fixture_id} not found in {league}")

    return fixture_difficulty


@router.get("/{league}/teams/{team_name}/fixtures", response_model=List[FixtureDifficulty])
async def get_team_fixtures(
        league: str,
        team_name: str,
        gameweek: Optional[int] = Query(None, ge=1, le=38, description="Filter by gameweek"),
        home_only: Optional[bool] = Query(False, description="Only home fixtures"),
        away_only: Optional[bool] = Query(False, description="Only away fixtures")
):
    """Get all fixtures for a specific team"""

    if league not in config.LEAGUES:
        raise HTTPException(status_code=400, detail=f"Unsupported league. Supported: {config.LEAGUES}")

    difficulties = _load_fixture_difficulties(league)

    # Filter by team
    team_fixtures = []
    for d in difficulties:
        if home_only and d.get('home_team', '').lower() == team_name.lower():
            team_fixtures.append(d)
        elif away_only and d.get('away_team', '').lower() == team_name.lower():
            team_fixtures.append(d)
        elif not home_only and not away_only and (
                d.get('home_team', '').lower() == team_name.lower() or
                d.get('away_team', '').lower() == team_name.lower()
        ):
            team_fixtures.append(d)

    # Filter by gameweek
    if gameweek is not None:
        team_fixtures = [f for f in team_fixtures if f.get('gameweek') == gameweek]

    if not team_fixtures:
        raise HTTPException(status_code=404, detail=f"No fixtures found for {team_name} in {league}")

    return team_fixtures