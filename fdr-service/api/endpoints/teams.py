from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import json
from pathlib import Path

from ..models.schemas import TeamRating, TeamRatingResponse, ErrorResponse
from ...src.core.config import Config

router = APIRouter()
config = Config()


def _load_team_ratings(league: str) -> List[dict]:
    """Load team ratings from file"""
    ratings_file = config.OUTPUT_DIR / f"{league}_team_ratings.json"

    if not ratings_file.exists():
        raise HTTPException(status_code=404, detail=f"Ratings not found for league: {league}")

    try:
        with open(ratings_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ratings: {str(e)}")


@router.get("/{league}/teams/ratings", response_model=List[TeamRating])
async def get_league_team_ratings(
        league: str,
        team: Optional[str] = Query(None, description="Filter by specific team"),
        min_rating: Optional[float] = Query(None, ge=0, le=100, description="Minimum rating filter"),
        max_rating: Optional[float] = Query(None, ge=0, le=100, description="Maximum rating filter")
):
    """Get team ratings for a specific league"""

    if league not in config.LEAGUES:
        raise HTTPException(status_code=400, detail=f"Unsupported league. Supported: {config.LEAGUES}")

    ratings = _load_team_ratings(league)

    # Apply filters
    if team:
        ratings = [r for r in ratings if r.get('team', '').lower() == team.lower()]
        if not ratings:
            raise HTTPException(status_code=404, detail=f"Team {team} not found in {league}")

    if min_rating is not None:
        ratings = [r for r in ratings if r.get('rating', 0) >= min_rating]

    if max_rating is not None:
        ratings = [r for r in ratings if r.get('rating', 100) <= max_rating]

    return ratings


@router.get("/{league}/teams/{team_name}/rating", response_model=TeamRating)
async def get_team_rating(league: str, team_name: str):
    """Get rating for a specific team"""

    if league not in config.LEAGUES:
        raise HTTPException(status_code=400, detail=f"Unsupported league. Supported: {config.LEAGUES}")

    ratings = _load_team_ratings(league)

    team_rating = next((r for r in ratings if r.get('team', '').lower() == team_name.lower()), None)

    if not team_rating:
        raise HTTPException(status_code=404, detail=f"Team {team_name} not found in {league}")

    return team_rating


@router.get("/leagues", response_model=List[str])
async def get_supported_leagues():
    """Get list of supported leagues"""
    return config.LEAGUES


@router.get("/{league}/teams", response_model=List[str])
async def get_league_teams(league: str):
    """Get list of teams in a league"""

    if league not in config.LEAGUES:
        raise HTTPException(status_code=400, detail=f"Unsupported league. Supported: {config.LEAGUES}")

    ratings = _load_team_ratings(league)
    teams = [r.get('team') for r in ratings if r.get('team')]

    return sorted(teams)