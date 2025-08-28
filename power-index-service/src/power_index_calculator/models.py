"""
Pydantic models for Power Index API responses
"""
from pydantic import BaseModel
from typing import Dict, Optional, Any


class TeamPPI(BaseModel):
    """Power Performance Index values for a single team"""
    overall_ppi: Optional[float] = None
    home_ppi: Optional[float] = None
    away_ppi: Optional[float] = None

class TeamMetrics (TeamPPI):
    overall_off: Optional[float] = None
    home_off: Optional[float] = None
    away_off: Optional[float] = None
    overall_def: Optional[float] = None
    home_def: Optional[float] = None
    away_def: Optional[float] = None

class LeagueConstants(BaseModel):
    """League-specific scaling constants for PPI calculation"""
    overall_c: Optional[float] = None
    home_c: Optional[float] = None
    away_c: Optional[float] = None
    overall_c_off: Optional[float] = None
    home_c_off: Optional[float] = None
    away_c_off: Optional[float] = None
    overall_c_def: Optional[float] = None
    home_c_def: Optional[float] = None
    away_c_def: Optional[float] = None


class LeagueResults(BaseModel):
    """Results for a single league"""
    league_constants: LeagueConstants
    teams: Dict[str, TeamPPI]
    calculation_info: Optional[Dict[str, Any]] = None


class PPIResponse(BaseModel):
    """Complete API response with results for all leagues"""
    success: bool
    message: str
    data: Optional[Dict[str, LeagueResults]] = None
    metadata: Optional[Dict[str, Any]] = None