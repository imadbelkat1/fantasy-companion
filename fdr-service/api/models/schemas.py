from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Explainer(BaseModel):
    """Feature explanation model"""
    feature: str
    impact: float
    description: str

class TeamRating(BaseModel):
    """Team rating response model"""
    team: str
    league: str
    rating: float = Field(..., ge=0, le=100, description="Overall team rating (0-100)")
    rating_low: Optional[float] = Field(None, ge=0, le=100, description="Lower bound of rating confidence interval")
    rating_high: Optional[float] = Field(None, ge=0, le=100, description="Upper bound of rating confidence interval")
    attack_rating: Optional[float] = Field(None, ge=0, le=100, description="Attack-specific rating")
    defense_rating: Optional[float] = Field(None, ge=0, le=100, description="Defense-specific rating")
    form_rating: Optional[float] = Field(None, ge=0, le=100, description="Recent form rating")
    raw_model_rating: Optional[float] = Field(None, description="Raw model prediction before adjustments")
    prior_anchor: Optional[float] = Field(None, description="Historical performance anchor")
    expected_goal_diff_vs_avg: Optional[float] = Field(None, description="Expected goal difference vs league average opponent")
    win_prob_vs_avg: Optional[float] = Field(None, ge=0, le=1, description="Win probability vs league average opponent")
    explainers: Optional[List[Explainer]] = Field(None, description="Top features driving the rating")
    confidence_level: Optional[str] = Field(None, description="Confidence level: High/Medium/Low")
    plain_language_summary: Optional[str] = Field(None, description="Plain language explanation")
    uncertainty: Optional[float] = Field(None, description="Rating uncertainty estimate")

class FixtureDifficulty(BaseModel):
    """Fixture difficulty response model"""
    fixture_id: str
    league: str
    date: Optional[str] = None
    gameweek: Optional[int] = None
    home_team: str
    away_team: str
    predicted_goal_diff_home: Optional[float] = Field(None, description="Predicted goal difference from home perspective")
    win_prob_home: Optional[float] = Field(None, ge=0, le=1, description="Home team win probability")
    draw_prob: Optional[float] = Field(None, ge=0, le=1, description="Draw probability")
    win_prob_away: Optional[float] = Field(None, ge=0, le=1, description="Away team win probability")
    fixture_difficulty_home: Optional[float] = Field(None, ge=1, le=5, description="Difficulty for home team (1=easy, 5=hard)")
    fixture_difficulty_away: Optional[float] = Field(None, ge=1, le=5, description="Difficulty for away team (1=easy, 5=hard)")
    home_xg_predicted: Optional[float] = Field(None, description="Predicted home team xG")
    away_xg_predicted: Optional[float] = Field(None, description="Predicted away team xG")
    explainers: Optional[List[Explainer]] = Field(None, description="Top features driving the prediction")

class TeamRatingResponse(BaseModel):
    """Response wrapper for team ratings"""
    teams: List[TeamRating]
    total_count: int
    league: str

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    status_code: int