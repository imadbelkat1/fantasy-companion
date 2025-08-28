"""
FastAPI routes for Power Index Service
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import json
import os
import sys

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from power_index_calculator.models import PPIResponse
from power_index_calculator.data_loader import load_fbref_data, extract_league_data
from power_index_calculator.calculator import calculate_all_leagues_ppi

app = FastAPI(
    title="Power Index Service",
    description="Calculate Power Performance Index (PPI) for football teams",
    version="1.0.0"
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Power Index Service is running"}


@app.get("/calculate-ppi", response_model=PPIResponse)
async def calculate_power_index():
    """
    Calculate Power Performance Index for all teams in all leagues

    Returns:
        PPIResponse: Complete PPI results for all leagues
    """
    try:
        # Load data
        raw_data = load_fbref_data()

        # Extract league data
        leagues_data = extract_league_data(raw_data)

        if not leagues_data:
            return PPIResponse(
                success=False,
                message="No valid league data found in the dataset",
                data=None
            )

        # Calculate PPI for all leagues
        results = calculate_all_leagues_ppi(leagues_data)

        if not results:
            return PPIResponse(
                success=False,
                message="Could not calculate PPI for any leagues",
                data=None
            )

        return PPIResponse(
            success=True,
            message=f"Successfully calculated triple metrics (PPI, OFF, DEF) for {len(results)} league(s)",
            data=results
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {str(e)}"
        )

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON data: {str(e)}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/calculate-ppi/{league}")
async def calculate_league_power_index(league: str):
    """
    Calculate Power Performance Index for a specific league

    Args:
        league: League name (e.g., 'epl', 'laliga')

    Returns:
        League-specific PPI results
    """
    try:
        # Load data
        raw_data = load_fbref_data()

        # Extract league data
        leagues_data = extract_league_data(raw_data)

        if league not in leagues_data:
            available_leagues = list(leagues_data.keys())
            raise HTTPException(
                status_code=404,
                detail=f"League '{league}' not found. Available leagues: {available_leagues}"
            )

        # Calculate PPI for specific league
        from power_index_calculator.calculator import calculate_league_ppi
        result = calculate_league_ppi(leagues_data[league])

        return PPIResponse(
            success=True,
            message=f"Successfully calculated triple metrics (PPI, OFF, DEF) for {league}",
            data={league: result}
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {str(e)}"
        )

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON data: {str(e)}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)