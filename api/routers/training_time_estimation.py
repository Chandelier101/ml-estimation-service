from fastapi import APIRouter, HTTPException
from models import TrainingEstimationRequest
from services import training_time_estimator

router = APIRouter()

@router.post("/estimate-training-time", response_model=dict)
async def estimate_training_time(request: TrainingEstimationRequest):
    """Estimate training time based on the request."""
    try:
        estimation = training_time_estimator.estimate(request)
        return estimation
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
