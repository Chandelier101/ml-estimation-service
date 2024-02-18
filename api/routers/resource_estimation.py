from fastapi import APIRouter, HTTPException
from models import ResourceEstimationRequest
from services import res_estimator

router = APIRouter()

@router.post("/estimate-resource", response_model=dict)
async def estimate_resource(request: ResourceEstimationRequest):
    """Estimate resources and time for 1-batch based on the request."""
    try:
        estimation = res_estimator.estimate(request)
        return estimation
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
