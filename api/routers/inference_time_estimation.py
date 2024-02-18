from fastapi import APIRouter, HTTPException
from models import InferenceEstimationRequest
from services import inference_time_estimator

router = APIRouter()

@router.post("/estimate-inference-time", response_model=dict)
async def estimate_inference_time(request: InferenceEstimationRequest):
    """Estimate inference time based on the request."""
    try:
        estimation = inference_time_estimator.estimate(request)
        return estimation
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
