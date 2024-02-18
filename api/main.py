"""
Main module for FastAPI application, defining and setting up the application instance and its routes.
"""

from fastapi import FastAPI
from .routers import resource_estimation_router, training_time_estimation_router, inference_time_estimation_router

app = FastAPI(title="ML Estimation Service")

app.include_router(resource_estimation_router, tags=["Resource Estimation"], prefix="/api/v1")
app.include_router(training_time_estimation_router, tags=["Training Time Estimation"], prefix="/api/v1")
app.include_router(inference_time_estimation_router, tags=["Inference Time Estimation"], prefix="/api/v1")
