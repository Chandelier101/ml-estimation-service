"""
Defines Pydantic models for various estimation request types within the application, specifying validation and metadata.
"""
from typing import Optional, Tuple
from pydantic import BaseModel, Field


class ResourceEstimationRequest(BaseModel):
    """
    Request model for resource estimation, containing details about the ML model and its training or inference setup.
    """
    task_type: str = Field(..., description="The type of the deep learning model, e.g., 'detection', 'segmentation'")
    architecture: str = Field(..., description="Model architecture, e.g., 'resnet50', 'fcn_resnet50'")
    parameters: int = Field(..., description="Number of parameters in the model")
    dataset_size: int = Field(..., description="Size of the dataset in number of images")
    batch_size: int = Field(..., description="Batch size used for training")
    image_size: Tuple[int, int] = Field(..., description="Size of the image for inference in pixels (width, height)")
    application_type: str = Field(..., description="Application type, e.g., 'training', 'inference'")
    threading: bool = Field(..., description="Use threading to monitor at fixed intervals")

class TrainingEstimationRequest(BaseModel):
    """
    Request model for training time estimation, including epochs and resource estimation results.
    """
    epochs: int = Field(..., description="Number of epochs for training")
    resource_estimation_results: dict = Field(..., description="Results from Resource Estimation")
    resource_estimation: ResourceEstimationRequest = Field(..., description="Resource estimation details")

class InferenceEstimationRequest(BaseModel):
    """
    Request model for inference time estimation, optionally including resource estimation details.
    """
    resource_estimation_results: dict = Field(..., description="Results from Resource Estimation")
    resource_estimation: Optional[ResourceEstimationRequest] = Field(None, description="Optional resource estimation details")
    threading: bool = Field(..., description="Use threading to monitor at fixed intervals")
