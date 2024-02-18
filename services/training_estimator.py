"""
Module defining the TrainingTimeEstimator class for estimating training time based on resource usage and model details.
"""
import math
from models.estimation_models import TrainingEstimationRequest

class TrainingTimeEstimator:
    def __init__(self) -> None:
        self.training_time = 0

    def estimate(self, request: TrainingEstimationRequest) -> dict:
        """Estimate the total training time based on the request details."""
        resource_estimation_results = request.resource_estimation_results['Batch Time']

        # Training Time (Forward Pass Time + Backward Pass Time)
        estimated_training_time = sum(resource_estimation_results[i] for i in resource_estimation_results)

        # Total Number of Batches
        num_batches = math.ceil(request.resource_estimation.dataset_size/request.resource_estimation.batch_size)

        # Total Training Time (Training Time per batch * Number of Batches * Number of epochs)
        self.training_time = estimated_training_time * num_batches * request.epochs

        return {"Total Training Time": self.training_time}
