"""
Initializes service layer by importing and instantiating the main service classes for resource, training,
and inference time estimation.
"""

from .resource_estimator import ResourceEstimator
from .training_estimator import TrainingTimeEstimator
from .inference_estimator import InferenceTimeEstimator

# Initializing the services
res_estimator = ResourceEstimator()
training_time_estimator = TrainingTimeEstimator()
inference_time_estimator = InferenceTimeEstimator()
