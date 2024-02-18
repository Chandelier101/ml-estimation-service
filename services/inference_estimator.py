"""
Module defining the InferenceTimeEstimator class, responsible for estimating inference time and monitoring resource usage.
"""
import time
import torch
from utils.data_collector import collect_cpu_metrics, collect_gpu_metrics
from models.pytorch_models import load_model
from models.estimation_models import InferenceEstimationRequest

class InferenceTimeEstimator:
    def __init__(self) -> None:
        # Initialize estimator state
        self.metrics = {
            "Peak GPU Memory": None,
            "Peak GPU Utilization": 0,
            "Peak RAM Memory": 0,
            "Peak CPU Utilization": 0,
            "Batch Time": {'forward_pass': 0, 'backward_pass': 0},
            "RAM before training": 0,
            "VRAM before training": 0}

        self.inference_time_seconds = 0
        self.monitoring = True
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def monitor_resources(self):
        # Get GPU metrics
        if self.device == 'cuda':
            total_gpu_load, total_gpu_memory = collect_gpu_metrics()
            self.metrics['Peak GPU Utilization'] = max(self.metrics['Peak GPU Utilization'], total_gpu_load)
            self.metrics['Peak GPU Memory'] = max(self.metrics['Peak GPU Memory'], total_gpu_memory)
        # Get CPU and RAM metrics
        cpu_load, cpu_memory = collect_cpu_metrics()
        self.metrics['Peak CPU Utilization'] = max(self.metrics['Peak CPU Utilization'], cpu_load)
        self.metrics['Peak RAM Memory'] = max(self.metrics['Peak RAM Memory'], cpu_memory)

    def estimate(self, request: InferenceEstimationRequest) -> dict:

        self.metrics['RAM before training'] = collect_cpu_metrics()[1]

        if self.device=='cuda':
            self.metrics['VRAM before training'] = collect_gpu_metrics()[1]
        self.model = load_model(request.resource_estimation.architecture).to(device=self.device)

        image = torch.rand(1, 3, request.resource_estimation.image_size[-2], request.resource_estimation.image_size[-1]).to(device=self.device)
        self.model.eval()
        self.monitor_resources()

        # Simulate a forward pass
        start_time = time.time()
        self.model(image)
        self.inference_time_seconds = time.time() - start_time
        self.monitor_resources()

        return {"1-Image Inference Time":
                self.inference_time_seconds,
                f"Batch of {request.resource_estimation.batch_size} Inference Time":
                request.resource_estimation_results['Batch Time']}
    def cleanup(self):
        # Clear model and reset state
        self.model = None
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        self.metrics = {
            "Peak GPU Memory": None,
            "Peak GPU Utilization": 0,
            "Peak RAM Memory": 0,
            "Peak CPU Utilization": 0,
            "Batch Time": {'forward_pass': 0, 'backward_pass': 0},
            "RAM before training": 0,
            "VRAM before training": 0}
        if self.device == 'cuda':
            torch.cuda.synchronize()
        print("Resources and state have been cleaned up.")
