"""
Module defining the ResourceEstimator class for estimating resource usage during model training.
"""
import threading
import time
import torch
from torch import optim
from models.estimation_models import ResourceEstimationRequest
from models.pytorch_models import load_model
from utils.data_collector import collect_cpu_metrics, collect_gpu_metrics

class ResourceEstimator:
    def __init__(self) -> None:
        # Initialize estimator state
        self.metrics = {
            "Peak GPU Memory": None,
            "Peak GPU Utilization": 0,
            "Peak RAM Memory": 0,
            "Peak CPU Utilization": 0,
            "Batch Time": {'forward_pass': 0, 'backward_pass': 0},
            "RAM before training": 0,
            "VRAM before training": 0
        }
        self.use_threading = False
        self.monitoring = True
        self.model = None
        self.pretrained = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def monitor_resources(self, interval=1):
        while self.monitoring:
            # Get GPU metrics
            if self.device == 'cuda':
                total_gpu_load, total_gpu_memory = collect_gpu_metrics()
                self.metrics['Peak GPU Utilization'] = max(self.metrics['Peak GPU Utilization'], total_gpu_load)
                self.metrics['Peak GPU Memory'] = max(self.metrics['Peak GPU Memory'], total_gpu_memory)
            # Get CPU and RAM metrics
            cpu_load, cpu_memory = collect_cpu_metrics()
            self.metrics['Peak CPU Utilization'] = max(self.metrics['Peak CPU Utilization'], cpu_load)
            self.metrics['Peak RAM Memory'] = max(self.metrics['Peak RAM Memory'], cpu_memory)
            if self.use_threading:
                time.sleep(interval)
            else:
                break

    def estimate(self, request: ResourceEstimationRequest) -> dict:
        self.use_threading = request.threading
        self.metrics['RAM before training'] = collect_cpu_metrics()[1]
        if self.device == 'cuda':
            self.metrics['VRAM before training'] = collect_gpu_metrics()[1]
        self.model = load_model(request.architecture, pretrained=True).to(self.device)
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        batch_data = torch.rand(request.batch_size, 3, request.image_size[-1], request.image_size[-2]).to(self.device)
        self.model.eval()

        if 'inference' in request.application_type:
            labels = None
        else:
            dummy_input = torch.rand(1, 3, self.model.transform.min_size[0], 1).to(device=self.device)
            labels = self.model(dummy_input) * request.batch_size
            del dummy_input
            self.model.train()

        # Monitor with/without threading
        if self.use_threading:
            monitor_thread = threading.Thread(target=self.monitor_resources, args=(0.01,))
            monitor_thread.start()
        else:
            self.monitor_resources()

        # Simulate a single forward pass
        start_time = time.time()
        optimizer.zero_grad()
        outputs = self.model(batch_data, labels)
        self.metrics['Batch Time']['forward_pass'] = time.time() - start_time
        if not self.use_threading:
            self.monitor_resources()

        # Simulate a backward pass
        if 'training' in request.application_type:
            loss = sum(loss for loss in outputs.values())
            start_time = time.time()
            loss.backward()
            optimizer.step()
            self.metrics['Batch Time']['backward_pass'] = time.time() - start_time

        # Stop monitoring
        if self.use_threading:
            self.monitoring = False
            monitor_thread.join()
        else:
            self.monitor_resources()

        return self.metrics

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
            "VRAM before training": 0
        }
        if self.device == 'cuda':
            torch.cuda.synchronize()
        print("Resources and state have been cleaned up.")
