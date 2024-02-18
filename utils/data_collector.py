import psutil
from GPUtil import getGPUs

def collect_gpu_metrics():
    """
    Collects and calculates the total GPU load and memory usage from all available GPUs.

    Returns:
        tuple: A tuple containing two elements:
            - total_gpu_load (float): The average load across all GPUs, expressed as a percentage.
            - total_gpu_memory (float): The total memory used across all GPUs, in MB.
    """
    gpus = getGPUs()
    total_gpu_load, total_gpu_memory = 0, 0

    for gpu in gpus:
        total_gpu_load += gpu.load
        total_gpu_memory += gpu.memoryUsed

    if gpus:
        total_gpu_load = (total_gpu_load / len(gpus)) * 100  # Convert to percentage
    else:
        total_gpu_load = 0  # Default to 0 if no GPUs found

    return total_gpu_load, total_gpu_memory

def collect_cpu_metrics():
    """
    Collects CPU utilization percentage and total used virtual memory.

    Returns:
        tuple: A tuple containing two elements:
            - cpu_percent (float): The percentage of CPU utilization.
            - used_memory (float): The amount of used virtual memory in bytes.
    """
    cpu_percent = psutil.cpu_percent()
    used_memory = psutil.virtual_memory().used
    return cpu_percent, used_memory
