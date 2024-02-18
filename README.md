<hr>

# ML Estimation Service

## Overview
This project implements a microservice-based ML application aimed at estimating computing time and resource consumption for training or inference of deep learning models. Specifically, it focuses on Detection and Instance Segmentation models using PyTorch. The application provides estimates for CPU, physical memory, GPU memory, and GPU computation/utilization required for model training, as well as time estimates for training and inference processes.

## Features
- **Resource Estimation**: Given a deep learning model, its architecture type, number of parameters, application type, and dataset size, it estimates necessary resources such as CPU, memory, and GPU utilization.
- **Training Time Estimation**: Using the model details and dataset size, it estimates the time required to train the model.
- **Inference Time Estimation**: It estimates the time required to make a prediction on a given image using the specified model.

## System Design
The application is structured as a microservice, utilizing Python and FastAPI for its implementation. The design focuses on modularity, simplicity, and testability, ensuring high code quality and maintainability.
Every API call to the resource estimator service outputs the following:<br>
```
{"Peak GPU Memory",
"Peak GPU Utilization",
"Peak RAM Memory",
"Peak CPU Utilization",
"Batch Time": {'forward_pass', 'backward_pass'},
"RAM before training",
"VRAM before training"}
```
Recorded 1 time for a model architecture for batch_size
<br>
Every API call to the training time estimator returns the following:
```
{"Total Training Time"}
```
i.e the Total training time computed as (forward_pass + backward_pass) x num_batches x epochs. <br>
Every API call to the inference time estimator returns the following:
```
{"1-Image Inference Time",
"Batch of {batch_size} Inference Time"}
```

## API Endpoints
- /api/v1/estimate-resource: Estimates resources for training or inference.
- /api/v1/estimate-training-time: Estimates the training time for a given model.
- /api/v1/estimate-inference-time: Estimates the inference time for a given image.

# Getting Started
## Prerequisites
- Docker and Docker Compose (for containerized environment)
- Python 3.8+ (for local development)

### Installation
1. Clone the repository:
```
git clone <repository-url>
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Run the application
```
uvicorn api.main:app --reload
```
### Docker Setup
1. Build the Docker image:

2. Run using Docker Compose:


### Example Usage via curl
```
curl -X 'POST' \
  'http://localhost:8000/api/v1/estimate-resource' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task_type": "detection",
  "architecture": "fasterrcnn_resnet50_fpn",
  "parameters": 25000000,
  "dataset_size": 1000,
  "batch_size": 4,
  "image_size": [256, 256],
  "application_type": "training",
  "threading": false
  ...
}'
```
- **dataset_size** - count of samples in the training set
- **parameters** - total parameters of the model
- **batch_size** - total samples in a batch of images for training/inference
- **image_size** - height, width of the image
- **threading** - for continuous/fixed interval monitoring
- **application_type** - <training/inference>
- **task_type** - <detection/instance>

### Testing
The project includes unit tests for API and service layers. To run tests:
#### Test Estimator Service
```
python \path\to\project\tests\test_services\test_estimator_service.py
```
#### Test API:
```
python \path\to\project]tests\test_api\test_estimator_api.py --test_case_path \Optional\Path\to\test_cases.csv
```
### Limitations
- Only allows for step-wise hardware monitoring (i.e before training, at end of forward pass, backpropogation.etc) OR threaded fixed interval monitoring.
- Only Object Detection and Instance Segmentation models from PyTorch are supported.
  
### Future Scope
- Incorporating PyTorch Hooks for more efficient/granular monitoring than threading which may affect performance.
- Support for a more general span of models
