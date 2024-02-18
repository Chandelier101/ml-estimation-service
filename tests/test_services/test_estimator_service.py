import warnings
from models import ResourceEstimationRequest, InferenceEstimationRequest, TrainingEstimationRequest
from services.resource_estimator import ResourceEstimator
from services.training_estimator import TrainingTimeEstimator
from services.inference_estimator import InferenceTimeEstimator

warnings.filterwarnings("ignore")

def test_data():
    """
    Retrieve test cases from a given path or return default test cases if no path is provided.
    """
    return [{'test_case':{
    "task_type": "detection",
    "architecture": "fasterrcnn_resnet50_fpn",
    "parameters": 25000000,
    "dataset_size": 1000,
    "batch_size": 2,
    "image_size": (224, 224),
    "application_type": "training",
    "threading":False}, 'num_samples':1}]

def run_tests(case_data, num_samples=1):
    """
    Run Test Cases num_sample times
    """
    print('\n\n')
    print("####"*30)
    print(case_data)
    print("####"*30)
    train_logs = {}
    estimator = ResourceEstimator()
    training_estimator = TrainingTimeEstimator()
    inference_estimator = InferenceTimeEstimator()
    # Instantiate the Pydantic model with test data
    request = ResourceEstimationRequest(**case_data)
    for _ in range(num_samples):
        # Use estimators directly instead of test functions
        results = estimator.estimate(request)
        estimator.cleanup()
        train_request = TrainingEstimationRequest(epochs=5, resource_estimation_results=results, resource_estimation=request)
        training_results = training_estimator.estimate(train_request)
        # Update results and logs
        results.update(training_results)
        for metric, value in results.items():
            if metric in train_logs:
                train_logs[metric].append(value)
            else:
                train_logs[metric] = [value]
    print('Train logging complete')
    test_logs = {}
    case_data["application_type"] = "inference"
    request = ResourceEstimationRequest(**case_data)
    for _ in range(num_samples):
        # Use estimators directly instead of test functions
        results = estimator.estimate(request)
        estimator.cleanup()

        inference_request = InferenceEstimationRequest(resource_estimation_results=results, resource_estimation=request, threading=False)
        inference_results = inference_estimator.estimate(inference_request)
        inference_estimator.cleanup()

        # Update results and logs
        results.update(inference_results)
        for metric, value in results.items():
            if metric in test_logs:
                test_logs[metric].append(value)
            else:
                test_logs[metric] = [value]

    print('Test logging complete')
    print('####' * 20)
    print(train_logs)
    print('Test Logs')
    print('####' * 20)
    print(test_logs)

    assert len(train_logs.keys()) > 0, "Training logs should not be empty"
    assert len(test_logs.keys()) > 0, "Test logs should not be empty"
    return {"TRAIN": train_logs, "TEST": test_logs}

for case in test_data():
    result = run_tests(case['test_case'], num_samples=case['num_samples'])
    print(result)
