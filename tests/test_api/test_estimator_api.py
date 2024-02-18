"""
Module to test resource estimation API with various test cases.
"""
import argparse
import ast
import asyncio
from httpx import AsyncClient
import pandas as pd
from api.main import app

def get_test_cases(path=None):
    """
    Retrieve test cases from a given path or return default test cases if no path is provided.
    """
    if path is None:
        return [{
            "task_type": "detection",
            "architecture": "fasterrcnn_resnet50_fpn",
            "parameters": 25000000,
            "dataset_size": 1000,
            "batch_size": 2,
            "image_size": (800, 1),
            "application_type": "training",
            "epochs": 20,
            "threading": True,
            "num_samples": 2
        }]
    test_cases = []
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        test_case = row.to_dict()
        test_case['image_size'] = ast.literal_eval(test_case['image_size'])
        test_case['dataset_size'] = int(test_case['dataset_size'])
        test_case['batch_size'] = int(test_case['batch_size'])
        test_case['parameters'] = int(test_case['parameters'])
        test_case['epochs'] = int(test_case['epochs'])
        test_case['num_samples'] = int(test_case['num_samples'])
        test_cases.append(test_case)
    return test_cases

async def test_resource_estimation(test_case):
    """
    Test resource estimation for a given test case.
    """
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/v1/estimate-resource", json=test_case)
    return response.json()

async def test_training_estimation(test_case):
    """
    Test training estimation for a given test case.
    """
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/v1/estimate-training-time", json=test_case)
    return response.json()

async def test_inference_estimation(test_case):
    """
    Test inference estimation for a given test case.
    """
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/v1/estimate-inference-time", json=test_case)
    return response.json()

async def run_cases(local_test_cases):
    """
    Run a series of test cases for resource, training, and inference estimation.
    """
    for idx, test_case in enumerate(local_test_cases):
        print('+=+'*50)
        print('Test Case:', idx)
        print('+=+'*50)
        for _ in range(test_case['num_samples']):
            resource_estimation_input = {k: v for k, v in test_case.items() if k not in ('epochs', 'num_samples')}
            resource_estimation_result = await test_resource_estimation(resource_estimation_input)

            training_estimation_input = {'epochs': test_case['epochs'], 'resource_estimation_results': resource_estimation_result, 'resource_estimation': resource_estimation_input}
            inference_estimation_input = {'threading': test_case['threading'], 'resource_estimation_results': resource_estimation_result, 'resource_estimation': resource_estimation_input}

            training_estimation_result = await test_training_estimation(training_estimation_input)
            inference_estimation_result = await test_inference_estimation(inference_estimation_input)
            print('#'*50)
            print(resource_estimation_result)
            print('#'*50)
            print(inference_estimation_result)
            print('#'*50)
            print(training_estimation_result)
            print('\n\n')
    print('Test Successful')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Case Arguments')
    parser.add_argument('--test_case_path', type=str, default=None, help='Pass Test Case Path')
    args = parser.parse_args()
    global_test_cases = get_test_cases(args.test_case_path)
    asyncio.run(run_cases(global_test_cases))
