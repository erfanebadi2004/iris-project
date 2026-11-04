import pytest
from fastapi.testclient import TestClient
from model_serving import app

client = TestClient(app)

def test_prediction_model():
    payload = {"sample": [
        {"sepal_length":2.7, "sepal_width": 4.9, "petal_length": 7.3, "petal_width": 1.7}
    ]}
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200