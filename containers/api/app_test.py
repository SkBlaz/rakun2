import pytest
from fastapi.testclient import TestClient
from fastapi import UploadFile
from main import app, PDFInput

client = TestClient(app)

def test_get_keywords():
    data = {
        "text": "Your sample text here.",
        "hyperparameters": {"num_keywords": 5, "alpha": 0.5}
    }
    response = client.post("/get_keywords/", json=data)
    assert response.status_code == 200
    data = response.json()
    assert "keywords" in data

if __name__ == "__main__":
    pytest.main()
