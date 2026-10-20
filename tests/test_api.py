from fastapi.testclient import TestClient

from heart_ml import api


def test_root_and_health_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(api, "_model", None)
    client = TestClient(api.app)
    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_clear_503_when_model_not_trained(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(api, "_model", None)
    payload = {
        "age": 57, "sex": 1, "cp": 2, "trestbps": 130, "chol": 236,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.0,
        "slope": 1, "ca": 0, "thal": 2,
    }
    response = TestClient(api.app).post("/predict", json=payload)
    assert response.status_code == 503
