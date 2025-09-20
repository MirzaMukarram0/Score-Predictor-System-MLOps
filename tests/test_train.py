# tests/test_server.py
import os
import sys
import pickle
import importlib

import numpy as np
from sklearn.linear_model import LinearRegression


def _dump_dummy_model(tmp_path):
    """Train a tiny model and pickle it to tmp_path/student_model.pkl."""
    X = np.array(
        [
            [1, 60, 50, 0],
            [2, 65, 55, 1],
            [3, 70, 60, 1],
            [4, 75, 65, 2],
        ]
    )
    y = np.array([52, 58, 61, 68])
    model = LinearRegression().fit(X, y)
    p = tmp_path / "student_model.pkl"
    with open(p, "wb") as f:
        pickle.dump(model, f)
    return str(p)


def _reload_server_with_model_path(model_path):
    """Set MODEL_PATH env var before importing app.server.

    This ensures the server module loads the test model instead of any
    committed model on disk.
    """
    os.environ["MODEL_PATH"] = model_path
    # remove loaded module if present so importlib reloads it fresh
    sys.modules.pop("app.server", None)
    return importlib.import_module("app.server")


def test_predict_success(tmp_path):
    model_path = _dump_dummy_model(tmp_path)
    server = _reload_server_with_model_path(model_path)
    client = server.app.test_client()

    # valid features -> should return 200 and a numeric prediction
    payload = {"features": [4, 75, 65, 2]}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)


def test_predict_bad_input(tmp_path):
    model_path = _dump_dummy_model(tmp_path)
    server = _reload_server_with_model_path(model_path)
    client = server.app.test_client()

    # missing "features" -> 400
    resp = client.post(
        "/predict",
        json={"foo": "bar"},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data
