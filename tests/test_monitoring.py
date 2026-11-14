import json

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from service.app import create_app
from service.drift import (
    build_reference,
    histogram,
    population_stability_index,
    validate_reference,
)
from service.predict import feature_frame, load_model, save_model

PAYLOAD = {"area": 45.5, "rooms": 2, "floor": 3, "total_floors": 9}


def test_histogram_keeps_values_outside_training_range():
    assert histogram([1, 10, 11, 20, 21, 1000], [10, 20]) == [1, 2, 3]


def test_psi_handles_empty_bins_and_distribution_shift():
    assert population_stability_index([10, 0, 20], [10, 0, 20]) == pytest.approx(0)
    assert population_stability_index([10, 0, 20], [0, 30, 0]) > 1


def test_reference_checks_data_and_model_versions():
    frame = pd.DataFrame([PAYLOAD])
    reference = build_reference(frame, "data", "model")
    validate_reference(reference, "data", "model")
    with pytest.raises(ValueError, match="не соответствует"):
        validate_reference(reference, "data", "other-model")
    reference["features"]["area"]["counts"] = [-1, 2]
    with pytest.raises(ValueError, match="гистограмма"):
        validate_reference(reference, "data", "model")


def test_reference_must_be_an_object():
    with pytest.raises(ValueError, match="JSON-объектом"):
        validate_reference(None, "data", "model")


@pytest.fixture
def monitored_app(tmp_path):
    frame = pd.DataFrame([{**PAYLOAD, "area": area} for area in [30, 45, 60]])
    model = DummyRegressor(strategy="median").fit(feature_frame(frame), [10_000_000, 20_000_000, 30_000_000])
    path = tmp_path / "model.joblib"
    save_model(model, "baseline", path, "test-data")
    bundle = load_model(path)
    reference = build_reference(frame, "test-data", bundle["artifact_sha256"])
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    return create_app({
        "TESTING": True, "MODEL_NAME": str(path), "APP_TOKEN": "monitor-test-token",
        "DRIFT_REFERENCE_PATH": str(reference_path), "DRIFT_WINDOW_SIZE": 4, "DRIFT_MIN_SAMPLES": 2,
    })


def test_only_successes_enter_bounded_drift_window(monitored_app):
    client = monitored_app.test_client()
    metrics = monitored_app.extensions["housing_monitoring"]
    registry = metrics.registry
    headers = {"Authorization": "Bearer monitor-test-token"}
    assert registry.get_sample_value("housing_drift_ready") == 0
    client.post("/api/numbers", json=PAYLOAD, headers=headers)
    assert registry.get_sample_value("housing_drift_ready") == 0
    client.post("/api/numbers", json=PAYLOAD, headers=headers)
    assert registry.get_sample_value("housing_drift_ready") == 1
    client.post("/api/numbers", json=PAYLOAD)  # 401 не входит в распределение.
    client.post("/api/numbers", json={**PAYLOAD, "area": -1}, headers=headers)
    assert registry.get_sample_value("housing_drift_window_samples") == 2
    for _ in range(5):
        client.post("/api/numbers", json=PAYLOAD, headers=headers)
    assert registry.get_sample_value("housing_drift_window_samples") == 4
    assert len(metrics.window) == 4
    assert registry.get_sample_value("housing_feature_psi", {"feature": "area"}) >= 0


def test_metrics_include_errors_and_latency_without_secrets(monitored_app):
    client = monitored_app.test_client()
    client.post("/api/numbers", json=PAYLOAD)
    client.get("/unknown-secret-path")
    client.get("/another-path")
    response = client.get("/metrics")
    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "housing_http_requests_total" in text
    assert 'status="401"' in text
    assert "housing_http_request_duration_seconds_bucket" in text
    assert "monitor-test-token" not in text
    assert "unknown-secret-path" not in text
    registry = monitored_app.extensions["housing_monitoring"].registry
    assert registry.get_sample_value("housing_http_requests_total", {"method": "GET", "route": "unmatched", "status": "404"}) == 2
    assert registry.get_sample_value("housing_http_requests_total", {"method": "GET", "route": "/metrics", "status": "200"}) is None


def test_each_app_has_an_independent_registry(monitored_app):
    another = create_app({"MODEL_NAME": "missing-model-for-test.joblib"})
    assert another.extensions["housing_monitoring"].registry is not monitored_app.extensions["housing_monitoring"].registry


def test_mismatching_reference_disables_drift(monitored_app, tmp_path):
    path = tmp_path / "mismatch.json"
    reference = monitored_app.extensions["housing_monitoring"].reference.copy()
    reference["model_sha256"] = "not-current-model"
    path.write_text(json.dumps(reference))
    app = create_app({**monitored_app.config, "DRIFT_REFERENCE_PATH": str(path)})
    assert app.extensions["housing_monitoring"].reference is None
    assert app.test_client().get("/health").status_code == 200
