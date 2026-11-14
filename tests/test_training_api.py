import copy

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from service.app import create_app
from service.predict import FEATURES, feature_frame, load_model, save_model
from train import evaluate, select_model, validate_split

PAYLOAD = {"area": 45.5, "rooms": 2, "floor": 3, "total_floors": 9}
HEADERS = {"Authorization": "Bearer unit-test-token"}


@pytest.fixture
def bundle_path(tmp_path):
    frame = pd.DataFrame([{**PAYLOAD, "area": area} for area in [30, 45, 60]])
    model = DummyRegressor(strategy="median").fit(feature_frame(frame), [10_000_000, 20_000_000, 100_000_000])
    path = tmp_path / "model.joblib"
    save_model(model, "baseline", path, "test-data-hash")
    return path


@pytest.fixture
def client(bundle_path):
    app = create_app({"TESTING": True, "MODEL_NAME": str(bundle_path), "APP_TOKEN": "unit-test-token"})
    return app.test_client()


def test_model_roundtrip_uses_train_median(bundle_path):
    bundle = load_model(bundle_path)
    assert bundle["features"] == list(FEATURES)
    assert bundle["estimator"].predict(feature_frame(pd.DataFrame([PAYLOAD])))[0] == 20_000_000


def test_old_feature_schema_is_rejected(bundle_path):
    bundle = joblib.load(bundle_path)
    bundle["features"] = ["total_meters", "rooms_1"]
    joblib.dump(bundle, bundle_path)
    with pytest.raises(ValueError, match="Схема"):
        load_model(bundle_path)


def test_serialized_estimator_schema_is_checked(bundle_path):
    bundle = joblib.load(bundle_path)
    bundle["estimator"].feature_names_in_ = np.array(["wrong"] * 4)
    joblib.dump(bundle, bundle_path)
    with pytest.raises(ValueError, match="Обученная модель"):
        load_model(bundle_path)


def test_api_returns_model_prediction_and_readiness(client):
    response = client.post("/api/numbers", json=PAYLOAD, headers=HEADERS)
    assert response.status_code == 200
    assert response.json == {"status": "success", "data": 20_000_000}
    assert client.get("/health").json == {"status": "ready", "model": "baseline"}
    assert client.get("/").status_code == 200
    # Значения существующей HTML-формы приходят строками.
    strings = {key: str(value) for key, value in PAYLOAD.items()}
    assert client.post("/api/numbers", json=strings, headers=HEADERS).json == response.json


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Bearer неверный"}])
def test_authentication_error_is_json(client, headers):
    response = client.post("/api/numbers", json=PAYLOAD, headers=headers)
    assert response.status_code == 401
    assert response.json["status"] == "error"


@pytest.mark.parametrize("payload", [
    None, [], "text", {}, {"area": 30},
    {**PAYLOAD, "area": -1}, {**PAYLOAD, "area": "NaN"},
    {**PAYLOAD, "area": "Infinity"}, {**PAYLOAD, "area": True},
    {**PAYLOAD, "rooms": 1.5}, {**PAYLOAD, "rooms": -1},
    {**PAYLOAD, "floor": 10}, {**PAYLOAD, "floor": 0},
    {**PAYLOAD, "total_floors": None}, {**PAYLOAD, "price": 100},
])
def test_invalid_payloads_are_400(client, payload):
    import json
    response = client.post("/api/numbers", data=json.dumps(payload), content_type="application/json", headers=HEADERS)
    assert response.status_code == 400
    assert response.json["status"] == "error"


def test_malformed_json_and_wrong_content_type(client):
    assert client.post("/api/numbers", data="{", content_type="application/json", headers=HEADERS).status_code == 400
    assert client.post("/api/numbers", data="text", headers=HEADERS).status_code == 415


def test_missing_model_is_not_ready(tmp_path):
    app = create_app({"TESTING": True, "MODEL_NAME": str(tmp_path / "missing.joblib"), "APP_TOKEN": "unit-test-token"})
    with app.test_client() as client:
        assert client.get("/health").status_code == 503
        assert client.post("/api/numbers", json=PAYLOAD, headers=HEADERS).status_code == 503


@pytest.mark.parametrize("bad_prediction", [-1, float("inf"), float("nan")])
def test_invalid_model_output_is_503(client, bad_prediction):
    class BrokenModel:
        def predict(self, frame):
            return [bad_prediction]
    client.application.config["MODEL_BUNDLE"]["estimator"] = BrokenModel()
    assert client.post("/api/numbers", json=PAYLOAD, headers=HEADERS).status_code == 503


def synthetic_split():
    train = pd.DataFrame([
        {**PAYLOAD, "area": 30 + i, "price": 1_000_000 + 100_000 * i,
         "id": str(i), "address": f"дом {i // 2}", "cv_fold": i // 2}
        for i in range(10)
    ])
    test = pd.DataFrame([{**PAYLOAD, "price": 50_000_000, "id": "100", "address": "другой дом"}])
    return train, test


def test_baseline_cv_fits_each_fold_without_holdout():
    train, test = synthetic_split()
    validate_split(train, test, 5)
    model, result, prediction = evaluate("baseline", train, test, 5)
    assert prediction[0] == train.price.median()
    for entry in result["folds"]:
        fit = train[train.cv_fold != entry["fold"]]
        val = train[train.cv_fold == entry["fold"]]
        assert entry["mae_rub"] == np.abs(val.price - fit.price.median()).mean()
    changed = test.assign(price=1)
    again, _, _ = evaluate("baseline", train, changed, 5)
    assert again.constant_[0][0] == model.constant_[0][0]


def test_training_rejects_house_leakage():
    train, test = synthetic_split()
    test["address"] = train.address.iloc[0]
    with pytest.raises(ValueError, match="Утечка"):
        validate_split(train, test, 5)
    test["address"] = "другой дом"
    train.loc[1, "cv_fold"] = 1
    with pytest.raises(ValueError, match="разные validation"):
        validate_split(train, test, 5)


def test_selection_requires_cv_and_holdout_improvement():
    result = {
        "baseline": {"cv_mae_mean_rub": 10, "holdout_mae_rub": 10},
        "linear": {"cv_mae_mean_rub": 5, "holdout_mae_rub": 5, "positive_finite_predictions": True},
    }
    assert select_model(result) == "linear"
    for change in [
        {"cv_mae_mean_rub": 11}, {"holdout_mae_rub": 11},
        {"positive_finite_predictions": False},
    ]:
        modified = copy.deepcopy(result)
        modified["linear"].update(change)
        assert select_model(modified) == "baseline"
