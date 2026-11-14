"""Общая схема признаков и загрузка собственных обученных артефактов."""

import math
from pathlib import Path

import joblib
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("area", "rooms", "floor", "total_floors")
MODELS = {
    "baseline_const": "models/baseline.joblib",
    "lin_reg": "models/linear.joblib",
    "selected": "models/selected.joblib",
}


def feature_frame(frame):
    """Никакие ID, цены, адреса или fold-метки не попадают в модель."""
    return frame.loc[:, list(FEATURES)].astype(float)


def request_frame(data):
    if not isinstance(data, dict) or set(data) != set(FEATURES):
        raise ValueError("Нужен JSON-объект с area, rooms, floor, total_floors")
    values = {}
    for field in FEATURES:
        raw = data[field]
        if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
            raise ValueError(f"{field}: требуется число")
        try:
            value = float(raw)
        except (ValueError, OverflowError):
            raise ValueError(f"{field}: требуется число") from None
        if not math.isfinite(value):
            raise ValueError(f"{field}: требуется конечное число")
        if field != "area" and not value.is_integer():
            raise ValueError(f"{field}: требуется целое число")
        if value < (0 if field == "rooms" else 1) and field != "area":
            raise ValueError(f"{field}: недопустимое значение")
        if field == "area" and value <= 0:
            raise ValueError("area должна быть положительной")
        values[field] = value
    if values["floor"] > values["total_floors"]:
        raise ValueError("Этаж квартиры выше этажности дома")
    return feature_frame(pd.DataFrame([values]))


def save_model(estimator, name, path, data_hash):
    bundle = {
        "schema_version": 1,
        "features": list(FEATURES),
        "sklearn_version": sklearn.__version__,
        "model_name": name,
        "data_dvc_md5": data_hash,
        "estimator": estimator,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model(name):
    path = Path(MODELS.get(str(name), name))
    if not path.is_absolute():
        path = ROOT / path
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise ValueError("Артефакт модели имеет неподдерживаемый формат")
    if bundle.get("features") != list(FEATURES):
        raise ValueError("Схема признаков артефакта не совпадает с API")
    if bundle.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Версия scikit-learn не совпадает с окружением обучения")
    estimator = bundle.get("estimator")
    if not callable(getattr(estimator, "predict", None)):
        raise ValueError("В артефакте отсутствует predict")
    if list(getattr(estimator, "feature_names_in_", [])) != list(FEATURES):
        raise ValueError("Обученная модель имеет другую схему признаков")
    return bundle
