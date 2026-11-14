"""Гистограммы признаков и PSI для ограниченного окна запросов."""

import numpy as np

DRIFT_FEATURES = ("area", "rooms")


def histogram(values, edges):
    bins = np.searchsorted(edges, np.asarray(values, dtype=float), side="right")
    return np.bincount(bins, minlength=len(edges) + 1).tolist()


def build_reference(frame, data_hash, model_sha256):
    if frame.empty:
        raise ValueError("Нельзя строить reference по пустой выборке")
    area_edges = np.unique(np.quantile(frame["area"], [0.2, 0.4, 0.6, 0.8])).tolist()
    features = {}
    for field, edges in (("area", area_edges), ("rooms", [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])):
        values = frame[field].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Reference содержит неконечные признаки")
        features[field] = {"edges": edges, "counts": histogram(values, edges)}
    return {
        "schema_version": 1, "data_dvc_md5": data_hash, "model_sha256": model_sha256,
        "training_rows": len(frame), "features": features,
    }


def validate_reference(reference, data_hash, model_sha256):
    if not isinstance(reference, dict):
        raise ValueError("Reference должен быть JSON-объектом")
    if (
        reference.get("schema_version") != 1 or reference.get("data_dvc_md5") != data_hash
        or reference.get("model_sha256") != model_sha256
    ):
        raise ValueError("Reference не соответствует данным модели")
    if set(reference.get("features", {})) != set(DRIFT_FEATURES):
        raise ValueError("Неизвестные признаки reference")
    for spec in reference["features"].values():
        edges, counts = np.asarray(spec["edges"]), np.asarray(spec["counts"])
        if (
            edges.ndim != 1 or counts.ndim != 1
            or not np.isfinite(edges).all() or not np.isfinite(counts).all()
            or len(counts) != len(edges) + 1 or (np.diff(edges) <= 0).any()
            or (counts < 0).any() or counts.sum() <= 0
            or counts.sum() != reference["training_rows"]
        ):
            raise ValueError("Некорректная гистограмма reference")


def population_stability_index(reference_counts, current_counts):
    # Полунаблюдение в каждом бакете, чтобы корректно обработать нулевые частоты.
    reference = np.asarray(reference_counts, dtype=float) + 0.5
    current = np.asarray(current_counts, dtype=float) + 0.5
    reference /= reference.sum()
    current /= current.sum()
    return float(np.sum((current - reference) * np.log(current / reference)))
