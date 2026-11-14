"""Метрики HTTP и окно валидных прогнозов без хранения токенов/адресов."""

import json
import threading
import time
from collections import deque
from pathlib import Path

from flask import Response, g, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

try:
    from .drift import (
        DRIFT_FEATURES,
        histogram,
        population_stability_index,
        validate_reference,
    )
except ImportError:
    from drift import (
        DRIFT_FEATURES,
        histogram,
        population_stability_index,
        validate_reference,
    )


class Monitoring:
    def __init__(self, reference=None, window_size=200, min_samples=30):
        if not 1 <= min_samples <= window_size:
            raise ValueError("Нужно 1 <= DRIFT_MIN_SAMPLES <= DRIFT_WINDOW_SIZE")
        self.registry = CollectorRegistry()
        self.requests = Counter("housing_http_requests", "HTTP responses", ["method", "route", "status"], registry=self.registry)
        self.duration = Histogram("housing_http_request_duration_seconds", "HTTP response time", ["method", "route"], registry=self.registry)
        self.model_ready = Gauge("housing_model_ready", "Whether the model is loaded", registry=self.registry)
        self.drift_ready = Gauge("housing_drift_ready", "Enough samples and matching reference", registry=self.registry)
        self.window_count = Gauge("housing_drift_window_samples", "Successful predictions in bounded window", registry=self.registry)
        self.reference_count = Gauge("housing_drift_reference_samples", "Number of training reference records", registry=self.registry)
        self.psi = Gauge("housing_feature_psi", "Smoothed population stability index, not a significance test", ["feature"], registry=self.registry)
        self.reference = reference
        self.window = deque(maxlen=window_size)
        self.min_samples = min_samples
        self.lock = threading.Lock()
        self.reference_count.set(reference["training_rows"] if reference else 0)

    def observe(self, frame):
        with self.lock:
            self.window.append({field: float(frame.iloc[0][field]) for field in DRIFT_FEATURES})
            self.window_count.set(len(self.window))
            ready = self.reference is not None and len(self.window) >= self.min_samples
            self.drift_ready.set(int(ready))
            if ready:
                for field in DRIFT_FEATURES:
                    spec = self.reference["features"][field]
                    counts = histogram([row[field] for row in self.window], spec["edges"])
                    self.psi.labels(feature=field).set(population_stability_index(spec["counts"], counts))


def install_monitoring(app, root):
    reference = None
    bundle = app.config["MODEL_BUNDLE"]
    path = Path(app.config.get("DRIFT_REFERENCE_PATH", "reports/reference.json"))
    if not path.is_absolute():
        path = root / path
    if bundle and path.is_file():
        try:
            reference = json.loads(path.read_text(encoding="utf-8"))
            validate_reference(reference, bundle["data_dvc_md5"], bundle["artifact_sha256"])
        except (ValueError, KeyError, TypeError):
            app.logger.warning("Reference не подходит модели; анализ дрифта отключён")
            reference = None
    monitor = Monitoring(reference, int(app.config.get("DRIFT_WINDOW_SIZE", 200)), int(app.config.get("DRIFT_MIN_SAMPLES", 30)))
    monitor.model_ready.set(int(bundle is not None))
    app.extensions["housing_monitoring"] = monitor

    @app.before_request
    def start_timer():
        g.monitoring_started = time.perf_counter()

    @app.after_request
    def record_response(response):
        # Не учитываем scrape, чтобы он не создавал видимость пользовательского трафика.
        if request.endpoint != "metrics":
            route = request.url_rule.rule if request.url_rule else "unmatched"
            method = request.method if request.method in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"} else "OTHER"
            monitor.requests.labels(method=method, route=route, status=str(response.status_code)).inc()
            monitor.duration.labels(method=method, route=route).observe(time.perf_counter() - g.monitoring_started)
        return response

    @app.get("/metrics")
    def metrics():
        monitor.model_ready.set(int(app.config["MODEL_BUNDLE"] is not None))
        return Response(generate_latest(monitor.registry), content_type=CONTENT_TYPE_LATEST)

    return monitor
