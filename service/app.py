"""Flask API для выбранного обученного артефакта."""

import argparse
import hmac
import logging
import math
import os

from dotenv import dotenv_values
from flask import Flask, render_template, request
from flask_cors import CORS
from flask_httpauth import HTTPTokenAuth
from werkzeug.exceptions import BadRequest

try:
    from .monitoring import install_monitoring
    from .predict import ROOT, load_model, request_frame
except ImportError:
    from monitoring import install_monitoring
    from predict import ROOT, load_model, request_frame


def create_app(overrides=None):
    app = Flask(__name__)
    config = {
        **dotenv_values(ROOT / ".env.example"),
        **{k: v for k, v in dotenv_values(ROOT / ".env").items() if v is not None},
        **{k: os.environ[k] for k in (
            "APP_TOKEN", "MODEL_NAME", "ORIGIN", "DRIFT_REFERENCE_PATH",
            "DRIFT_WINDOW_SIZE", "DRIFT_MIN_SAMPLES",
        ) if k in os.environ},
        **(overrides or {}),
    }
    app.config.update(config)
    name = app.config.get("MODEL_NAME", "selected")
    try:
        bundle = load_model(name)
    except FileNotFoundError:
        # Приложение может запуститься до dvc pull, но не объявляет себя готовым.
        app.logger.warning("Модель отсутствует; выполните dvc pull или dvc repro")
        bundle = None
    app.config["MODEL_BUNDLE"] = bundle
    monitor = install_monitoring(app, ROOT)
    CORS(app, resources={r"/api/*": {"origins": config.get("ORIGIN", "http://localhost:5000")}})
    auth = HTTPTokenAuth(scheme="Bearer")

    @auth.verify_token
    def verify_token(token):
        expected = app.config.get("APP_TOKEN")
        if (
            isinstance(token, str) and isinstance(expected, str) and expected
            and hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))
        ):
            return "user"
        return None

    @auth.error_handler
    def authentication_error(status):
        return {"status": "error", "data": "Нужен действительный Bearer-токен"}, status

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        model = app.config["MODEL_BUNDLE"]
        if model is None:
            return {"status": "error", "data": "Модель недоступна"}, 503
        return {"status": "ready", "model": model["model_name"]}

    @app.post("/api/numbers")
    @auth.login_required
    def process_numbers():
        if not request.is_json:
            return {"status": "error", "data": "Ожидается application/json"}, 415
        try:
            data = request.get_json()
            frame = request_frame(data)
        except (BadRequest, ValueError) as exc:
            message = "Неверный JSON" if isinstance(exc, BadRequest) else str(exc)
            return {"status": "error", "data": message}, 400
        model = app.config["MODEL_BUNDLE"]
        if model is None:
            return {"status": "error", "data": "Модель недоступна"}, 503
        try:
            predictions = model["estimator"].predict(frame)
            if len(predictions) != 1:
                raise ValueError("Неверный размер ответа модели")
            price = float(predictions[0])
            if not math.isfinite(price) or price <= 0 or round(price) <= 0:
                raise ValueError("Недопустимый прогноз")
        except Exception:
            app.logger.exception("Ошибка предсказания")
            return {"status": "error", "data": "Не удалось получить допустимый прогноз"}, 503
        monitor.observe(frame)
        return {"status": "success", "data": round(price)}

    return app


app = create_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сервис оценки стоимости квартиры")
    parser.add_argument("-m", "--model", default=None, help="selected, baseline_const, lin_reg или путь к артефакту")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    server = create_app({"MODEL_NAME": args.model}) if args.model else app
    server.run(host="127.0.0.1", port=args.port, debug=False)
