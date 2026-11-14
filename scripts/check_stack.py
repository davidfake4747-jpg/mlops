"""Проверка локального API → Prometheus → Grafana без вывода секретов."""

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def fetch(url, headers=None, data=None):
    request = urllib.request.Request(url, headers=headers or {}, data=data)
    try:
        response = urllib.request.urlopen(request, timeout=5)
    except urllib.error.HTTPError as error:
        response = error
    return response.code, response.read()


def wait_json(url, predicate, headers=None):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            status, body = fetch(url, headers)
            value = json.loads(body)
            if status == 200 and predicate(value):
                return value
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError(f"Не получен ожидаемый ответ: {url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic", action="store_true", help="Создать 30 демонстрационных запросов")
    args = parser.parse_args()
    config = {**dotenv_values(ROOT / ".env"), **os.environ}
    token = config["APP_TOKEN"]
    password = config["GRAFANA_ADMIN_PASSWORD"]
    auth = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    wait_json("http://127.0.0.1:5000/health", lambda x: x.get("status") == "ready")
    if args.traffic:
        for index in range(30):
            payload = {"area": 35.5 + 10 * (index % 5), "rooms": index % 4, "floor": 3, "total_floors": 12}
            status, body = fetch("http://127.0.0.1:5000/api/numbers", auth, json.dumps(payload).encode())
            if status != 200 or json.loads(body).get("status") != "success":
                raise RuntimeError(f"Ошибка предсказания: HTTP {status}")
        invalid = json.dumps({"area": -1, "rooms": 2, "floor": 3, "total_floors": 12}).encode()
        if fetch("http://127.0.0.1:5000/api/numbers", auth, invalid)[0] != 400:
            raise RuntimeError("API должен отклонять отрицательную площадь")
        if fetch("http://127.0.0.1:5000/api/numbers", data=b"{}")[0] != 401:
            raise RuntimeError("API должен требовать токен")
    status, body = fetch("http://127.0.0.1:5000/metrics")
    text = body.decode()
    if status != 200 or "housing_model_ready 1.0" not in text:
        raise RuntimeError("Метрики API недоступны")
    if token in text or password in text:
        raise RuntimeError("Секрет обнаружен в метриках")
    query = urllib.parse.urlencode({"query": 'up{job="housing-api"}'})
    wait_json("http://127.0.0.1:9090/api/v1/query?" + query,
              lambda x: any(r["value"][1] == "1" for r in x.get("data", {}).get("result", [])))
    if args.traffic:
        query = urllib.parse.urlencode({"query": "housing_drift_ready"})
        wait_json("http://127.0.0.1:9090/api/v1/query?" + query,
                  lambda x: any(r["value"][1] == "1" for r in x.get("data", {}).get("result", [])))
    grafana_auth = {"Authorization": "Basic " + base64.b64encode(("admin:" + password).encode()).decode()}
    dashboard = wait_json("http://127.0.0.1:3000/api/dashboards/uid/housing-api",
                          lambda x: len(x.get("dashboard", {}).get("panels", [])) == 8, grafana_auth)
    wait_json("http://127.0.0.1:3000/api/datasources/uid/housing-prometheus/health",
              lambda x: x.get("status") == "OK", grafana_auth)
    print("API готов; метрики читаются Prometheus; Grafana загрузила 8 панелей")
    print("Дашборд: http://127.0.0.1:3000" + dashboard["meta"]["url"])


if __name__ == "__main__":
    main()
