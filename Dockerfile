FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-runtime.txt requirements-monitoring.lock.txt ./
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt \
    && useradd --create-home --uid 10001 app
COPY service ./service
COPY .env.example ./
COPY reports/reference.json ./reports/reference.json
USER app
EXPOSE 5000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=2)"
# Один worker сохраняет единое окно дрифта и единый реестр метрик.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--access-logfile", "-", "service.app:app"]
