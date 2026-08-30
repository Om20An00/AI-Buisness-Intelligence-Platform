FROM python:3.12-slim

WORKDIR /app

# System deps for psycopg2 / xgboost runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV MODEL_DIR=/app/ml_artifacts
RUN mkdir -p /app/ml_artifacts

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
