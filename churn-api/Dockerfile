# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app
COPY app.py /app/app.py
COPY requirements-app.txt /app/requirements.txt

# System deps for pandas/pyarrow/scikit-learn sometimes needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# env defaults (overridden at deploy)
ENV PORT=8080
ENV MODEL_LOCAL_DIR=/app/models

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
