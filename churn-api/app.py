import os
import json
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query, Body
from typing import Optional
from pydantic import BaseModel
from google.cloud import bigquery, storage

# --------- Config via env ---------
PROJECT_ID      = os.getenv("PROJECT_ID", "YOUR_PROJECT_ID")
BQ_DATASET      = os.getenv("BQ_DATASET", "dmml_assignment_churn_raw")
BQ_FEATURES_TBL = os.getenv("BQ_TABLE", "churn_features")  # or vw_latest_features
MODEL_VERSION   = os.getenv("MODEL_VERSION")               # optional (int as string)
REGISTRY_TABLE  = os.getenv("MODEL_REGISTRY_TABLE", "model_registry")
MODEL_LOCAL_DIR = os.getenv("MODEL_LOCAL_DIR", "./model_artifacts")
FEATURE_METADATA_TABLE = os.getenv("FEATURE_METADATA_TABLE", "feature_metadata")
os.makedirs(MODEL_LOCAL_DIR, exist_ok=True)

# --------- Globals (populated on startup) ---------
model = None
model_info = {
    "version": None,
    "artifact_uri": None,
    "loaded_ok": False,
}

# --------- Helpers for model discovery/load ---------
def _get_bq_client():
    return bigquery.Client(project=PROJECT_ID)

def _get_storage_client():
    return storage.Client(project=PROJECT_ID)

def _gcs_download(gs_uri: str, local_dir: str) -> str:
    # gs://bucket/path/to/file
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gs_uri}")
    _, rest = gs_uri.split("gs://", 1)
    bucket_name, blob_path = rest.split("/", 1)
    client = _get_storage_client()
    bkt = client.bucket(bucket_name)
    blob = bkt.blob(blob_path)
    local_path = os.path.join(local_dir, os.path.basename(blob_path))
    blob.download_to_filename(local_path)
    return local_path

def _select_model_from_registry(version: Optional[int]) -> dict:
    """
    Returns {"version": int, "artifact_uri": str} or raises if not found/ready.
    """
    bq = _get_bq_client()
    if version is not None:
        sql = f"""
        SELECT version, artifact_uri
        FROM `{PROJECT_ID}.{BQ_DATASET}.{REGISTRY_TABLE}`
        WHERE version = @v AND status = 'READY'
        ORDER BY updated_at DESC
        LIMIT 1
        """
        df = bq.query(
            sql, job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("v", "INT64", int(version))]
            )
        ).to_dataframe()
    else:
        sql = f"""
        SELECT version, artifact_uri
        FROM `{PROJECT_ID}.{BQ_DATASET}.{REGISTRY_TABLE}`
        WHERE is_latest = TRUE AND status = 'READY'
        ORDER BY updated_at DESC
        LIMIT 1
        """
        df = bq.query(sql).to_dataframe()

    if df.empty:
        raise RuntimeError("No READY model found in model_registry for the requested selection")
    v = int(df.iloc[0]["version"])
    uri = df.iloc[0]["artifact_uri"]
    if not uri:
        raise RuntimeError("Selected model has empty artifact_uri")
    return {"version": v, "artifact_uri": uri}

def _load_model_at_start():
    global model, model_info
    # parse env MODEL_VERSION if provided
    ver = None
    if MODEL_VERSION:
        try:
            ver = int(MODEL_VERSION)
        except Exception:
            raise RuntimeError(f"MODEL_VERSION must be an integer, got: {MODEL_VERSION}")

    sel = _select_model_from_registry(ver)
    local_path = _gcs_download(sel["artifact_uri"], MODEL_LOCAL_DIR)
    mdl = joblib.load(local_path)
    model = mdl
    model_info.update({
        "version": sel["version"],
        "artifact_uri": sel["artifact_uri"],
        "local_path": local_path,
        "loaded_ok": True
    })

def fetch_feature_metadata(feature_name: Optional[str] = None,
                           version: Optional[str] = None,
                           source: Optional[str] = None,
                           limit: int = 200) -> pd.DataFrame:
    """
    Reads feature metadata rows from BigQuery.
    Supports optional filters by feature_name (exact), version (e.g. 'v1'), and source.
    """
    bq = _get_bq_client()

    where = []
    params = []
    if feature_name:
        where.append("feature_name = @fname")
        params.append(bigquery.ScalarQueryParameter("fname", "STRING", feature_name))
    if version:
        where.append("version = @ver")
        params.append(bigquery.ScalarQueryParameter("ver", "STRING", version))
    if source:
        where.append("source = @src")
        params.append(bigquery.ScalarQueryParameter("src", "STRING", source))

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
      SELECT feature_name, dtype, description, source, version, created_date
      FROM `{PROJECT_ID}.{BQ_DATASET}.{FEATURE_METADATA_TABLE}`
      {where_sql}
      ORDER BY feature_name
      LIMIT @lim
    """
    params.append(bigquery.ScalarQueryParameter("lim", "INT64", int(limit)))

    job = bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    return job.to_dataframe()


# ---------- Load model on startup ----------
try:
    _load_model_at_start()
except Exception as e:
    # Don't crash; expose via /health
    print("WARNING: model load failed at startup:", e)

# --------- FastAPI app ---------
app = FastAPI(title="Churn Prediction API", version="1.0.0")

@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "degraded",
        "project_id": PROJECT_ID,
        "dataset": BQ_DATASET,
        "features_table": BQ_FEATURES_TBL,
        "model_loaded": model is not None,
        "model_info": model_info,
    }

# --------- Helper: fetch features by customer_id ---------
def fetch_features(customer_id: str) -> pd.DataFrame:
    client = _get_bq_client()
    sql = f"""
      SELECT * EXCEPT(ingest_date)
      FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_FEATURES_TBL}`
      WHERE CAST(customer_id AS STRING) = @cust
      ORDER BY ingest_date DESC
      LIMIT 1
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("cust", "STRING", customer_id)]
        ),
    )
    df = job.result().to_dataframe(create_bqstorage_client=False)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"customer_id '{customer_id}' not found")
    return df

# The model expects the same columns as training (features-only)
def to_model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["churn_label", "customer_id", "ingest_date"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    return X

def fetch_registry(version: Optional[int] = None) -> pd.DataFrame:
    client = _get_bq_client()
    if version is None:
        sql = f"""
          SELECT version, status, is_latest, artifact_uri, created_at, updated_at
          FROM `{PROJECT_ID}.{BQ_DATASET}.{REGISTRY_TABLE}`
          ORDER BY version DESC
          LIMIT 20
        """
        return client.query(sql).to_dataframe()
    else:
        sql = f"""
          SELECT version, status, is_latest, artifact_uri, created_at, updated_at
          FROM `{PROJECT_ID}.{BQ_DATASET}.{REGISTRY_TABLE}`
          WHERE version = @v
          ORDER BY updated_at DESC
          LIMIT 1
        """
        return client.query(
            sql,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("v", "INT64", version)]
            )
        ).to_dataframe()


# --------- Route: predict by customer_id (BigQuery source) ---------
@app.get("/predict")
def predict_by_customer(customer_id: str = Query(..., description="Customer ID to score")):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    df = fetch_features(customer_id)
    X = to_model_inputs(df)
    try:
        proba = float(model.predict_proba(X)[:, 1][0])
        
        print(X)
        print(proba)
        pred = int(proba >= 0.5)
        print(pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    return {
        "customer_id": customer_id,
        "churn_probability": round(proba, 6),
        "prediction": pred,
        "model_version": model_info.get("version"),
    }

# --------- Route: predict from raw JSON features ---------
class FeaturePayload(BaseModel):
    tenure: Optional[float] = None
    monthly_charges: Optional[float] = None
    total_charges: Optional[float] = None
    is_senior: Optional[bool] = None
    has_internet: Optional[bool] = None
    services_count: Optional[int] = None
    avg_charges_per_tenure: Optional[float] = None
    charges_per_service: Optional[float] = None
    internet_type: Optional[str] = None
    contract_type: Optional[str] = None
    payment_method: Optional[str] = None
    contract_length_months: Optional[float] = None
    tenure_bucket: Optional[str] = None
    is_high_spender: Optional[bool] = None

@app.post("/predict")
def predict_from_json(payload: FeaturePayload = Body(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    X = pd.DataFrame([payload.dict()])
    try:
        proba = float(model.predict_proba(X)[:, 1][0])
        pred = int(proba >= 0.5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
    return {
        "churn_probability": round(proba, 6),
        "prediction": pred,
        "model_version": model_info.get("version"),
    }

@app.get("/registry")
def registry_all():
    """
    Return metadata for recent model versions (default: last 20).
    """
    try:
        df = fetch_registry()
        return json.loads(df.to_json(orient="records", date_format="iso"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch registry: {e}")


@app.get("/registry/{version}")
def registry_by_version(version: int):
    """
    Return metadata for a specific model version.
    """
    try:
        df = fetch_registry(version)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Model version {version} not found")
        return json.loads(df.to_json(orient="records", date_format="iso"))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch registry: {e}")

@app.get("/features/metadata")
def list_feature_metadata(version: Optional[str] = None,
                          source: Optional[str] = None,
                          limit: int = 200):
    """
    List feature metadata (optionally filter by version/source).
    """
    try:
        df = fetch_feature_metadata(version=version, source=source, limit=limit)
        return json.loads(df.to_json(orient="records", date_format="iso"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch feature metadata: {e}")


@app.get("/features/metadata/{feature_name}")
def get_feature_metadata(feature_name: str, version: Optional[str] = None):
    """
    Get metadata for a single feature (optionally for a specific version).
    """
    try:
        df = fetch_feature_metadata(feature_name=feature_name, version=version, limit=1)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No metadata found for '{feature_name}'")
        return json.loads(df.to_json(orient="records", date_format="iso"))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch feature metadata: {e}")
