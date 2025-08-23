#!/usr/bin/env python3

import argparse, json, os, sys, shutil, uuid, tempfile, time
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---------------- BigQuery helpers ----------------

def write_status(bq: bigquery.Client, project: str, dataset: str, run_id: str, step: str, status: str, message: str = ""):
    sql = f"""
    INSERT INTO `{project}.{dataset}.pipeline_status` (run_id, step, status, message, created_at)
    VALUES (@run_id, @step, @status, @message, CURRENT_TIMESTAMP())
    """
    bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("run_id","STRING",run_id),
            bigquery.ScalarQueryParameter("step","STRING",step),
            bigquery.ScalarQueryParameter("status","STRING",status),
            bigquery.ScalarQueryParameter("message","STRING",message),
        ]
    )).result()

def assert_prev_success(bq: bigquery.Client, project: str, dataset: str, run_id: str, prev_step: str):
    sql = f"""
    SELECT status
    FROM `{project}.{dataset}.pipeline_status`
    WHERE run_id=@run_id AND step=@prev_step
    ORDER BY created_at DESC
    LIMIT 1
    """
    df = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("run_id","STRING",run_id),
            bigquery.ScalarQueryParameter("prev_step","STRING",prev_step),
        ]
    )).to_dataframe()
    if df.empty or df.iloc[0]["status"] != "SUCCESS":
        raise RuntimeError(f"Previous step '{prev_step}' not SUCCESS for run_id={run_id}")

def ensure_model_registry(bq: bigquery.Client, project: str, dataset: str):
    sql = f"""
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.model_registry` (
      run_id        STRING,
      version       INT64,
      status        STRING,          -- RUNNING/TRAINING/READY/FAILED
      artifact_uri  STRING,          -- gs://.../model.joblib
      metrics_json  STRING,          -- JSON text of metrics
      is_latest     BOOL,
      created_at    TIMESTAMP,
      updated_at    TIMESTAMP
    )
    """
    bq.query(sql).result()

def get_or_bind_version_for_run(bq: bigquery.Client, project: str, dataset: str, run_id: str) -> int:
    """
    Single-row-per-run behavior:
    - If a row exists for run_id with version IS NULL -> assign next version to THAT ROW (no new row).
    - Else if a row exists with a version -> reuse that version, set status='TRAINING'.
    - Else (no row) -> insert new row with next version, status='TRAINING'.
    Returns the version to use.
    """
    # 1) Does a row exist for this run?
    q = f"""
      SELECT run_id, version
      FROM `{project}.{dataset}.model_registry`
      WHERE run_id=@run_id
      ORDER BY updated_at DESC, created_at DESC
      LIMIT 1
    """
    df = bq.query(q, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id","STRING",run_id)]
    )).to_dataframe()

    # Compute next version number
    v_next = int(bq.query(
        f"SELECT COALESCE(MAX(version),0)+1 AS v FROM `{project}.{dataset}.model_registry`"
    ).to_dataframe().iloc[0]["v"])

    if df.empty:
        # No row -> insert with new version
        ins = f"""
        INSERT INTO `{project}.{dataset}.model_registry`
          (run_id, version, status, artifact_uri, metrics_json, is_latest, created_at, updated_at)
        VALUES (@run_id, @version, 'TRAINING', NULL, NULL, FALSE, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())
        """
        bq.query(ins, job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id","STRING",run_id),
                bigquery.ScalarQueryParameter("version","INT64",v_next),
            ]
        )).result()
        return v_next

    # Row exists
    cur_version = df.iloc[0]["version"]
    if pd.isna(cur_version):
        # Upgrade existing NULL-version row to the next version
        upd = f"""
        UPDATE `{project}.{dataset}.model_registry`
        SET version=@version, status='TRAINING', updated_at=CURRENT_TIMESTAMP()
        WHERE run_id=@run_id AND version IS NULL
        """
        bq.query(upd, job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id","STRING",run_id),
                bigquery.ScalarQueryParameter("version","INT64",v_next),
            ]
        )).result()
        return v_next
    else:
        # Reuse existing version for this run; set TRAINING
        upd = f"""
        UPDATE `{project}.{dataset}.model_registry`
        SET status='TRAINING', updated_at=CURRENT_TIMESTAMP()
        WHERE run_id=@run_id AND version=@version
        """
        bq.query(upd, job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id","STRING",run_id),
                bigquery.ScalarQueryParameter("version","INT64",int(cur_version)),
            ]
        )).result()
        return int(cur_version)

def update_registry_success(bq: bigquery.Client, project: str, dataset: str,
                            run_id: str, version: int, artifact_uri: str, metrics_json: str):
    # Clear previous latest (except this row)
    clr = f"""
    UPDATE `{project}.{dataset}.model_registry`
    SET is_latest=FALSE, updated_at=CURRENT_TIMESTAMP()
    WHERE is_latest=TRUE AND NOT (run_id=@run_id AND version=@version)
    """
    bq.query(clr, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("run_id","STRING",run_id),
            bigquery.ScalarQueryParameter("version","INT64",version),
        ]
    )).result()

    # Update THIS run's row
    upd = f"""
    UPDATE `{project}.{dataset}.model_registry`
    SET status='READY', artifact_uri=@uri, metrics_json=@mj,
        is_latest=TRUE, updated_at=CURRENT_TIMESTAMP()
    WHERE run_id=@run_id AND version=@version
    """
    bq.query(upd, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("uri","STRING",artifact_uri),
            bigquery.ScalarQueryParameter("mj","STRING",metrics_json),
            bigquery.ScalarQueryParameter("run_id","STRING",run_id),
            bigquery.ScalarQueryParameter("version","INT64",version),
        ]
    )).result()

def update_registry_failed(bq: bigquery.Client, project: str, dataset: str, run_id: str, version: int, msg: str):
    upd = f"""
    UPDATE `{project}.{dataset}.model_registry`
    SET status='FAILED', metrics_json=@msg, updated_at=CURRENT_TIMESTAMP()
    WHERE run_id=@run_id AND version=@version
    """
    bq.query(upd, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("msg","STRING",msg[:90000]),
            bigquery.ScalarQueryParameter("run_id","STRING",run_id),
            bigquery.ScalarQueryParameter("version","INT64",version),
        ]
    )).result()

# ---------------- Data / Training ----------------

def load_features(project: str, dataset: str, table: str, today_only: bool = True) -> pd.DataFrame:
    bq = bigquery.Client(project=project)
    where_clause = "WHERE ingest_date = CURRENT_DATE()" if today_only else ""
    sql = f"SELECT * EXCEPT(ingest_date) FROM `{project}.{dataset}.{table}` {where_clause}"
    return bq.query(sql).to_dataframe(create_bqstorage_client=False)

def train_and_eval(df: pd.DataFrame):
    y = df["churn_label"]
    X = df.drop(columns=["churn_label", "customer_id"])

    # Core domain categoricals
    cat_cols = [c for c in ["internet_type","contract_type","payment_method","tenure_bucket"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    candidates = {
        "logreg": LogisticRegression(max_iter=1000, solver="lbfgs"),  # n_jobs not used by lbfgs
        "rf": RandomForestClassifier(n_estimators=250, random_state=42)
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    best_name, best_pipe, best_auc = None, None, -1.0
    metrics_all = {}

    for name, clf in candidates.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        metrics = {
            "accuracy":  float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred)),
            "recall":    float(recall_score(y_test, pred)),
            "f1":        float(f1_score(y_test, pred)),
            "roc_auc":   float(roc_auc_score(y_test, proba)),
        }
        metrics_all[name] = metrics
        if metrics["roc_auc"] > best_auc:
            best_auc, best_name, best_pipe = metrics["roc_auc"], name, pipe

    return best_name, best_pipe, metrics_all

# ---------------- GCS helpers ----------------

def upload_file_to_gcs(local_path: str, bucket: str, dest_uri_prefix: str) -> str:
    """Upload a single file to GCS. Returns full gs:// URI."""
    client = storage.Client()
    bkt = client.bucket(bucket)
    fname = os.path.basename(local_path)
    blob = bkt.blob(f"{dest_uri_prefix.rstrip('/')}/{fname}")
    blob.upload_from_filename(local_path)
    return f"gs://{bucket}/{dest_uri_prefix.rstrip('/')}/{fname}"

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--features_table", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--today_only", action="store_true", default=True)
    args = ap.parse_args()

    bq = bigquery.Client(project=args.project)
    step = "train"

    # temp run folder to avoid uploading old files
    tmpdir = tempfile.mkdtemp(prefix=f"model_run_{args.run_id}_")
    try:
        # Gate & status
        assert_prev_success(bq, args.project, args.dataset, args.run_id, "features")
        write_status(bq, args.project, args.dataset, args.run_id, step, "STARTED")

        # Registry prep
        ensure_model_registry(bq, args.project, args.dataset)
        version = get_or_bind_version_for_run(bq, args.project, args.dataset, args.run_id)

        # Load data
        df = load_features(args.project, args.dataset, args.features_table, today_only=args.today_only)
        if df.empty:
            raise RuntimeError("No feature rows found for training (check today's partition).")

        # Train
        best_name, best_pipe, metrics_all = train_and_eval(df)

        # Persist locally
        ts = time.strftime("%Y%m%d_%H%M%S")
        model_fname = f"model_v{version}_{best_name}_{ts}.joblib"
        metrics_fname = f"metrics_v{version}_{ts}.json"
        model_path = str(Path(tmpdir) / model_fname)
        metrics_path = str(Path(tmpdir) / metrics_fname)
        joblib.dump(best_pipe, model_path)

        metrics_payload = {"version": version, "best_model": best_name, "candidates": metrics_all}
        with open(metrics_path, "w") as f:
            json.dump(metrics_payload, f, indent=2)

        # Upload to GCS: gs://bucket/<dd-MM-YYYY>/output/model/
        date_str = datetime.now().strftime("%d-%m-%Y")
        dest_prefix = f"{date_str}/output/model"
        model_uri = upload_file_to_gcs(model_path, args.bucket, dest_prefix)
        metrics_uri = upload_file_to_gcs(metrics_path, args.bucket, dest_prefix)

        # Update registry (set latest=true; ready)
        update_registry_success(
            bq, args.project, args.dataset, args.run_id, version,
            artifact_uri=model_uri,
            metrics_json=json.dumps({**metrics_payload, "model_uri": model_uri, "metrics_uri": metrics_uri})
        )

        write_status(bq, args.project, args.dataset, args.run_id, step, "SUCCESS", f"v{version} READY: {model_uri}")
        print(f"[done] Best={best_name}  v{version}")
        print(f"[model] {model_uri}")
        print(f"[metrics] {metrics_uri}")

    except Exception as e:
        # Try to mark registry as failed (if version was created)
        try:
            if 'version' in locals():
                update_registry_failed(bq, args.project, args.dataset, args.run_id, version, str(e))
        except Exception:
            pass
        write_status(bq, args.project, args.dataset, args.run_id, step, "FAILED", str(e))
        print("FAILED:", e, file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

if __name__ == "__main__":
    main()
