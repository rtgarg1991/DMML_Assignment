#!/usr/bin/env python3
"""
load_to_bigquery.py  (NO STAGING, schema grows automatically)

- Expects CSV at: gs://<bucket>/<dd-MM-yyyy>/churn_raw.csv
- Table: YOUR_PROJECT_ID.dmml_assignment_churn_raw.churn_raw
- Uses autodetect + ALLOW_FIELD_ADDITION so new CSV columns are added
- ingest_date and ingest_ts are filled by table defaults set in DDL above
- Also ensures/updates model_registry with a 'RUNNING' row for this run_id
"""

import argparse
import sys
from datetime import datetime
from google.cloud import bigquery, exceptions

def write_status(project, run_id, step, status, message=""):
    client = bigquery.Client(project=project)
    client.query(f"""
      INSERT INTO `{project}.dmml_assignment_churn_raw.pipeline_status`
        (run_id, step, status, message, created_at)
      VALUES (@run_id, @step, @status, @message, CURRENT_TIMESTAMP())
    """, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("run_id","STRING",run_id),
            bigquery.ScalarQueryParameter("step","STRING",step),
            bigquery.ScalarQueryParameter("status","STRING",status),
            bigquery.ScalarQueryParameter("message","STRING",message),
        ])
    ).result()

# --- NEW: model_registry helpers ---

def ensure_model_registry(project: str):
    client = bigquery.Client(project=project)
    client.query(f"""
      CREATE TABLE IF NOT EXISTS `{project}.dmml_assignment_churn_raw.model_registry` (
        run_id       STRING,
        version      INT64,
        status       STRING,        -- RUNNING/TRAINING/READY/FAILED
        artifact_uri STRING,        -- gs://.../model.joblib (filled at train step)
        metrics_json STRING,        -- JSON metrics (filled at train step)
        is_latest    BOOL,
        created_at   TIMESTAMP,
        updated_at   TIMESTAMP
      )
    """).result()

def upsert_registry_running(project: str, run_id: str):
    """Create/refresh a 'RUNNING' row for this run (version is still NULL)."""
    client = bigquery.Client(project=project)
    client.query(f"""
      MERGE `{project}.dmml_assignment_churn_raw.model_registry` T
      USING (SELECT @run_id AS run_id) S
      ON T.run_id = S.run_id AND T.version IS NULL
      WHEN MATCHED THEN
        UPDATE SET status='RUNNING', updated_at=CURRENT_TIMESTAMP()
      WHEN NOT MATCHED THEN
        INSERT (run_id, version, status, artifact_uri, metrics_json, is_latest, created_at, updated_at)
        VALUES (@run_id, NULL, 'RUNNING', NULL, NULL, FALSE, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())
    """, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id","STRING",run_id)]
    )).result()

def main(args):
    # mark pipeline status
    write_status(args.project, args.run_id, "load_raw", "STARTED")

    # ensure model_registry exists & add/refresh RUNNING entry for this run
    ensure_model_registry(args.project)
    upsert_registry_running(args.project, args.run_id)

    client = bigquery.Client(project=args.project)
    table_id = f"{args.project}.{args.dataset}.{args.table}"

    # Build today's URI: gs://bucket/dd-MM-yyyy/churn_raw.csv
    gcs_uri = f"gs://{args.bucket}/{args.date_folder}/churn_raw.csv"
    print(f"[info] Loading: {gcs_uri}")

    # Ensure table exists (created earlier with ingest_date partitioning)
    try:
        client.get_table(table_id)
        print(f"[info] Table exists: {table_id}")
    except Exception as e:
        print("Table not found. Create it first with ingest_date partitioning, then re-run.", file=sys.stderr)
        raise

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=False,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],  # let schema expand
        allow_quoted_newlines=True,
        ignore_unknown_values=True,
        max_bad_records=1000,
    )

    job = client.load_table_from_uri(gcs_uri, table_id, job_config=job_config)
    print(f"[job] {job.job_id}")
    job.result()
    print("[done] Load completed.")

    # Show row count
    cnt = client.query(f"SELECT COUNT(*) AS c FROM `{table_id}`").to_dataframe().iloc[0,0]
    print(f"[count] {table_id} now has {cnt} rows.")
    write_status(args.project, args.run_id, "load_raw", "SUCCESS", f"loaded {cnt} rows")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--dataset", default="dmml_assignment_churn_raw")
    ap.add_argument("--table", default="churn_raw")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--date_folder", required=True, help="dd-MM-YYYY")
    args = ap.parse_args()
    try:
        main(args)
    except Exception as e:
        print("FAILED:", e, file=sys.stderr)
        # best-effort status update
        try:
            write_status(args.project, args.run_id, "load_raw", "FAILED", str(e))
        except Exception:
            pass
        sys.exit(1)
