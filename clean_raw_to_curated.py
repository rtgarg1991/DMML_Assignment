#!/usr/bin/env python3
import argparse
import sys
from google.cloud import bigquery

# ---------- SQL templates ----------

CREATE_TABLE_SQL_TMPL = """
CREATE OR REPLACE TABLE `{p}.{cd}.{clean}`
PARTITION BY ingest_date AS
SELECT
  SAFE_CAST(customer_id AS INT64) AS customer_id,
  TRIM(gender) AS gender,
  SAFE_CAST(senior_citizen AS INT64) AS senior_citizen,
  TRIM(partner) AS partner,
  TRIM(dependents) AS dependents,
  SAFE_CAST(tenure AS INT64) AS tenure,
  TRIM(phone_service) AS phone_service,
  TRIM(multiple_lines) AS multiple_lines,
  TRIM(internet_service) AS internet_service,
  TRIM(online_security) AS online_security,
  TRIM(online_backup) AS online_backup,
  TRIM(device_protection) AS device_protection,
  TRIM(tech_support) AS tech_support,
  TRIM(streaming_tv) AS streaming_tv,
  TRIM(streaming_movies) AS streaming_movies,
  TRIM(contract) AS contract,
  TRIM(paperless_billing) AS paperless_billing,
  TRIM(payment_method) AS payment_method,
  SAFE_CAST(monthly_charges AS FLOAT64) AS monthly_charges,
  SAFE_CAST(NULLIF(TRIM(CAST(total_charges AS STRING)), '') AS FLOAT64) AS total_charges,
  TRIM(churn) AS churn,
  IF(TRIM(churn) = 'Yes', 1, 0) AS churn_label,
  CURRENT_DATE() AS ingest_date
FROM `{p}.{cd}.{raw}`
WHERE customer_id IS NOT NULL AND ingest_date = CURRENT_DATE();
"""

INSERT_SQL_TMPL = """
INSERT INTO `{p}.{cd}.{clean}` (
  customer_id, gender, senior_citizen, partner, dependents, tenure,
  phone_service, multiple_lines, internet_service, online_security, online_backup,
  device_protection, tech_support, streaming_tv, streaming_movies, contract,
  paperless_billing, payment_method, monthly_charges, total_charges, churn, churn_label, ingest_date
)
SELECT
  SAFE_CAST(customer_id AS INT64) AS customer_id,
  TRIM(gender) AS gender,
  SAFE_CAST(senior_citizen AS INT64) AS senior_citizen,
  TRIM(partner) AS partner,
  TRIM(dependents) AS dependents,
  SAFE_CAST(tenure AS INT64) AS tenure,
  TRIM(phone_service) AS phone_service,
  TRIM(multiple_lines) AS multiple_lines,
  TRIM(internet_service) AS internet_service,
  TRIM(online_security) AS online_security,
  TRIM(online_backup) AS online_backup,
  TRIM(device_protection) AS device_protection,
  TRIM(tech_support) AS tech_support,
  TRIM(streaming_tv) AS streaming_tv,
  TRIM(streaming_movies) AS streaming_movies,
  TRIM(contract) AS contract,
  TRIM(paperless_billing) AS paperless_billing,
  TRIM(payment_method) AS payment_method,
  SAFE_CAST(monthly_charges AS FLOAT64) AS monthly_charges,
  SAFE_CAST(NULLIF(TRIM(CAST(total_charges AS STRING)), '') AS FLOAT64) AS total_charges,
  TRIM(churn) AS churn,
  IF(TRIM(churn) = 'Yes', 1, 0) AS churn_label,
  CURRENT_DATE() AS ingest_date
FROM `{p}.{cd}.{raw}`
WHERE customer_id IS NOT NULL AND ingest_date = CURRENT_DATE();
"""

CREATE_VIEW_SQL_TMPL = """
CREATE OR REPLACE VIEW `{p}.{cd}.vw_latest_clean` AS
SELECT *
FROM `{p}.{cd}.{clean}`
WHERE ingest_date = (SELECT MAX(ingest_date) FROM `{p}.{cd}.{clean}`);
"""

# ---------- Pipeline status helpers ----------

def write_status(client, project, dataset, run_id, step, status, message=""):
    sql = f"""
    INSERT INTO `{project}.{dataset}.pipeline_status`
      (run_id, step, status, message, created_at)
    VALUES (@run_id, @step, @status, @message, CURRENT_TIMESTAMP())
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("run_id","STRING",run_id),
            bigquery.ScalarQueryParameter("step","STRING",step),
            bigquery.ScalarQueryParameter("status","STRING",status),
            bigquery.ScalarQueryParameter("message","STRING",message),
        ]
    )
    client.query(sql, job_config=job_config).result()

def assert_prev_success(client, project, dataset, run_id, prev_step):
    sql = f"""
    SELECT status
    FROM `{project}.{dataset}.pipeline_status`
    WHERE run_id = @run_id AND step = @prev_step
    ORDER BY created_at DESC
    LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("run_id","STRING",run_id),
            bigquery.ScalarQueryParameter("prev_step","STRING",prev_step),
        ]
    )
    df = client.query(sql, job_config=job_config).to_dataframe()
    if df.empty or df.iloc[0]["status"] != "SUCCESS":
        raise RuntimeError(f"Previous step '{prev_step}' not SUCCESS for run_id={run_id}")

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw_table", required=True)
    ap.add_argument("--clean_table", required=True)
    ap.add_argument("--run_id", required=True)
    args = ap.parse_args()

    client = bigquery.Client(project=args.project)
    step = "clean"

    try:
        # Gate on previous step
        assert_prev_success(client, args.project, args.dataset, args.run_id, "load_raw")
        write_status(client, args.project, args.dataset, args.run_id, step, "STARTED")

        # Check if clean table exists and has >1 column
        table_exists = False
        has_schema = False
        try:
            tbl = client.get_table(f"{args.project}.{args.dataset}.{args.clean_table}")
            table_exists = True
            # if only 1 column (ingest_date), treat as empty schema
            if len(tbl.schema) > 1:
                has_schema = True
        except Exception:
            pass

        if not table_exists or not has_schema:
            # First run: create table with full schema
            sql = CREATE_TABLE_SQL_TMPL.format(
                p=args.project, cd=args.dataset, raw=args.raw_table, clean=args.clean_table
            )
            client.query(sql).result()
            print(f"[info] Created table {args.clean_table} with today's data")
        else:
            # Subsequent run: append only today's partition
            sql = INSERT_SQL_TMPL.format(
                p=args.project, cd=args.dataset, raw=args.raw_table, clean=args.clean_table
            )
            client.query(sql).result()
            print(f"[info] Appended today's partition to {args.clean_table}")

        # Create/update view
        vsql = CREATE_VIEW_SQL_TMPL.format(
            p=args.project, cd=args.dataset, clean=args.clean_table
        )
        client.query(vsql).result()

        write_status(client, args.project, args.dataset, args.run_id, step, "SUCCESS", "Clean table updated")
        print(f"[done] Clean table {args.clean_table} updated, view vw_latest_clean created.")

    except Exception as e:
        write_status(client, args.project, args.dataset, args.run_id, step, "FAILED", str(e))
        print("FAILED:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
