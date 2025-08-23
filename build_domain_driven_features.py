#!/usr/bin/env python3
import argparse, sys
from google.cloud import bigquery

# ---------- SQL templates ----------

CREATE_SQL_TMPL = """
CREATE OR REPLACE TABLE `{p}.{d}.{features}`
PARTITION BY ingest_date AS
WITH stats AS (
  SELECT APPROX_QUANTILES(monthly_charges, 100)[OFFSET(80)] AS p80_monthly
  FROM `{p}.{d}.{clean}`
  WHERE ingest_date = CURRENT_DATE()
)
SELECT
  c.customer_id,
  c.tenure,
  c.monthly_charges,
  c.total_charges,
  (c.senior_citizen = 1) AS is_senior,
  (c.internet_service != 'No') AS has_internet,
  CAST(c.online_security = 'Yes' AS INT64) +
  CAST(c.online_backup = 'Yes' AS INT64) +
  CAST(c.device_protection = 'Yes' AS INT64) +
  CAST(c.tech_support = 'Yes' AS INT64) +
  CAST(c.streaming_tv = 'Yes' AS INT64) +
  CAST(c.streaming_movies = 'Yes' AS INT64) AS services_count,
  SAFE_DIVIDE(c.total_charges, NULLIF(c.tenure, 0)) AS avg_charges_per_tenure,
  SAFE_DIVIDE(
    c.monthly_charges,
    NULLIF(
      CAST(c.online_security = 'Yes' AS INT64) +
      CAST(c.online_backup = 'Yes' AS INT64) +
      CAST(c.device_protection = 'Yes' AS INT64) +
      CAST(c.tech_support = 'Yes' AS INT64) +
      CAST(c.streaming_tv = 'Yes' AS INT64) +
      CAST(c.streaming_movies = 'Yes' AS INT64), 0)
  ) AS charges_per_service,
  c.internet_service AS internet_type,
  c.contract        AS contract_type,
  c.payment_method  AS payment_method,
  CASE
    WHEN c.contract = 'Month-to-month' THEN 1
    WHEN c.contract = 'One year'       THEN 12
    WHEN c.contract = 'Two year'       THEN 24
    ELSE NULL
  END AS contract_length_months,
  CASE
    WHEN c.tenure IS NULL            THEN 'unknown'
    WHEN c.tenure < 12               THEN '<12m'
    WHEN c.tenure BETWEEN 12 AND 24  THEN '12-24m'
    WHEN c.tenure BETWEEN 25 AND 48  THEN '25-48m'
    ELSE '>48m'
  END AS tenure_bucket,
  (c.monthly_charges > s.p80_monthly) AS is_high_spender,
  c.churn_label,
  CURRENT_DATE() AS ingest_date
FROM `{p}.{d}.{clean}` AS c
CROSS JOIN stats AS s
WHERE c.ingest_date = CURRENT_DATE();
"""

APPEND_SQL_TMPL = """
INSERT INTO `{p}.{d}.{features}` (
  customer_id, tenure, monthly_charges, total_charges, is_senior, has_internet,
  services_count, avg_charges_per_tenure, charges_per_service,
  internet_type, contract_type, payment_method, contract_length_months,
  tenure_bucket, is_high_spender, churn_label, ingest_date
)
WITH stats AS (
  SELECT APPROX_QUANTILES(monthly_charges, 100)[OFFSET(80)] AS p80_monthly
  FROM `{p}.{d}.{clean}`
  WHERE ingest_date = CURRENT_DATE()
)
SELECT
  c.customer_id,
  c.tenure,
  c.monthly_charges,
  c.total_charges,
  (c.senior_citizen = 1) AS is_senior,
  (c.internet_service != 'No') AS has_internet,
  CAST(c.online_security = 'Yes' AS INT64) +
  CAST(c.online_backup = 'Yes' AS INT64) +
  CAST(c.device_protection = 'Yes' AS INT64) +
  CAST(c.tech_support = 'Yes' AS INT64) +
  CAST(c.streaming_tv = 'Yes' AS INT64) +
  CAST(c.streaming_movies = 'Yes' AS INT64) AS services_count,
  SAFE_DIVIDE(c.total_charges, NULLIF(c.tenure, 0)) AS avg_charges_per_tenure,
  SAFE_DIVIDE(
    c.monthly_charges,
    NULLIF(
      CAST(c.online_security = 'Yes' AS INT64) +
      CAST(c.online_backup = 'Yes' AS INT64) +
      CAST(c.device_protection = 'Yes' AS INT64) +
      CAST(c.tech_support = 'Yes' AS INT64) +
      CAST(c.streaming_tv = 'Yes' AS INT64) +
      CAST(c.streaming_movies = 'Yes' AS INT64), 0)
  ) AS charges_per_service,
  c.internet_service AS internet_type,
  c.contract        AS contract_type,
  c.payment_method  AS payment_method,
  CASE
    WHEN c.contract = 'Month-to-month' THEN 1
    WHEN c.contract = 'One year'       THEN 12
    WHEN c.contract = 'Two year'       THEN 24
    ELSE NULL
  END AS contract_length_months,
  CASE
    WHEN c.tenure IS NULL            THEN 'unknown'
    WHEN c.tenure < 12               THEN '<12m'
    WHEN c.tenure BETWEEN 12 AND 24  THEN '12-24m'
    WHEN c.tenure BETWEEN 25 AND 48  THEN '25-48m'
    ELSE '>48m'
  END AS tenure_bucket,
  (c.monthly_charges > s.p80_monthly) AS is_high_spender,
  c.churn_label,
  CURRENT_DATE() AS ingest_date
FROM `{p}.{d}.{clean}` AS c
CROSS JOIN stats AS s
WHERE c.ingest_date = CURRENT_DATE();
"""

CREATE_VIEW_SQL_TMPL = """
CREATE OR REPLACE VIEW `{p}.{d}.vw_latest_features` AS
SELECT *
FROM `{p}.{d}.{features}`
WHERE ingest_date = (SELECT MAX(ingest_date) FROM `{p}.{d}.{features}`);
"""

# ---------- status helpers ----------
def write_status(client, project, dataset, run_id, step, status, message=""):
    sql = f"""
    INSERT INTO `{project}.{dataset}.pipeline_status`
      (run_id, step, status, message, created_at)
    VALUES (@run_id, @step, @status, @message, CURRENT_TIMESTAMP())
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("run_id","STRING",run_id),
        bigquery.ScalarQueryParameter("step","STRING",step),
        bigquery.ScalarQueryParameter("status","STRING",status),
        bigquery.ScalarQueryParameter("message","STRING",message),
    ])
    client.query(sql, job_config=cfg).result()

def assert_prev_success(client, project, dataset, run_id, prev_step):
    sql = f"""
    SELECT status
    FROM `{project}.{dataset}.pipeline_status`
    WHERE run_id = @run_id AND step = @prev_step
    ORDER BY created_at DESC
    LIMIT 1
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("run_id","STRING",run_id),
        bigquery.ScalarQueryParameter("prev_step","STRING",prev_step),
    ])
    df = client.query(sql, job_config=cfg).to_dataframe()
    if df.empty or df.iloc[0]["status"] != "SUCCESS":
        raise RuntimeError(f"Previous step '{prev_step}' not SUCCESS for run_id={run_id}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--clean_table", required=True)      # churn_clean (partitioned by ingest_date)
    ap.add_argument("--features_table", required=True)   # churn_features
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--gate_on", default="validate_clean", choices=["clean","validate_clean"],
                    help="Which prior step to gate on (default: validate_clean)")
    args = ap.parse_args()

    client = bigquery.Client(project=args.project)
    step = "features"

    try:
        # gate & status
        assert_prev_success(client, args.project, args.dataset, args.run_id, args.gate_on)
        write_status(client, args.project, args.dataset, args.run_id, step, "STARTED")

        # Does features table exist with schema?
        features_fqn = f"{args.project}.{args.dataset}.{args.features_table}"
        create_mode = True
        try:
            tbl = client.get_table(features_fqn)
            # If table exists with >1 column, we treat it as having schema
            if len(tbl.schema) > 1:
                create_mode = False
        except Exception:
            create_mode = True

        if create_mode:
            # First run or only ingest_date exists → create full table from today's clean
            sql = CREATE_SQL_TMPL.format(p=args.project, d=args.dataset,
                                         clean=args.clean_table, features=args.features_table)
            client.query(sql).result()
            action = "created"
        else:
            # Append today's partition
            sql = APPEND_SQL_TMPL.format(p=args.project, d=args.dataset,
                                         clean=args.clean_table, features=args.features_table)
            client.query(sql).result()
            action = "appended"

        # Refresh latest view
        vsql = CREATE_VIEW_SQL_TMPL.format(p=args.project, d=args.dataset, features=args.features_table)
        client.query(vsql).result()

        write_status(client, args.project, args.dataset, args.run_id, step, "SUCCESS", f"Features {action} & view updated")
        print(f"[done] Features {action}: `{features_fqn}`; view `vw_latest_features` updated.")

    except Exception as e:
        write_status(client, args.project, args.dataset, args.run_id, step, "FAILED", str(e))
        print("FAILED:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
