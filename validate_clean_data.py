#!/usr/bin/env python3
import argparse, os, csv, sys, json, shutil, uuid
from datetime import datetime, date
from google.cloud import bigquery, storage

# -------- Validation checks (templated) --------
QUERIES = {
    "row_count": """
        SELECT COUNT(*) AS cnt
        FROM `{p}.{d}.{t}`
        {flt}
    """,
    "nulls": """
        SELECT
          SUM(CASE WHEN tenure IS NULL THEN 1 ELSE 0 END) AS null_tenure,
          SUM(CASE WHEN gender IS NULL THEN 1 ELSE 0 END) AS null_gender,
          SUM(CASE WHEN monthly_charges IS NULL THEN 1 ELSE 0 END) AS null_monthly,
          SUM(CASE WHEN total_charges IS NULL THEN 1 ELSE 0 END) AS null_total,
          SUM(CASE WHEN senior_citizen IS NULL THEN 1 ELSE 0 END) AS null_senior_citizen,
          SUM(CASE WHEN churn_label IS NULL THEN 1 ELSE 0 END) AS null_churn_label
        FROM `{p}.{d}.{t}`
        {flt}
    """,
    "label_values": """
        SELECT churn_label, COUNT(*) AS cnt
        FROM `{p}.{d}.{t}`
        {flt}
        GROUP BY churn_label
        ORDER BY churn_label
    """,
    "out_of_range": """
        SELECT
          SUM(CASE WHEN tenure < 0 THEN 1 ELSE 0 END) AS bad_tenure,
          SUM(CASE WHEN gender NOT IN ('Female','Male') THEN 1 ELSE 0 END) AS bad_gender,
          SUM(CASE WHEN monthly_charges < 0 THEN 1 ELSE 0 END) AS bad_monthly,
          SUM(CASE WHEN total_charges < 0 THEN 1 ELSE 0 END) AS bad_total,
          SUM(CASE WHEN senior_citizen NOT IN (0,1) THEN 1 ELSE 0 END) AS bad_senior_citizen
        FROM `{p}.{d}.{t}`
        {flt}
    """,
    "basic_stats": """
        SELECT
          MIN(tenure) AS min_tenure, MAX(tenure) AS max_tenure, AVG(tenure) AS avg_tenure,
          MIN(monthly_charges) AS min_monthly, MAX(monthly_charges) AS max_monthly, AVG(monthly_charges) AS avg_monthly,
          MIN(total_charges) AS min_total, MAX(total_charges) AS max_total, AVG(total_charges) AS avg_total
        FROM `{p}.{d}.{t}`
        {flt}
    """,
    "distinct_counts": """
        SELECT
          COUNT(DISTINCT gender) AS distinct_gender,
          COUNT(DISTINCT internet_service) AS distinct_internet_service,
          COUNT(DISTINCT contract) AS distinct_contract,
          COUNT(DISTINCT payment_method) AS distinct_payment_method
        FROM `{p}.{d}.{t}`
        {flt}
    """,
    "duplicates": """
        SELECT COUNT(*) AS dupes
        FROM (
          SELECT customer_id
          FROM `{p}.{d}.{t}`
          {flt}
          GROUP BY customer_id
          HAVING COUNT(*) > 1
        )
    """,
    "partitions_present": """
        SELECT DISTINCT ingest_date
        FROM `{p}.{d}.{t}`
        ORDER BY ingest_date DESC
        LIMIT 5
    """
}

def upload_dir_to_gcs(local_dir: str, bucket_name: str, dest_prefix: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for fname in files:
            src = os.path.join(root, fname)
            rel = os.path.relpath(src, local_dir).replace("\\", "/")
            blob = bucket.blob(f"{dest_prefix.rstrip('/')}/{rel}")
            blob.upload_from_filename(src)
            print(f"[uploaded] {src} → gs://{bucket_name}/{dest_prefix}{rel}")

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--table", required=True)          # churn_clean
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--today_only", action="store_true", default=True)
    ap.add_argument("--out_dir", default="./output")
    ap.add_argument("--date", required=True)
    ap.add_argument("--date_folder", required=True)
    args = ap.parse_args()

    client = bigquery.Client(project=args.project)
    step = "validate_clean"

    # Use a temp subfolder for this run so we only upload fresh files
    tmp_dir = os.path.join(args.out_dir, f"_run_{args.run_id}_{uuid.uuid4().hex[:6]}")
    os.makedirs(tmp_dir, exist_ok=True)
    report_csv = os.path.join(tmp_dir, "validation_report.csv")

    try:
        # gate on clean
        assert_prev_success(client, args.project, args.dataset, args.run_id, "clean")
        write_status(client, args.project, args.dataset, args.run_id, step, "STARTED")

        where_clause = "WHERE ingest_date = DATE(@d)" if args.today_only else ""

        results = []
        for name, sql in QUERIES.items():
            q = sql.format(p=args.project, d=args.dataset, t=args.table, flt=where_clause)
            df = client.query(q, job_config=bigquery.QueryJobConfig(
                        query_parameters=[bigquery.ScalarQueryParameter("d","DATE", args.date)])
                ).to_dataframe()
            # make JSON-safe
            payload = df.to_dict(orient="records")
            results.append({"check": name, "result": payload})

        # write single CSV
        with open(report_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["check", "result_json"])
            for r in results:
                w.writerow([r["check"], json.dumps(r["result"], default=str)])

        print(f"[ok] wrote {report_csv}")
        for r in results:
            print(f"{r['check']}: {json.dumps(r['result'], indent=2, default=str)}")

        # upload ONLY this run's temp dir
        base_prefix = f"{args.date_folder}/output/"
        upload_dir_to_gcs(tmp_dir, args.bucket, base_prefix)
        print(f"[done] Uploaded to gs://{args.bucket}/{base_prefix}")

        write_status(client, args.project, args.dataset, args.run_id, step, "SUCCESS", "validation report uploaded")

    except Exception as e:
        write_status(client, args.project, args.dataset, args.run_id, step, "FAILED", str(e))
        print("FAILED:", e, file=sys.stderr)
        sys.exit(1)
    finally:
        # clean up temp dir
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

if __name__ == "__main__":
    sys.exit(main())
