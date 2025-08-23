#!/usr/bin/env python3
import argparse, os, sys, shutil, uuid
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery, storage

NUM_COLS_CANDIDATES = [
    "tenure", "monthly_charges", "total_charges", "services_count",
    "avg_charges_per_tenure", "charges_per_service", "contract_length_months"
]
CAT_COLS_CANDIDATES = [
    "gender", "internet_service", "contract_type", "payment_method",
    "partner", "dependents", "phone_service", "multiple_lines",
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "paperless_billing",
    "tenure_bucket", "internet_type"
]

# ---------- helpers ----------
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

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--table", required=True)          # typically churn_clean or vw_latest_clean
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--out_dir", default="./output/eda_artifacts")
    ap.add_argument("--today_only", action="store_true", default=True,
                    help="Filter ingest_date = CURRENT_DATE() (default True)")
    args = ap.parse_args()

    step = "eda"
    bq = bigquery.Client(project=args.project)

    # temp subfolder so we upload only this run's files
    run_tmp = os.path.join(args.out_dir, f"_run_{args.run_id}_{uuid.uuid4().hex[:6]}")
    os.makedirs(run_tmp, exist_ok=True)

    try:
        # gate on validate_clean
        assert_prev_success(bq, args.project, args.dataset, args.run_id, "validate_clean")
        write_status(bq, args.project, args.dataset, args.run_id, step, "STARTED")

        # ---- Load from BigQuery (today only) ----
        where_clause = "WHERE ingest_date = CURRENT_DATE()" if args.today_only else ""
        sql = f"SELECT * FROM `{args.project}.{args.dataset}.{args.table}` {where_clause}"
        df = bq.query(sql).to_dataframe()

        if df.empty:
            raise RuntimeError("No rows found for today's partition; nothing to analyze.")
        if "churn_label" not in df.columns:
            raise RuntimeError("Expected 'churn_label' column in the input table/view.")

        num_cols = [c for c in NUM_COLS_CANDIDATES if c in df.columns]
        cat_cols = [c for c in CAT_COLS_CANDIDATES if c in df.columns]

        # ---- Summaries ----
        if num_cols:
            df[num_cols].describe().to_csv(os.path.join(run_tmp, "summary_numeric.csv"))
        pd.DataFrame({c: [df[c].nunique()] for c in cat_cols}).T.rename(columns={0:"unique_count"}).to_csv(
            os.path.join(run_tmp, "summary_categorical_uniques.csv")
        )

        # ---- Correlation CSV + Heatmap ----
        if num_cols:
            corr = df[num_cols + ["churn_label"]].corr(method="pearson")
            corr.to_csv(os.path.join(run_tmp, "correlation_numeric.csv"))

            plt.figure(figsize=(8,6))
            plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
            plt.colorbar(label="Pearson r")
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
            plt.yticks(range(len(corr.index)), corr.index)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(run_tmp, "correlation_heatmap.png"))
            plt.close()

            print("Top correlations with churn_label (numeric):")
            print(corr["churn_label"].drop("churn_label").sort_values(ascending=False).head(10))
            print()

        # ---- Categorical churn rates ----
        cat_churn_rates = {}
        for col in cat_cols:
            churn_rate = df.groupby(col)["churn_label"].mean()
            cat_churn_rates[col] = churn_rate.to_dict()
        if cat_churn_rates:
            pd.DataFrame.from_dict(cat_churn_rates, orient="index").T.to_csv(
                os.path.join(run_tmp, "categorical_churn_rates.csv")
            )

        print("Sample churn rates by category:")
        for col in cat_cols[:5]:
            print(f"{col}:")
            print(df.groupby(col)["churn_label"].mean().sort_values(ascending=False).head(3))
            print()

        # ---- Plots ----
        for col in num_cols:
            plt.figure()
            df[col].dropna().hist(bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col); plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(os.path.join(run_tmp, f"{col}_hist.png"))
            plt.close()

            plt.figure()
            df[[col,"churn_label"]].dropna().boxplot(column=col, by="churn_label")
            plt.suptitle(""); plt.title(f"{col} vs churn_label")
            plt.ylabel(col); plt.xlabel("churn_label")
            plt.tight_layout()
            plt.savefig(os.path.join(run_tmp, f"{col}_vs_churn.png"))
            plt.close()

        for col in cat_cols:
            plt.figure(figsize=(7,4))
            df[col].fillna("Unknown").value_counts().plot(kind="bar")
            plt.title(f"Distribution of {col}")
            plt.ylabel("count"); plt.xlabel(col)
            plt.tight_layout()
            plt.savefig(os.path.join(run_tmp, f"{col}_bar.png"))
            plt.close()

            plt.figure(figsize=(7,4))
            churn_rate = df.groupby(col)["churn_label"].mean().sort_values(ascending=False)
            churn_rate.plot(kind="bar")
            plt.title(f"Churn rate by {col}")
            plt.ylabel("churn_rate"); plt.xlabel(col)
            plt.tight_layout()
            plt.savefig(os.path.join(run_tmp, f"{col}_churn_rate.png"))
            plt.close()

        # ---- Upload ONLY this run's files ----
        date_str = datetime.now().strftime("%d-%m-%Y")
        base_prefix = f"{date_str}/output/eda"
        upload_dir_to_gcs(run_tmp, args.bucket, base_prefix)
        print(f"\nEDA artifacts uploaded to: gs://{args.bucket}/{base_prefix}")

        write_status(bq, args.project, args.dataset, args.run_id, step, "SUCCESS", "EDA artifacts uploaded")

    except Exception as e:
        write_status(bq, args.project, args.dataset, args.run_id, step, "FAILED", str(e))
        print("FAILED:", e, file=sys.stderr)
        sys.exit(1)
    finally:
        # clean up temp folder
        try:
            shutil.rmtree(run_tmp)
        except Exception:
            pass

if __name__ == "__main__":
    main()
