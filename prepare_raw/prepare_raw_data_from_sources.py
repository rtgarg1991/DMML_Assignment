#!/usr/bin/env python3

import argparse
import os
import sys
import io
import re
import glob
import tempfile
from datetime import datetime, timezone

import pandas as pd
from google.cloud import storage

# You need kagglehub in requirements.txt
try:
    import kagglehub
except Exception as e:
    print("ERROR: 'kagglehub' not installed. Add it to requirements.txt", file=sys.stderr)
    raise

KAGGLE_DATASET = "blastchar/telco-customer-churn"

# Final schema we want in RAW (snake_case, with new customer_id first)
EXPECTED_COLS_SNAKE = [
    "gender","senior_citizen","partner","dependents","tenure",
    "phone_service","multiple_lines","internet_service","online_security",
    "online_backup","device_protection","tech_support","streaming_tv",
    "streaming_movies","contract","paperless_billing","payment_method",
    "monthly_charges","total_charges","churn", "ingest_date"
]

def log(msg: str):
    print(f"[prepare] {msg}")

def snake_case(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s.strip())
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = s.lower()
    return re.sub(r"_+", "_", s).strip("_")

def load_kaggle_df() -> pd.DataFrame:
    log("Downloading Kaggle dataset…")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csvs:
        raise RuntimeError(f"No CSV found under Kaggle path: {path}")
    # Usually: WA_Fn-UseC_-Telco-Customer-Churn.csv
    kaggle_csv = csvs[0]
    log(f"Found Kaggle CSV: {kaggle_csv}")
    df = pd.read_csv(kaggle_csv)
    log(f"Kaggle rows: {len(df)}")
    return df

def load_custom_from_gcs(bucket: str) -> pd.DataFrame:
    client = storage.Client()
    bkt = client.bucket(bucket)
    blob = bkt.blob("custom_data.csv")
    if not blob.exists():
        log("No custom_data.csv in bucket root; proceeding with Kaggle only.")
        return pd.DataFrame()
    log("Downloading custom_data.csv from GCS…")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        blob.download_to_filename(tmp.name)
        df = pd.read_csv(tmp.name)
    os.unlink(tmp.name)
    log(f"Custom rows: {len(df)}")
    return df

def normalize_columns_to_snake(df: pd.DataFrame) -> pd.DataFrame:
    # 1) snake_case headers
    df = df.rename(columns={c: snake_case(c) for c in df.columns})

    # 2) map kaggle-style names to our exact snake names
    rename_map = {
        "customerid": "customer_id",
        "seniorcitizen": "senior_citizen",
        "phoneservice": "phone_service",
        "multiplelines": "multiple_lines",
        "internetservice": "internet_service",
        "onlinesecurity": "online_security",
        "onlinebackup": "online_backup",
        "deviceprotection": "device_protection",
        "techsupport": "tech_support",
        "streamingtv": "streaming_tv",
        "streamingmovies": "streaming_movies",
        "paperlessbilling": "paperless_billing",
        "paymentmethod": "payment_method",
        "monthlycharges": "monthly_charges",
        "totalcharges": "total_charges",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 3) Tidy churn values to 'Yes'/'No'
    if "churn" in df.columns:
        df["churn"] = df["churn"].astype(str).str.strip().str.title()

    # 4) Ensure all expected columns exist (create NA if missing)
    for col in EXPECTED_COLS_SNAKE:
        if col not in df.columns:
            df[col] = pd.NA

    # 5) Reorder to expected; keep any extra columns at end (except old id)
    ordered = EXPECTED_COLS_SNAKE + [c for c in df.columns if c not in EXPECTED_COLS_SNAKE and c != "customer_id"]
    df = df[ordered + (["customer_id"] if "customer_id" in df.columns else [])]

    return df

def combine_and_assign_ids(kaggle_df: pd.DataFrame, custom_df: pd.DataFrame) -> pd.DataFrame:
    log("Normalizing Kaggle columns to snake_case…")
    k = normalize_columns_to_snake(kaggle_df)

    if not custom_df.empty:
        log("Normalizing custom columns to snake_case…")
        c = normalize_columns_to_snake(custom_df)
    else:
        c = pd.DataFrame(columns=k.columns)

    # Drop any existing customer_id from both sources
    for df in (k, c):
        if "customer_id" in df.columns:
            df.drop(columns=["customer_id"], inplace=True)

    # Concatenate: Kaggle first, then custom
    log("Combining Kaggle + custom data…")
    combined = pd.concat([k, c], ignore_index=True)

    # Coerce numeric-looking columns (safe — later steps use SAFE_CAST anyway)
    for col in ["tenure", "monthly_charges", "total_charges"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Assign new auto-increment customer_id starting at 1
    combined.insert(0, "customer_id", range(1, len(combined) + 1))

    # Keep only the final schema (id + expected cols)
    keep = ["customer_id"] + EXPECTED_COLS_SNAKE
    extras = [c for c in combined.columns if c not in keep]
    if extras:
        log(f"Dropping unexpected extra columns: {extras}")
        combined = combined[keep]

    # Quick sanity logs
    log(f"Combined rows: {len(combined)}")
    log("Null counts (key cols): " + str(combined[["tenure","monthly_charges","total_charges","churn"]].isna().sum().to_dict()))
    return combined

def upload_df_to_gcs_csv(df: pd.DataFrame, bucket: str, dest_path: str):
    client = storage.Client()
    bkt = client.bucket(bucket)
    blob = bkt.blob(dest_path)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    blob.upload_from_file(io.BytesIO(csv_bytes), content_type="text/csv")
    log(f"Uploaded → gs://{bucket}/{dest_path} (rows={len(df)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="Target GCS bucket (no gs://). custom_data.csv should be in bucket root.")
    ap.add_argument("--date", help="YYYY-MM-DD for ingest_date (BQ). Defaults to UTC today.")
    ap.add_argument("--date_folder", help="dd-MM-YYYY for GCS path. Defaults to UTC today.")

    args = ap.parse_args()

    log("Starting raw data preparation…")
    
    date_iso = args.date or datetime.utcnow().strftime("%Y-%m-%d")
    date_folder = args.date_folder or datetime.utcnow().strftime("%d-%m-%Y")
    log(f"Using date={date_iso}, date_folder={date_folder}")


    kaggle_df = load_kaggle_df()
    custom_df = load_custom_from_gcs(args.bucket)

    combined = combine_and_assign_ids(kaggle_df, custom_df)
    
    # --- ensure numeric types match the BQ table ---
    # integers
    for col in ["customer_id", "senior_citizen", "tenure"]:
        if col in combined.columns:
            combined[col] = (
                pd.to_numeric(combined[col], errors="coerce")
                .round(0)
                .astype("Int64")      # pandas nullable int (writes as integer, not 0.0)
            )

    # floats
    for col in ["monthly_charges", "total_charges"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("Float64")
            
    combined["ingest_date"] = date_iso    

    dest = f"{date_folder}/churn_raw.csv"
    upload_df_to_gcs_csv(combined, args.bucket, dest)

    log("Preview (first 5 rows):")
    log("\n" + combined.head(5).to_string(index=False))
    log("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FAILED: {e}")
        sys.exit(1)
