# dags/churn_pipeline_dag.py
from __future__ import annotations
import os, subprocess
from datetime import datetime, timedelta
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.sensors.gcs import GCSObjectExistenceSensor

PROJECT_ID   = os.environ.get("GCP_PROJECT") or Variable.get("GCP_PROJECT", default_var="YOUR_PROJECT_ID")
BUCKET       = Variable.get("CHURN_BUCKET")                 # e.g. dmml_assignment
DATASET      = Variable.get("CHURN_DATASET")                # dmml_assignment_churn_raw
RAW_TABLE    = "churn_raw"
CLEAN_TABLE  = "churn_clean"
FEATURES_TBL = "churn_features"

# All steps share one run_id = Airflow run_id
def _date_folder(execution_date: datetime) -> str:
    # our convention: dd-MM-YYYY
    return execution_date.strftime("%d-%m-%Y")

def _run(cmd: list[str]):
    # Run a python script that lives in /home/airflow/gcs/dags/
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

default_args = {
    "owner": "you",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="churn_pipeline_composer",
    start_date=datetime(2025, 1, 1),
    schedule_interval="30 4 * * *",   # run daily at 10:00 AM
    catchup=False,
    default_args=default_args,
    tags=["churn","dmml","composer"],
) as dag:

    def step_prepare_raw(**ctx):
        run_id = ctx["run_id"]
        _run([
            "python", "/home/airflow/gcs/dags/prepare_raw_data_from_sources.py",
            "--bucket", BUCKET,
        ])

    def step_load_raw(**ctx):
        run_id = ctx["run_id"]
        _run([
            "python", "/home/airflow/gcs/dags/load_to_bigquery.py",
            "--project", PROJECT_ID,
            "--dataset", DATASET,
            "--table", RAW_TABLE,
            "--bucket", BUCKET,
            "--run_id", run_id,
        ])

    def step_clean(**ctx):
        run_id = ctx["run_id"]
        _run([
            "python", "/home/airflow/gcs/dags/clean_raw_to_curated.py",
            "--project", PROJECT_ID,
            "--dataset", DATASET,
            "--raw_table", RAW_TABLE,
            "--clean_table", CLEAN_TABLE,
            "--run_id", run_id,
        ])

    def step_validate(**ctx):
        run_id = ctx["run_id"]
        _run([
            "python", "/home/airflow/gcs/dags/validate_clean_data.py",
            "--project", PROJECT_ID,
            "--dataset", DATASET,
            "--table", CLEAN_TABLE,
            "--bucket", BUCKET,
            "--run_id", run_id,
        ])

    def step_eda(**ctx):
        run_id = ctx["run_id"]
        _run([
            "python", "/home/airflow/gcs/dags/exploratory_data_analysis_from_bq.py",
            "--project", PROJECT_ID,
            "--dataset", DATASET,
            "--table", CLEAN_TABLE,
            "--bucket", BUCKET,
            "--run_id", run_id,
        ])

    def step_features(**ctx):
        run_id = ctx["run_id"]
        _run([
            "python", "/home/airflow/gcs/dags/build_domain_driven_features.py",
            "--project", PROJECT_ID,
            "--dataset", DATASET,
            "--clean_table", CLEAN_TABLE,
            "--features_table", FEATURES_TBL,
            "--run_id", run_id,
            "--gate_on", "validate_clean",
        ])

    def step_train(**ctx):
        run_id = ctx["run_id"]
        _run([
            "python", "/home/airflow/gcs/dags/train_model.py",
            "--project", PROJECT_ID,
            "--dataset", DATASET,
            "--features_table", FEATURES_TBL,
            "--bucket", BUCKET,
            "--run_id", run_id,
        ])

    t_raw_prepare   = PythonOperator(task_id="prepare_raw", python_callable=step_prepare_raw, provide_context=True)
    t_load          = PythonOperator(task_id="load_raw", python_callable=step_load_raw, provide_context=True)
    t_clean         = PythonOperator(task_id="clean_curated", python_callable=step_clean, provide_context=True)
    t_validate      = PythonOperator(task_id="validate_clean", python_callable=step_validate, provide_context=True)
    t_eda           = PythonOperator(task_id="eda", python_callable=step_eda, provide_context=True)
    t_feat          = PythonOperator(task_id="build_features", python_callable=step_features, provide_context=True)
    t_train         = PythonOperator(task_id="train_model", python_callable=step_train, provide_context=True)

    t_raw_prepare >> t_load >> t_clean >> t_validate >> t_eda >> t_feat >> t_train
