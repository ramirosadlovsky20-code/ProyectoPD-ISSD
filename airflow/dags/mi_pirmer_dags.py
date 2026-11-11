from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "ramiro",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="pipeline_pp1_spark_bash",
    default_args=default_args,
    description=(
        "Pipeline completo de PySpark orquestado "
        "mediante __main__.py y BashOperator"
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    correr_pipeline_completo = BashOperator(
        task_id="ejecutar_fase_todas",

        bash_command=(
            "python /opt/airflow/dags/__main__.py "
            "--fase todas --clean"
        ),

        cwd="/opt/airflow/project",
    )

    correr_pipeline_completo