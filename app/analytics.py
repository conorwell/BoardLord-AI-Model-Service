import json
import os
from datetime import datetime, timezone

from google.cloud import bigquery
from google.cloud.exceptions import Conflict

_client: bigquery.Client = None
_table_ref: str = None

_SCHEMA = [
    bigquery.SchemaField("client_id", "STRING"),
    bigquery.SchemaField("session_id", "STRING"),
    bigquery.SchemaField("timestamp", "TIMESTAMP"),
    bigquery.SchemaField("angle", "INTEGER"),
    bigquery.SchemaField("hold_count", "INTEGER"),
    bigquery.SchemaField("holds", "STRING"),        # JSON array of {position_id, role}
    bigquery.SchemaField("is_nomatch", "BOOLEAN"),
    bigquery.SchemaField("predicted_grade", "STRING"),
    bigquery.SchemaField("predicted_grade_class", "INTEGER"),
    bigquery.SchemaField("confidence", "FLOAT"),
    bigquery.SchemaField("response_time_ms", "INTEGER"),
    bigquery.SchemaField("error", "STRING"),        # NULL on success
]

_DATASET_ID = "board_lord_analytics"
_TABLE_ID = "prediction_events"


def init():
    global _client, _table_ref

    try:
        _client = bigquery.Client()
        project = _client.project
        _table_ref = f"{project}.{_DATASET_ID}.{_TABLE_ID}"

        dataset = bigquery.Dataset(f"{project}.{_DATASET_ID}")
        dataset.location = "US"
        try:
            _client.create_dataset(dataset)
            print(f"Created BigQuery dataset {_DATASET_ID}")
        except Conflict:
            pass  # already exists

        table = bigquery.Table(_table_ref, schema=_SCHEMA)
        try:
            _client.create_table(table)
            print(f"Created BigQuery table {_TABLE_ID}")
        except Conflict:
            pass  # already exists

        print("Analytics initialized.")
    except Exception as e:
        print(f"Analytics init failed (events will not be logged): {e}")
        _client = None


def log_prediction(
    *,
    client_id: str,
    session_id: str,
    angle: int,
    holds: list[dict],
    is_nomatch: bool,
    predicted_grade: str,
    predicted_grade_class: int,
    confidence: float,
    response_time_ms: int,
    error: str | None = None,
):
    if _client is None:
        return

    row = {
        "client_id": client_id or "unknown",
        "session_id": session_id or "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "angle": angle,
        "hold_count": len(holds),
        "holds": json.dumps(holds),
        "is_nomatch": is_nomatch,
        "predicted_grade": predicted_grade,
        "predicted_grade_class": predicted_grade_class,
        "confidence": confidence,
        "response_time_ms": response_time_ms,
        "error": error,
    }

    try:
        errors = _client.insert_rows_json(_table_ref, [row])
        if errors:
            print(f"BigQuery insert errors: {errors}")
    except Exception as e:
        print(f"Failed to log analytics event: {e}")
