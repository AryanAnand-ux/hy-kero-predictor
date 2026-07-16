"""
database.py
───────────
SQLite database persistence for prediction history.
Allows pagination, querying, and runtime prediction logging.
"""

import os
import sqlite3
import json
from pathlib import Path
import pandas as pd
import logging
from urllib.parse import urlparse

try:
    from .constants import DEFAULT_RMSE, CI_Z_SCORE_95
except ImportError:
    from constants import DEFAULT_RMSE, CI_Z_SCORE_95

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    class _Psycopg2Fallback:
        def __init__(self):
            self.connect = None
            self.extras = None

    psycopg2 = _Psycopg2Fallback()

logger = logging.getLogger("hykero.database")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "database.db"
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "predictions_history.csv"


def _get_database_url() -> str:
    return os.getenv("DATABASE_URL", "").strip()


def _get_database_path() -> Path:
    database_url = _get_database_url()
    if database_url.startswith("sqlite:///"):
        parsed = urlparse(database_url)
        if parsed.path:
            path_value = parsed.path
            if os.name == "nt" and len(path_value) >= 3 and path_value[0] == "/" and path_value[2] == ":":
                path_value = path_value[1:]
            elif path_value.startswith("//") and len(path_value) > 2:
                path_value = path_value[1:]
            elif path_value.startswith("/") and not path_value.startswith("//"):
                path_value = path_value[1:]
            return Path(path_value)
    return DB_PATH


DB_PATH = _get_database_path()


def _is_postgres() -> bool:
    database_url = _get_database_url()
    return database_url.startswith("postgresql://") or database_url.startswith("postgres://")


def _format_query(query: str) -> str:
    if _is_postgres():
        return query.replace("?", "%s")
    return query


def _row_value(row, key: str):
    if row is None:
        return None
    if hasattr(row, "keys"):
        try:
            return row[key]
        except (KeyError, IndexError, TypeError):
            try:
                return row[0]
            except (KeyError, IndexError, TypeError):
                return None
    return row[0] if isinstance(row, (list, tuple)) else None


def get_db_connection():
    """Return a connection to SQLite by default or PostgreSQL when DATABASE_URL points to one."""
    database_url = _get_database_url()
    if _is_postgres():
        if psycopg2.connect is None:
            raise RuntimeError("psycopg2 is required for PostgreSQL support. Install backend requirements first.")
        cursor_factory = None
        if getattr(psycopg2, "extras", None) is not None and getattr(psycopg2.extras, "RealDictCursor", None) is not None:
            cursor_factory = psycopg2.extras.RealDictCursor
        conn = psycopg2.connect(
            database_url,
            cursor_factory=cursor_factory,
        )
        conn.autocommit = False
        return conn

    conn = sqlite3.connect(str(_get_database_path()))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database schema and seeds it from CSV if empty."""
    if not _is_postgres():
        db_path = _get_database_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        if _is_postgres():
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    sample_ts TEXT NOT NULL,
                    shift TEXT NOT NULL,
                    actual DOUBLE PRECISION,
                    predicted DOUBLE PRECISION NOT NULL,
                    residual DOUBLE PRECISION,
                    confidence_lower DOUBLE PRECISION,
                    confidence_upper DOUBLE PRECISION,
                    sensors TEXT,
                    lag_flash_gc DOUBLE PRECISION,
                    lag2_flash_gc DOUBLE PRECISION,
                    lag3_flash_gc DOUBLE PRECISION,
                    is_runtime BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_ts TEXT NOT NULL,
                    shift TEXT NOT NULL,
                    actual REAL,
                    predicted REAL NOT NULL,
                    residual REAL,
                    confidence_lower REAL,
                    confidence_upper REAL,
                    sensors TEXT,
                    lag_flash_gc REAL,
                    lag2_flash_gc REAL,
                    lag3_flash_gc REAL,
                    is_runtime INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
        conn.commit()

        # Create indexes
        try:
            cursor.execute(_format_query("CREATE INDEX IF NOT EXISTS idx_predictions_sample_ts ON predictions(sample_ts)"))
            cursor.execute(_format_query("CREATE INDEX IF NOT EXISTS idx_predictions_shift ON predictions(shift)"))
            cursor.execute(_format_query("CREATE INDEX IF NOT EXISTS idx_predictions_sample_ts_shift ON predictions(sample_ts, shift)"))
            cursor.execute(_format_query("CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id)"))
            cursor.execute(_format_query("CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON chat_messages(conversation_id)"))
            conn.commit()
        except Exception as idx_err:
            logger.warning(f"Could not create indexes: {idx_err}")

        # Seed predictions history
        cursor.execute(_format_query("SELECT COUNT(*) as cnt FROM predictions"))
        count = _row_value(cursor.fetchone(), "cnt")

        if count == 0:
            if CSV_PATH.exists():
                logger.info(f"Seeding database from CSV: {CSV_PATH}")
                try:
                    df = pd.read_csv(CSV_PATH)
                    rmse = DEFAULT_RMSE
                    try:
                        metrics_path = CSV_PATH.parent / "model_metrics.json"
                        best_name_path = CSV_PATH.parent.parent / "models" / "best_model_name.pkl"
                        if metrics_path.exists() and best_name_path.exists():
                            import joblib
                            best_name = joblib.load(best_name_path)
                            with open(metrics_path, "r") as f:
                                metrics_data = json.load(f)
                            rmse = metrics_data.get(best_name, {}).get("test", {}).get("rmse", DEFAULT_RMSE)
                    except Exception as ex:
                        logger.warning(f"Could not load best model RMSE for seeding: {ex}")

                    ci_half = CI_Z_SCORE_95 * rmse
                    rows_to_insert = []
                    for _, row in df.iterrows():
                        sample_ts = str(row["sample_ts"])
                        shift_val = str(row["shift"])
                        actual = float(row["actual"]) if pd.notnull(row["actual"]) else None
                        predicted = float(row["predicted"])
                        residual = float(row["residual"]) if pd.notnull(row["residual"]) else None
                        conf_lower = round(predicted - ci_half, 2)
                        conf_upper = round(predicted + ci_half, 2)

                        rows_to_insert.append((
                            sample_ts, shift_val, actual, predicted, residual,
                            conf_lower, conf_upper, "{}", None, None, None, 0
                        ))

                    cursor.executemany(_format_query("""
                        INSERT INTO predictions (
                            sample_ts, shift, actual, predicted, residual,
                            confidence_lower, confidence_upper, sensors,
                            lag_flash_gc, lag2_flash_gc, lag3_flash_gc, is_runtime
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """), rows_to_insert)
                    conn.commit()
                    logger.info(f"Successfully seeded {len(rows_to_insert)} records.")
                except Exception as e:
                    logger.error(f"Error seeding database: {e}")
    finally:
        conn.close()


def insert_prediction(sample_ts: str, shift: str, predicted: float, confidence_lower: float,
                      confidence_upper: float, sensors: dict, lag_flash_gc: float | None = None,
                      lag2_flash_gc: float | None = None, lag3_flash_gc: float | None = None,
                      actual: float | None = None) -> int:
    """Inserts a new prediction record into the database."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        residual = None
        if actual is not None:
            residual = round(actual - predicted, 2)

        sensors_json = json.dumps(sensors)

        if _is_postgres():
            cursor.execute(_format_query("""
                INSERT INTO predictions (
                    sample_ts, shift, actual, predicted, residual,
                    confidence_lower, confidence_upper, sensors,
                    lag_flash_gc, lag2_flash_gc, lag3_flash_gc, is_runtime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                RETURNING id
            """), (
                sample_ts, shift, actual, predicted, residual,
                confidence_lower, confidence_upper, sensors_json,
                lag_flash_gc, lag2_flash_gc, lag3_flash_gc
            ))
            new_id = _row_value(cursor.fetchone(), "id")
        else:
            cursor.execute(_format_query("""
                INSERT INTO predictions (
                    sample_ts, shift, actual, predicted, residual,
                    confidence_lower, confidence_upper, sensors,
                    lag_flash_gc, lag2_flash_gc, lag3_flash_gc, is_runtime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """), (
                sample_ts, shift, actual, predicted, residual,
                confidence_lower, confidence_upper, sensors_json,
                lag_flash_gc, lag2_flash_gc, lag3_flash_gc
            ))
            new_id = cursor.lastrowid
        conn.commit()
        return new_id
    finally:
        conn.close()


def get_predictions(shift: str | None = None, start_date: str | None = None,
                    end_date: str | None = None, limit: int = 200, offset: int = 0):
    """Queries prediction history with filtering and pagination."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []

        if shift:
            query += " AND UPPER(shift) = ?"
            params.append(shift.upper())

        if start_date:
            if _is_postgres():
                query += " AND sample_ts >= ?"
            else:
                query += " AND datetime(sample_ts) >= datetime(?)"
            params.append(start_date)

        if end_date:
            if _is_postgres():
                query += " AND sample_ts <= ?"
            else:
                query += " AND datetime(sample_ts) <= datetime(?)"
            params.append(end_date)

        count_query = query.replace("SELECT *", "SELECT COUNT(*) as cnt")
        cursor.execute(_format_query(count_query), params)
        total_count = _row_value(cursor.fetchone(), "cnt")

        if _is_postgres():
            query += " ORDER BY sample_ts DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
        else:
            query += " ORDER BY datetime(sample_ts) DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        cursor.execute(_format_query(query), params)
        rows = cursor.fetchall()

        result = []
        for r in rows:
            sensors_dict = {}
            try:
                if r["sensors"]:
                    sensors_dict = json.loads(r["sensors"])
            except Exception:
                pass

            result.append({
                "id": r["id"],
                "sample_ts": r["sample_ts"],
                "shift": r["shift"],
                "actual": r["actual"],
                "predicted": r["predicted"],
                "residual": r["residual"],
                "confidence_lower": r["confidence_lower"],
                "confidence_upper": r["confidence_upper"],
                "sensors": sensors_dict,
                "lag_flash_gc": r["lag_flash_gc"],
                "lag2_flash_gc": r["lag2_flash_gc"],
                "lag3_flash_gc": r["lag3_flash_gc"],
                "is_runtime": bool(r["is_runtime"])
            })

        return total_count, result
    finally:
        conn.close()


def get_prediction_stats():
    """Calculates statistics for history_stats API endpoint."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Single aggregate query: includes second-moment columns for SQL-based std computation
        cursor.execute(_format_query("""
            SELECT
                COUNT(*) as total,
                AVG(actual) as act_mean,
                AVG(actual * actual) as act_mean_sq,
                AVG(predicted) as pred_mean,
                AVG(residual) as res_mean,
                MIN(actual) as act_min,
                MAX(actual) as act_max,
                AVG((actual - predicted) * (actual - predicted)) as mse
            FROM predictions
        """))
        row = cursor.fetchone()

        total = row["total"] if row["total"] else 0

        if total == 0:
            return {
                "total_predictions": 0,
                "actual_mean": 0.0,
                "actual_std": 0.0,
                "actual_min": 0.0,
                "actual_max": 0.0,
                "predicted_mean": 0.0,
                "residual_mean": 0.0,
                "residual_std": 0.0
            }

        # Compute population std via Var = E[X²] - E[X]²  (no Python row fetch needed)
        act_std = 0.0
        if row["act_mean_sq"] is not None and row["act_mean"] is not None:
            act_std = max(0.0, row["act_mean_sq"] - row["act_mean"] ** 2) ** 0.5

        res_std = 0.0
        if row["mse"] is not None and row["res_mean"] is not None:
            res_std = max(0.0, row["mse"] - row["res_mean"] ** 2) ** 0.5

        return {
            "total_predictions": total,
            "actual_mean": round(row["act_mean"], 2) if row["act_mean"] else 0.0,
            "actual_std": round(act_std, 2),
            "actual_min": round(row["act_min"], 2) if row["act_min"] else 0.0,
            "actual_max": round(row["act_max"], 2) if row["act_max"] else 0.0,
            "predicted_mean": round(row["pred_mean"], 2) if row["pred_mean"] else 0.0,
            "residual_mean": round(row["res_mean"], 2) if row["res_mean"] else 0.0,
            "residual_std": round(res_std, 2)
        }
    finally:
        conn.close()




def create_conversation(session_id: str, title: str = "New Conversation") -> int:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if _is_postgres():
            cursor.execute(_format_query("INSERT INTO conversations (session_id, title) VALUES (?, ?) RETURNING id"), (session_id, title))
            new_id = _row_value(cursor.fetchone(), "id")
        else:
            cursor.execute(_format_query("INSERT INTO conversations (session_id, title) VALUES (?, ?)"), (session_id, title))
            new_id = cursor.lastrowid
        conn.commit()
        return new_id
    finally:
        conn.close()


def get_conversations_by_session(session_id: str) -> list:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            _format_query("SELECT * FROM conversations WHERE session_id = ? ORDER BY created_at DESC"),
            (session_id,)
        )
        rows = cursor.fetchall()
        return [{"id": r["id"], "session_id": r["session_id"], "title": r["title"], "created_at": r["created_at"]} for r in rows]
    finally:
        conn.close()


def check_conversation_exists(conversation_id: int) -> bool:
    """Check if a conversation exists in the database by its ID."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            _format_query("SELECT COUNT(*) as cnt FROM conversations WHERE id = ?"),
            (conversation_id,)
        )
        row = cursor.fetchone()
        count = _row_value(row, "cnt")
        return bool(count and count > 0)
    finally:
        conn.close()


def delete_conversation(conversation_id: int):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if not _is_postgres():
            cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute(_format_query("DELETE FROM conversations WHERE id = ?"), (conversation_id,))
        conn.commit()
    finally:
        conn.close()


def insert_chat_message(conversation_id: int, role: str, content: str) -> int:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if _is_postgres():
            cursor.execute(_format_query("INSERT INTO chat_messages (conversation_id, role, content) VALUES (?, ?, ?) RETURNING id"), (conversation_id, role, content))
            new_id = _row_value(cursor.fetchone(), "id")
        else:
            cursor.execute(_format_query("INSERT INTO chat_messages (conversation_id, role, content) VALUES (?, ?, ?)"), (conversation_id, role, content))
            new_id = cursor.lastrowid
        conn.commit()
        return new_id
    finally:
        conn.close()


def get_chat_history(conversation_id: int) -> list:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            _format_query("SELECT * FROM chat_messages WHERE conversation_id = ? ORDER BY created_at ASC"),
            (conversation_id,)
        )
        rows = cursor.fetchall()
        return [{"id": r["id"], "conversation_id": r["conversation_id"], "role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in rows]
    finally:
        conn.close()
