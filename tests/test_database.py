import importlib
import sys
import os
from pathlib import Path
import pytest

# Add backend/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import database


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    # Redirect database path and csv path to temp/nonexistent locations for testing
    test_db_path = tmp_path / "test_database.db"
    original_db_path = database.DB_PATH
    original_csv_path = database.CSV_PATH
    
    database.DB_PATH = test_db_path
    database.CSV_PATH = Path("/nonexistent/predictions_history.csv")
    
    # Initialize schema without seeding
    database.init_db()
    
    yield
    
    # Restore paths
    database.DB_PATH = original_db_path
    database.CSV_PATH = original_csv_path


def test_db_init_and_insertion():
    # Insert a dummy record
    sensors = {"MF_HK_Draw_T": 224.0, "MF_FlashZone_T": 365.0}
    new_id = database.insert_prediction(
        sample_ts="2026-06-25 12:00:00",
        shift="E",
        predicted=76.5,
        confidence_lower=71.0,
        confidence_upper=82.0,
        sensors=sensors,
        lag_flash_gc=75.0,
        lag2_flash_gc=74.0,
        lag3_flash_gc=76.0,
        actual=77.0
    )
    
    assert new_id is not None
    assert new_id > 0

    # Retrieve record
    total, records = database.get_predictions(shift="E", limit=10)
    assert total == 1
    assert len(records) == 1
    
    record = records[0]
    assert record["shift"] == "E"
    assert record["predicted"] == 76.5
    assert record["actual"] == 77.0
    assert record["residual"] == 0.5  # actual - predicted
    assert record["sensors"]["MF_HK_Draw_T"] == 224.0
    assert record["lag_flash_gc"] == 75.0


def test_db_pagination():
    # Insert multiple records
    for i in range(15):
        database.insert_prediction(
            sample_ts=f"2026-06-25 10:{i:02d}:00",
            shift="M",
            predicted=70.0 + i,
            confidence_lower=65.0 + i,
            confidence_upper=75.0 + i,
            sensors={}
        )
        
    # Get page 1 (first 10)
    total, page1 = database.get_predictions(limit=10, offset=0)
    assert total == 15
    assert len(page1) == 10
    
    # Get page 2 (remaining 5)
    total, page2 = database.get_predictions(limit=10, offset=10)
    assert total == 15
    assert len(page2) == 5


def test_db_stats():
    # Insert predictions with actual values to calculate stats
    database.insert_prediction("2026-06-25 06:00:00", "M", predicted=75.0, confidence_lower=70.0, confidence_upper=80.0, sensors={}, actual=77.0)
    database.insert_prediction("2026-06-25 14:00:00", "E", predicted=80.0, confidence_lower=75.0, confidence_upper=85.0, sensors={}, actual=79.0)
    
    stats = database.get_prediction_stats()
    
    assert stats["total_predictions"] == 2
    assert stats["actual_mean"] == 78.0  # (77 + 79) / 2
    assert stats["predicted_mean"] == 77.5  # (75 + 80) / 2
    assert stats["residual_mean"] == 0.5  # ((77-75) + (79-80)) / 2 = (2 - 1) / 2 = 0.5
    # Residual mean in query is AVG(residual). Inserted residuals:
    # 77.0 - 75.0 = 2.0
    # 79.0 - 80.0 = -1.0
    # AVG is 0.5. Let's check stats keys.


def test_database_url_override_uses_custom_sqlite_path(tmp_path, monkeypatch):
    custom_db = tmp_path / "custom.sqlite"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{custom_db}")
    importlib.reload(database)

    database.init_db()

    assert custom_db.exists()
    conn = database.get_db_connection()
    assert conn is not None
    conn.close()

    importlib.reload(database)


def test_postgres_connection_uses_psycopg2_parameters(monkeypatch):
    class FakeCursor:
        def __init__(self):
            self.executed_queries = []
            self.lastrowid = 1

        def execute(self, query, params=None):
            self.executed_queries.append((query, params))
            if "INSERT INTO predictions" in query:
                self.lastrowid = 1

        def executemany(self, query, params):
            self.executed_queries.append((query, params))

        def fetchone(self):
            return {"cnt": 0}

        def fetchall(self):
            return []

    class FakeConnection:
        def __init__(self):
            self.cursor_obj = FakeCursor()

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            return None

        def close(self):
            return None

        @property
        def autocommit(self):
            return False

        @autocommit.setter
        def autocommit(self, value):
            self._autocommit = value

    fake_conn = FakeConnection()

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/app")
    importlib.reload(database)
    monkeypatch.setattr(database.psycopg2, "connect", lambda url, cursor_factory=None: fake_conn)

    database.init_db()
    database.insert_prediction(
        sample_ts="2026-06-25 12:00:00",
        shift="E",
        predicted=76.5,
        confidence_lower=71.0,
        confidence_upper=82.0,
        sensors={},
    )

    insert_query = next(q for q in fake_conn.cursor_obj.executed_queries if "INSERT INTO predictions" in q[0])
    assert "%s" in insert_query[0]
    assert "?" not in insert_query[0]
