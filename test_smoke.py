from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_and_assets():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["history_rows"] == 1348
    assert body["feature_count"] == 6

def test_forecast_has_expected_horizon_and_dates():
    response = client.get("/api/forecast?horizon=7")
    assert response.status_code == 200
    body = response.json()
    assert body["horizon"] == 7
    assert len(body["daily_forecast"]) == 7
    assert body["forecast_start"] == "2014-09-30"
    assert body["forecast_end"] == "2014-10-06"
    assert body["total_forecast_revenue"] >= 0

def test_horizon_limit():
    response = client.get("/api/forecast?horizon=91")
    assert response.status_code == 422


def test_history_csv_has_expected_columns():
    response = client.get("/api/history")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1348
    assert body[0]["Date"] == "2011-01-21"
    assert body[-1]["Date"] == "2014-09-29"
    assert "ProductCount" in body[0]
