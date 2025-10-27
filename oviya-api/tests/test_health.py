from fastapi.testclient import TestClient
from app.main import app


def test_healthz():
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ok"






