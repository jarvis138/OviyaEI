from fastapi.testclient import TestClient
from app.main import app


def test_healthz():
    """
    Verify the application's /healthz endpoint responds with HTTP 200 and a JSON body where "status" equals "ok".
    
    This test creates a TestClient for the FastAPI app, sends a GET request to /healthz, and asserts the response status code and JSON payload.
    """
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ok"