from core.utils import auth
from fastapi import HTTPException, status, FastAPI, Depends
from fastapi.testclient import TestClient
import pytest

@pytest.fixture(autouse=True)
def setup_auth(monkeypatch):
    monkeypatch.setattr(auth, "BEARER_TOKEN", "test_token")

@pytest.mark.parametrize(
    "authorization",
    ["Bearer test_token", "bearer test_token", "BEARER test_token"]
)
def test_verify_bearer_returns_token(authorization):
    assert auth.verify_bearer(authorization) == "test_token"

@pytest.mark.parametrize(
    "authorization, expected_status, expected_detail",
    [
        (None, status.HTTP_401_UNAUTHORIZED, "Missing Auth Token"),
        ("", status.HTTP_401_UNAUTHORIZED, "Missing Auth Token"),
        ("   ", status.HTTP_401_UNAUTHORIZED, "Missing Auth Token"),
        ("Bearer", status.HTTP_401_UNAUTHORIZED, "Invalid Bearer Token"),
        ("Bearer ", status.HTTP_401_UNAUTHORIZED, "Invalid Bearer Token"),
        ("Basic test_token", status.HTTP_401_UNAUTHORIZED, "Invalid Bearer Token"),
        ("Bearer wrong_token", status.HTTP_401_UNAUTHORIZED, "Invalid Bearer Token"),
    ],
)
def test_verify_bearer_rejects_bad_header(authorization, expected_status, expected_detail):
    with pytest.raises(HTTPException) as exc_info:
        auth.verify_bearer(authorization)
    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail

@pytest.mark.parametrize("server_token", [None, ""])
def test_verify_bearer_rejects_missing_server_token(monkeypatch, server_token):
    monkeypatch.setattr(auth, "BEARER_TOKEN", server_token)
    with pytest.raises(HTTPException) as exc_info:
        auth.verify_bearer("Bearer test_token")
    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc_info.value.detail == "Server auth not configured"

def test_missing_token_returns_401():
    app = FastAPI()

    @app.get("/protected")
    def handle_protected_endpoint(token: str = Depends(auth.verify_bearer)):
        return {"message": "Access granted", "token": token}

    client = TestClient(app)
    response = client.get("/protected")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == {"detail": "Missing Auth Token"}