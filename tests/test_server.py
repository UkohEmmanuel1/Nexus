from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.inference.server import create_app
from src.utils.key_manager import KeyManager


@pytest.fixture(autouse=True)
def patch_key_manager_db(tmp_path, monkeypatch):
    """Ensure all tests use a temporary SQLite database."""
    db_file = tmp_path / "test_nexus_keys.db"
    original_init = KeyManager.__init__

    def mock_init(self, db_path=None):
        original_init(self, db_path=str(db_file))

    monkeypatch.setattr(KeyManager, "__init__", mock_init)


@pytest.fixture
def mock_engine(mock_tokenizer):
    engine = MagicMock()
    engine.device = "cpu"
    engine.tokenizer = mock_tokenizer

    def mock_generate(prompt_or_msg, **kwargs):
        if kwargs.get("stream"):
            return ["mock", " stream", " output"]
        if kwargs.get("thinking_mode"):
            return "mock output", "mock thinking trace"
        return "mock output"

    engine.generate.side_effect = mock_generate
    return engine


def test_key_manager_lifecycle():
    km = KeyManager()

    # 1. Validating non-existent key
    assert not km.validate_key("invalid_key")

    # 2. Generate key
    key = km.generate_key("test_user")
    assert key.startswith("nexus_sk_")

    # 3. Validate generated key
    assert km.validate_key(key)

    # 4. Revoke key
    assert km.revoke_key(key)

    # 5. Revoking again returns False
    assert not km.revoke_key(key)

    # 6. Validate revoked key
    assert not km.validate_key(key)


def test_server_authentication_and_routes(mock_engine, monkeypatch):
    # Set up master api key and admin key in environment
    monkeypatch.setenv("NEXUS_API_KEY", "master_api_key")
    monkeypatch.setenv("NEXUS_ADMIN_KEY", "admin_api_key")

    app = create_app(mock_engine, api_key="master_api_key")
    client = TestClient(app)

    # 1. Public route check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    res = client.get("/models")
    assert res.status_code == 200
    assert "models" in res.json()

    # 2. Protected route check (unauthorized)
    res = client.post("/chat", json={"message": "hello"})
    assert res.status_code == 401

    # 3. Access using master api key (Bearer)
    res = client.post(
        "/chat", json={"message": "hello"}, headers={"Authorization": "Bearer master_api_key"}
    )
    assert res.status_code == 200
    assert res.json()["response"] == "mock output"

    # 4. Generate key endpoint check (Requires NEXUS_ADMIN_KEY)
    # 4a. Without admin key should fail
    res = client.post("/v1/keys/generate", json={"name": "developer_1"})
    assert res.status_code == 401

    # 4b. With correct admin key should succeed
    res = client.post(
        "/v1/keys/generate", json={"name": "developer_1"}, headers={"X-API-Key": "admin_api_key"}
    )
    assert res.status_code == 200
    generated_data = res.json()
    assert "key" in generated_data
    generated_key = generated_data["key"]
    assert generated_data["name"] == "developer_1"

    # 4c. List keys without admin key should fail
    res = client.get("/v1/keys")
    assert res.status_code == 401

    # 4d. List keys with admin key should succeed
    res = client.get("/v1/keys", headers={"X-API-Key": "admin_api_key"})
    assert res.status_code == 200
    keys_list = res.json()
    assert len(keys_list) >= 1
    assert any(k["key"] == generated_key for k in keys_list)

    # 5. Access protected route with the newly generated key
    res = client.post("/chat", json={"message": "hello"}, headers={"X-API-Key": generated_key})
    assert res.status_code == 200
    assert res.json()["response"] == "mock output"

    # 6. Revoke key (requires check_auth, so we pass master_api_key or the valid key itself)
    res = client.post(
        "/v1/keys/revoke",
        json={"key": generated_key},
        headers={"Authorization": "Bearer master_api_key"},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # 7. Access protected route with the revoked key should fail
    res = client.post("/chat", json={"message": "hello"}, headers={"X-API-Key": generated_key})
    assert res.status_code == 401


def test_server_rate_limiter(mock_engine, monkeypatch):
    # Setup server with low rate limit window of 1 request
    app = create_app(mock_engine, api_key="master_key", rate_limit=2)
    client = TestClient(app)

    headers = {"Authorization": "Bearer master_key"}

    # First request
    res = client.post("/chat", json={"message": "first"}, headers=headers)
    assert res.status_code == 200

    # Second request
    res = client.post("/chat", json={"message": "second"}, headers=headers)
    assert res.status_code == 200

    # Third request within 60s should trigger rate limit (429)
    res = client.post("/chat", json={"message": "third"}, headers=headers)
    assert res.status_code == 429
    assert res.json()["detail"] == "rate_limit_exceeded"
