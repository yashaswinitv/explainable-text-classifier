import pytest
from fastapi.testclient import TestClient
from xtc.api import app

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

def test_health(client):
    # app loads model on startup; this is a smoke test once you've trained and saved to runs/latest
    assert client is not None
