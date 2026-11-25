"""E2E tests need a live server (uvicorn inference.app:app --port 8000). Skip cleanly otherwise."""
import socket
import pytest


def _server_up(host="127.0.0.1", port=8000):
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    if _server_up():
        return
    skip = pytest.mark.skip(reason="no server on 127.0.0.1:8000 (start: uvicorn inference.app:app --port 8000)")
    for item in items:
        if "tests/e2e" in str(item.fspath).replace("\\", "/"):
            item.add_marker(skip)
