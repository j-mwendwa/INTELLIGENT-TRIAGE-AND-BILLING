"""
tests/unit/test_api_health.py — Health endpoint contract tests.
"""

from __future__ import annotations


def test_health_endpoint_is_namespaced_under_api_v1(api_client):
    response = api_client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
