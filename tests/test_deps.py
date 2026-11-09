import pytest
from fastapi import HTTPException
from redis.exceptions import RedisError
from starlette.requests import Request

from app.api import deps


def make_request(**headers) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [
                (key.lower().encode(), value.encode()) for key, value in headers.items()
            ],
        }
    )


class FakeRedis:
    def __init__(self):
        self.counts = {}
        self.expiry = {}

    def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def expire(self, key, seconds):
        self.expiry[key] = seconds


def test_current_user_is_none_when_auth_is_disabled(monkeypatch):
    monkeypatch.setattr(deps.settings, "auth_mode", "none")

    assert deps.current_user(make_request()) is None


def test_current_user_reads_trusted_proxy_header(monkeypatch):
    monkeypatch.setattr(deps.settings, "auth_mode", "proxy")
    monkeypatch.setattr(deps.settings, "auth_user_header", "X-Auth-Request-User")

    request = make_request(**{"X-Auth-Request-User": "alice@example.com"})

    assert deps.current_user(request) == "alice@example.com"


def test_current_user_requires_header_when_auth_is_proxy(monkeypatch):
    monkeypatch.setattr(deps.settings, "auth_mode", "proxy")

    with pytest.raises(HTTPException) as error:
        deps.current_user(make_request())

    assert error.value.status_code == 401


def test_rate_limit_rejects_requests_over_the_window(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr(deps, "get_redis_client", lambda: client)
    monkeypatch.setattr(deps.settings, "rate_limit_per_minute", 2)

    deps.rate_limit("alice")
    deps.rate_limit("alice")
    with pytest.raises(HTTPException) as error:
        deps.rate_limit("alice")

    assert error.value.status_code == 429
    assert error.value.headers == {"Retry-After": "60"}


def test_rate_limit_fails_open_when_redis_is_unavailable(monkeypatch):
    class BrokenRedis:
        def incr(self, key):
            raise RedisError("down")

    monkeypatch.setattr(deps, "get_redis_client", lambda: BrokenRedis())
    monkeypatch.setattr(deps.settings, "rate_limit_per_minute", 1)

    deps.rate_limit("alice")


def test_require_admin_accepts_the_configured_key(monkeypatch):
    monkeypatch.setattr(deps.settings, "admin_api_key", "secret")

    deps.require_admin(make_request(), "secret")


def test_require_admin_accepts_a_proxy_admin_role(monkeypatch):
    monkeypatch.setattr(deps.settings, "auth_mode", "proxy")
    monkeypatch.setattr(deps.settings, "auth_roles_header", "X-Auth-Request-Roles")
    monkeypatch.setattr(
        deps.settings, "auth_admin_roles", "it-support-admins, platform"
    )
    request = make_request(
        **{"X-Auth-Request-Roles": "employee, IT-Support-Admins"}
    )

    deps.require_admin(request, "")


@pytest.mark.parametrize(
    ("auth_mode", "roles"),
    [("proxy", "employee"), ("none", "it-support-admins")],
)
def test_require_admin_rejects_missing_or_untrusted_roles(monkeypatch, auth_mode, roles):
    monkeypatch.setattr(deps.settings, "auth_mode", auth_mode)
    monkeypatch.setattr(deps.settings, "auth_admin_roles", "it-support-admins")
    request = make_request(**{"X-Auth-Request-Roles": roles})

    with pytest.raises(HTTPException) as error:
        deps.require_admin(request, "")

    assert error.value.status_code == 401


def test_require_admin_rejects_an_empty_key_when_no_key_is_configured(monkeypatch):
    monkeypatch.setattr(deps.settings, "admin_api_key", "")
    monkeypatch.setattr(deps.settings, "auth_mode", "none")

    with pytest.raises(HTTPException) as error:
        deps.require_admin(make_request(), "")

    assert error.value.status_code == 401
