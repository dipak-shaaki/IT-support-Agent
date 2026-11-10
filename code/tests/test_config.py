import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_production_requires_non_default_admin_key():
    with pytest.raises(ValidationError, match="ADMIN_API_KEY"):
        Settings(app_env="production", admin_api_key="change-me")


def test_production_rejects_wildcard_cors():
    with pytest.raises(ValidationError, match="CORS_ORIGINS"):
        Settings(app_env="production", admin_api_key="unique-secret", cors_origins="*")


def test_production_requires_proxy_authentication():
    with pytest.raises(ValidationError, match="AUTH_MODE=proxy"):
        Settings(
            app_env="production",
            admin_api_key="unique-secret",
            cors_origins="https://support.example.com",
            auth_mode="none",
        )


def test_production_accepts_proxy_authentication():
    settings = Settings(
        app_env="production",
        admin_api_key="unique-secret",
        cors_origins="https://support.example.com",
        auth_mode="proxy",
    )

    assert settings.auth_mode == "proxy"
