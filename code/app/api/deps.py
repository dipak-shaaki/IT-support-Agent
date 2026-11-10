import logging
import secrets
import time

from fastapi import Depends, Header, HTTPException, Request
from redis.exceptions import RedisError

from app.core.config import get_settings
from app.core.metrics import RATE_LIMITED
from app.services.conversation import get_redis_client

logger = logging.getLogger(__name__)
settings = get_settings()


def _admin_roles(header_value: str) -> set[str]:
    return {
        role.strip().casefold()
        for role in header_value.split(",")
        if role.strip()
    }


def require_admin(request: Request, x_admin_key: str = Header(default="")) -> None:
    if settings.admin_api_key and secrets.compare_digest(
        x_admin_key, settings.admin_api_key
    ):
        return
    if settings.auth_mode == "proxy":
        granted = _admin_roles(request.headers.get(settings.auth_roles_header, ""))
        allowed = _admin_roles(settings.auth_admin_roles)
        if granted & allowed:
            return
    raise HTTPException(status_code=401, detail="Invalid admin key")


def current_user(request: Request) -> str | None:
    """Identity supplied by a trusted reverse proxy; None when auth is disabled."""
    if settings.auth_mode == "none":
        return None
    user = request.headers.get(settings.auth_user_header, "").strip()
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def rate_limit(user: str | None = Depends(current_user)) -> None:
    # shortcut: fixed one-minute window; move to a sliding window if bursts matter
    identity = user or "anonymous"
    key = f"ratelimit:{identity}:{int(time.time() // 60)}"
    try:
        client = get_redis_client()
        count = client.incr(key)
        if count == 1:
            client.expire(key, 60)
    except RedisError:
        # Fail open: a Redis outage must not take chat down with it.
        logger.warning("Rate limiter unavailable; allowing request", exc_info=True)
        return
    if count > settings.rate_limit_per_minute:
        RATE_LIMITED.inc()
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please retry shortly.",
            headers={"Retry-After": "60"},
        )
