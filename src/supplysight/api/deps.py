"""FastAPI dependencies: bearer-token auth and role checks."""

from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from supplysight.services.auth import allowed, decode_token

_bearer = HTTPBearer(auto_error=False)


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    """Decode the bearer token or raise 401."""
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        return decode_token(creds.credentials)
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from exc


def require(permission: str):
    """Dependency factory: user must hold ``permission``."""

    def _check(user: dict = Depends(current_user)) -> dict:
        if not allowed(user["role"], permission):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Role '{user['role']}' lacks permission '{permission}'")
        return user

    return _check
