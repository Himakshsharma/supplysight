"""Demo authentication: PBKDF2-hashed users, JWT (HS256) tokens and role permissions.

Demo users (password = username + "123"): buyer, planner, manager, admin.
Replace with SSO / SAP user roles in a real deployment.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime, timedelta

import jwt

from supplysight.config import get_settings

ROLES = ("buyer", "planner", "manager", "admin")
PERMISSIONS: dict[str, set[str]] = {
    "anomaly:read": {"buyer", "manager", "admin"},
    "anomaly:score": {"buyer", "manager", "admin"},
    "anomaly:review": {"manager", "admin"},
    "forecast:read": {"planner", "buyer", "manager", "admin"},
    "copilot:ask": set(ROLES),
    "system:metrics": {"manager", "admin"},
    "overview:read": set(ROLES),
}


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()


_USERS = {u: {"role": u, "salt": f"ss-{u}", "hash": _hash(f"{u}123", f"ss-{u}"), "name": n}
          for u, n in [("buyer", "Riya Sharma (Buyer)"), ("planner", "Arjun Mehta (MRP Planner)"),
                       ("manager", "Neha Gupta (Purchase Manager)"), ("admin", "System Admin")]}


def authenticate(username: str, password: str) -> dict | None:
    """Return the user record if the credentials are valid."""
    u = _USERS.get(username)
    if u and hmac.compare_digest(u["hash"], _hash(password, u["salt"])):
        return {"username": username, "role": u["role"], "name": u["name"]}
    return None


def create_token(username: str, role: str, minutes: int | None = None) -> str:
    """Sign a JWT for the user."""
    s = get_settings()
    exp = datetime.now(UTC) + timedelta(minutes=minutes or s.jwt_expiry_minutes)
    return jwt.encode({"sub": username, "role": role, "exp": exp}, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str) -> dict:
    """Verify and decode a JWT (raises jwt.InvalidTokenError)."""
    s = get_settings()
    claims = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    if claims.get("role") not in ROLES:
        raise jwt.InvalidTokenError("unknown role")
    return claims


def allowed(role: str, permission: str) -> bool:
    """Role-based permission check."""
    return role in PERMISSIONS.get(permission, set())
