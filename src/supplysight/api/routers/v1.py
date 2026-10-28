"""Versioned API routes (/api/v1)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool

from supplysight.api.deps import current_user, require
from supplysight.api.schemas import (
    AskRequest,
    AskResponse,
    Flag,
    ForecastResponse,
    LoginRequest,
    POScoreRequest,
    POScoreResponse,
    ReviewRequest,
    ReviewResponse,
    TokenResponse,
)
from supplysight.config import get_settings
from supplysight.copilot.agent import Copilot
from supplysight.services import core
from supplysight.services.auth import authenticate, create_token

router = APIRouter(prefix="/api/v1")
_copilot = Copilot()


# ---------------------------------------------------------------- auth
@router.post("/auth/login", response_model=TokenResponse, tags=["auth"])
def login(body: LoginRequest) -> TokenResponse:
    """Exchange demo credentials for a JWT (roles: buyer, planner, manager, admin)."""
    user = authenticate(body.username, body.password)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password")
    mins = get_settings().jwt_expiry_minutes
    return TokenResponse(
        access_token=create_token(user["username"], user["role"], mins),
        role=user["role"],
        name=user["name"],
        expires_in_minutes=mins,
    )


@router.get("/auth/me", tags=["auth"])
def me(user: dict = Depends(current_user)) -> dict:
    """Return the caller's identity from the token."""
    return {"username": user["sub"], "role": user["role"]}


# ---------------------------------------------------------------- anomaly
@router.post("/anomaly/score", response_model=POScoreResponse, tags=["anomaly"])
async def score(body: POScoreRequest, user: dict = Depends(require("anomaly:score"))) -> POScoreResponse:
    """Score one PO item before release: risk level, score and top-3 reasons."""
    try:
        res = await run_in_threadpool(core.score_po, body.model_dump())
    except core.NotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return POScoreResponse(**res)


@router.get("/anomaly/flags", response_model=list[Flag], tags=["anomaly"])
def flags(
    risk: str | None = Query(None, description="High, Medium or High,Medium"),
    date_from: str | None = Query(None, alias="from", description="PO date from (YYYY-MM-DD)"),
    date_to: str | None = Query(None, alias="to", description="PO date to (YYYY-MM-DD)"),
    review_status: str | None = Query(None, alias="status", pattern="^(open|reviewed)$"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    user: dict = Depends(require("anomaly:read")),
) -> list[dict]:
    """List flagged POs (Medium/High by default), highest score first."""
    return core.list_flags(risk, date_from, date_to, review_status, limit, offset)


@router.post("/anomaly/flags/{flag_id}/review", response_model=ReviewResponse, tags=["anomaly"])
def review(flag_id: str, body: ReviewRequest, user: dict = Depends(require("anomaly:review"))) -> dict:
    """Record an analyst verdict; verdicts become labels for the next retraining."""
    try:
        return core.review_flag(flag_id, body.verdict, body.comment, user["sub"])
    except core.NotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.get("/anomaly/duplicate-vendors", tags=["anomaly"])
def duplicate_vendors(user: dict = Depends(require("anomaly:read"))) -> list[dict]:
    """Suspected duplicate vendor pairs (identifiers masked unless admin)."""
    from supplysight.copilot.guardrails import mask_value

    rows = core.duplicate_vendors()
    if user["role"] != "admin":
        for r in rows:
            for k in ("gstin_a", "gstin_b", "bank_a", "bank_b"):
                r[k] = mask_value(r[k])
    return rows


# ---------------------------------------------------------------- forecast
@router.get("/forecast/reorder-risk", tags=["forecast"])
def reorder_risk(
    risk: str | None = Query(None, description="High, Medium, Low or a comma list"),
    limit: int = Query(500, ge=1, le=2000),
    user: dict = Depends(require("forecast:read")),
) -> list[dict]:
    """Recommended reorder points vs current MINBE, with stock-out risk."""
    return core.reorder_risk(risk, limit)


@router.get("/forecast/{matnr}", response_model=ForecastResponse, tags=["forecast"])
def forecast(
    matnr: str,
    plant: str | None = Query(None, pattern="^(1000|2000)$"),
    horizon: int = Query(8, ge=1, le=12),
    user: dict = Depends(require("forecast:read")),
) -> dict:
    """Weekly P10/P50/P90 demand forecast with recent history and reorder recommendation."""
    try:
        return core.get_forecast(matnr, plant, horizon)
    except core.NotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


# ---------------------------------------------------------------- copilot
@router.post("/copilot/ask", response_model=AskResponse, tags=["copilot"])
async def ask(body: AskRequest, user: dict = Depends(require("copilot:ask"))) -> dict:
    """Ask the procurement copilot (SQL / SOP / insight routes, with citations)."""
    ans = await run_in_threadpool(_copilot.ask, body.question, user["role"])
    return ans.to_dict()


# ---------------------------------------------------------------- overview & system
@router.get("/overview", tags=["overview"])
def overview(user: dict = Depends(require("overview:read"))) -> dict:
    """Dashboard KPIs."""
    return core.overview()


@router.get("/models", tags=["system"])
def models(user: dict = Depends(require("system:metrics"))) -> dict:
    """Model registry, production versions and the latest drift summary."""
    return core.model_health()
