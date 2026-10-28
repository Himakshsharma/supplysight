"""Pydantic v2 request/response models for the public API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(examples=["manager"])
    password: str = Field(examples=["manager123"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    name: str
    expires_in_minutes: int


class POScoreRequest(BaseModel):
    """One purchase-order item to score before release."""

    lifnr: str = Field(description="Vendor (LFA1-LIFNR)", examples=["500014"])
    matnr: str = Field(description="Material (MARA-MATNR)", examples=["100001"])
    werks: str = Field("1000", description="Plant", examples=["1000"])
    menge: float = Field(gt=0, description="Order quantity", examples=[400])
    netpr: float = Field(gt=0, description="Net unit price (INR)", examples=[125.0])
    created_by: str = Field("API_USER", examples=["RSHARMA"])
    created_at: str | None = Field(
        None, description="ISO timestamp; defaults to the next business day", examples=["2026-10-04T23:30:00"]
    )
    eindt: str | None = Field(None, description="Delivery date (YYYY-MM-DD)")
    lead_time_days: int | None = Field(14, ge=0, le=365)
    pr_menge: float | None = Field(None, gt=0, description="Requisition quantity")
    ebeln: str | None = None


class POScoreResponse(BaseModel):
    score: float
    risk_level: Literal["Low", "Medium", "High"]
    reasons: list[str]
    components: dict[str, float]
    model_version: str


class Flag(BaseModel):
    id: str
    ebeln: str
    ebelp: str
    bedat: str
    lifnr: str
    vendor: str
    matnr: str
    maktx: str
    werks: str
    netwr: float
    score: float
    risk_level: str
    reasons: list[str]
    rule_score: float
    iforest_score: float
    ae_score: float
    model_version: str | None = None
    verdict: str | None = None
    comment: str | None = None
    reviewer: str | None = None
    reviewed_at: str | None = None


class ReviewRequest(BaseModel):
    verdict: Literal["true_issue", "false_positive"]
    comment: str = Field("", max_length=500)


class ReviewResponse(BaseModel):
    id: str
    ebeln: str
    ebelp: str
    verdict: str
    comment: str
    reviewer: str
    reviewed_at: str


class ForecastPoint(BaseModel):
    week: str
    yhat: float
    p10: float
    p90: float


class ForecastResponse(BaseModel):
    matnr: str
    maktx: str
    plant: str
    unit: str
    history: list[dict]
    forecast: list[ForecastPoint]
    model_version: str
    reorder: dict | None


class AskRequest(BaseModel):
    question: str = Field(
        min_length=2, max_length=1000, examples=["Which vendor supplies glycerin and what is the lead time?"]
    )


class AskResponse(BaseModel):
    answer: str
    route: str
    citations: list[dict]
    sql: str | None
    rows: list[dict]
    columns: list[str]
    row_count: int
    mode: str
    blocked: bool
    latency_ms: float


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
