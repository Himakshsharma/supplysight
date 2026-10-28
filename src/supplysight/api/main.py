"""FastAPI application: `uvicorn supplysight.api.main:app --reload`."""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from supplysight import __version__
from supplysight.api.routers.v1 import router
from supplysight.db import table_exists
from supplysight.mlops import Registry
from supplysight.pipeline import ANOMALY_MODEL, FORECAST_MODEL
from supplysight.services.core import METRICS, new_request_id
from supplysight.utils import get_logger

log = get_logger("supplysight.api")

app = FastAPI(
    title="SupplySight API",
    version=__version__,
    description="AI-powered procurement and demand intelligence for SAP ERP (MM/PP). **All data is synthetic.**",
    openapi_tags=[{"name": n} for n in ("auth", "anomaly", "forecast", "copilot", "overview", "system")],
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Request id, structured access log and latency metrics."""
    rid = request.headers.get("x-request-id") or new_request_id()
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # pragma: no cover - turned into a 500 JSON body below
        log.exception("unhandled error rid=%s", rid)
        response = JSONResponse(
            {"error": {"code": "internal_error", "message": "Internal server error", "request_id": rid}},
            status_code=500,
        )
    ms = (time.perf_counter() - t0) * 1000
    route = request.scope.get("route")
    METRICS.record(getattr(route, "path", request.url.path), response.status_code, ms)
    response.headers["x-request-id"] = rid
    log.info("%s %s %s %.1fms rid=%s", request.method, request.url.path, response.status_code, ms, rid)
    return response


def _error(request: Request, status_code: int, code: str, message) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message, "request_id": getattr(request.state, "request_id", None)}},
        status_code=status_code,
    )


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
    """Consistent error envelope for HTTP errors."""
    codes = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 422: "validation_error"}
    resp = _error(request, exc.status_code, codes.get(exc.status_code, "error"), exc.detail)
    for k, v in (exc.headers or {}).items():
        resp.headers[k] = v
    return resp


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Consistent error envelope for validation errors."""
    details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
    return _error(request, 422, "validation_error", details)


@app.get("/health", tags=["system"])
def health() -> dict:
    """Liveness + readiness: database tables and production models present."""
    reg = Registry()
    models = {n: (reg.get_version(n, "Production") or {}).get("version") for n in (ANOMALY_MODEL, FORECAST_MODEL)}
    db_ok = table_exists("ekpo") and table_exists("po_scores")
    return {
        "status": "ok" if db_ok and all(models.values()) else "degraded",
        "database": db_ok,
        "models": models,
        "version": __version__,
    }


@app.get("/metrics", tags=["system"])
def metrics() -> dict:
    """Request counts, error counts, p50/p95 latency per route and production model versions."""
    reg = Registry()
    snap = METRICS.snapshot()
    snap["models"] = {
        n: (reg.get_version(n, "Production") or {}).get("version") for n in (ANOMALY_MODEL, FORECAST_MODEL)
    }
    return snap


app.include_router(router)
