from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from api.dependencies import get_neo4j_client
from api.schemas.response import HealthResponse
from src.common.logger import get_logger
from src.knowledge_graph.neo4j_client import Neo4jClient

logger = get_logger(__name__)

router = APIRouter()

_NEO4J_TIMEOUT_SECONDS = 0.5


async def check_neo4j(client: Neo4jClient) -> bool:
    try:
        await asyncio.wait_for(
            client.execute("RETURN 1"), timeout=_NEO4J_TIMEOUT_SECONDS
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Neo4j health check failed", extra={"error": str(exc)})
        return False


def check_llm_available(request: Request) -> bool:
    return hasattr(request.app.state, "llm_client")


@router.get("")
async def health_check(
    request: Request,
    neo4j_client: Neo4jClient | None = Depends(get_neo4j_client),
) -> HealthResponse:
    neo4j_ok = False
    if neo4j_client is not None:
        neo4j_ok = await check_neo4j(neo4j_client)
    llm_ok = check_llm_available(request)
    return HealthResponse(
        status="healthy" if neo4j_ok else "degraded",
        version="0.1.0",
        neo4j_connected=neo4j_ok,
        llm_available=llm_ok,
    )


@router.get("/ready")
async def readiness_check(
    request: Request,
    neo4j_client: Neo4jClient | None = Depends(get_neo4j_client),
) -> JSONResponse:
    neo4j_ok = False
    if neo4j_client is not None:
        neo4j_ok = await check_neo4j(neo4j_client)
    llm_ok = check_llm_available(request)
    status_code = (
        status.HTTP_200_OK if neo4j_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    payload = HealthResponse(
        status="healthy" if neo4j_ok else "degraded",
        version="0.1.0",
        neo4j_connected=neo4j_ok,
        llm_available=llm_ok,
    )
    return JSONResponse(status_code=status_code, content=_model_dump(payload))


@router.get("/live")
async def liveness_check(request: Request) -> HealthResponse:
    llm_ok = check_llm_available(request)
    neo4j_connected = getattr(request.app.state, "neo4j_connected", False)
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        neo4j_connected=neo4j_connected,
        llm_available=llm_ok,
    )


def _model_dump(payload: HealthResponse) -> dict[str, Any]:
    return payload.model_dump()
