from __future__ import annotations

import importlib
import os
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.common.config import DEFAULT_CONFIG_PATH, load_config
from src.common.logger import get_logger, setup_logging
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.llm.client_factory import create_llm_client

logger = get_logger(__name__)


def _load_router(module_path: str) -> APIRouter:
    try:
        module = importlib.import_module(module_path)
        router = getattr(module, "router")
    except ModuleNotFoundError:
        return APIRouter()
    if isinstance(router, APIRouter):
        return router
    return APIRouter()


health_router = _load_router("api.routers.health")
qa_router = _load_router("api.routers.qa")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.getenv("APP_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    config = load_config(config_path)
    setup_logging(config.logging.level)
    logger.info("Starting application", extra={"config_path": config_path})

    neo4j_client = Neo4jClient(config.graph)
    neo4j_connected = False
    try:
        await neo4j_client.connect()
        neo4j_connected = True
        logger.info("Neo4j connected successfully")
    except Exception as exc:
        logger.warning(
            "Neo4j connection failed, QA features will be unavailable: %s",
            exc,
        )

    llm_client = create_llm_client(config.llm)
    llm_started = False
    try:
        if hasattr(llm_client, "start"):
            await llm_client.start()
            llm_started = True
            logger.info("LLM client started successfully")
    except Exception as exc:
        logger.warning("LLM client start failed: %s", exc)

    app.state.config = config
    app.state.neo4j_client = neo4j_client if neo4j_connected else None
    app.state.llm_client = llm_client
    app.state.neo4j_connected = neo4j_connected
    app.state.llm_started = llm_started

    yield

    if llm_started and hasattr(llm_client, "stop"):
        await llm_client.stop()
    if neo4j_connected:
        await neo4j_client.close()
    logger.info("Application shutdown complete")


app = FastAPI(title="Multi-hop QA System", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(qa_router, prefix="/qa", tags=["qa"])
app.include_router(health_router, prefix="/health", tags=["health"])
