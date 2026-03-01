"""Async Neo4j client wrapper for graph operations."""

from __future__ import annotations

from typing import Any

from src.common.config import GraphConfig
from src.common.exceptions import GraphError


class Neo4jClient:
    def __init__(self, config: GraphConfig) -> None:
        self._config = config
        self._driver: Any | None = None

    async def connect(self, timeout: float = 5.0) -> None:
        if self._driver is not None:
            return
        try:
            import asyncio

            from neo4j import AsyncGraphDatabase  # type: ignore
        except ModuleNotFoundError as exc:
            message = (
                "Neo4j driver is required for graph access. "
                "Install it with `uv sync --group graph` or `uv add neo4j`."
            )
            raise GraphError(message) from exc
        driver = AsyncGraphDatabase.driver(
            self._config.uri,
            auth=(self._config.user, self._config.password),
            connection_timeout=timeout,
        )
        try:
            await asyncio.wait_for(driver.verify_connectivity(), timeout=timeout)
        except (TimeoutError, OSError) as exc:
            await driver.close()
            raise GraphError(f"Failed to connect to Neo4j: {exc}") from exc
        self._driver = driver

    async def close(self) -> None:
        if self._driver is None:
            return
        await self._driver.close()
        self._driver = None

    async def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self._driver is None:
            await self.connect()
        if self._driver is None:
            raise GraphError("Neo4j driver not initialized.")
        async with self._driver.session(database=self._config.database) as session:
            result = await session.run(query, params or {})
            records = await result.data()
        return records

    async def __aenter__(self) -> Neo4jClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()
