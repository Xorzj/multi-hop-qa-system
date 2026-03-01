from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from src.common.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Neo4jDockerConfig:
    compose_file: Path
    container_name: str
    http_port: int
    bolt_port: int
    username: str
    password: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _compose_file() -> Path:
    return _project_root() / "docker-compose.yml"


def _load_config() -> Neo4jDockerConfig:
    compose_path = _compose_file()
    if not compose_path.exists():
        raise FileNotFoundError(f"docker-compose.yml not found at {compose_path}")

    container_name = (
        _extract_compose_value(compose_path, "container_name") or "multihop-qa-neo4j"
    )
    auth_value = _extract_compose_value(compose_path, "NEO4J_AUTH") or "neo4j/neo4j"
    username, password = _parse_auth(auth_value)
    http_port = _extract_port(compose_path, "7474", "7474") or 7474
    bolt_port = _extract_port(compose_path, "7687", "7687") or 7687

    return Neo4jDockerConfig(
        compose_file=compose_path,
        container_name=container_name,
        http_port=http_port,
        bolt_port=bolt_port,
        username=username,
        password=password,
    )


def _extract_compose_value(compose_path: Path, key: str) -> str | None:
    try:
        with compose_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if key in line:
                    parts = line.split(":", 1)
                    if len(parts) != 2:
                        continue
                    value = parts[1].strip().strip('"').strip("'")
                    return value or None
    except OSError as exc:
        logger.error("Failed to read docker-compose.yml", extra={"error": str(exc)})
        return None
    return None


def _extract_port(
    compose_path: Path, host_port: str, container_port: str
) -> int | None:
    pattern = f"{host_port}:{container_port}"
    try:
        with compose_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if pattern in line:
                    digits = "".join(ch for ch in line.split(":", 1)[0] if ch.isdigit())
                    if digits:
                        return int(digits)
    except OSError as exc:
        logger.error(
            "Failed to parse ports from docker-compose.yml", extra={"error": str(exc)}
        )
        return None
    return None


def _parse_auth(auth_value: str) -> tuple[str, str]:
    if "/" in auth_value:
        user, password = auth_value.split("/", 1)
        return user, password
    return auth_value, ""


def _compose_command() -> list[str]:
    if _command_available(["docker", "compose", "version"]):
        return ["docker", "compose"]
    if _command_available(["docker-compose", "version"]):
        return ["docker-compose"]
    raise RuntimeError("Docker Compose is not available.")


def _command_available(command: list[str]) -> bool:
    try:
        result = subprocess.run(
            command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return result.returncode == 0
    except OSError:
        return False


def _ensure_docker_running() -> None:
    if not _command_available(["docker", "info"]):
        raise RuntimeError("Docker daemon is not running.")


def is_neo4j_running() -> bool:
    config = _load_config()
    if not _command_available(["docker", "info"]):
        logger.warning("Docker daemon is not running.")
        return False
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", config.container_name],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        logger.error("Failed to inspect Docker container", extra={"error": str(exc)})
        return False
    return result.stdout.strip() == "true"


def start_neo4j() -> None:
    config = _load_config()
    _ensure_docker_running()
    compose_cmd = _compose_command()
    logger.info("Starting Neo4j container", extra={"container": config.container_name})
    try:
        subprocess.run(
            compose_cmd + ["-f", str(config.compose_file), "up", "-d", "neo4j"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to start Neo4j container.") from exc


def stop_neo4j(clean: bool = False) -> None:
    config = _load_config()
    _ensure_docker_running()
    compose_cmd = _compose_command()
    command = ["-f", str(config.compose_file), "stop", "neo4j"]
    if clean:
        command = ["-f", str(config.compose_file), "down", "-v"]
    logger.info(
        "Stopping Neo4j container",
        extra={"container": config.container_name, "clean": clean},
    )
    try:
        subprocess.run(compose_cmd + command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to stop Neo4j container.") from exc


def wait_for_neo4j(timeout_seconds: int = 120, poll_interval: int = 5) -> None:
    config = _load_config()
    _ensure_docker_running()

    start_time = time.time()
    while True:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Health.Status}}",
                    config.container_name,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise RuntimeError("Failed to inspect Neo4j health status.") from exc

        status = result.stdout.strip()
        if status == "healthy":
            logger.info(
                "Neo4j is healthy",
                extra={
                    "http": f"http://localhost:{config.http_port}",
                    "bolt": f"bolt://localhost:{config.bolt_port}",
                },
            )
            return
        if status == "unhealthy":
            raise RuntimeError("Neo4j container reported unhealthy status.")
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError("Timed out waiting for Neo4j to become healthy.")
        time.sleep(poll_interval)


def get_connection_info() -> dict[str, str | int]:
    config = _load_config()
    return {
        "http_url": f"http://localhost:{config.http_port}",
        "bolt_url": f"bolt://localhost:{config.bolt_port}",
        "username": config.username,
        "password": config.password,
    }


__all__ = [
    "Neo4jDockerConfig",
    "get_connection_info",
    "is_neo4j_running",
    "start_neo4j",
    "stop_neo4j",
    "wait_for_neo4j",
]
