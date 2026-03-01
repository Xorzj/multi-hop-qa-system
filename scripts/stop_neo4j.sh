#!/usr/bin/env bash
set -euo pipefail

info() {
	echo "[neo4j] $*"
}

error() {
	echo "[neo4j][error] $*" >&2
}

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="${project_root}/docker-compose.yml"

if [[ ! -f "${compose_file}" ]]; then
	error "docker-compose.yml not found at ${compose_file}"
	exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
	error "Docker is not installed. Please install Docker first."
	exit 1
fi

if ! docker info >/dev/null 2>&1; then
	error "Docker daemon is not running. Start Docker and try again."
	exit 1
fi

compose_cmd=()
if docker compose version >/dev/null 2>&1; then
	compose_cmd=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
	compose_cmd=(docker-compose)
else
	error "Docker Compose is not available. Install docker compose or docker-compose."
	exit 1
fi

container_name="$(awk -F: '/container_name:/ {gsub(/["'"'"']/, "", $2); gsub(/^[ \t]+/, "", $2); print $2; exit}' "${compose_file}")"
if [[ -z "${container_name}" ]]; then
	container_name="multihop-qa-neo4j"
fi

if [[ "${1:-}" == "--clean" ]]; then
	info "Stopping Neo4j and removing volumes..."
	"${compose_cmd[@]}" -f "${compose_file}" down -v
	exit 0
fi

if docker ps -a --format "{{.Names}}" | grep -qx "${container_name}"; then
	info "Stopping Neo4j container '${container_name}'..."
	"${compose_cmd[@]}" -f "${compose_file}" stop neo4j
else
	info "Neo4j container '${container_name}' does not exist."
fi
