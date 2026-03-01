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

auth_value="$(awk -F: '/NEO4J_AUTH:/ {gsub(/["'"'"']/, "", $2); gsub(/^[ \t]+/, "", $2); print $2; exit}' "${compose_file}")"
auth_value="${auth_value:-neo4j/neo4j}"
neo4j_user="${auth_value%%/*}"
neo4j_password="${auth_value#*/}"

http_port="$(awk -F: '/7474:7474/ {gsub(/[^0-9]/, "", $1); print $1; exit}' "${compose_file}")"
bolt_port="$(awk -F: '/7687:7687/ {gsub(/[^0-9]/, "", $1); print $1; exit}' "${compose_file}")"
http_port="${http_port:-7474}"
bolt_port="${bolt_port:-7687}"

if docker ps -a --format "{{.Names}}" | grep -qx "${container_name}"; then
	is_running="$(docker inspect -f '{{.State.Running}}' "${container_name}" 2>/dev/null || true)"
	if [[ "${is_running}" == "true" ]]; then
		info "Neo4j container '${container_name}' is already running."
	else
		info "Starting existing Neo4j container '${container_name}'..."
		"${compose_cmd[@]}" -f "${compose_file}" up -d neo4j
	fi
else
	info "Starting Neo4j container using docker-compose..."
	"${compose_cmd[@]}" -f "${compose_file}" up -d neo4j
fi

info "Waiting for Neo4j to become healthy..."
start_time="$(date +%s)"
timeout_seconds=120
while true; do
	health_status="$(docker inspect -f '{{.State.Health.Status}}' "${container_name}" 2>/dev/null || true)"
	if [[ "${health_status}" == "healthy" ]]; then
		break
	fi
	if [[ "${health_status}" == "unhealthy" ]]; then
		error "Neo4j reported unhealthy status."
		exit 1
	fi
	now="$(date +%s)"
	if ((now - start_time > timeout_seconds)); then
		error "Timed out waiting for Neo4j health check."
		exit 1
	fi
	sleep 5
done

info "Neo4j is ready."
info "HTTP: http://localhost:${http_port}"
info "Bolt: bolt://localhost:${bolt_port}"
info "User: ${neo4j_user}"
info "Password: ${neo4j_password}"
