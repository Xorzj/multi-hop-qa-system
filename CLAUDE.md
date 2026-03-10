# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Setup
```bash
# Install all dependencies
uv sync --group dev --group dl --group llm --group graph --group api --group data

# Start Neo4j (Docker)
bash scripts/start_neo4j.sh

# Stop Neo4j (with optional cleanup)
bash scripts/stop_neo4j.sh        # Stop only
bash scripts/stop_neo4j.sh --clean # Stop and remove volumes
```

### Testing & Quality
```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/unit/test_llm.py -v

# Run a single test by name
uv run pytest -k "test_function_name" -v

# With coverage
uv run pytest --cov=src

# Code formatting
uv run ruff format .

# Linting with auto-fix
uv run ruff check --fix .

# Type checking
uv run mypy src/ api/

# All quality checks at once
uv run ruff format . && uv run ruff check --fix . && uv run mypy src/ api/
```

### Running the Demo
```bash
# End-to-end demo: document → knowledge graph → multi-hop QA
uv run python -m scripts.run_demo

# With custom document
uv run python -m scripts.run_demo --doc path/to/file.docx

# With specific question
uv run python -m scripts.run_demo --question "你的问题"

# Skip offline phase (graph already built)
uv run python -m scripts.run_demo --skip-offline

# Verbose output (full content, reasoning steps, XML)
uv run python -m scripts.run_demo --verbose
```

### Running the API
```bash
uv run uvicorn api.main:app --reload --port 8000
```

## Architecture Overview

This is a domain-specific multi-hop question answering system built around Neo4j knowledge graph and LLM-based reasoning.

### Two-Phase Pipeline

**Offline Phase**: Documents → Knowledge Graph
1. DocumentLoader: Converts .docx to markdown using markitdown, strips base64 images
2. EntityExtractor: Extracts entities using LLM (chunk-based, supports domain schema constraints)
3. TripleExtractor: Extracts triples (subject-predicate-object) with context entities
4. GraphBuilder: Creates Neo4j nodes and relationships

**Online Phase**: Question → Answer with Evidence Chain
1. QuestionParser: Parses intent, extracts entities and relation hints
2. ReasoningOrchestrator: Multi-hop agentic reasoning over graph
3. ContextAssembler: Builds prompt from evidence chain
4. AnswerGenerator: Generates final answer with confidence

### Key Architectural Components

#### Schema-Driven Extraction (`src/data_processing/schema_inducer.py`)
The `SchemaInducer` uses a "Teacher LLM" to analyze document samples and generate a `DomainSchema` (the "Domain Constitution"). This schema defines:
- Entity types with definitions and examples
- Relation types with source/target type constraints
- Valid (source_type, relation, target_type) combinations

Once generated, the schema is saved to `config/domain_schema.json` and used to constrain extraction via the `schema_path` config option.

#### Agentic Multi-Hop Reasoning (`src/reasoning/reasoning_orchestrator.py`)
The `ReasoningOrchestrator` implements goal-oriented multi-hop reasoning:
1. **Entity Resolution**: Three-phase resolution (exact match → substring match → LLM-assisted iterative refinement)
2. **Target-Oriented Path Finding**: First entity is start point, remaining entities are goals
3. **Agentic Hop Decisions**: LLM decides whether to continue and which entities to explore next
4. **Evidence Chain Filtering**: Only preserves path-connected edges to goal entities
5. **Soft-Hint Fallback**: Relation hints are suggestions, not hard constraints - retried without filter if no results

#### LLM Provider Abstraction (`src/llm/`)
- `BaseLLMClient`: Abstract interface with `generate()`, `chat()`, `start()`, `stop()`
- `LocalLLMClient`: Wraps `InferenceEngine` for on-device inference
- `ZhipuClient`: API client for Zhipu AI models
- `OpenAIClient`: Client for OpenAI-compatible APIs (currently the active provider, configured via `base_url` and `api_key`)
- `client_factory.create_llm_client()`: Factory pattern for provider selection

#### Knowledge Graph Layer (`src/knowledge_graph/`)
- `Neo4jClient`: Async Neo4j driver wrapper with connection management
- `CypherBuilder`: Builds type-safe Cypher queries
- `GraphRetriever`: High-level graph operations (get_neighbors, get_path, search_nodes)
- `GraphBuilder`: Creates nodes/relationships from extraction results with auto-missing node creation

#### API Layer (`api/`)
FastAPI with dependency injection pattern (`api/dependencies.py`):
- Sync endpoint: `/qa/sync` - immediate response
- Async endpoints: `/qa/submit`, `/qa/status/{id}`, `/qa/result/{id}` - task queue pattern
- Returns XML-formatted evidence chain for tracing

## Configuration

Main config file: `config/config.toml`

Key sections:
- `[llm]`: Provider selection (local/openai/zhipu), model paths, `base_url`/`api_key` for API providers
- `[graph]`: Neo4j connection (supports `${NEO4J_PASSWORD}` env interpolation)
- `[extraction]`: Chunk sizes, LLM params for extraction, `schema_path` for domain constraints

Secret management:
- `config/auth.json`: Stores secrets (checked first before env vars)
- Environment variables: `${VAR_NAME}` in config.toml resolves to auth.json or os.environ

Neo4j browser: http://localhost:7474 (user: neo4j / password from NEO4J_PASSWORD)

## Code Standards

- **Type annotations**: All function signatures must have type annotations
- **Async-first**: All I/O operations use `async/await`
- **Exception handling**: Only catch specific exceptions, no bare `except`
- **Imports**: stdlib → third-party → local (blank line separated); enforced by ruff rule `I`
- **Ruff rules**: `E`, `F`, `I`, `UP` enabled; line-length 88; target Python 3.12
- **Pytest**: `asyncio_mode = "auto"` — async test functions run automatically without `@pytest.mark.asyncio`

## Git Commit Convention

Format: `<type>: <description>` (in Chinese or English, matching existing commits)
Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

## Important Design Decisions

1. **Domain-Agnostic with Schema Induction**: The system doesn't hardcode domain knowledge. The `SchemaInducer` discovers the ontology from samples, making it portable across verticals.

2. **Async-First**: All components are async. Neo4j queries use the async driver. LLM calls are async with retry logic in `post_with_retry()`.

3. **Evidence Chain Filtering**: The reasoning orchestrator aggressively filters edges to maintain path connectivity. LLM-suggested entities not in graph neighbors are filtered out (logged as warnings).

4. **Soft vs Hard Constraints**: Relation hints in questions are soft hints (retry without filter if no results). Schema constraints from `schema_path` are hard constraints used during extraction.

5. **Entity Resolution is Multi-Phase**: Don't rely on exact graph matches. The orchestrator tries exact, substring, then LLM-assisted reformulation to handle naming variations.
