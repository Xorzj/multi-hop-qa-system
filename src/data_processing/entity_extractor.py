from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.common.config import ExtractionConfig
from src.common.logger import get_logger
from src.data_processing.document_loader import Section
from src.llm.base_client import BaseLLMClient, GenerationParams

DEFAULT_ENTITY_TYPES: list[str] = []


@dataclass
class Entity:
    """Named entity extracted from document text."""

    name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    source_span: tuple[int, int] | None = None


class EntityExtractor:
    """Extract domain entities from document sections using an LLM client.

    Processes sections *sequentially* so that already-extracted entities from
    earlier sections can be fed as incremental context to later ones, reducing
    hallucinations and improving cross-section coherence.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        entity_types: list[str] | None = None,
        extraction_config: ExtractionConfig | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._entity_types = list(entity_types or DEFAULT_ENTITY_TYPES)
        self._logger = get_logger(__name__)

        cfg = extraction_config
        self._max_retries = cfg.max_retries if cfg else 3
        self._max_context = cfg.max_context_entities if cfg else 15

        self._generation_params = GenerationParams(
            max_new_tokens=cfg.max_new_tokens if cfg else 2048,
            temperature=cfg.temperature if cfg else 0.05,
            top_p=cfg.top_p if cfg else 0.1,
        )

    # ── Public API ───────────────────────────────────────────────

    async def extract(self, sections: list[Section]) -> list[Entity]:
        """Extract entities from document sections with incremental context."""

        if not sections:
            return []

        all_entities: list[Entity] = []
        for idx, section in enumerate(sections):
            ctx = all_entities[-self._max_context :] if all_entities else []
            heading = " > ".join(section.heading_chain) or f"section-{idx}"
            self._logger.info(
                f"Extracting entities from [{heading}] "
                f"({len(section.content)} chars, context={len(ctx)} entities)"
            )
            section_entities = await self._extract_section(section, ctx)
            self._logger.info(f"  → {len(section_entities)} entities from [{heading}]")
            all_entities.extend(section_entities)

        deduped = self._deduplicate_entities(all_entities)
        self._logger.info(
            f"Entity extraction done: {len(all_entities)} raw → {len(deduped)} unique"
        )
        return deduped

    # ── Section-level extraction with retry ──────────────────────

    async def _extract_section(
        self, section: Section, context_entities: list[Entity]
    ) -> list[Entity]:
        """Extract entities from one section, retrying on failure."""

        prompt = self._build_prompt(section, context_entities)

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._llm_client.generate(
                    prompt=prompt, params=self._generation_params
                )
                entities = self._parse_response(response)
                if entities or attempt > 0:
                    return entities
                # First attempt returned empty – retry once to be sure
                self._logger.debug("Empty entity list on first attempt, retrying...")
            except Exception as exc:
                last_error = exc
                self._logger.warning(
                    f"Entity extraction attempt {attempt + 1} failed: {exc}"
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))

        if last_error:
            self._logger.error(
                f"All {self._max_retries} extraction attempts failed: {last_error}"
            )
        return []

    # ── Prompt construction ──────────────────────────────────────

    def _build_prompt(self, section: Section, context_entities: list[Entity]) -> str:
        """Build prompt with heading context and incremental entity context."""

        parts: list[str] = [
            "你是专业的知识图谱实体抽取专家。请从下面的文本中抽取所有重要的命名实体。\n",
            "要求：\n"
            "1. 抽取所有专业术语、技术名词、组织名称、产品名称、标准名称等\n"
            "2. 每个实体必须有 name 和 type 字段\n"
            "3. 只抽取当前段落中明确提到的实体，不要编造\n"
            "4. 只返回JSON数组，不要有任何其他文字\n",
        ]

        # Heading context
        if section.heading_chain:
            parts.append(f"当前章节位置：{' > '.join(section.heading_chain)}\n")

        # Incremental context
        if context_entities:
            names = ", ".join(e.name for e in context_entities)
            parts.append(f"已知实体（来自前文，避免重复抽取）：{names}\n")

        # Entity-type guidance
        if self._entity_types:
            parts.append(
                f"可参考的实体类型提示（仅作指导）：{'、'.join(self._entity_types)}\n"
            )

        parts.append(
            "输出格式示例：\n"
            '[{"name": "SDH", "type": "技术", "aliases": ["同步数字体系"]}, '
            '{"name": "STM-1", "type": "速率等级"}]\n\n'
            f"待抽取文本：\n{section.content}\n\n"
            "请直接返回JSON数组："
        )

        return "\n".join(parts)

    # ── Response parsing (unchanged) ─────────────────────────────

    def _parse_response(self, response: str) -> list[Entity]:
        """Parse the LLM JSON response into a list of Entity objects."""

        payload = None

        # Direct parse
        try:
            payload = json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Regex: tightest JSON array
        if payload is None:
            match = re.search(r"\[\s*\{.*\}\s*\]", response, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        # Bracket search
        if payload is None:
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end > start:
                try:
                    payload = json.loads(response[start : end + 1])
                except json.JSONDecodeError:
                    pass

        # Truncated JSON repair: close array after last complete object
        if payload is None:
            start = response.find("[")
            if start != -1:
                last_brace = response.rfind("}")
                if last_brace > start:
                    try:
                        payload = json.loads(response[start : last_brace + 1] + "]")
                        self._logger.info(
                            "Repaired truncated JSON:"
                            f" extracted {len(payload)} entities"
                        )
                    except json.JSONDecodeError:
                        pass

        if payload is None:
            self._logger.warning(f"Failed to parse entity response: {response[:200]}")
            return []

        if not isinstance(payload, list):
            self._logger.warning("Entity extraction response is not a list")
            return []

        entities: list[Entity] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            entity_type = item.get("type") or item.get("entity_type")
            if not isinstance(name, str) or not isinstance(entity_type, str):
                continue
            if not name.strip():
                continue
            aliases = item.get("aliases")
            properties = item.get("properties")
            alias_list = (
                [alias for alias in aliases if isinstance(alias, str)]
                if isinstance(aliases, list)
                else []
            )
            property_map = properties if isinstance(properties, dict) else {}
            entities.append(
                Entity(
                    name=name.strip(),
                    entity_type=entity_type.strip(),
                    aliases=alias_list,
                    properties=property_map,
                )
            )
        return entities

    # ── Deduplication (unchanged) ────────────────────────────────

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """Merge duplicate entities by name."""

        merged: dict[str, Entity] = {}
        for entity in entities:
            key = entity.name.strip()
            if not key:
                continue
            existing = merged.get(key)
            if existing is None:
                merged[key] = entity
                continue
            for alias in entity.aliases:
                if alias not in existing.aliases and alias != existing.name:
                    existing.aliases.append(alias)
            for prop_key, prop_value in entity.properties.items():
                existing.properties.setdefault(prop_key, prop_value)
        return list(merged.values())
