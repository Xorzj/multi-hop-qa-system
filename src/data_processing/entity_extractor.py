from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.common.config import ExtractionConfig
from src.common.logger import get_logger
from src.data_processing.document_loader import Section
from src.data_processing.relation_types import (
    DEFAULT_RELATION_TYPES,
    RelationType,
    build_relation_type_prompt,
)
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


@dataclass
class IncrementalRelation:
    """Relation extracted during incremental extraction.
    
    Produced when entities and relations are extracted together.
    """

    subject: str
    predicate: str
    object: str
    relation_type: str = ""
    confidence: float = 1.0


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
                self._logger.debug(
                    f"LLM response for [section-{section.index}]:\n{response}"
                )
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
            "你是专业的知识图谱实体抽取专家。请从下面的文本中抽取所有重要的实体，"
            "包括但不限于以下类别：\n"
            "  - 具体事物：技术名词、产品名称、标准、工具、材料、药物\n"
            "  - 抽象概念：疾病、症状、过程、机制、现象、理论、方法\n"
            "  - 组织与人物：机构、公司、人名、团队\n"
            "  - 度量与属性：指标、参数、性能特征\n\n"
            "特别注意：不要遗漏因果链条中的中间实体。例如 'A损伤B，形成C，"
            "导致D' 中 A、B、C、D 都是需要抽取的实体。\n",
            "要求：\n"
            "1. 完整抽取文本中所有专业术语、概念、事物、过程和现象\n"
            "2. 每个实体必须有 name 和 type 字段\n"
            "3. 只抽取当前段落中明确提到的实体，不要编造\n"
            "4. 只返回JSON数组，不要有任何其他文字\n"
            "5. 宁可多抽、不要遗漏，尤其是因果链条中的中间节点\n",
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
            "输出格式示例（涵盖多领域）：\n"
            '[{"name": "动脉粥样硬化", "type": "病理过程", "aliases": ["动脉硬化"]}, '
            '{"name": "血管内皮", "type": "组织结构"}, '
            '{"name": "分布式数据库", "type": "技术"}, '
            '{"name": "光伏发电", "type": "技术"}]\n\n'
            f"待抽取文本：\n{section.content}\n\n"
            "请直接返回JSON数组："
        )

        return "\n".join(parts)

    # ── Response parsing ──────────────────────────────────────

    def _parse_response(self, response: str) -> list[Entity]:
        """Parse the LLM JSON response into a list of Entity objects.
        
        Handles thinking-model responses where JSON may be wrapped in
        <think> tags or markdown code blocks.
        """

        # Pre-process: strip thinking tags
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()
        if not cleaned:
            cleaned = response  # fallback if everything was inside <think>

        payload = None

        # Strategy: markdown code block (common in thinking model output)
        code_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?\s*```', cleaned)
        if code_match:
            try:
                payload = json.loads(code_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Direct parse
        if payload is None:
            try:
                payload = json.loads(cleaned.strip())
            except json.JSONDecodeError:
                pass

        # Regex: tightest JSON array
        if payload is None:
            match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        # Bracket search
        if payload is None:
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start != -1 and end > start:
                try:
                    payload = json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    pass

        # Truncated JSON repair: close array after last complete object
        if payload is None:
            start = cleaned.find("[")
            if start != -1:
                last_brace = cleaned.rfind("}")
                if last_brace > start:
                    try:
                        payload = json.loads(cleaned[start : last_brace + 1] + "]")
                        self._logger.info(
                            "Repaired truncated JSON:"
                            f" extracted {len(payload)} entities"
                        )
                    except json.JSONDecodeError:
                        pass

        if payload is None:
            self._logger.warning(f"Failed to parse entity response: {cleaned[:200]}")
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

    # ── Incremental Extraction (entities + relations in one call) ──

    async def extract_incremental(
        self,
        sections: list[Section],
        relation_types: list[RelationType] | None = None,
    ) -> tuple[list[Entity], list[IncrementalRelation]]:
        """Extract entities AND relations together in one LLM call per section.

        This is the **unified incremental extraction** mode: each section
        produces both new entities and relations referencing predefined
        relation types, using previously-extracted entities as context.

        Returns:
            Tuple of (deduplicated entities, all relations).
        """
        if not sections:
            return [], []

        rtypes = relation_types or DEFAULT_RELATION_TYPES
        all_entities: list[Entity] = []
        all_relations: list[IncrementalRelation] = []

        for idx, section in enumerate(sections):
            ctx = all_entities[-self._max_context :] if all_entities else []
            heading = (" > ".join(section.heading_chain)
                       or f"section-{idx}")
            self._logger.info(
                f"Incremental extraction from [{heading}] "
                f"({len(section.content)} chars, context={len(ctx)} entities)"
            )

            entities, relations = await self._extract_section_incremental(
                section, ctx, rtypes
            )
            self._logger.info(
                f"  → {len(entities)} entities, "
                f"{len(relations)} relations from [{heading}]"
            )
            all_entities.extend(entities)
            all_relations.extend(relations)

        deduped = self._deduplicate_entities(all_entities)
        self._logger.info(
            f"Incremental extraction done: {len(all_entities)} raw entities → "
            f"{len(deduped)} unique, {len(all_relations)} relations"
        )
        return deduped, all_relations

    async def _extract_section_incremental(
        self,
        section: Section,
        context_entities: list[Entity],
        relation_types: list[RelationType],
    ) -> tuple[list[Entity], list[IncrementalRelation]]:
        """Run unified extraction on one section with retry."""

        prompt = self._build_incremental_prompt(
            section, context_entities, relation_types
        )

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._llm_client.generate(
                    prompt=prompt, params=self._generation_params
                )
                self._logger.debug(
                    f"LLM incremental response for "
                    f"[section-{section.index}]:\n{response}"
                )
                entities, relations = self._parse_incremental_response(response)
                if entities or relations or attempt > 0:
                    return entities, relations
                self._logger.debug(
                    "Empty incremental result on first attempt, retrying..."
                )
            except Exception as exc:
                last_error = exc
                self._logger.warning(
                    f"Incremental extraction attempt {attempt + 1} failed: {exc}"
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))

        if last_error:
            self._logger.error(
                f"All {self._max_retries} incremental attempts failed: {last_error}"
            )
        return [], []

    def _build_incremental_prompt(
        self,
        section: Section,
        context_entities: list[Entity],
        relation_types: list[RelationType],
    ) -> str:
        """Build unified extraction prompt (entities + relations)."""

        relation_block = build_relation_type_prompt(relation_types)

        parts: list[str] = [
            "你是专业的知识图谱构建专家。请从下面的文本中同时抽取【实体】和【关系】。\n",
            "实体抽取要求：\n"
            "1. 完整抽取文本中所有专业术语、概念、事物、过程和现象\n"
            "2. 不要遗漏因果链条中的中间实体\n"
            "3. 每个实体必须有 name 和 type 字段\n"
            "4. 只抽取当前段落中明确提到的实体，不要编造\n",
            "关系抽取要求：\n"
            "1. 使用下方定义的关系类型（relation_type 字段必须是定义中的 name 值）\n"
            "2. predicate 字段用中文描述具体关系动词（如 '导致'、'包含'）\n"
            "3. subject 和 object 必须是本次或已知实体列表中的实体名称\n"
            "4. 关系必须在文本中有明确依据\n",
            relation_block,
        ]

        # Heading context
        if section.heading_chain:
            parts.append(
                f"当前章节位置：{' > '.join(section.heading_chain)}\n"
            )

        # Known entities from previous sections
        if context_entities:
            ctx_lines = [f"- {e.name}（{e.entity_type}）"
                         for e in context_entities]
            parts.append(
                "已知实体（来自前文，可在关系中引用，不需要重新抽取）：\n"
                + "\n".join(ctx_lines)
                + "\n"
            )

        parts.append(
            '输出格式（严格JSON，不要有任何其他文字）：\n'
            '{\n'
            '  "new_entities": [\n'
            '    {"name": "实体名", "type": "类型", "aliases": []}\n'
            '  ],\n'
            '  "relations": [\n'
            '    {"subject": "主语实体名", "predicate": "关系动词", '
            '"object": "宾语实体名", "relation_type": "类型name值", '
            '"confidence": 0.9}\n'
            '  ]\n'
            '}\n\n'
            f"待抽取文本：\n{section.content}\n\n"
            "请直接返回JSON对象："
        )

        return "\n".join(parts)

    def _parse_incremental_response(
        self, response: str
    ) -> tuple[list[Entity], list[IncrementalRelation]]:
        """Parse unified extraction response containing entities and relations."""

        # Pre-process: strip thinking tags
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()
        if not cleaned:
            cleaned = response

        payload = self._extract_json_object(cleaned)

        if payload is None:
            # Fallback: try parsing as entity-only array (graceful degradation)
            entities = self._parse_response(response)
            return entities, []

        # Guard: if payload has no expected keys, fall back to entity-only
        has_incremental_keys = (
            "new_entities" in payload
            or "entities" in payload
            or "relations" in payload
        )
        if not has_incremental_keys:
            entities = self._parse_response(response)
            return entities, []

        # Parse entities
        entities: list[Entity] = []
        raw_entities = payload.get("new_entities") or payload.get("entities") or []
        if isinstance(raw_entities, list):
            for item in raw_entities:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                entity_type = item.get("type") or item.get("entity_type")
                if not isinstance(name, str) or not isinstance(entity_type, str):
                    continue
                if not name.strip():
                    continue
                aliases = item.get("aliases")
                alias_list = (
                    [a for a in aliases if isinstance(a, str)]
                    if isinstance(aliases, list)
                    else []
                )
                entities.append(
                    Entity(
                        name=name.strip(),
                        entity_type=entity_type.strip(),
                        aliases=alias_list,
                    )
                )

        # Parse relations
        relations: list[IncrementalRelation] = []
        raw_relations = payload.get("relations") or []
        if isinstance(raw_relations, list):
            for item in raw_relations:
                if not isinstance(item, dict):
                    continue
                subject = item.get("subject")
                predicate = item.get("predicate") or item.get("relation")
                obj = item.get("object")
                relation_type = item.get("relation_type", "")
                confidence = item.get("confidence", 1.0)
                if not isinstance(subject, str) or not isinstance(obj, str):
                    continue
                if not isinstance(predicate, str):
                    continue
                if not isinstance(confidence, (int, float)):
                    confidence = 1.0
                if not isinstance(relation_type, str):
                    relation_type = ""
                relations.append(
                    IncrementalRelation(
                        subject=subject.strip(),
                        predicate=predicate.strip(),
                        object=obj.strip(),
                        relation_type=relation_type.strip(),
                        confidence=float(confidence),
                    )
                )

        return entities, relations

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        """Try multiple strategies to extract a JSON object from LLM output."""

        # Strategy 1: markdown code block
        code_match = re.search(
            r'```(?:json)?\s*\n?([\s\S]*?)\n?\s*```', text
        )
        if code_match:
            try:
                result = json.loads(code_match.group(1).strip())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # Strategy 2: direct parse
        try:
            result = json.loads(text.strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Strategy 3: find outermost { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                result = json.loads(text[start : end + 1])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return None
