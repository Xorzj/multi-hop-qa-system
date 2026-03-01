from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.common.config import ExtractionConfig
from src.common.logger import get_logger
from src.data_processing.document_loader import Section
from src.data_processing.entity_extractor import Entity
from src.llm.base_client import BaseLLMClient, GenerationParams


@dataclass
class Triple:
    """Relationship triple extracted from document text."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    source: str | None = None


class TripleExtractor:
    """Extract subject-predicate-object triples from document sections.

    Improvements over the naive approach:
    * Only entities *mentioned* in each section are passed to the LLM.
    * Predefined relation types constrain the output.
    * Self-loop and orphan triples are filtered out.
    * Extraction runs in parallel across sections.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        relation_types: list[str] | None = None,
        extraction_config: ExtractionConfig | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._logger = get_logger(__name__)

        cfg = extraction_config
        self._relation_types = list(
            relation_types or (cfg.relation_types if cfg else [])
        )
        self._max_retries = cfg.max_retries if cfg else 3

        self._generation_params = GenerationParams(
            max_new_tokens=cfg.max_new_tokens if cfg else 2048,
            temperature=cfg.temperature if cfg else 0.05,
            top_p=cfg.top_p if cfg else 0.1,
        )

    # ── Public API ───────────────────────────────────────────────

    async def extract(
        self,
        sections: list[Section],
        entities: list[Entity],
    ) -> list[Triple]:
        """Extract triples from document sections (parallel)."""

        if not sections or not entities:
            return []

        entity_name_set = {e.name for e in entities}

        tasks = [
            self._extract_section(section, entities, entity_name_set)
            for section in sections
        ]
        results = await asyncio.gather(*tasks)
        flattened = [triple for group in results for triple in group]

        # Post-processing: filter then deduplicate
        filtered = self._filter_triples(flattened, entity_name_set)
        deduped = self._deduplicate_triples(filtered)
        self._logger.info(
            f"Triple extraction done: {len(flattened)} raw → "
            f"{len(filtered)} filtered → {len(deduped)} unique"
        )
        return deduped

    # ── Section-level extraction ─────────────────────────────────

    async def _extract_section(
        self,
        section: Section,
        all_entities: list[Entity],
        entity_name_set: set[str],
    ) -> list[Triple]:
        """Extract triples from one section, passing only relevant entities."""

        relevant = self._filter_entities_for_section(section, all_entities)
        if len(relevant) < 2:
            # Need at least 2 entities to form a relationship
            return []

        heading = " > ".join(section.heading_chain) or f"section-{section.index}"
        self._logger.info(
            f"Extracting triples from [{heading}] "
            f"({len(section.content)} chars, {len(relevant)} entities)"
        )

        prompt = self._build_prompt(section, relevant)
        triples = await self._generate_with_retry(prompt)

        self._logger.info(f"  → {len(triples)} triples from [{heading}]")
        return triples

    async def _generate_with_retry(self, prompt: str) -> list[Triple]:
        """Call the LLM with retry on failure."""

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._llm_client.generate(
                    prompt=prompt, params=self._generation_params
                )
                triples = self._parse_response(response)
                if triples or attempt > 0:
                    return triples
                self._logger.debug("Empty triple list on first attempt, retrying...")
            except Exception as exc:
                last_error = exc
                self._logger.warning(
                    f"Triple extraction attempt {attempt + 1} failed: {exc}"
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))

        if last_error:
            self._logger.error(
                f"All {self._max_retries} extraction attempts failed: {last_error}"
            )
        return []

    # ── Entity filtering per section ─────────────────────────────

    @staticmethod
    def _filter_entities_for_section(
        section: Section, entities: list[Entity]
    ) -> list[Entity]:
        """Return only entities whose name appears in the section text."""

        text = section.content
        return [e for e in entities if e.name in text]

    # ── Prompt construction ──────────────────────────────────────

    def _build_prompt(self, section: Section, entities: list[Entity]) -> str:
        """Build prompt with heading context, filtered entities, and relation types."""

        parts: list[str] = [
            "你是专业的知识图谱关系抽取专家。"
            "请分析文本并识别给定实体之间的关系三元组。\n",
            "要求：\n"
            "1. 只在给定的实体列表内寻找关系，不要引入未列出的实体\n"
            "2. 主语和宾语必须来自给定实体列表\n"
            "3. 关系必须在文本中有明确依据，不要猜测\n"
            "4. 只返回JSON数组，不要有任何其他文字\n",
        ]

        # Heading context
        if section.heading_chain:
            parts.append(f"当前章节位置：{' > '.join(section.heading_chain)}\n")

        # Relation type constraint
        if self._relation_types:
            types_str = "、".join(self._relation_types)
            parts.append(f"关系类型限定（必须使用以下类型之一）：{types_str}\n")

        # Entity list
        entity_lines = [f"- {e.name}（{e.entity_type}）" for e in entities]
        parts.append(
            "实体列表（仅在以下实体之间建立关系）：\n" + "\n".join(entity_lines) + "\n"
        )

        parts.append(
            "输出格式示例：\n"
            '[{"subject": "SDH", "predicate": "包含", "object": "STM-1", '
            '"confidence": 0.9}]\n\n'
            f"待抽取文本：\n{section.content}\n\n"
            "请直接返回JSON数组："
        )

        return "\n".join(parts)

    # ── Response parsing ─────────────────────────────────────────

    def _parse_response(self, response: str) -> list[Triple]:
        """Parse the LLM JSON response into a list of Triple objects."""

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
                            f"Repaired truncated JSON: extracted {len(payload)} triples"
                        )
                    except json.JSONDecodeError:
                        pass

        if payload is None:
            self._logger.warning(
                f"Failed to parse triple extraction response: {response[:200]}"
            )
            return []

        if not isinstance(payload, list):
            self._logger.warning("Triple extraction response is not a list")
            return []

        triples: list[Triple] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            subject = item.get("subject")
            predicate = item.get("predicate") or item.get("relation")
            object_value = item.get("object")
            if not isinstance(subject, str) or not isinstance(predicate, str):
                continue
            if not isinstance(object_value, str):
                continue
            confidence = item.get("confidence", 1.0)
            properties = item.get("properties")
            if not isinstance(confidence, (int, float)):
                confidence = 1.0
            property_map = properties if isinstance(properties, dict) else {}
            triples.append(
                Triple(
                    subject=subject.strip(),
                    predicate=predicate.strip(),
                    object=object_value.strip(),
                    confidence=float(confidence),
                    properties=property_map,
                )
            )
        return triples

    # ── Post-processing: filter + deduplicate ────────────────────

    def _filter_triples(
        self, triples: list[Triple], entity_names: set[str]
    ) -> list[Triple]:
        """Remove self-loops, orphans, and invalid relation types."""

        filtered: list[Triple] = []
        for t in triples:
            # 1. Self-loop: subject == object
            if t.subject == t.object:
                self._logger.debug(f"Filtered self-loop: ({t.subject}) → ({t.object})")
                continue

            # 2. Orphan: subject or object not in entity list
            if t.subject not in entity_names or t.object not in entity_names:
                self._logger.debug(
                    f"Filtered orphan triple: ({t.subject})"
                    f" --[{t.predicate}]--> ({t.object})"
                )
                continue

            # 3. Relation type validation (if types are defined)
            if self._relation_types and t.predicate not in self._relation_types:
                self._logger.debug(
                    f"Filtered invalid relation type '{t.predicate}': "
                    f"({t.subject}) → ({t.object})"
                )
                continue

            filtered.append(t)

        removed = len(triples) - len(filtered)
        if removed:
            self._logger.info(f"Filtered {removed} invalid triples")
        return filtered

    def _deduplicate_triples(self, triples: list[Triple]) -> list[Triple]:
        """Merge duplicate triples by (subject, predicate, object)."""

        merged: dict[tuple[str, str, str], Triple] = {}
        for triple in triples:
            key = (
                triple.subject.strip(),
                triple.predicate.strip(),
                triple.object.strip(),
            )
            if not all(key):
                continue
            existing = merged.get(key)
            if existing is None:
                merged[key] = triple
                continue
            existing.confidence = max(existing.confidence, triple.confidence)
            for prop_key, prop_value in triple.properties.items():
                existing.properties.setdefault(prop_key, prop_value)
        return list(merged.values())
