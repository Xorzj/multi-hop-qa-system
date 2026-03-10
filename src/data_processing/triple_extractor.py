from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.common.config import ExtractionConfig
from src.common.logger import get_logger
from src.data_processing.document_loader import Section
from src.data_processing.entity_extractor import Entity
from src.data_processing.relation_types import (
    RelationType,
    build_relation_type_prompt,
)
from src.llm.base_client import BaseLLMClient, GenerationParams

if TYPE_CHECKING:
    from src.data_processing.schema_inducer import DomainSchema


@dataclass
class Triple:
    """Relationship triple extracted from document text."""

    subject: str
    predicate: str
    object: str
    relation_type: str = ""
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
        relation_types: list[str] | list[RelationType] | None = None,
        extraction_config: ExtractionConfig | None = None,
        domain_schema: DomainSchema | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._logger = get_logger(__name__)
        self._domain_schema = domain_schema

        cfg = extraction_config

        # Accept both list[str] (legacy) and list[RelationType]
        raw_types = relation_types or (cfg.relation_types if cfg else [])
        self._relation_type_names: list[str] = []
        self._structured_types: list[RelationType] = []
        for rt in raw_types:
            if isinstance(rt, RelationType):
                self._structured_types.append(rt)
                self._relation_type_names.append(rt.name)
            elif isinstance(rt, str):
                self._relation_type_names.append(rt)

        self._max_retries = cfg.max_retries if cfg else 3

        self._generation_params = GenerationParams(
            max_new_tokens=cfg.max_new_tokens if cfg else 4096,
            temperature=cfg.temperature if cfg else 0.05,
            top_p=cfg.top_p if cfg else 0.1,
            enable_thinking=False,
            system_message=(
                "你是专业的知识图谱关系抽取专家。"
                "你的任务是从给定文本中识别实体之间的关系三元组。"
                "请以JSON数组格式输出结果，"
                "每个元素包含 subject、predicate、object 字段。"
            ),
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

        # Fill source provenance from section metadata
        source_id = section.chunk_id or f"sec_{section.index}"
        source_text = section.content[:200]
        for triple in triples:
            if not triple.source:
                triple.source = source_id
            triple.properties.setdefault("source_chunk_id", source_id)
            triple.properties.setdefault("source_text", source_text)

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
                self._logger.debug(f"LLM response for triple extraction:\n{response}")
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
        """Build prompt with heading context, filtered entities, and relation types.

        When a DomainSchema is available, uses schema-constrained relation types
        and valid combination constraints.
        """

        if self._domain_schema and self._domain_schema.relation_types:
            return self._build_schema_prompt(section, entities)
        return self._build_open_prompt(section, entities)

    def _build_schema_prompt(self, section: Section, entities: list[Entity]) -> str:
        """Build prompt constrained by DomainSchema relation types + constraints."""
        assert self._domain_schema is not None

        parts: list[str] = [
            "你是专业的知识图谱关系抽取专家。请分析文本并识别实体之间的关系三元组。\n",
            "要求：\n"
            "1. 优先从给定的实体列表中寻找关系\n"
            "2. subject和object必须是简短的实体名称"
            "（通常2-6个字），不要使用描述性短语\n"
            "3. 关系必须在文本中有明确依据\n"
            "4. relation_type 必须使用下方定义的关系类型 name 值\n"
            "5. 用JSON数组格式输出\n"
            "6. 注意抽取因果链条中的中间环节\n",
        ]

        if section.heading_chain:
            chain = " > ".join(section.heading_chain)
            parts.append(f"当前章节位置：{chain}\n")

        # Relation types from schema
        schema_rts = self._domain_schema.to_relation_types()
        parts.append(build_relation_type_prompt(schema_rts))
        parts.append(
            "说明：predicate 字段用中文描述具体关系动词，"
            "relation_type 字段必须使用上述类型的 name 值\n"
        )

        # Constraint rules from schema
        constraint_prompt = self._domain_schema.build_constraint_prompt()
        if constraint_prompt:
            parts.append(constraint_prompt)

        # Entity list
        entity_lines = [f"- {e.name}（{e.entity_type}）" for e in entities]
        parts.append(
            "已知实体列表（优先在这些实体之间建立关系）：\n"
            + "\n".join(entity_lines)
            + "\n"
        )

        # Output format example
        example_items: list[str] = []
        for rt in self._domain_schema.relation_types[:2]:
            if rt.examples:
                example_items.append(
                    f'{{"subject": "示例A", "predicate": "示例关系", '
                    f'"object": "示例B", "relation_type": "{rt.name}", '
                    f'"confidence": 0.9}}'
                )
        if not example_items:
            example_items = [
                '{"subject": "A", "predicate": "导致", '
                '"object": "B", "relation_type": "correlate", '
                '"confidence": 0.9}'
            ]

        parts.append(
            f"输出格式示例：\n[{', '.join(example_items)}]\n\n"
            f"待抽取文本：\n{section.content}\n\n"
            "请直接返回JSON数组（实体名称不要加括号类型）："
        )

        return "\n".join(parts)

    def _build_open_prompt(self, section: Section, entities: list[Entity]) -> str:
        """Build generic prompt without schema constraints (original behavior)."""

        parts: list[str] = [
            "你是专业的知识图谱关系抽取专家。请分析文本并识别实体之间的关系三元组。\n",
            "要求：\n"
            "1. 优先从给定的实体列表中寻找关系\n"
            "2. 如果文本中明确提到了未列在实体列表中的重要实体"
            "（尤其是因果链条中的中间节点），"
            "可以将其作为三元组的一部分输出\n"
            "3. subject和object必须是简短的实体名称"
            "（通常2-6个字），不要使用描述性短语\n"
            "   ✓ 正确：“血管内皮”“动脉粥样硬化”\n"
            "   ✗ 错误：“病变发生的核心部位”"
            "“主要的治疗手段”\n"
            "4. 关系必须在文本中有明确依据，不要猜测\n"
            "5. 最终输出为JSON数组格式\n"
            "6. 关系要具体、有信息量，"
            "避免笼统的 '相关'、'有关' 等模糊词汇\n"
            "7. 重要：subject和object字段只填写实体名称，"
            '例如"高血压"而不是"高血压（疾病）"\n'
            "8. 注意抽取因果链条中的中间环节，"
            "不要跳过中间步骤\n"
            "   例如：高血压→损伤→血管内皮"
            "→促进→动脉粥样硬化→导致→冠心病\n"
            "   不要直接写：高血压→导致→冠心病\n",
        ]

        if section.heading_chain:
            chain = " > ".join(section.heading_chain)
            parts.append(f"当前章节位置：{chain}\n")

        # Structured relation types (preferred)
        if self._structured_types:
            parts.append(build_relation_type_prompt(self._structured_types))
            parts.append(
                "说明：predicate 字段用中文描述具体关系动词，"
                "relation_type 字段必须使用上述类型的 name 值\n"
            )
        elif self._relation_type_names:
            types_str = "、".join(self._relation_type_names)
            parts.append(f"关系类型限定（必须使用以下类型之一）：{types_str}\n")
        else:
            parts.append(
                "推荐的关系类型（仅作参考，可自由使用文本中出现的准确动词）：\n"
                "  因果类：导致、引起、促进、抑制、诱发、加剧、缓解\n"
                "  作用类：损伤、修复、消耗、产生、转化、分解\n"
                "  组成类：包含、属于、组成、依赖\n"
                "  描述类：是一种、用于、特征为、表现为\n"
            )

        # Entity list
        entity_lines = [f"- {e.name}（{e.entity_type}）" for e in entities]
        parts.append(
            "已知实体列表（优先在这些实体之间建立关系，"
            "但也可补充文本中明确提到的遗漏实体）：\n" + "\n".join(entity_lines) + "\n"
        )

        # Output format
        if self._structured_types:
            parts.append(
                "输出格式示例：\n"
                '[{"subject": "高血压", "predicate": "损伤", '
                '"object": "血管内皮", "relation_type": "correlate", '
                '"confidence": 0.95}, '
                '{"subject": "动脉粥样硬化", "predicate": "诱发", '
                '"object": "冠心病", "relation_type": "derivative", '
                '"confidence": 0.9}]\n\n'
                f"待抽取文本：\n{section.content}\n\n"
                "请直接返回JSON数组（实体名称不要加括号类型）："
            )
        else:
            parts.append(
                "输出格式示例（注意关系要具体，不要用'相关'）：\n"
                '[{"subject": "高血压", "predicate": "损伤", '
                '"object": "血管内皮", "confidence": 0.95}, '
                '{"subject": "动脉粥样硬化", "predicate": "诱发", '
                '"object": "冠心病", "confidence": 0.9}, '
                '{"subject": "分布式数据库", "predicate": "依赖", '
                '"object": "CAP定理", "confidence": 0.85}]\n\n'
                f"待抽取文本：\n{section.content}\n\n"
                "请直接返回JSON数组（实体名称不要加括号类型）："
            )

        return "\n".join(parts)

    # ── Response parsing ─────────────────────────────────────────

    def _parse_response(self, response: str) -> list[Triple]:
        """Parse the LLM JSON response into a list of Triple objects.

        Handles thinking-model responses where JSON may be wrapped in
        <think> tags or markdown code blocks.
        """

        # Pre-process: strip thinking tags
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()
        if not cleaned:
            cleaned = response  # fallback if everything was inside <think>

        # Strip CoT analysis from models that output reasoning
        # in content field without <think> tags (e.g. glm-4.7-flash)
        cleaned = self._strip_cot_analysis(cleaned)

        payload = None

        # Strategy: markdown code block (common in thinking model output)
        code_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", cleaned)
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
                            f"Repaired truncated JSON: extracted {len(payload)} triples"
                        )
                    except json.JSONDecodeError:
                        pass

        if payload is None:
            self._logger.warning(
                f"Failed to parse triple extraction response: {cleaned[:200]}"
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
                    relation_type=str(item.get("relation_type", "")).strip(),
                    confidence=float(confidence),
                    properties=property_map,
                )
            )
        return triples

    # ── Post-processing: filter + deduplicate ────────────────────

    @staticmethod
    def _strip_cot_analysis(text: str) -> str:
        """Strip chain-of-thought analysis to find JSON content.

        Thinking models may output reasoning as plain markdown
        in the content field. This locates the first JSON-like
        content ([ or {) and returns from that point onward.
        """
        stripped = text.strip()
        if stripped.startswith(("[", "{")):
            return text
        bracket = stripped.find("[")
        brace = stripped.find("{")
        if bracket == -1 and brace == -1:
            return text  # no JSON-like content found
        candidates = [i for i in (bracket, brace) if i >= 0]
        start = min(candidates)
        return stripped[start:]

    @staticmethod
    def _normalize_entity_name(name: str) -> str:
        """Strip parenthetical type annotations from entity names.

        LLM sometimes returns 'SDH（技术）' instead of 'SDH'.
        This normalizes to just the entity name for matching.
        """
        # Strip Chinese parentheses: “SDH（技术）” -> “SDH”
        name = re.sub(r"（[^）]*）$", "", name)
        # Strip ASCII parentheses: “SDH(技术)” -> “SDH”
        name = re.sub(r"\([^)]*\)$", "", name)
        return name.strip()

    @staticmethod
    def _is_descriptive_phrase(name: str) -> bool:
        """Check if name looks like a descriptive phrase, not a proper entity.

        Rejects things like "病变发生的核心部位", "主要的治疗手段".
        Keeps things like "动脉粥样硬化", "血管内皮", "DWDM".
        """
        name = name.strip()
        # Too long for Chinese: > 10 chars is likely a phrase
        if len(name) > 10 and not re.search(r"[A-Za-z]", name):
            return True
        # Contains structural particles suggesting a clause/phrase
        _clause_markers = ("的", "了", "过", "着", "地", "得")
        if any(m in name for m in _clause_markers):
            return True
        return False

    def _filter_triples(
        self, triples: list[Triple], entity_names: set[str]
    ) -> list[Triple]:
        """Remove self-loops and invalid relation types.

        When a DomainSchema is available, also validates relation_type
        against the schema's relation type names.
        New entities discovered during triple extraction are kept —
        the graph builder's auto_create_missing_nodes handles them.
        """

        # Build valid relation type set: prefer schema, fallback to configured
        if self._domain_schema and self._domain_schema.relation_types:
            valid_type_names = {rt.name for rt in self._domain_schema.relation_types}
        elif self._relation_type_names:
            valid_type_names = set(self._relation_type_names)
        else:
            valid_type_names = set()

        filtered: list[Triple] = []
        for t in triples:
            # Normalize subject/object to strip type annotations
            subject = self._normalize_entity_name(t.subject)
            obj = self._normalize_entity_name(t.object)

            # 1. Self-loop: subject == object
            if subject == obj:
                self._logger.debug(f"Filtered self-loop: ({subject}) → ({obj})")
                continue

            # 2. Entity name quality check for NEW entities
            #    Known entities pass through; new ones must look
            #    like proper entity names, not descriptive phrases
            bad_entity = False
            for name, label in ((subject, "subject"), (obj, "object")):
                if name not in entity_names:
                    if self._is_descriptive_phrase(name):
                        self._logger.debug(
                            f"Filtered descriptive phrase as {label}: '{name}'"
                        )
                        bad_entity = True
                        break
                    self._logger.info(f"New entity discovered in triple: '{name}'")
            if bad_entity:
                continue
            # 3. Relation type validation
            if valid_type_names:
                # Check both predicate and relation_type
                type_ok = (
                    t.predicate in valid_type_names
                    or t.relation_type in valid_type_names
                )
                if not type_ok:
                    self._logger.debug(
                        f"Filtered invalid relation type "
                        f"'{t.predicate}/{t.relation_type}': "
                        f"({subject}) → ({obj})"
                    )
                    continue

            # Store normalized triple
            filtered.append(
                Triple(
                    subject=subject,
                    predicate=t.predicate,
                    object=obj,
                    relation_type=t.relation_type,
                    confidence=t.confidence,
                    properties=t.properties,
                    source=t.source,
                )
            )

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

    # ── Cross-section extraction ──────────────────────────────

    async def extract_cross_section(
        self,
        sections: list[Section],
        entities: list[Entity],
    ) -> list[Triple]:
        """Extract relations that span across different sections.

        Builds a condensed summary of each section's key entities
        and asks the LLM to identify inter-section relationships.
        """
        if len(sections) < 2 or not entities:
            return []

        prompt = self._build_cross_section_prompt(sections, entities)

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._llm_client.generate(
                    prompt=prompt, params=self._generation_params
                )
                triples = self._parse_response(response)
                if triples:
                    entity_names = {e.name for e in entities}
                    filtered = self._filter_triples(triples, entity_names)
                    self._logger.info(
                        f"Cross-section extraction: {len(filtered)} relations found"
                    )
                    return filtered
            except Exception as exc:
                last_error = exc
                self._logger.warning(
                    f"Cross-section attempt {attempt + 1} failed: {exc}"
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))

        if last_error:
            self._logger.error(
                f"All {self._max_retries} cross-section attempts failed: {last_error}"
            )
        return []

    def _build_cross_section_prompt(
        self,
        sections: list[Section],
        entities: list[Entity],
    ) -> str:
        """Build prompt for cross-section relation extraction."""

        parts: list[str] = [
            "你是专业的知识图谱关系抽取专家。"
            "请分析以下多个章节的内容摘要，"
            "找出跨章节的实体关系。\n",
            "要求：\n"
            "1. 只抽取跨越不同章节的关系，同一章节内的关系不需要\n"
            "2. 关系必须在多个章节的内容中有依据\n"
            "3. 只返回JSON数组，不要有任何其他文字，不要输出分析过程\n",
        ]

        # Relation type guidance
        if self._structured_types:
            parts.append(build_relation_type_prompt(self._structured_types))
        elif self._relation_type_names:
            types_str = "、".join(self._relation_type_names)
            parts.append(f"关系类型限定：{types_str}\n")

        # Section summaries
        parts.append("章节内容摘要：\n")
        for section in sections:
            heading = " > ".join(section.heading_chain) or f"section-{section.index}"
            preview = section.content[:300]
            parts.append(f"--- [{heading}] ---")
            parts.append(preview)
            parts.append("")

        # Entity list
        entity_lines = [f"- {e.name}（{e.entity_type}）" for e in entities]
        parts.append("已知实体列表：\n" + "\n".join(entity_lines) + "\n")

        # Output format
        if self._structured_types:
            parts.append(
                "输出格式：\n"
                '[{"subject": "实体A", "predicate": "关系", '
                '"object": "实体B", "relation_type": "类name", '
                '"confidence": 0.8}]\n\n'
                "请直接返回JSON数组："
            )
        else:
            parts.append(
                "输出格式：\n"
                '[{"subject": "实体A", "predicate": "关系", '
                '"object": "实体B", "confidence": 0.8}]\n\n'
                "请直接返回JSON数组："
            )

        return "\n".join(parts)
