"""Teacher Annotator: Generate high-quality extraction annotations with CoT traces.

Uses a strong API model (the "Teacher") to annotate document chunks with
entities and triples, including chain-of-thought reasoning traces. The
output serves as training data for a smaller "Student" model via LoRA
fine-tuning.

Typical usage::

    annotator = TeacherAnnotator(llm_client, domain_schema=schema)
    annotations = await annotator.annotate(sections, sample_ratio=0.2)
    annotator.save_jsonl(annotations, Path("data/teacher_annotations.jsonl"))
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.common.logger import get_logger
from src.data_processing.document_loader import Section
from src.llm.base_client import BaseLLMClient, GenerationParams

if TYPE_CHECKING:
    from src.data_processing.schema_inducer import DomainSchema

logger = get_logger(__name__)


@dataclass
class TeacherAnnotation:
    """Single annotation produced by the Teacher model."""

    chunk_id: str
    chunk_text: str
    thinking_trace: str
    entities: list[dict[str, Any]]
    triples: list[dict[str, Any]]
    heading_chain: list[str] = field(default_factory=list)


class TeacherAnnotator:
    """Annotate document chunks using a strong Teacher LLM with CoT reasoning."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        domain_schema: DomainSchema | None = None,
        generation_params: GenerationParams | None = None,
    ) -> None:
        self._llm = llm_client
        self._schema = domain_schema
        self._params = generation_params or GenerationParams(
            max_new_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            enable_thinking=True,
            system_message=(
                "你是专业的知识图谱标注专家。请仔细分析给定文本，"
                "先输出你的推理过程，然后再输出结构化的抽取结果。"
            ),
        )
        self._logger = logger

    async def annotate(
        self,
        sections: list[Section],
        sample_ratio: float = 0.2,
        seed: int = 42,
    ) -> list[TeacherAnnotation]:
        """Annotate a stratified sample of sections.

        Args:
            sections: All document sections.
            sample_ratio: Fraction of sections to annotate (0-1).
            seed: Random seed for reproducible sampling.

        Returns:
            List of TeacherAnnotation, one per sampled section.
        """
        sampled = self._stratified_sample(sections, sample_ratio, seed)
        self._logger.info(
            "Teacher annotation: %d/%d sections sampled (ratio=%.1f%%)",
            len(sampled),
            len(sections),
            sample_ratio * 100,
        )

        annotations: list[TeacherAnnotation] = []
        for idx, section in enumerate(sampled):
            self._logger.info(
                "Annotating section %d/%d [%s] (%d chars)",
                idx + 1,
                len(sampled),
                section.chunk_id or f"sec_{section.index}",
                len(section.content),
            )
            annotation = await self._annotate_section(section)
            if annotation is not None:
                annotations.append(annotation)

        self._logger.info(
            "Teacher annotation complete: %d annotations produced",
            len(annotations),
        )
        return annotations

    async def _annotate_section(self, section: Section) -> TeacherAnnotation | None:
        """Annotate one section with CoT reasoning."""
        prompt = self._build_prompt(section)
        try:
            response = await self._llm.generate(prompt=prompt, params=self._params)
        except Exception as exc:
            self._logger.error(
                "Teacher annotation failed for %s: %s", section.chunk_id, exc
            )
            return None

        thinking, result = self._parse_response(response)
        if result is None:
            self._logger.warning(
                "Failed to parse Teacher annotation for %s",
                section.chunk_id,
            )
            return None

        return TeacherAnnotation(
            chunk_id=section.chunk_id or f"sec_{section.index}",
            chunk_text=section.content,
            thinking_trace=thinking,
            entities=result.get("entities", []),
            triples=result.get("triples", []),
            heading_chain=list(section.heading_chain),
        )

    def _build_prompt(self, section: Section) -> str:
        """Build Teacher annotation prompt with CoT instructions."""
        parts: list[str] = [
            "请仔细阅读以下文本，完成知识图谱的实体和关系抽取。\n",
            "## 步骤\n"
            "1. 先分析文本的主题和关键信息（推理过程）\n"
            "2. 识别所有重要实体（名称、类型、别名）\n"
            "3. 识别实体间的关系（主语、谓词、宾语、关系类型）\n"
            "4. 检查是否遗漏因果链条中的中间实体\n",
        ]

        if self._schema and self._schema.entity_types:
            parts.append("## 领域实体类型\n")
            parts.append(self._schema.build_entity_type_prompt())
        if self._schema and self._schema.relation_types:
            constraint_prompt = self._schema.build_constraint_prompt()
            if constraint_prompt:
                parts.append(constraint_prompt)

        if section.heading_chain:
            parts.append(f"## 章节位置\n{' > '.join(section.heading_chain)}\n")

        parts.append(
            "## 输出格式\n"
            "先输出推理过程，然后输出JSON结果块：\n"
            "```json\n"
            "{\n"
            '  "entities": [\n'
            '    {"name": "实体名", "type": "类型", "aliases": []}\n'
            "  ],\n"
            '  "triples": [\n'
            '    {"subject": "主语", "predicate": "谓词", '
            '"object": "宾语", "relation_type": "类型", '
            '"confidence": 0.9}\n'
            "  ]\n"
            "}\n"
            "```\n\n"
            f"## 待标注文本\n{section.content}\n\n"
            "请开始分析："
        )
        return "\n".join(parts)

    def _parse_response(self, response: str) -> tuple[str, dict[str, Any] | None]:
        """Parse Teacher response into thinking trace + structured result."""
        import re

        # Extract thinking trace from <think> tags if present
        think_match = re.search(r"<think>([\s\S]*?)</think>", response)
        thinking = think_match.group(1).strip() if think_match else ""

        # Remove thinking tags for JSON extraction
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()

        # If no explicit thinking tags, everything before JSON block is thinking
        if not thinking:
            json_start = cleaned.find("```")
            if json_start > 0:
                thinking = cleaned[:json_start].strip()

        # Extract JSON from code block
        code_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", cleaned)
        if code_match:
            try:
                result = json.loads(code_match.group(1).strip())
                if isinstance(result, dict):
                    return thinking, result
            except json.JSONDecodeError:
                pass

        # Direct JSON parse fallback
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            if not thinking:
                thinking = cleaned[:brace_start].strip()
            try:
                result = json.loads(cleaned[brace_start : brace_end + 1])
                if isinstance(result, dict):
                    return thinking, result
            except json.JSONDecodeError:
                pass

        return thinking or cleaned[:500], None

    @staticmethod
    def _stratified_sample(
        sections: list[Section],
        ratio: float,
        seed: int,
    ) -> list[Section]:
        """Stratified sampling: evenly spaced + random fill."""
        n = max(1, int(len(sections) * ratio))
        n = min(n, len(sections))

        if n >= len(sections):
            return list(sections)

        # Evenly spaced indices
        step = len(sections) / n
        indices = {int(i * step) for i in range(n)}

        # Random fill if not enough
        rng = random.Random(seed)
        remaining = [i for i in range(len(sections)) if i not in indices]
        while len(indices) < n and remaining:
            pick = rng.choice(remaining)
            remaining.remove(pick)
            indices.add(pick)

        return [sections[i] for i in sorted(indices)]

    @staticmethod
    def save_jsonl(annotations: list[TeacherAnnotation], path: Path) -> None:
        """Save annotations to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ann in annotations:
                record = {
                    "chunk_id": ann.chunk_id,
                    "chunk_text": ann.chunk_text,
                    "thinking_trace": ann.thinking_trace,
                    "entities": ann.entities,
                    "triples": ann.triples,
                    "heading_chain": ann.heading_chain,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Saved %d annotations to %s", len(annotations), path)

    @staticmethod
    def load_jsonl(path: Path) -> list[TeacherAnnotation]:
        """Load annotations from JSONL file."""
        annotations: list[TeacherAnnotation] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                annotations.append(
                    TeacherAnnotation(
                        chunk_id=data["chunk_id"],
                        chunk_text=data["chunk_text"],
                        thinking_trace=data.get("thinking_trace", ""),
                        entities=data.get("entities", []),
                        triples=data.get("triples", []),
                        heading_chain=data.get("heading_chain", []),
                    )
                )
        return annotations
