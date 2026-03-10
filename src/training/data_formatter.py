"""Training Data Formatter: Convert Teacher annotations to ChatML instruction format.

Transforms TeacherAnnotation JSONL records into Qwen ChatML conversations
suitable for QLoRA fine-tuning of the Student extraction model.

Two instruction modes:
1. **Entity extraction**: chunk_text → entities JSON
2. **Triple extraction**: chunk_text + entities → triples JSON

Typical usage::

    from src.training.data_formatter import AnnotationFormatter

    formatter = AnnotationFormatter()
    samples = formatter.format_annotations(annotations)
    formatter.save_jsonl(samples, Path("data/sft_train.jsonl"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.logger import get_logger
from src.data_processing.teacher_annotator import TeacherAnnotation

logger = get_logger(__name__)


@dataclass
class ChatMLSample:
    """A single ChatML-formatted training sample."""

    conversation: str  # Full ChatML string
    task_type: str  # "entity_extraction" or "triple_extraction"
    chunk_id: str


class AnnotationFormatter:
    """Convert Teacher annotations to ChatML instruction format for SFT."""

    ENTITY_SYSTEM = (
        "你是专业的知识图谱实体抽取专家。"
        "请从给定文本中识别所有重要的实体。"
        "以JSON数组格式输出结果，每个元素包含 name 和 type 字段。"
    )

    TRIPLE_SYSTEM = (
        "你是专业的知识图谱关系抽取专家。"
        "请根据给定文本和实体列表，识别实体之间的关系三元组。"
        "以JSON数组格式输出结果，"
        "每个元素包含 subject、predicate、object 字段。"
    )

    def format_annotations(
        self,
        annotations: list[TeacherAnnotation],
        include_entity_task: bool = True,
        include_triple_task: bool = True,
    ) -> list[ChatMLSample]:
        """Convert annotations to ChatML training samples.

        Each annotation can produce up to 2 samples:
        one for entity extraction and one for triple extraction.
        """
        samples: list[ChatMLSample] = []

        for ann in annotations:
            if include_entity_task and ann.entities:
                sample = self._format_entity_sample(ann)
                if sample:
                    samples.append(sample)

            if include_triple_task and ann.triples and ann.entities:
                sample = self._format_triple_sample(ann)
                if sample:
                    samples.append(sample)

        logger.info(
            "Formatted %d training samples from %d annotations",
            len(samples),
            len(annotations),
        )
        return samples

    def _format_entity_sample(self, ann: TeacherAnnotation) -> ChatMLSample | None:
        """Format entity extraction as ChatML conversation."""
        heading_ctx = ""
        if ann.heading_chain:
            heading_ctx = f"\n当前章节位置：{' > '.join(ann.heading_chain)}\n"

        user_msg = (
            f"请从下面的文本中抽取所有重要的实体。\n"
            f"{heading_ctx}"
            f"\n待抽取文本：\n{ann.chunk_text}\n\n"
            f"请直接返回JSON数组："
        )

        # Build assistant response: clean entities JSON
        entities_json = json.dumps(ann.entities, ensure_ascii=False, indent=2)
        assistant_msg = entities_json

        conversation = self._build_chatml(
            system=self.ENTITY_SYSTEM,
            user=user_msg,
            assistant=assistant_msg,
        )
        return ChatMLSample(
            conversation=conversation,
            task_type="entity_extraction",
            chunk_id=ann.chunk_id,
        )

    def _format_triple_sample(self, ann: TeacherAnnotation) -> ChatMLSample | None:
        """Format triple extraction as ChatML conversation."""
        heading_ctx = ""
        if ann.heading_chain:
            heading_ctx = f"\n当前章节位置：{' > '.join(ann.heading_chain)}\n"

        entity_lines = []
        for e in ann.entities:
            name = e.get("name", "")
            etype = e.get("type", "")
            entity_lines.append(f"- {name}（{etype}）")
        entity_list = "\n".join(entity_lines)

        user_msg = (
            f"请根据文本和实体列表，识别实体之间的关系三元组。\n"
            f"{heading_ctx}"
            f"\n已知实体列表：\n{entity_list}\n"
            f"\n待抽取文本：\n{ann.chunk_text}\n\n"
            f"请直接返回JSON数组："
        )

        triples_json = json.dumps(ann.triples, ensure_ascii=False, indent=2)
        assistant_msg = triples_json

        conversation = self._build_chatml(
            system=self.TRIPLE_SYSTEM,
            user=user_msg,
            assistant=assistant_msg,
        )
        return ChatMLSample(
            conversation=conversation,
            task_type="triple_extraction",
            chunk_id=ann.chunk_id,
        )

    @staticmethod
    def _build_chatml(system: str, user: str, assistant: str) -> str:
        """Build a Qwen ChatML formatted conversation string."""
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>"
        )

    @staticmethod
    def save_jsonl(samples: list[ChatMLSample], path: Path) -> None:
        """Save ChatML samples to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                record = {
                    "text": sample.conversation,
                    "task_type": sample.task_type,
                    "chunk_id": sample.chunk_id,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Saved %d samples to %s", len(samples), path)

    @staticmethod
    def load_jsonl(path: Path) -> list[dict[str, Any]]:
        """Load JSONL training data records."""
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
