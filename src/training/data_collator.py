from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.common.logger import get_logger
from src.data_processing.document_loader import Document
from src.llm.base_client import BaseLLMClient

_LOGGER = get_logger(__name__)


@runtime_checkable
class DatasetProtocol(Protocol):
    @classmethod
    def from_dict(cls, data: dict[str, list[list[int]]]) -> DatasetProtocol: ...


@dataclass
class DAPTSample:
    text: str
    source: str | None = None


@dataclass
class SFTSample:
    question: str
    answer: str
    context: str | None = None
    source: str | None = None


class DAPTCollator:
    def __init__(
        self,
        tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct",
        max_length: int = 2048,
        stride: int = 512,
    ) -> None:
        self._tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.stride = stride
        self._tokenizer: Any | None = None

    def collate(self, samples: list[DAPTSample]) -> DatasetProtocol:
        texts: list[str] = []
        for sample in samples:
            texts.extend(self._chunk_text(sample.text))

        if not texts:
            _LOGGER.warning("No DAPT chunks produced from samples")
            return self._dataset_from_dict(
                {"input_ids": [], "attention_mask": [], "labels": []}
            )

        tokenized = self._tokenize(texts)
        return self._dataset_from_dict(tokenized)

    def _chunk_text(self, text: str) -> list[str]:
        tokenizer = self._get_tokenizer()
        tokenized = tokenizer(text, add_special_tokens=False)
        input_ids: list[int] = tokenized["input_ids"]
        if not input_ids:
            return []

        step = self.max_length - self.stride
        if step <= 0:
            step = self.max_length

        chunks: list[str] = []
        for start in range(0, len(input_ids), step):
            end = start + self.max_length
            chunk_ids = input_ids[start:end]
            if not chunk_ids:
                continue
            chunk_text = tokenizer.decode(
                chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if chunk_text.strip():
                chunks.append(chunk_text)
        return chunks

    def _tokenize(self, texts: list[str]) -> dict[str, list[list[int]]]:
        tokenizer = self._get_tokenizer()
        tokenized = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        input_ids: list[list[int]] = tokenized["input_ids"]
        attention_mask: list[list[int]] = tokenized["attention_mask"]
        labels = [ids.copy() for ids in input_ids]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name, use_fast=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self._tokenizer = tokenizer
        return self._tokenizer

    def _dataset_from_dict(self, data: dict[str, list[list[int]]]) -> DatasetProtocol:
        dataset_class = self._get_dataset_class()
        return dataset_class.from_dict(data)

    def _get_dataset_class(self) -> type[DatasetProtocol]:
        try:
            from datasets import Dataset  # type: ignore
        except ModuleNotFoundError as exc:
            message = (
                "datasets is required for training data collation. "
                "Install it with `uv sync --group llm` or `uv add datasets`."
            )
            raise ModuleNotFoundError(message) from exc
        return Dataset


class SFTCollator:
    def __init__(
        self,
        tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct",
        max_length: int = 2048,
    ) -> None:
        self._tokenizer_name = tokenizer_name
        self.max_length = max_length
        self._tokenizer: Any | None = None

    def collate(self, samples: list[SFTSample]) -> DatasetProtocol:
        conversations = [self._format_conversation(sample) for sample in samples]
        if not conversations:
            _LOGGER.warning("No SFT conversations produced from samples")
            return self._dataset_from_dict(
                {"input_ids": [], "attention_mask": [], "labels": []}
            )

        tokenized = self._tokenize(conversations)
        return self._dataset_from_dict(tokenized)

    def _format_conversation(self, sample: SFTSample) -> str:
        question = sample.question
        if sample.context:
            question = f"{question}\n\nContext:\n{sample.context}"

        return (
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{sample.answer}<|im_end|>"
        )

    def _tokenize(self, conversations: list[str]) -> dict[str, list[list[int]]]:
        tokenizer = self._get_tokenizer()
        tokenized = tokenizer(
            conversations,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        input_ids: list[list[int]] = tokenized["input_ids"]
        attention_mask: list[list[int]] = tokenized["attention_mask"]
        labels = [ids.copy() for ids in input_ids]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name, use_fast=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self._tokenizer = tokenizer
        return self._tokenizer

    def _dataset_from_dict(self, data: dict[str, list[list[int]]]) -> DatasetProtocol:
        dataset_class = self._get_dataset_class()
        return dataset_class.from_dict(data)

    def _get_dataset_class(self) -> type[DatasetProtocol]:
        try:
            from datasets import Dataset  # type: ignore
        except ModuleNotFoundError as exc:
            message = (
                "datasets is required for training data collation. "
                "Install it with `uv sync --group llm` or `uv add datasets`."
            )
            raise ModuleNotFoundError(message) from exc
        return Dataset


def load_documents_as_dapt(documents: list[Document]) -> list[DAPTSample]:
    return [
        DAPTSample(text=document.content, source=str(document.source_path))
        for document in documents
    ]


async def generate_qa_pairs(
    text: str,
    llm_client: BaseLLMClient,
) -> list[SFTSample]:
    _LOGGER.info(
        "QA pair generation is not implemented yet",
        extra={"text_length": len(text), "provider": llm_client.provider},
    )
    return []
