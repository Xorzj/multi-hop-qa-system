from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.common.logger import get_logger

if TYPE_CHECKING:
    from datasets import Dataset  # type: ignore[import-not-found]
else:
    type Dataset = Any

type BitsAndBytesConfig = Any
type LoraConfig = Any
type TrainingArguments = Any

_LOGGER = get_logger(__name__)


@dataclass
class SFTConfig:
    output_dir: str
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    base_adapter_path: str | None = None
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    max_length: int = 2048
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    use_4bit: bool = True
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    warmup_ratio: float = 0.03


class _MaskedSFTCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        marker = "<|im_start|>assistant\n"
        self._assistant_marker_ids: list[int] = tokenizer(
            marker, add_special_tokens=False
        )["input_ids"]

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, list[list[int]]]:
        texts: list[str] = []
        pretokenized: list[dict[str, Any]] = []
        for sample in samples:
            if "text" in sample:
                texts.append(sample["text"])
                continue
            if "conversation" in sample:
                texts.append(sample["conversation"])
                continue
            if "input_ids" in sample:
                pretokenized.append(sample)
                continue
            raise ValueError("SFT sample missing text/conversation/input_ids")

        if texts:
            tokenized = self._tokenizer(
                texts,
                add_special_tokens=False,
                truncation=True,
                max_length=self._max_length,
                padding="max_length",
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
        else:
            input_ids = [sample["input_ids"] for sample in pretokenized]
            attention_mask = [
                sample.get("attention_mask", self._build_attention_mask(sample))
                for sample in pretokenized
            ]

        labels: list[list[int]] = []
        for ids, mask in zip(input_ids, attention_mask, strict=True):
            labels.append(self._mask_labels(ids, mask))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _build_attention_mask(self, sample: dict[str, Any]) -> list[int]:
        pad_token_id = self._tokenizer.pad_token_id
        return [
            0 if token_id == pad_token_id else 1 for token_id in sample["input_ids"]
        ]

    def _mask_labels(
        self, input_ids: list[int], attention_mask: list[int]
    ) -> list[int]:
        start_index = self._find_assistant_start(input_ids)
        labels = [-100 for _ in input_ids]
        if start_index is None:
            return labels
        for idx in range(start_index, len(input_ids)):
            if attention_mask[idx] == 1:
                labels[idx] = input_ids[idx]
        return labels

    def _find_assistant_start(self, input_ids: list[int]) -> int | None:
        marker = self._assistant_marker_ids
        if not marker:
            return None
        max_start = len(input_ids) - len(marker)
        for idx in range(max_start + 1):
            if input_ids[idx : idx + len(marker)] == marker:
                return idx + len(marker)
        return None


class SFTTrainer:
    def __init__(self, config: SFTConfig) -> None:
        """Initialize trainer with configuration."""

        self.config = config
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._trainer: Any | None = None

    def setup(self) -> None:
        """Load tokenizer and model with QLoRA adapters for SFT."""

        from peft import (  # type: ignore
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer

        quantization_config = (
            self._setup_quantization() if self.config.use_4bit else None
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            quantization_config=quantization_config,
        )
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if self.config.use_4bit:
            model = prepare_model_for_kbit_training(model)

        self._model = model
        self._load_base_adapter()
        if self._model is None:
            raise RuntimeError("Failed to initialize base model for SFT.")
        self._model = get_peft_model(self._model, self._setup_lora())

        _LOGGER.info(
            "SFT trainer setup complete", extra={"model": self.config.model_name}
        )

    def train(self, dataset: Dataset) -> None:
        """Run SFT training loop on the provided dataset."""

        if self._model is None or self._tokenizer is None:
            self.setup()

        from transformers import Trainer  # type: ignore

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Trainer setup failed to initialize model or tokenizer.")

        data_collator = _MaskedSFTCollator(self._tokenizer, self.config.max_length)
        training_args = self._setup_training_args()
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        trainer.train()
        self._trainer = trainer

    def save(self, path: str | None = None) -> None:
        """Save LoRA adapter weights to the target path."""

        if self._model is None:
            raise RuntimeError(
                "Model is not initialized. Call setup() or train() first."
            )
        output_path = path or self.config.output_dir
        self._model.save_pretrained(output_path)
        _LOGGER.info("Saved LoRA adapters", extra={"path": output_path})

    def _load_base_adapter(self) -> None:
        """Load and merge an existing DAPT adapter if configured."""

        if not self.config.base_adapter_path:
            return
        if self._model is None:
            raise RuntimeError(
                "Base model must be initialized before loading adapters."
            )

        from peft import PeftModel  # type: ignore

        peft_model = PeftModel.from_pretrained(
            self._model, self.config.base_adapter_path
        )
        self._model = peft_model.merge_and_unload()
        _LOGGER.info(
            "Merged base adapter",
            extra={"path": self.config.base_adapter_path},
        )

    def _setup_quantization(self) -> BitsAndBytesConfig:
        """Create 4-bit quantization configuration for QLoRA."""

        import torch
        from transformers import BitsAndBytesConfig  # type: ignore

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    def _setup_lora(self) -> LoraConfig:
        """Create PEFT LoRA configuration targeting attention and MLP projections."""

        from peft import LoraConfig  # type: ignore

        return LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    def _setup_training_args(self) -> TrainingArguments:
        """Build HuggingFace training arguments for SFT."""

        from transformers import TrainingArguments  # type: ignore

        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            warmup_ratio=self.config.warmup_ratio,
            bf16=self.config.bf16,
            remove_unused_columns=False,
            report_to=[],
        )
