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
class DAPTConfig:
    output_dir: str
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    max_length: int = 2048
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    use_4bit: bool = True
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    warmup_ratio: float = 0.03


class DAPTTrainer:
    def __init__(self, config: DAPTConfig) -> None:
        """Initialize trainer with configuration."""

        self.config = config
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._trainer: Any | None = None

    def setup(self) -> None:
        """Load tokenizer and model with QLoRA adapters."""

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
        model = get_peft_model(model, self._setup_lora())  # type: ignore[assignment]

        self._model = model
        _LOGGER.info(
            "DAPT trainer setup complete", extra={"model": self.config.model_name}
        )

    def train(self, dataset: Dataset) -> None:
        """Run DAPT training loop on the provided dataset."""

        if self._model is None or self._tokenizer is None:
            self.setup()

        from transformers import (  # type: ignore
            DataCollatorForLanguageModeling,
            Trainer,
        )

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Trainer setup failed to initialize model or tokenizer.")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )
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
        """Build HuggingFace training arguments for DAPT."""

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
