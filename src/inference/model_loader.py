from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_4bit: bool = True
    bf16: bool = True
    device_map: str = "auto"
    max_memory: dict[int, str] | None = None


class ModelLoader:
    def __init__(self, config: ModelConfig) -> None:
        """Initialize the loader with configuration only."""
        self.config = config
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """Return True when the base model has been loaded."""
        return self._model is not None and self._tokenizer is not None

    def _setup_quantization(self) -> Any:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        bits_and_bytes_config = getattr(transformers, "BitsAndBytesConfig")

        return bits_and_bytes_config(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if self.config.bf16
            else torch.float16,
        )

    def load(self) -> None:
        """Load the base model and tokenizer with optional 4-bit quantization."""
        if self.is_loaded:
            logger.info("Model already loaded")
            return

        transformers = importlib.import_module("transformers")
        auto_model = getattr(transformers, "AutoModelForCausalLM")
        auto_tokenizer = getattr(transformers, "AutoTokenizer")

        quantization_config = None
        if self.config.use_4bit:
            quantization_config = self._setup_quantization()

        logger.info("Loading base model", extra={"model": self.config.model_name})
        self._tokenizer = auto_tokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        self._model = auto_model.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            device_map=self.config.device_map,
            quantization_config=quantization_config,
            max_memory=self.config.max_memory,
        )

    def load_adapter(
        self, adapter_path: str | Path, adapter_name: str = "default"
    ) -> None:
        """Load a LoRA adapter and register it under a name."""
        self._ensure_loaded()

        path = str(adapter_path)
        logger.info(
            "Loading adapter",
            extra={"adapter": adapter_name, "path": path},
        )
        model = self._require_model()
        model.load_adapter(path, adapter_name=adapter_name)

    def switch_adapter(self, adapter_name: str) -> None:
        """Switch the active adapter without reloading the base model."""
        self._ensure_loaded()

        logger.info("Switching adapter", extra={"adapter": adapter_name})
        model = self._require_model()
        model.set_adapter(adapter_name)

    def unload_adapter(self, adapter_name: str) -> None:
        """Unload a named adapter from memory."""
        self._ensure_loaded()

        logger.info("Unloading adapter", extra={"adapter": adapter_name})
        model = self._require_model()
        model.delete_adapter(adapter_name)

    def list_adapters(self) -> list[str]:
        """Return the list of loaded adapter names."""
        self._ensure_loaded()

        model = self._require_model()
        adapters = getattr(model, "peft_config", {})
        if isinstance(adapters, dict):
            return list(adapters.keys())
        return []

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate a response with the active adapter."""
        self._ensure_loaded()

        tokenizer = self._require_tokenizer()
        model = self._require_model()
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )
        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        return response.strip()

    def unload(self) -> None:
        """Release model and tokenizer references and free GPU memory."""
        if not self.is_loaded:
            return

        self._model = None
        self._tokenizer = None

        import gc

        torch = importlib.import_module("torch")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before use")

    def _require_model(self) -> Any:
        self._ensure_loaded()
        return self._model

    def _require_tokenizer(self) -> Any:
        self._ensure_loaded()
        return self._tokenizer
