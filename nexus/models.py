"""Nexus DSL - Configuration Models.

Typed dataclasses that represent a parsed Nexus configuration,
with validation and sensible defaults for every field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Supported enum-like values
# ---------------------------------------------------------------------------

Purpose = Literal["train", "host", "fine-tune"]
SourceType = Literal["huggingface", "local"]
Precision = Literal["4bit", "8bit", "16bit", "32bit"]
Compression = Literal["qlora", "lora", "full", "none"]
LoRATargets = Literal["all", "attention", "mlp", "custom"]
ContextMethod = Literal["yarn", "ntk", "linear", "none"]
DatasetFormat = Literal["alpaca", "chat", "text", "sharegpt", "json"]

VALID_PURPOSE: set[str] = {"train", "host", "fine-tune"}
VALID_SOURCES: set[str] = {"huggingface", "local"}
VALID_PRECISIONS: set[str] = {"4bit", "8bit", "16bit", "32bit"}
VALID_COMPRESSIONS: set[str] = {"qlora", "lora", "full", "none"}
VALID_CONTEXT_METHODS: set[str] = {"yarn", "ntk", "linear", "none"}
VALID_FORMATS: set[str] = {"alpaca", "chat", "text", "sharegpt", "json"}
VALID_LORA_TARGETS: set[str] = {"all", "attention", "mlp", "custom"}

# Mapping from compression → recommended LoRA target modules
LORA_TARGET_MAP: dict[str, list[str]] = {
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "all": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "custom": [],
}

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Base model and context settings."""
    name: str = "nexus-model"
    base: str = ""
    source: str = "huggingface"
    local_path: str = ""
    context: int = 2048
    context_extend: int = 0
    context_method: str = "yarn"
    rope_theta: float = 1_000_000.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.base:
            errors.append("model.base is required (e.g., 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit')")
        if self.source not in VALID_SOURCES:
            errors.append(f"model.source must be one of: {', '.join(sorted(VALID_SOURCES))}")
        if self.source == "local" and not self.local_path:
            errors.append("model.local_path is required when model.source = 'local'")
        if self.context < 128:
            errors.append(f"model.context must be ≥ 128 (got {self.context})")
        if self.context_extend and self.context_extend < self.context:
            errors.append(f"model.context_extend ({self.context_extend}) must be ≥ model.context ({self.context})")
        if self.context_method not in VALID_CONTEXT_METHODS:
            errors.append(f"model.context_method must be one of: {', '.join(sorted(VALID_CONTEXT_METHODS))}")
        return errors


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class LoraConfig:
    """LoRA / QLoRA adapter settings."""
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    targets: str = "all"
    target_modules: list[str] = field(default_factory=list)

    def resolve_targets(self) -> list[str]:
        if self.target_modules:
            return self.target_modules
        return LORA_TARGET_MAP.get(self.targets, LORA_TARGET_MAP["all"])

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.rank < 1 or self.rank > 512:
            errors.append(f"train.lora.rank must be 1-512 (got {self.rank})")
        if self.alpha < 1:
            errors.append(f"train.lora.alpha must be ≥ 1 (got {self.alpha})")
        if not (0.0 <= self.dropout <= 1.0):
            errors.append(f"train.lora.dropout must be 0.0-1.0 (got {self.dropout})")
        if self.targets not in VALID_LORA_TARGETS:
            errors.append(f"train.lora.targets must be one of: {', '.join(sorted(VALID_LORA_TARGETS))}")
        return errors


@dataclass
class TrainConfig:
    """Training hyper-parameters and optimizer settings."""
    purpose: str = "train"
    precision: str = "4bit"
    compression: str = "qlora"
    lora: LoraConfig = field(default_factory=LoraConfig)
    steps: int = 100
    epochs: int = 0
    batch_size: int = 2
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    optimizer: str = "adamw_8bit"
    scheduler: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 1
    save_steps: int = 50
    eval_steps: int = 50
    eval_strategy: str = "steps"
    seed: int = 42

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.purpose not in VALID_PURPOSE:
            errors.append(f"purpose must be one of: {', '.join(sorted(VALID_PURPOSE))}")
        if self.precision not in VALID_PRECISIONS:
            errors.append(f"train.precision must be one of: {', '.join(sorted(VALID_PRECISIONS))}")
        if self.compression not in VALID_COMPRESSIONS:
            errors.append(f"train.compression must be one of: {', '.join(sorted(VALID_COMPRESSIONS))}")
        if self.purpose != "host":
            if self.steps < 1 and self.epochs < 1:
                errors.append("At least one of train.steps or train.epochs must be > 0")
            if self.batch_size < 1:
                errors.append(f"train.batch.size must be ≥ 1 (got {self.batch_size})")
            if self.learning_rate <= 0:
                errors.append(f"train.learning.rate must be > 0 (got {self.learning_rate})")
        return errors


# ---------------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Dataset source and formatting."""
    source: str = "huggingface"
    path: str = ""
    format: str = "text"
    split: str = "train"
    text_field: str = "text"
    local_path: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.source not in VALID_SOURCES:
            errors.append(f"dataset.source must be one of: {', '.join(sorted(VALID_SOURCES))}")
        if self.source == "huggingface":
            if not self.path:
                errors.append("dataset.path is required when dataset.source = 'huggingface'")
        else:
            if not self.local_path:
                errors.append("dataset.local_path is required when dataset.source = 'local'")
        if self.format not in VALID_FORMATS:
            errors.append(f"dataset.format must be one of: {', '.join(sorted(VALID_FORMATS))}")
        return errors


# ---------------------------------------------------------------------------
# Output Configuration
# ---------------------------------------------------------------------------

@dataclass
class OutputConfig:
    """Where to save the trained model."""
    dir: str = "./outputs"
    name: str = "nexus-finetuned"
    push_to_hub: bool = False
    hub_id: str = ""

    def resolve_path(self) -> Path:
        return Path(self.dir) / self.name

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.dir:
            errors.append("output.dir is required")
        if not self.name:
            errors.append("output.name is required")
        return errors


# ---------------------------------------------------------------------------
# Root Configuration
# ---------------------------------------------------------------------------

@dataclass
class NexusConfig:
    """Complete Nexus DSL configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> list[str]:
        """Run all sub-validations. Returns a flat list of error messages."""
        errors = (
            self.model.validate()
            + self.train.validate()
            + self.output.validate()
        )
        # Dataset is only required when training/fine-tuning
        if self.train.purpose != "host":
            errors += self.dataset.validate()
        return errors

    @property
    def is_valid(self) -> bool:
        return len(self.validate()) == 0
