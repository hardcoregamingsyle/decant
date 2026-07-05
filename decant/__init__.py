"""Decant — A minimal DSL for specifying AI model training and hosting.

Parses .dc configuration files and generates VRAM-optimized Python code
using Unsloth, QLoRA, and Hugging Face Transformers.
"""

from decant.models import NexusConfig, ModelConfig, TrainConfig, LoraConfig, DatasetConfig, OutputConfig
from decant.parser import parse_file, parse_string
from decant.generator import generate_code
from decant.executor import run_training

__all__ = [
    "NexusConfig", "ModelConfig", "TrainConfig", "LoraConfig",
    "DatasetConfig", "OutputConfig",
    "parse_file", "parse_string",
    "generate_code",
    "run_training",
]

__version__ = "0.1.0"
