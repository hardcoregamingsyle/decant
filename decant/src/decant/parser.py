"""Decant DSL — Parser.

Uses the Lark parsing library to parse .nx/.dc configuration files
into typed NexusConfig dataclasses with full validation.
"""

from __future__ import annotations

import ast
from pathlib import Path

from lark import Lark, Tree, Token

from decant.models import (
    NexusConfig, ModelConfig, TrainConfig, LoraConfig,
    DatasetConfig, OutputConfig,
)

# Load the grammar from the file shipped alongside this module
_GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"
_PARSER: Lark | None = None


def _get_parser() -> Lark:
    global _PARSER
    if _PARSER is None:
        grammar = _GRAMMAR_PATH.read_text()
        _PARSER = Lark(grammar, parser="lalr", maybe_placeholders=False)
    return _PARSER


# ---------------------------------------------------------------------------
# Value coercion helpers
# ---------------------------------------------------------------------------

def _coerce_value(node: Token | Tree):
    """Turn a Lark Token or Tree into a Python value."""
    # If it's a Tree (e.g. string/number/boolean), unwrap to find the Token
    if isinstance(node, Tree):
        if node.children:
            return _coerce_value(node.children[0])
        return None
    # It's a Token
    text = node.value.strip()
    if text.startswith('"') or text.startswith("'"):
        return ast.literal_eval(text)
    if text.startswith("["):
        return _parse_list(node)
    if text.startswith("{"):
        return _parse_inline_map(node)
    if text in ("true", "false"):
        return text == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _parse_list(token: Token) -> list:
    """Parse a Lark list tree into a Python list."""
    items = []
    for child in token.children:
        if isinstance(child, Tree) and child.data == "value":
            items.append(_coerce_value(child.children[0]))
        elif isinstance(child, Token) and child.type not in ("_NL", "COMMA"):
            items.append(_coerce_value(child))
    return items


def _parse_inline_map(token: Token) -> dict:
    """Parse an inline map { ... } into a Python dict."""
    result = {}
    for child in token.children:
        if isinstance(child, Tree) and child.data == "inline_pair":
            key_parts = [c.value for c in child.children if isinstance(c, Token) and c.type == "CNAME"]
            value_node = [c for c in child.children if isinstance(c, Tree) and c.data == "value"]
            if key_parts and value_node:
                key = ".".join(key_parts)
                result[key] = _coerce_value(value_node[0].children[0])
    return result


# ---------------------------------------------------------------------------
# Nested-dict builder & helpers
# ---------------------------------------------------------------------------

def _build_nested(flat: dict[str, object]) -> object:
    """Convert flat dotted keys into nested dicts.

    Example:
      {"model.base": "foo", "model.context": 2048}
      → {"model": {"base": "foo", "context": 2048}}
    """
    root: dict = {}

    for dotted_key, value in flat.items():
        parts = dotted_key.split(".")
        target = root
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            elif not isinstance(target[part], dict):
                raise ValueError(f"Cannot nest under non-dict key '{part}' (value={target[part]})")
            target = target[part]
        target[parts[-1]] = value

    return root


def _deep_get(d: dict, *keys, default=None):
    """Traverse nested dicts to resolve values from dotted .nx keys.

    Example:
        _deep_get(train_data, "gradient.accumulation", "gradient_accumulation", default=4)
        This will check:
          1. train_data["gradient"]["accumulation"]  (nested path)
          2. train_data["gradient_accumulation"]     (flat key)
          3. return default=4 if neither found
    """
    for key in keys:
        parts = key.split(".")
        cur = d
        found = True
        for part in parts:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                found = False
                break
        if found:
            return cur
    return default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_string(source: str, source_name: str = "<string>") -> NexusConfig:
    """Parse a Decant DSL string into a validated NexusConfig."""
    parser = _get_parser()
    tree = parser.parse(source)

    flat: dict[str, object] = {}
    for statement in tree.children:
        if isinstance(statement, Tree) and statement.data == "assignment":
            # key
            key_parts = []
            # value
            value_token = None
            for child in statement.children:
                if isinstance(child, Tree) and child.data == "dotted_key":
                    key_parts = [c.value for c in child.children if isinstance(c, Token)]
                elif isinstance(child, Tree) and child.data == "value":
                    value_token = child.children[0]

            if key_parts and value_token is not None:
                key = ".".join(key_parts)
                flat[key] = _coerce_value(value_token)

    nested = _build_nested(flat)

    # Be flexible: allow top-level alias "model.source" or "model.base"
    # as well as a standalone "purpose" key
    model_data: dict = nested.get("model", {})
    train_data: dict = nested.get("train", {})
    dataset_data: dict = nested.get("dataset", {})
    output_data: dict = nested.get("output", {})

    # Support standalone "purpose" as shorthand for train.purpose
    if "purpose" in nested and "purpose" not in train_data:
        _val = nested["purpose"]
        if isinstance(_val, str):
            train_data["purpose"] = _val

    # Move compression → train.compression
    # (the nested dict mapping already handles this)

    # Assemble
    lora_data: dict = train_data.pop("lora", {})
    lora = LoraConfig(
        rank=lora_data.get("rank", 16),
        alpha=lora_data.get("alpha", 16),
        dropout=lora_data.get("dropout", 0.0),
        targets=lora_data.get("targets", "all"),
        target_modules=lora_data.get("target_modules", []),
    )

    config = NexusConfig(
        model=ModelConfig(
            name=model_data.get("name", "decant-model"),
            base=model_data.get("base", ""),
            source=model_data.get("source", "huggingface"),
            local_path=_deep_get(model_data, "local_path", "local-path", default=""),
            context=int(model_data.get("context", 2048)),
            context_extend=int(_deep_get(model_data, "context_extend", "context-extend", default=0)),
            context_method=_deep_get(model_data, "context_method", "context-method", default="yarn"),
            rope_theta=float(_deep_get(model_data, "rope_theta", "rope-theta", default=1_000_000.0)),
        ),
        train=TrainConfig(
            purpose=train_data.get("purpose", "train"),
            precision=train_data.get("precision", "4bit"),
            compression=train_data.get("compression", "qlora"),
            lora=lora,
            steps=int(train_data.get("steps", 100)),
            epochs=int(train_data.get("epochs", 0)),
            batch_size=int(_deep_get(train_data, "batch.size", "batch_size", default=2)),
            gradient_accumulation=int(_deep_get(train_data, "gradient.accumulation", "gradient_accumulation", "gradient-accumulation", default=4)),
            learning_rate=float(_deep_get(train_data, "learning.rate", "learning_rate", "learning-rate", default=2e-4)),
            warmup_steps=int(_deep_get(train_data, "warmup.steps", "warmup_steps", "warmup-steps", default=5)),
            optimizer=train_data.get("optimizer", "adamw_8bit"),
            scheduler=train_data.get("scheduler", "cosine"),
            weight_decay=float(_deep_get(train_data, "weight.decay", "weight_decay", "weight-decay", default=0.0)),
            max_grad_norm=float(_deep_get(train_data, "max.grad.norm", "max_grad_norm", "max-grad-norm", default=1.0)),
            logging_steps=int(_deep_get(train_data, "logging.steps", "logging_steps", "logging-steps", default=1)),
            save_steps=int(_deep_get(train_data, "save.steps", "save_steps", "save-steps", default=50)),
            eval_steps=int(_deep_get(train_data, "eval.steps", "eval_steps", "eval-steps", default=50)),
            seed=int(train_data.get("seed", 42)),
        ),
        dataset=DatasetConfig(
            source=dataset_data.get("source", "huggingface"),
            path=dataset_data.get("path", dataset_data.get("paths", "")),
            format=dataset_data.get("format", "text"),
            split=dataset_data.get("split", "train"),
            text_field=_deep_get(dataset_data, "text_field", "text-field", default="text"),
            local_path=_deep_get(dataset_data, "local_path", "local-path", default=""),
        ),
        output=OutputConfig(
            dir=output_data.get("dir", "./outputs"),
            name=output_data.get("name", "decant-finetuned"),
            push_to_hub=_deep_get(output_data, "push_to_hub", "push-to-hub", default=False),
            hub_id=_deep_get(output_data, "hub_id", "hub-id", default=""),
        ),
    )

    errors = config.validate()
    if errors:
        raise ValueError(
            f"Configuration validation failed ({len(errors)} error(s)):\n"
            + "\n".join(f"  ✗ {e}" for e in errors)
        )

    return config


def parse_file(path: str | Path) -> NexusConfig:
    """Parse a .nx file from disk into a validated NexusConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Decant file not found: {path}")
    return parse_string(path.read_text(), source_name=str(path))
