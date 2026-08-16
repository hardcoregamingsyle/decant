"""Decant DSL — Python Code Generator.

Translates a validated NexusConfig into VRAM-optimized Python code
using Unsloth, QLoRA, Hugging Face Transformers, TRL, and YaRN/ RoPE scaling.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

from decant.models import (
    NexusConfig,
    TrainConfig,
    LoraConfig,
    ModelConfig,
    DatasetConfig,
    OutputConfig,
)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_IMPORTS = '''\
import torch
import gc
import sys
import traceback
from pathlib import Path
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
'''


def _model_loading_block(model: ModelConfig, precision: str = "4bit") -> str:
    """Generate the model loading block with optional 4-bit / 8-bit / 16-bit."""
    if precision == "4bit":
        load_in_4bit = "True"
        load_in_8bit = "False"
    elif precision == "8bit":
        load_in_4bit = "False"
        load_in_8bit = "True"
    else:
        load_in_4bit = "False"
        load_in_8bit = "False"

    return f'''\
    # -- Load model & tokenizer -----------------------------------------------------------
    model_name = "{model.base}"

    max_seq_length = {model.context_extend or model.context}

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit={load_in_4bit},
        load_in_8bit={load_in_8bit},
        device_map="auto",
    )'''


def _context_extension_block(model: ModelConfig) -> str:
    """Generate YaRN / RoPE scaling block if context extension is configured."""
    if not model.context_extend or model.context_extend <= model.context:
        return "    # No context extension needed\n    pass"

    if model.context_method == "yarn":
        scaling_config = (
            f'        "rope_type": "yarn",\n'
            f'        "factor": {model.context_extend / model.context:.2f},\n'
            f'        "original_max_position_embeddings": {model.context},\n'
            f'        "attention_factor": {model.context_extend / model.context:.2f},\n'
        )
        return f'''\
    # -- Apply YaRN context extension -------------------------------------------------
    # Scaling from {model.context} -> {model.context_extend} tokens ({model.context_extend // model.context}x)
    from transformers import DynamicCache
    model.config.rope_scaling = {{
{scaling_config}
    }}
    model.config.max_position_embeddings = {model.context_extend}
    tokenizer.model_max_length = {model.context_extend}
'''
    elif model.context_method == "ntk":
        return f'''\
    # -- Apply NTK-aware RoPE scaling --------------------------------------------------
    model.config.rope_scaling = {{
        "rope_type": "linear",
        "factor": {model.context_extend / model.context:.2f},
    }}
    model.config.max_position_embeddings = {model.context_extend}
    tokenizer.model_max_length = {model.context_extend}
'''
    elif model.context_method == "linear":
        return f'''\
    # -- Apply linear RoPE scaling -----------------------------------------------------
    model.config.rope_scaling = {{
        "rope_type": "linear",
        "factor": {model.context_extend / model.context:.2f},
    }}
    model.config.max_position_embeddings = {model.context_extend}
    tokenizer.model_max_length = {model.context_extend}
'''
    return "    pass"


def _lora_block(lora: LoraConfig) -> str:
    """Generate LoRA / QLoRA PEFT adapter block."""
    target_modules = lora.resolve_targets()
    modules_str = repr(target_modules) if target_modules else "None"
    return f'''\
    # -- Apply LoRA / QLoRA adapters --------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r={lora.rank},
        target_modules={modules_str},
        lora_alpha={lora.alpha},
        lora_dropout={lora.dropout},
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state={42},
        use_rslora=True,
        loftq_config=None,
    )'''


def _dataset_block(dataset: DatasetConfig, train: TrainConfig) -> str:
    """Generate dataset loading + formatting block."""
    if dataset.source == "huggingface":
        path_expr = f'"{dataset.path}"'
    else:
        path_expr = f'"{dataset.local_path}"'

    blocks = []
    blocks.append(f'''\
    # -- Load dataset -------------------------------------------------------------------
    dataset = load_dataset(
        {path_expr},
        split="{dataset.split}",
        trust_remote_code=True,
    )''')

    if dataset.format in ("alpaca", "sharegpt"):
        alpaca_fmt = 'f"### Instruction:\\n{instruction}\\n\\n### Input:\\n{inp}\\n\\n### Output:\\n{output}"'
        alpaca_fmt_no_input = 'f"### Instruction:\\n{instruction}\\n\\n### Output:\\n{output}"'
        blocks.append(f'''\
    # -- Format dataset for instruction tuning -----------------------------------------
    def format_func(examples):
        texts = []
        for i in range(len(examples.get("instruction", [""] * len(next(iter(examples.values())))))):
            try:
                instruction = examples.get("instruction", [""])[i] or ""
                inp = examples.get("input", examples.get("input_text", examples.get("context", [""])))[i] or ""
                output = examples.get("output", examples.get("response", [""]))[i] or ""
                if inp:
                    texts.append({alpaca_fmt})
                else:
                    texts.append({alpaca_fmt_no_input})
            except (IndexError, KeyError):
                texts.append("")
        return {{"text": texts}}

    dataset = dataset.map(format_func, batched=True, remove_columns=dataset.column_names)
''')
    elif dataset.format == "chat":
        blocks.append(f'''\
    # -- Format for chat -----------------------------------------------------------------
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    def format_chat(examples):
        texts = []
        for conv in examples.get("conversations", examples.get("messages", [])):
            if isinstance(conv, list):
                text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                texts.append(text)
            else:
                texts.append(str(conv))
        return {{"text": texts}}

    dataset = dataset.map(format_chat, batched=True)
''')
    else:
        blocks.append(f'''\
    # -- Use raw text field ------------------------------------------------------------
    if "{dataset.text_field}" in dataset.column_names:
        dataset = dataset.rename_column("{dataset.text_field}", "text")
    elif "text" not in dataset.column_names:
        dataset = dataset.select_columns([dataset.column_names[0]])
        dataset = dataset.rename_column(dataset.column_names[0], "text")
''')

    return "\n".join(blocks)


def _training_block(train: TrainConfig, model: ModelConfig, output_dir: str) -> str:
    """Generate SFTTrainer + TrainingArguments block."""
    max_seq = model.context_extend or model.context

    # Determine effective training steps
    if train.epochs > 0:
        max_steps = -1
        num_epochs = train.epochs
    else:
        max_steps = train.steps
        num_epochs = 1

    return f'''\
    # -- Training arguments -------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir="{output_dir}",
        per_device_train_batch_size={train.batch_size},
        gradient_accumulation_steps={train.gradient_accumulation},
        warmup_steps={train.warmup_steps},
        num_train_epochs={num_epochs},
        max_steps={max_steps},
        learning_rate={train.learning_rate},
        optim="{train.optimizer}",
        weight_decay={train.weight_decay},
        max_grad_norm={train.max_grad_norm},
        logging_steps={train.logging_steps},
        save_steps={train.save_steps},
        eval_strategy="no",
        seed={train.seed},
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={{"use_reentrant": False}},
        dataloader_num_workers=0,
        group_by_length=True,
        lr_scheduler_type="{train.scheduler}",
        ddp_find_unused_parameters=False,
        save_only_model=True,
    )

    # -- Trainer ------------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length={max_seq},
        dataset_num_proc=1,
        packing=False,
        args=training_args,
    )

    # -- Train! -------------------------------------------------------------------------
    trainer_stats = trainer.train()

    # -- Save ---------------------------------------------------------------------------
    model.save_pretrained_merged(
        "{output_dir}",
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"[OK] Model saved to {{output_dir}}")
'''


def _output_block(output: OutputConfig) -> str:
    """Generate final output save / push block."""
    save_path = output.resolve_path()
    blocks = []
    blocks.append(f'''\
    output_path = "{save_path}"
    model.save_pretrained_merged(output_path, tokenizer, save_method="merged_16bit")
    print(f"✓ Model saved to {{output_path}}")
''')

    if output.push_to_hub and output.hub_id:
        blocks.append(f'''\
    # -- Push to Hugging Face Hub --------------------------------------------------------
    model.push_to_hub_merged("{output.hub_id}", tokenizer, save_method="merged_16bit")
    print(f"✓ Model pushed to Hugging Face Hub: {{output.hub_id}}")
''')

    return "\n".join(blocks)


def _error_wrapper_block() -> str:
    """Generate the plain-English error handler block (no emoji for Windows compat)."""
    return '''\
# -- Plain-English error handler --------------------------------------------------------
def handle_error(exc: Exception, tb: str) -> str:
    exc_name = type(exc).__name__
    msg = str(exc).lower()

    if "cuda" in msg or "out of memory" in msg or "memory" in msg:
        return (
            "[VRAM] Your GPU ran out of VRAM (video memory).\\n"
            "   Try these fixes (from easiest to hardest):\\n"
            "   1. Lower train.batch.size to 1\\n"
            "   2. Use train.precision = '4bit' (you are probably already using this)\\n"
            "   3. Lower model.context or model.context_extend\\n"
            "   4. Set train.compression to 'qlora' and lower train.lora.rank to 8 or 4\\n"
            "   5. Enable gradient checkpointing (it's already on by default)\\n"
            "   Your GPU needs about 2-3x the model size in VRAM for training."
        )
    if "no module named" in msg:
        missing = msg.split("no module named")[-1].strip().strip("'\\\"")
        return (
            f"[MISSING] Missing Python package: '{missing}'\\n"
            f"   Run: pip install {missing}"
        )
    if "connection" in msg or "timeout" in msg or "resolve" in msg:
        return (
            "[NETWORK] Network error - could not download the model or dataset.\\n"
            "   Check your internet connection and try again.\\n"
            "   If Hugging Face is blocked in your region, use model.source = 'local'."
        )
    if "not found" in msg or "does not exist" in msg:
        return (
            "[NOT FOUND] File or resource not found.\\n"
            "   Check that the model name or dataset path is spelled correctly.\\n"
            "   For Hugging Face models, use format: 'org/model-name'"
        )
    if "permission" in msg or "access" in msg:
        return (
            "[ACCESS] Permission denied.\\n"
            "   You may need to log in: huggingface-cli login\\n"
            "   Or the model/dataset may require special access."
        )
    if "tokenizer" in msg or "eos" in msg or "bos" in msg:
        return (
            "[TOKENIZER] Tokenizer issue - the model's tokenizer couldn't be loaded correctly.\\n"
            "   Try a different model or check the model name."
        )
    if "shape" in msg or "dimension" in msg or "size mismatch" in msg:
        return (
            "[TENSOR] Tensor shape mismatch - often caused by context window issues.\\n"
            "   Try reducing model.context or model.context_extend."
        )
    if "dataset" in msg and ("format" in msg or "column" in msg):
        return (
            "[DATASET] Dataset format error - the dataset columns don't match what's expected.\\n"
            "   Try dataset.format = 'text' for raw text, or 'alpaca' for instruction data."
        )

    # Default: show the actual error but in a friendlier wrapper
    return (
        f"[ERROR] An unexpected error occurred: {exc_name}\\n"
        f"   {str(exc)[:200]}\\n"
        f"\\n"
        f"   For support, share the full traceback above."
    )
'''


# ---------------------------------------------------------------------------
# Main code generation
# ---------------------------------------------------------------------------

def generate_code(config: NexusConfig, output_dir: Optional[str] = None) -> str:
    """Generate a complete, VRAM-optimized Python training script.

    Parameters
    ----------
    config : NexusConfig
        The validated Decant configuration.
    output_dir : str, optional
        Override the output directory for the saved model.

    Returns
    -------
    str
        A standalone Python script ready to execute.
    """
    parts = [
        '"""Auto-generated by Decant DSL - VRAM-optimized training."""\n',
        _IMPORTS,
        "",
        _error_wrapper_block(),
        "",
        "\n# -- Main training function --------------------------------------------------------\n",
        "def main():\n",
    ]

    _append(parts, "    try:\n", 0)
    _append(parts, _model_loading_block(config.model, config.train.precision), 1)
    _append(parts, "\n", 0)

    ctx = _context_extension_block(config.model)
    _append(parts, ctx, 1)
    _append(parts, "\n", 0)

    if config.train.compression in ("qlora", "lora"):
        _append(parts, _lora_block(config.train.lora), 1)
        _append(parts, "\n", 0)

    if config.train.purpose != "host":
        _append(parts, _dataset_block(config.dataset, config.train), 1)
        _append(parts, "\n", 0)

        trainer_code = _training_block(config.train, config.model, str(config.output.resolve_path()))
        _append(parts, trainer_code, 1)

        _append(parts, _output_block(config.output), 1)
    else:
        _append(parts, "        # Host mode: model loaded, no training needed.\n", 0)
        _append(parts, f'        print("[OK] Model loaded: {config.model.base}")\n', 0)
        _append(parts, f'        print(f"  Context window: {{model.config.max_position_embeddings}} tokens")\n', 0)

    _append(parts, "\n", 0)

    # Error handling
    _append(parts, "    except Exception as e:\n", 0)
    _append(parts, "        tb = traceback.format_exc()\n", 0)
    _append(parts, "        print('\\n[FAIL] Training failed - here\\'s what happened in plain English:')\n", 0)
    _append(parts, '        print("-" * 50)\n', 0)
    _append(parts, "        print(handle_error(e, tb))\n", 0)
    _append(parts, "        print('\\nFull technical details:')\n", 0)
    _append(parts, "        print(tb)\n", 0)
    _append(parts, "        sys.exit(1)\n", 0)

    _append(parts, '\n\nif __name__ == "__main__":\n', 0)
    _append(parts, "    main()\n", 0)

    return "".join(parts)


def _append(parts: list, text: str, indent_level: int) -> None:
    """Append text to parts, handling indentation consistently."""
    if indent_level:
        text = textwrap.indent(text, "    " * indent_level)
    parts.append(text)


def write_generated_code(config: NexusConfig, path: str | Path) -> Path:
    """Generate code and write it to a file. Returns the output path."""
    path = Path(path)
    code = generate_code(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    return path
