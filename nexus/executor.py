"""Nexus DSL — Execution Engine.

Handles running the generated Python training scripts with
plain-English error reporting for end users.
"""

from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

from nexus.models import NexusConfig
from nexus.generator import generate_code, write_generated_code


ERROR_SUFFIX_HINTS: dict[str, str] = {
    "killed": (
        "The training process was killed (probably ran out of system memory).\n"
        "  → Use a smaller model, lower precision, or reduce context length."
    ),
    "segfault": (
        "A segmentation fault occurred — usually a driver issue.\n"
        "  → Make sure your NVIDIA drivers and CUDA are up to date."
    ),
    "core dumped": (
        "The system ran out of memory and the kernel killed the process.\n"
        "  → Close other applications, lower batch size, or use a smaller model."
    ),
}


def _translate_exit_code(returncode: int) -> str:
    """Map exit codes to plain-English messages."""
    mapping = {
        -9: "💀 The process was killed by the system (SIGKILL) — ran out of memory!",
        -6: "💥 The process aborted (SIGABRT) — possible CUDA/driver crash.",
        -11: "🚫 Segmentation fault — likely a GPU driver or hardware issue.",
        137: "💀 Out of memory (process was killed by the OOM killer).",
        139: "🚫 Segmentation fault (exit code 139 — GPU/driver problem).",
    }
    return mapping.get(
        returncode,
        f"⚠️ Process exited with code {returncode}. See error output above.",
    )


def run_training(
    config: NexusConfig,
    generated_path: Optional[str | Path] = None,
    timeout: Optional[int] = None,
) -> int:
    """Run the generated training script for the given NexusConfig.

    Parameters
    ----------
    config : NexusConfig
        Validated Nexus configuration.
    generated_path : str or Path, optional
        Where to write the generated Python script (default: temp file).
    timeout : int, optional
        Maximum runtime in seconds (no limit if None).

    Returns
    -------
    int
        Exit code of the training process (0 = success).
    """
    # Write the generated script
    if generated_path is None:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="nexus_"
        )
        generated_path = tmp.name
        tmp.close()  # close so subprocess can write to it
        # Write code to it
        code = generate_code(config)
        Path(generated_path).write_text(code)
    else:
        generated_path = Path(generated_path)
        write_generated_code(config, generated_path)

    script_path = str(generated_path)

    print(f"\n{'='*60}")
    print(f"  Nexus — Training Pipeline")
    print(f"  Config: {config.model.base}")
    print(f"  Output: {config.output.resolve_path()}")
    print(f"{'='*60}\n")

    # Determine Python executable
    python_exe = sys.executable

    try:
        process = subprocess.Popen(
            [python_exe, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output in real-time
        for line in process.stdout or []:
            print(line, end="")

        process.wait(timeout=timeout)

        if process.returncode != 0:
            print(f"\n{'─'*50}")
            print("  ❌ Training failed!")
            print(f"  {_translate_exit_code(process.returncode)}")
            print(f"{'─'*50}\n")
        else:
            print(f"\n{'─'*50}")
            print("  ✅ Training completed successfully!")
            print(f"{'─'*50}\n")

        return process.returncode

    except subprocess.TimeoutExpired:
        process.kill()
        print(f"\n{'─'*50}")
        print("  ⏰ Training timed out!")
        print(f"  The process was killed after {timeout}s.")
        print(f"{'─'*50}\n")
        return -1

    except FileNotFoundError:
        print(f"\n{'─'*50}")
        print("  ❌ Python not found!")
        print(f"  Could not find: {python_exe}")
        print(f"{'─'*50}\n")
        return -2

    except Exception as e:
        print(f"\n{'─'*50}")
        print("  ❌ Failed to launch training!")
        print(f"  {str(e)}")
        print(f"{'─'*50}\n")
        traceback.print_exc()
        return -3


def run_build_only(
    config: NexusConfig,
    output_path: str | Path,
) -> Path:
    """Generate the Python script without executing it. Returns the output path."""
    output_path = Path(output_path)
    return write_generated_code(config, output_path)
