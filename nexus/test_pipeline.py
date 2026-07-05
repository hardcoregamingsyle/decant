import sys
sys.path.insert(0, "/home/daytona/codebase")

print("=" * 60)
print("TEST 1: Import modules")
print("=" * 60)
from nexus.parser import parse_string, parse_file
from nexus.generator import generate_code, write_generated_code
print("  ✓ All modules imported successfully")
print()

print("=" * 60)
print("TEST 2: Parse example .nx file (train-llama3-8b)")
print("=" * 60)
src = open("/home/daytona/codebase/nexus/examples/train-llama3-8b.nx").read()
config = parse_string(src)
print(f"  Model:    {config.model.base}")
print(f"  Purpose:  {config.train.purpose}")
print(f"  Precision: {config.train.precision}")
print(f"  Compression: {config.train.compression}")
print(f"  LoRA rank: {config.train.lora.rank}")
print(f"  Dataset:  {config.dataset.path}")
print(f"  Context:  {config.model.context} -> {config.model.context_extend}")
print(f"  Valid:    {config.is_valid}")
print()

print("=" * 60)
print("TEST 3: Generate Python code")
print("=" * 60)
code = generate_code(config)
lines = code.split("\n")
print(f"  Generated {len(lines)} lines of Python code")
print(f"  First line: {lines[0][:60]}")
print()

print("=" * 60)
print("TEST 4: Validate generated Python syntax")
print("=" * 60)
try:
    compile(code, "nexus_generated.py", "exec")
    print("  ✓ Generated code is syntactically valid Python!")
except SyntaxError as e:
    print(f"  ✗ Syntax error: {e}")
    print()
    # Show the problematic area
    if hasattr(e, 'lineno') and e.lineno:
        context_start = max(0, e.lineno - 5)
        context_end = min(len(lines), e.lineno + 3)
        print(f"  Context around line {e.lineno}:")
        for i in range(context_start, context_end):
            marker = " >>>" if i == e.lineno - 1 else "    "
            print(f"  {marker} {i+1}: {lines[i]}")
    sys.exit(1)
print()

print("=" * 60)
print("TEST 5: Parse host example")
print("=" * 60)
src2 = open("/home/daytona/codebase/nexus/examples/host-mistral.nx").read()
config2 = parse_string(src2)
print(f"  Model:    {config2.model.base}")
print(f"  Purpose:  {config2.train.purpose}")
code2 = generate_code(config2)
compile(code2, "nexus_host.py", "exec")
print(f"  ✓ Host config generated valid Python ({len(code2.split(chr(10)))} lines)")
print()

print("=" * 60)
print("TEST 6: Parse local dataset example")
print("=" * 60)
src3 = open("/home/daytona/codebase/nexus/examples/train-local-dataset.nx").read()
config3 = parse_string(src3)
print(f"  Model:    {config3.model.base}")
print(f"  Dataset:  {config3.dataset.local_path}")
print(f"  Format:   {config3.dataset.format}")
code3 = generate_code(config3)
compile(code3, "nexus_local.py", "exec")
print(f"  ✓ Local dataset config generated valid Python ({len(code3.split(chr(10)))} lines)")
print()

print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
