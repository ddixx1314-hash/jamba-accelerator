# AGENTS.md

## 1. Project Overview

This repository is a small Chisel/Scala hardware accelerator learning project.

Project path:

```text
/home/dong/jamba-accelerator
```

Long-term goal:

Build a step-by-step hardware accelerator prototype for Mamba/Jamba-like neural network blocks using Chisel.

Current learning goal:

The user wants to learn Chisel while building the project. The user has some digital logic and Verilog background, but is new to Chisel. Therefore, do not only write code. Explain the hardware idea, Chisel implementation, tests, and Verilog correspondence.

## 2. Current Stack

Use the existing project stack unless there is a strong reason to change it.

Current stack:

- Scala 2.13.x
- Chisel 6.x
- chiseltest
- SBT
- JDK 17
- Verilator
- Python golden models
- pytest

Do not modify `build.sbt` unless necessary.

## 3. Current Repository Structure

Expected structure:

```text
src/main/scala/basic/
src/test/scala/basic/
python/golden/
python/tests/
scripts/
docs/
generated/
```

Current known modules:

```text
Counter
PE
```

Current known Python golden model:

```text
python/golden/mamba_ops.py
```

## 4. Development Philosophy

Work in small, testable steps.

Do not jump directly to a full Jamba accelerator. Build the project in this order:

```text
Counter
PE
DotProduct
SmallGemm4x4
VectorOps
RmsNormStats
MambaStateUpdate
TinyMambaBlock
CausalConv1D
SelectiveScanTiny
AttentionDecodeTiny
TinyJambaBlock
```

If the user does not specify the next task, recommend the next smallest missing module in this sequence.

## 5. Teaching Mode

Act as both a coding agent and a Chisel/hardware accelerator tutor.

For every new hardware module, follow this workflow:

1. Explain what the module does.
2. Explain why it is needed.
3. Explain where it fits in Mamba/Jamba/AI accelerator design.
4. Show a simple dataflow diagram.
5. Implement the Chisel module.
6. Implement the corresponding test.
7. Run the test.
8. If the test fails, explain the error and fix it.
9. After the test passes, explain the key Chisel code.
10. Explain the Verilog correspondence:
    - `Input` / `Output`
    - `Wire`
    - `Reg` / `RegInit`
    - `when`
    - `Vec`
    - `Bundle`
    - `for` loops
    - `reduce`
    - combinational logic
    - sequential logic
11. Give 3 short understanding-check questions for the user.

Keep explanations clear and suitable for someone with basic Verilog knowledge.

## 6. Code Rules

General rules:

- Each task should change only what is necessary.
- Do not perform large refactors unless explicitly requested.
- Do not generate a full Jamba top module unless explicitly requested.
- Prefer simple, readable Chisel over overly clever code.
- Prefer fixed-point or integer versions first.
- Avoid BF16/FP16 in early modules.
- Make bit widths explicit, especially for multiplication and accumulation.
- New Chisel modules should usually go under:

```text
src/main/scala/basic/
```

- New tests should usually go under:

```text
src/test/scala/basic/
```

Naming style:

```text
ModuleName.scala
ModuleNameSpec.scala
```

Example:

```text
DotProduct.scala
DotProductSpec.scala
```

## 7. Testing Rules

Every new hardware module must have a test.

Run:

```bash
sbt test
```

If Python golden models or Python tests are changed, run:

```bash
./scripts/run_test.sh
```

Tests should cover:

- normal cases
- zero inputs
- negative inputs when using `SInt`
- enable/clear/reset behavior when applicable
- at least one meaningful hand-computed expected result

Do not claim the task is finished unless tests pass, or clearly explain why they could not be run.

## 8. Documentation Rules

Maintain a concise learning note file:

```text
docs/learning_notes.md
```

After each completed module, append a short note using this format:

```markdown
## ModuleName

### Function
Briefly describe what the module does.

### Role in the Accelerator
Explain how this module relates to Mamba/Jamba/AI acceleration.

### Chisel Concepts
List the Chisel concepts used.

### Verilog Correspondence
Explain the equivalent Verilog ideas.

### Common Pitfalls
List 1-3 common mistakes.
```

Keep notes concise. Do not write long textbook explanations.

## 9. Current Stage Restrictions

The project is currently in the learning/prototype stage.

Unless the user explicitly asks, do not implement:

- full Jamba2 Mini
- full AXI4
- DDR controller
- full MoE
- full Attention engine
- BF16/FP16 units
- 256K context support
- large memory system
- large project restructuring

## 10. Recommended Next Task

The recommended next task is usually `DotProduct` if it does not already exist.

Target module:

```text
src/main/scala/basic/DotProduct.scala
```

Target test:

```text
src/test/scala/basic/DotProductSpec.scala
```

Requirements:

- parameters:
  - `length = 4`
  - `dataWidth = 8`
  - `accWidth = 32`
- inputs:
  - `a: Vec(length, SInt(dataWidth.W))`
  - `b: Vec(length, SInt(dataWidth.W))`
- output:
  - `y: SInt(accWidth.W)`
- function:

```text
y = sum(a(i) * b(i))
```

Required tests:

```text
[1, 2, 3, 4] · [5, 6, 7, 8] = 70
[1, -2, 3, -4] · [5, 6, -7, -8] = 4
[0, 0, 0, 0] · [0, 0, 0, 0] = 0
```

Do not modify `Counter`, `PE`, or `build.sbt` while doing this task unless necessary.
