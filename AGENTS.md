# Repository Guidelines

## Project Structure & Module Organization
`vllm/` contains the Python runtime, API server entrypoints, schedulers, and model integrations for this fork. Put feature work near the owning subsystem, for example `vllm/entrypoints/`, `vllm/engine/`, or `vllm/v1/`. `csrc/` and `cmake/` hold CUDA/C++ extensions and native build logic. `tests/` mirrors runtime areas such as `tests/entrypoints/`, `tests/multimodal/`, `tests/v1/`, and `tests/encoder_decoder/`. Use `examples/` for runnable templates and clients, `benchmarks/` for performance scripts, and `.buildkite/` plus the top-level Dockerfiles for CI and platform-specific builds.

## Build, Test, and Development Commands
Install the recommended CUDA dev environment with `pip install -r requirements-cuda.txt && pip install -e .`. Switch to another `requirements-*.txt` when targeting CPU, ROCm, XPU, or TPU. This fork expects `VLLM_USE_V1=1`; export it before local runs. Start the OpenAI-compatible server with `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-VL-7B-Instruct --gpu-memory-utilization 0.8 --port 7778 --enforce-eager`. Run tests with `python -m pytest tests/`, or narrow scope during iteration, for example `python -m pytest tests/encoder_decoder/`. For linting, install `requirements-lint.txt`, then run `pre-commit install --hook-type pre-commit --hook-type commit-msg` and `pre-commit run --all-files`.

## Coding Style & Naming Conventions
Use 4-space indentation in Python and keep lines within the repo's 80-column Ruff setting. Formatting and static checks are enforced with `yapf`, `ruff`, `isort`, `mypy`, `codespell`, and `shellcheck` through `pre-commit`. Follow existing names: `snake_case` for files, functions, and tests; `PascalCase` for classes; `UPPER_CASE` for constants. Keep shell helpers descriptive, such as `client_qwen2vl_7b.sh`. Native code under `csrc/` follows the checked-in `.clang-format`.

## Testing Guidelines
Tests use `pytest`. Reuse the markers defined in `pyproject.toml`, including `distributed`, `cpu_model`, `quant_model`, and `optional`, so hardware-specific coverage stays filterable. Place new tests beside the code they exercise and prefer focused regression tests before adding large end-to-end cases. There is no declared global coverage threshold in this repo; bug fixes and scheduler or model changes should ship with a targeted test.

## Commit & Pull Request Guidelines
Recent history favors short, lower-case, imperative subjects such as `update readme` and `add client script`. Keep the first line concise and specific. DCO sign-off is expected: either install the commit-msg hook or add `Signed-off-by: Your Name <you@example.com>` manually. Pull requests should describe the behavior change, note hardware or runtime assumptions, link related issues, and include benchmark data or request/response examples when modifying performance-sensitive or serving paths.
