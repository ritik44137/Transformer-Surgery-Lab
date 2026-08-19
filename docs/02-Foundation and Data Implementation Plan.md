# Foundation and Data Implementation Plan
\## Purpose

This document covers the first two major build layers of Transformer Surgery Lab:  
\- project foundation and environment setup  
\- configuration, runtime utilities, and data pipeline

These stages are where the repo either becomes stable and extensible, or quietly becomes messy. The goal here is to make the later model and experiment work almost boring to build.

Every step below has exactly one \*\*primary file\*\*. Small dependent edits are allowed if needed to keep the repository coherent, but the implementation target for the step is always one file.

\---

\## Phase 1 — Foundation and Environment

\### Goal

Create a clean, reproducible, container-first development environment for WSL Ubuntu 22.04 and establish the repository’s public face and core skeleton.

\### Phase gate

This phase is done only when:  
\- the repo has a credible public scaffold  
\- Docker images build successfully  
\- the development shell is usable  
\- dependencies install cleanly  
\- basic developer commands are ergonomic

\---

\## Step 1.1 — \`README.md\`

\### Why this file matters

The README is the first review surface. Even before the code exists, it should signal taste, direction, and discipline. A rushed README makes the whole project feel smaller than it is.

\### What to implement

Create the \*\*initial public README\*\*, not the final one. It should explain:  
\- the project in one sentence  
\- why transformer component swapping is the core idea  
\- the four comparison axes  
\- the intended outputs: experiments, dashboard, lessons learned  
\- the environment assumption: WSL Ubuntu 22.04 + Docker  
\- a short roadmap section  
\- placeholders for future results

\### What not to do

\- do not include fake metrics  
\- do not overclaim performance  
\- do not make it too long before the repo exists

\### Acceptance criteria

\- concise and professional  
\- understandable in under a minute  
\- clearly more polished than a typical student project README

\### Prompt for coding agent

Create a polished initial README.md for Transformer Surgery Lab. This is a config-driven PyTorch project for controlled transformer architecture experiments. The main swaps are LayerNorm vs RMSNorm, sinusoidal vs RoPE, ReLU vs SwiGLU, and MHA vs GQA. The README should sound serious, resume-grade, and technically grounded, while staying concise. It should mention that development is done in WSL Ubuntu 22.04 with Docker and include placeholders for experiment results, dashboard visuals, and lessons learned.

\### Validation

Read the README top to bottom and ask: if a recruiter saw only this file, would they understand why the repo matters?

\---

\## Step 1.2 — \`.gitignore\`

\### Why this file matters

Research repos get messy fast. If this file is weak, the repo will accumulate run outputs, caches, data artifacts, and editor junk.

\### What to implement

Ignore:  
\- Python caches and build artifacts  
\- experiment directories and checkpoints  
\- tokenized data artifacts  
\- local raw datasets  
\- test and lint caches  
\- notebook leftovers  
\- editor and OS junk  
\- optional local env files

\### Acceptance criteria

\- safe for a research repo with experiments and dashboard work  
\- structured with readable sections

\### Prompt for coding agent

Write a production-quality .gitignore for Transformer Surgery Lab, a PyTorch research repo with Docker, experiment outputs, checkpoints, local datasets, tokenizer artifacts, dashboard assets, Python caches, test caches, and common editor junk. Keep it complete but cleanly organized.

\### Validation

Skim it and confirm nothing obviously important is being ignored incorrectly, especially source files or config files.

\---

\## Step 1.3 — \`.dockerignore\`

\### Why this file matters

Without this, the Docker context becomes slow, large, and error-prone.

\### What to implement

Exclude:  
\- git metadata  
\- experiment outputs  
\- local datasets  
\- caches  
\- compiled artifacts  
\- screenshots or generated assets not needed for build

\### Acceptance criteria

\- build context remains lean  
\- no accidental inclusion of runs or data

\### Prompt for coding agent

Create a clean .dockerignore for Transformer Surgery Lab. Exclude git metadata, local datasets, experiment outputs, checkpoints, Python caches, test caches, editor files, and other unnecessary Docker build-context content.

\### Validation

Compare it mentally with \`.gitignore\` and ensure the Docker build context is even stricter where appropriate.

\---

\## Step 1.4 — \`requirements.txt\`

\### Why this file matters

The project should stay lean. A bloated dependency file makes the repo harder to trust.

\### What to implement

Only runtime dependencies needed for:  
\- PyTorch training  
\- YAML config loading  
\- tokenization support  
\- data export/loading  
\- plotting  
\- Streamlit dashboarding

\### Acceptance criteria

\- minimal and justified packages only  
\- no “just in case” libraries

\### Prompt for coding agent

Create a lean requirements.txt for Transformer Surgery Lab. Include only runtime dependencies needed for PyTorch training, YAML config loading, tokenization support, data handling, plotting, and a Streamlit dashboard. Keep the list disciplined and realistic for a small but polished ML systems repo.

\### Validation

Ask whether every dependency has a clear home in the technical design.

\---

\## Step 1.5 — \`requirements-dev.txt\`

\### Why this file matters

This file defines whether development feels professional or sloppy.

\### What to implement

Include:  
\- pytest  
\- black  
\- ruff  
\- optional mypy  
\- optional ipython

\### Acceptance criteria

\- focused dev tooling only  
\- no duplication of runtime libs unless intentionally pinned

\### Prompt for coding agent

Create requirements-dev.txt for Transformer Surgery Lab with focused development tools for testing, formatting, linting, optional typing, and a strong interactive shell. Keep it lean and professional.

\### Validation

Check that the list matches how you actually intend to work.

\---

\## Step 1.6 — \`docker/Dockerfile\`

\### Why this file matters

This is the canonical environment. If this file is confusing or flaky, everything downstream suffers.

\### What to implement

Create the main development image with:  
\- clean Python 3.11-capable environment  
\- project working directory  
\- dependency installation from requirements files  
\- useful OS packages only  
\- readable layer ordering  
\- strong defaults for containerized development

\### Design note

Prefer a simple image that is easy to reason about over a clever GPU-specific monster. If CUDA-specific work is needed later, keep the structure understandable.

\### Acceptance criteria

\- image builds cleanly  
\- repo mounts cleanly in compose  
\- shell and Python commands work predictably

\### Prompt for coding agent

Write docker/Dockerfile for Transformer Surgery Lab. Target development in WSL Ubuntu 22.04 with Docker. The image should support Python 3.11, PyTorch development, pytest, linting, and Streamlit. Keep it readable, minimal, and production-quality, with good Docker layering and sensible environment defaults.

\### Validation

Build the image and verify a simple Python import command works inside it.

\---

\## Step 1.7 — \`docker/Dockerfile.cpu\`

\### Why this file matters

A CPU path makes smoke testing, onboarding, and portability much easier.

\### What to implement

Mirror the main Dockerfile structure as closely as possible, but avoid assumptions about GPU availability.

\### Acceptance criteria

\- consistent feel with the main image  
\- useful for tests and smoke runs

\### Prompt for coding agent

Write docker/Dockerfile.cpu as a CPU-only companion image for Transformer Surgery Lab. Keep the structure parallel to the main Dockerfile so the environments feel consistent.

\### Validation

Build it and confirm it can run Python and install dependencies cleanly.

\---

\## Step 1.8 — \`docker/entrypoint.sh\`

\### Why this file matters

Entrypoints often become hidden complexity. This one should stay nearly invisible.

\### What to implement

A minimal shell script with:  
\- strict shell flags  
\- predictable execution behavior  
\- no swallowed errors

\### Acceptance criteria

\- transparent behavior  
\- safe by default

\### Prompt for coding agent

Create docker/entrypoint.sh for Transformer Surgery Lab. Use strict shell settings, keep it minimal, and execute the passed command cleanly without hiding failures.

\### Validation

Run a command that intentionally fails and ensure the failure is visible.

\---

\## Step 1.9 — \`docker/docker-compose.yml\`

\### Why this file matters

This is the developer ergonomics file. It should make the common workflow obvious.

\### What to implement

Define at least:  
\- \`dev\` service  
\- \`dashboard\` service

Optionally include:  
\- \`train\` service

Include:  
\- repo volume mount  
\- working directory  
\- Streamlit port exposure  
\- environment values if useful

\### Acceptance criteria

\- easy to open a shell  
\- easy to launch dashboard later  
\- readable to a solo developer

\### Prompt for coding agent

Write docker/docker-compose.yml for Transformer Surgery Lab with at least a dev service and a dashboard service. Mount the repository, set the working directory, expose the Streamlit port, and keep the file clear for a solo workflow on WSL + Docker.

\### Validation

Open the dev shell using compose and confirm you land in the correct directory.

\---

\## Step 1.10 — \`Makefile\`

\### Why this file matters

The Makefile turns the repo from “technically runnable” into “pleasant to use.”

\### What to implement

Add practical targets for:  
\- build  
\- build-cpu  
\- shell  
\- test  
\- lint  
\- format  
\- prepare-data  
\- smoke  
\- train  
\- benchmark  
\- dashboard

\### Acceptance criteria

\- command names are obvious  
\- recipes are short and transparent  
\- no magical hidden behavior

\### Prompt for coding agent

Write a clean Makefile for Transformer Surgery Lab with practical targets for Docker build, CPU build, shell, tests, lint, format, data preparation, smoke test, train, benchmark, and dashboard. Keep it ergonomic and easy to read.

\### Validation

Run at least the build and shell targets successfully.

\---

\## Step 1.11 — \`pyproject.toml\`

\### Why this file matters

This is where formatting and linting stop being inconsistent personal taste and become project defaults.

\### What to implement

Configure:  
\- black  
\- ruff  
\- pytest  
\- optional mypy

\### Acceptance criteria

\- simple and modern  
\- aligned with the repo’s small, readable code style

\### Prompt for coding agent

Create pyproject.toml for Transformer Surgery Lab with clean configuration for black, ruff, pytest, and optional mypy. Keep it minimal and aligned with a polished research-engineering repository.

\### Validation

Run formatting and lint commands against the repo scaffold.

\---

\## Phase 1 gate checklist

Do not leave Phase 1 until all of this is true:  
\- Dockerfile builds  
\- CPU Dockerfile builds  
\- compose shell works  
\- Make targets exist and are usable  
\- formatter/linter config is in place  
\- the repo already looks credible from the outside

\---

\## Phase 2 — Config, Utilities, and Data Pipeline

\### Goal

Build the stable runtime backbone that all later training and experiment logic depends on.

\### Phase gate

This phase is done only when:  
\- configs load predictably  
\- runtime helpers are in place  
\- dataset preparation works  
\- train and validation dataloaders produce correct batches

\---

\## Step 2.1 — \`src/tsl/\_\_init\_\_.py\`

\### Why this file matters

It formalizes the package boundary early and keeps imports sane.

\### What to implement

A minimal package init file. Do not overpopulate it.

\### Prompt for coding agent

Create src/tsl/\_\_init\_\_.py for Transformer Surgery Lab. Keep it minimal and package-oriented.

\### Validation

Confirm the package imports cleanly inside the container.

\---

\## Step 2.2 — \`src/tsl/constants.py\`

\### Why this file matters

Stable names are critical in an experiment repo. Without them, filenames, metric keys, and variant names drift.

\### What to implement

Define canonical constants for:  
\- artifact filenames  
\- directory names  
\- metric keys  
\- canonical variant names: layernorm, rmsnorm, sinusoidal, rope, relu, swiglu, mha, gqa

\### Acceptance criteria

\- no random string duplication elsewhere will be needed later  
\- names are consistent with the README and config design

\### Prompt for coding agent

Implement src/tsl/constants.py for Transformer Surgery Lab. Define stable constants for metric keys, artifact filenames, directory names, and canonical variant names like layernorm, rmsnorm, sinusoidal, rope, relu, swiglu, mha, and gqa.

\### Validation

Read the file and check whether it can actually serve as a single source of truth for naming.

\---

\## Step 2.3 — \`src/tsl/config.py\`

\### Why this file matters

This is the heart of the project. If config loading is weak, “config-driven” becomes a marketing phrase instead of a real design choice.

\### What to implement

Implement a YAML config loader that can:  
\- load a base config  
\- merge one or more override configs  
\- return a resolved dictionary or typed object  
\- validate required top-level sections lightly  
\- be easy to save back out as a resolved config

\### Design rule

Keep it simple enough to explain in an interview on a whiteboard.

\### Acceptance criteria

\- deterministic merge behavior  
\- clear error messages  
\- no overengineered config framework

\### Prompt for coding agent

Implement src/tsl/config.py for Transformer Surgery Lab. Build a clean YAML config loader that can merge a base config with override configs and return a resolved configuration object or dictionary. Keep it easy to explain, avoid overengineering, and include sensible validation for required sections.

\### Validation

Create a tiny config merge example manually and verify the output looks exactly as expected.

\---

\## Step 2.4 — \`src/tsl/utils/seed.py\`

\### Why this file matters

Controlled experiments mean seed discipline.

\### What to implement

A clear helper that sets seeds for:  
\- Python  
\- NumPy  
\- PyTorch

\### Acceptance criteria

\- deterministic setup path is obvious  
\- no hidden side effects

\### Prompt for coding agent

Implement src/tsl/utils/seed.py for Transformer Surgery Lab. Expose a clear function that sets deterministic random seeds across Python, NumPy, and PyTorch, with concise docs and practical defaults.

\### Validation

Run a tiny repeatability check.

\---

\## Step 2.5 — \`src/tsl/utils/device.py\`

\### Why this file matters

Device handling is where many small repos become brittle.

\### What to implement

Provide helpers for:  
\- selecting CPU vs CUDA  
\- exposing small runtime info  
\- avoiding surprise device behavior

\### Acceptance criteria

\- simple API  
\- easy to inspect at runtime

\### Prompt for coding agent

Implement src/tsl/utils/device.py for Transformer Surgery Lab. It should choose CPU vs CUDA cleanly, expose a small runtime info helper, and avoid surprising behavior.

\### Validation

Run it on both CPU and, if available, CUDA environments.

\---

\## Step 2.6 — \`src/tsl/utils/io.py\`

\### Why this file matters

Every artifact system becomes worse if file I/O is duplicated everywhere.

\### What to implement

Helpers for:  
\- creating directories  
\- writing JSON  
\- writing JSONL  
\- reading JSON and YAML  
\- handling artifact paths safely

\### Acceptance criteria

\- small and reliable  
\- enough to support tracking and config writing later

\### Prompt for coding agent

Implement src/tsl/utils/io.py for Transformer Surgery Lab. Include small, reliable helpers for creating directories, reading and writing JSON, JSONL, and YAML, and handling run artifact paths cleanly.

\### Validation

Write and read a small temporary artifact successfully.

\---

\## Step 2.7 — \`src/tsl/utils/logging\_utils.py\`

\### Why this file matters

Training logs should be readable and consistent, not improvised script prints.

\### What to implement

A lightweight logger setup for scripts and library code.

\### Acceptance criteria

\- readable console logging  
\- not overabstracted

\### Prompt for coding agent

Implement src/tsl/utils/logging\_utils.py for Transformer Surgery Lab. Create a simple, professional logger setup suitable for training scripts and dashboard-side artifact loading. Avoid noisy abstractions.

\### Validation

Use the logger from a small script and confirm formatting is sane.

\---

\## Step 2.8 — \`configs/default.yaml\`

\### Why this file matters

This file establishes the repository’s default assumptions and keeps later experiment configs small.

\### What to implement

Include sane defaults for:  
\- run metadata  
\- data settings  
\- model settings  
\- training settings  
\- benchmark settings  
\- artifact paths

\### Acceptance criteria

\- values are realistic for a small transformer lab  
\- later configs will mostly override, not duplicate, this file

\### Prompt for coding agent

Create configs/default.yaml for Transformer Surgery Lab. It should contain sane defaults for run metadata, data settings, model settings, training settings, benchmark settings, and output paths. Keep values realistic for a small experimental transformer project.

\### Validation

Read it and ask whether it would actually reduce duplication in the experiment configs.

\---

\## Step 2.9 — \`data/README.md\`

\### Why this file matters

Data layout confusion wastes time. This file should eliminate that.

\### What to implement

Explain:  
\- expected raw data location  
\- processed artifact location  
\- tokenizer artifact location  
\- what gets ignored from git  
\- what scripts populate these folders

\### Acceptance criteria

\- a new contributor could understand data layout without guessing

\### Prompt for coding agent

Write data/README.md for Transformer Surgery Lab. Explain the expected raw, processed, and tokenizer artifact layout, how datasets should be prepared, and what files are usually ignored from git.

\### Validation

Check whether it matches the actual planned directories from the tech spec.

\---

\## Step 2.10 — \`src/tsl/data/tokenizer.py\`

\### Why this file matters

Tokenizer handling needs to be stable without turning into a whole project of its own.

\### What to implement

Provide utilities to:  
\- load a fixed tokenizer artifact  
\- optionally train one once if configured  
\- tokenize text into IDs reproducibly  
\- save/load tokenizer paths cleanly

\### Design note

Keep the public interface simple. The goal is repeatable experiments, not tokenizer innovation.

\### Acceptance criteria

\- stable tokenizer workflow  
\- clear separation between tokenizer lifecycle and dataset preprocessing

\### Prompt for coding agent

Implement src/tsl/data/tokenizer.py for Transformer Surgery Lab. It should provide clean utilities to either train or load a tokenizer artifact for a small language modeling dataset, while keeping the interface simple, reproducible, and compatible with a fixed-tokenizer experiment policy.

\### Validation

Verify the same text always tokenizes the same way with the same saved artifact.

\---

\## Step 2.11 — \`src/tsl/data/preprocess.py\`

\### Why this file matters

This file creates the actual experiment substrate. If preprocessing is sloppy, the experiments are contaminated before the model even starts.

\### What to implement

Build preprocessing that can:  
\- read raw text or dataset records  
\- normalize text lightly if needed  
\- tokenize text  
\- create train/validation splits  
\- save processed token data  
\- record metadata about what was produced

\### Acceptance criteria

\- deterministic enough for repeated local runs  
\- output format is straightforward for the dataset class later

\### Prompt for coding agent

Implement src/tsl/data/preprocess.py for Transformer Surgery Lab. It should prepare a small text dataset for causal language modeling: read raw text, apply light normalization if needed, tokenize it, create train and validation splits, and save processed artifacts suitable for PyTorch training.

\### Validation

Run it on a tiny input sample and inspect the written outputs manually.

\---

\## Step 2.12 — \`src/tsl/data/dataset.py\`

\### Why this file matters

This is the boundary between stored token data and actual training batches.

\### What to implement

Create a causal LM dataset that returns:  
\- input token windows of fixed length  
\- shifted target token windows for next-token prediction

\### Acceptance criteria

\- shape behavior is simple and predictable  
\- no unnecessary abstraction

\### Prompt for coding agent

Implement src/tsl/data/dataset.py for Transformer Surgery Lab. Build a causal language modeling dataset that returns fixed-length token windows and shifted targets for next-token prediction. Keep it simple, efficient, and easy to test.

\### Validation

Print one sample and verify input-target alignment manually.

\---

\## Step 2.13 — \`src/tsl/data/datamodule.py\`

\### Why this file matters

This file keeps training scripts from knowing too much about file formats or batching logic.

\### What to implement

Provide functions or a small class that:  
\- loads processed artifacts  
\- constructs train and validation datasets  
\- creates dataloaders based on config

\### Acceptance criteria

\- train and val loaders are easy to obtain from one interface  
\- batch shapes are predictable

\### Prompt for coding agent

Implement src/tsl/data/datamodule.py for Transformer Surgery Lab. It should load tokenized data artifacts, construct train and validation datasets, and return PyTorch dataloaders based on the resolved config.

\### Validation

Iterate one batch from each loader and confirm tensor shapes.

\---

\## Step 2.14 — \`scripts/prepare\_data.py\`

\### Why this file matters

The repo needs one obvious way to prepare data. This script is that interface.

\### What to implement

The script should:  
\- load config  
\- call tokenizer/preprocess utilities  
\- write outputs to expected locations  
\- log useful progress

\### Acceptance criteria

\- one command prepares data end to end  
\- logs are readable and practical

\### Prompt for coding agent

Implement scripts/prepare\_data.py for Transformer Surgery Lab. It should load config, prepare the chosen dataset and tokenizer artifacts, write outputs to the expected directories, and log useful progress messages.

\### Validation

Run the script and confirm the expected processed files are created.

\---

\## Step 2.15 — \`configs/data/tinystories.yaml\`

\### Why this file matters

This config anchors the main experiment dataset and reduces ambiguity.

\### What to implement

Define:  
\- dataset name  
\- raw and processed paths  
\- tokenizer path  
\- sequence length  
\- split policy  
\- other data-specific knobs

\### Acceptance criteria

\- realistic and specific enough to be actually used immediately  
\- aligns with the preprocessing script

\### Prompt for coding agent

Create configs/data/tinystories.yaml for Transformer Surgery Lab with realistic paths and settings for a TinyStories-style small language modeling dataset. It should align with the preprocessing and dataloader pipeline.

\### Validation

Use it directly with the data preparation script.

\---

\## Phase 2 gate checklist

Do not leave Phase 2 until all of this is true:  
\- config loading and merging work  
\- seed and device helpers work  
\- artifact I/O helpers work  
\- data layout is documented  
\- tokenizer path is stable  
\- preprocessing runs  
\- train and validation dataloaders produce correct batches

\---