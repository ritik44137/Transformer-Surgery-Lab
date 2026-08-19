# Model and Training Implementation Plan
\## Purpose

This document covers the first model-complete stage of Transformer Surgery Lab:  
\- baseline transformer module implementation  
\- model assembly  
\- training loop and optimizer stack  
\- artifact writing and checkpointing  
\- baseline training entry points

The key objective here is to establish a \*\*fully working baseline system\*\* before adding any architectural swaps. This is one of the most important discipline points in the entire project. If the baseline is not solid first, every later comparison becomes harder to trust.

Every step below has one \*\*primary file\*\*. Small dependent edits are allowed only if required to keep the repository coherent.

\---

\## Phase 3 — Baseline Model Core

\### Goal

Build a small decoder-only transformer that can perform a correct forward pass for causal language modeling and is cleanly structured for later component swapping.

\### Phase gate

This phase is done only when:  
\- the baseline model builds from config  
\- logits have correct shape  
\- causal masking is correct  
\- one backward pass succeeds  
\- the code structure clearly exposes future swap points

\---

\## Step 3.1 — \`src/tsl/model/norms.py\`

\### Why this file matters

Normalization is one of the four core experiment axes, so the baseline implementation must already expose a clear abstraction boundary.

\### What to implement

Start with:  
\- a clean LayerNorm implementation path  
\- a lightweight variant-selection-friendly structure  
\- good docstrings and shape clarity

\### Design note

Even before RMSNorm exists, this file should already make it obvious where RMSNorm will go.

\### Acceptance criteria

\- easy to compare and explain  
\- not overabstracted  
\- safe for later extension

\### Prompt for coding agent

Implement src/tsl/model/norms.py for Transformer Surgery Lab. Start with a clean LayerNorm module and structure the file so RMSNorm can be added later without redesigning everything. Include concise docs, expected tensor shapes, and readable naming.

\### Validation

Run it on a toy tensor and confirm output shape and finite values.

\---

\## Step 3.2 — \`src/tsl/model/positional.py\`

\### Why this file matters

This file eventually becomes home to one of the most conceptually interesting swaps: sinusoidal vs RoPE.

\### What to implement

For now, implement:  
\- sinusoidal positional encoding  
\- clean separation between additive positional encoding logic and future rotary helpers

\### Design note

Do not entangle positional logic with the attention module yet.

\### Acceptance criteria

\- sinusoidal path is correct and simple  
\- later RoPE support can be added without rewriting the file from scratch

\### Prompt for coding agent

Implement src/tsl/model/positional.py for Transformer Surgery Lab. Start with sinusoidal positional encoding for a decoder-only transformer and structure the file so RoPE can be added later cleanly without mixing concerns.

\### Validation

Check output shape and verify encodings vary by position.

\---

\## Step 3.3 — \`src/tsl/model/feedforward.py\`

\### Why this file matters

Feed-forward blocks look simple, but they are one of the strongest architecture-swap surfaces in the project.

\### What to implement

Create the baseline feed-forward block using:  
\- linear up-projection  
\- ReLU activation  
\- linear down-projection  
\- dropout if configured

\### Design note

Structure it so SwiGLU can be added later without clutter.

\### Acceptance criteria

\- clear baseline MLP block  
\- hidden-dimension usage is explicit

\### Prompt for coding agent

Implement src/tsl/model/feedforward.py for Transformer Surgery Lab. Start with a clean ReLU-based transformer MLP block and structure the file so a SwiGLU variant can be added later without messy refactoring.

\### Validation

Run one toy forward pass and verify shape preservation.

\---

\## Step 3.4 — \`src/tsl/model/attention.py\`

\### Why this file matters

Attention is where readability matters most. This is one of the first files a technically strong reviewer may inspect.

\### What to implement

Implement standard causal multi-head self-attention with:  
\- query, key, value projections  
\- head reshaping  
\- causal mask  
\- softmax attention  
\- output projection  
\- optional dropout

\### Design note

Keep tensor-shape flow very readable. GQA will be added later, so do not hardcode assumptions too rigidly.

\### Acceptance criteria

\- shape logic is understandable  
\- causal mask is correct  
\- code is interview-friendly

\### Prompt for coding agent

Implement src/tsl/model/attention.py for Transformer Surgery Lab. Start with standard causal multi-head self-attention for a decoder-only transformer. Make the code readable, add shape assertions where useful, and structure it so grouped-query attention can be added later.

\### Validation

Run a toy forward pass and verify output shape matches input hidden shape.

\---

\## Step 3.5 — \`src/tsl/model/block.py\`

\### Why this file matters

This file makes the model feel like a transformer rather than a pile of parts.

\### What to implement

Build a pre-norm transformer block with:  
\- normalization before attention  
\- residual connection around attention  
\- normalization before feed-forward  
\- residual connection around feed-forward

\### Acceptance criteria

\- structure is explicit and easy to follow  
\- later variant injection is straightforward

\### Prompt for coding agent

Implement src/tsl/model/block.py for Transformer Surgery Lab. Build a pre-norm decoder transformer block using injected normalization, attention, and feed-forward modules. Keep the control flow explicit and easy to explain.

\### Validation

Pass a toy tensor through the block and confirm shape preservation.

\---

\## Step 3.6 — \`src/tsl/model/embeddings.py\`

\### Why this file matters

It keeps token embeddings and positional plumbing separate from the full model assembly.

\### What to implement

Implement:  
\- token embeddings  
\- baseline positional addition path where needed  
\- embedding dropout if configured

\### Acceptance criteria

\- clean interface for transformer model assembly  
\- no attention-specific logic inside this file

\### Prompt for coding agent

Implement src/tsl/model/embeddings.py for Transformer Surgery Lab. It should handle token embeddings cleanly and support the baseline positional-encoding path without entangling model assembly logic.

\### Validation

Verify a batch of token IDs becomes the expected hidden-state shape.

\---

\## Step 3.7 — \`src/tsl/model/transformer.py\`

\### Why this file matters

This is the baseline model itself. It should feel small, clean, and confident.

\### What to implement

Assemble:  
\- embeddings  
\- a repeated stack of transformer blocks  
\- final normalization  
\- LM head  
\- forward method that returns logits

\### Design note

Keep it close in spirit to nanoGPT: compact enough to read quickly, but not monolithic.

\### Acceptance criteria

\- full model forward pass works  
\- logits shape is correct for causal LM  
\- code is substantially readable in one sitting

\### Prompt for coding agent

Implement src/tsl/model/transformer.py for Transformer Surgery Lab. Assemble a small decoder-only language model using embeddings, repeated transformer blocks, final normalization, and an LM head. Keep it minimal and nanoGPT-like in spirit, but modular enough for architecture swaps.

\### Validation

Run a forward pass on toy token IDs and confirm logits shape is \`\[batch, seq, vocab\]\`.

\---

\## Step 3.8 — \`src/tsl/model/factory.py\`

\### Why this file matters

This file is where “config-driven” becomes real for the model.

\### What to implement

Build a model factory that:  
\- reads resolved model config  
\- selects the appropriate module classes or paths  
\- assembles the full transformer  
\- stays explicit enough for debugging

\### Acceptance criteria

\- model creation from config works cleanly  
\- no hidden registry magic that makes the project harder to explain

\### Prompt for coding agent

Implement src/tsl/model/factory.py for Transformer Surgery Lab. It should read the resolved model config and construct the correct model components using clean variant selection logic. Keep it explicit, readable, and easy to debug.

\### Validation

Build the baseline model solely from config and confirm it matches expectations.

\---

\## Step 3.9 — \`configs/model/base\_small.yaml\`

\### Why this file matters

The baseline dimensions should be small enough to iterate on but large enough to produce meaningful experiment output.

\### What to implement

Define:  
\- vocab size placeholder or resolved expectation  
\- hidden size  
\- number of layers  
\- number of heads  
\- feed-forward dimension  
\- max sequence length  
\- dropout  
\- baseline variant choices

\### Acceptance criteria

\- realistic for repeated local or notebook-GPU experiments  
\- consistent with the model factory assumptions

\### Prompt for coding agent

Create configs/model/base\_small.yaml for Transformer Surgery Lab with sane small-model defaults appropriate for repeated local or Colab/Kaggle experiments. Include baseline variant selections that match LayerNorm, sinusoidal encoding, ReLU MLP, and standard MHA.

\### Validation

Load the config and build the model from it without manual patching.

\---

\## Phase 3 gate checklist

Do not leave Phase 3 until all of this is true:  
\- baseline model builds from config  
\- toy forward pass works  
\- logits shape is correct  
\- one backward pass works  
\- the model code is still clean enough to explain file by file

\---

\## Phase 4 — Training, Tracking, and Baseline Run Pipeline

\### Goal

Turn the baseline model into a trainable, checkpointed, artifact-producing experiment system.

\### Phase gate

This phase is done only when:  
\- a baseline run can train end to end  
\- train and validation metrics are written  
\- checkpoints save and load  
\- summary artifacts exist for future comparison tools

\---

\## Step 4.1 — \`src/tsl/train/losses.py\`

\### Why this file matters

Loss logic should be simple and explicit rather than hidden in the trainer.

\### What to implement

Create the causal LM loss helper for:  
\- logits  
n- next-token targets  
\- optional shape checks

\### Acceptance criteria

\- easy to test independently  
\- no ambiguity about token shifting assumptions

\### Prompt for coding agent

Implement src/tsl/train/losses.py for Transformer Surgery Lab. Define the causal language modeling loss cleanly for logits and next-token targets, with concise documentation and straightforward shape checks where useful.

\### Validation

Use toy logits and targets and confirm the function returns a scalar loss.

\---

\## Step 4.2 — \`src/tsl/train/optimizer.py\`

\### Why this file matters

This file separates optimization policy from trainer orchestration.

\### What to implement

Create an optimizer factory centered on AdamW. If parameter grouping is added, keep it minimal and understandable.

\### Acceptance criteria

\- simple, explicit construction path  
\- easy to inspect when debugging

\### Prompt for coding agent

Implement src/tsl/train/optimizer.py for Transformer Surgery Lab. Build a clean optimizer factory centered on AdamW with only lightweight parameter grouping support if it is genuinely useful.

\### Validation

Build an optimizer on the baseline model and inspect the parameter groups.

\---

\## Step 4.3 — \`src/tsl/train/scheduler.py\`

\### Why this file matters

A clear scheduler keeps training behavior reproducible and documented.

\### What to implement

Implement:  
\- warmup support  
\- optional cosine decay  
\- a simple interface the trainer can call

\### Acceptance criteria

\- not framework-heavy  
\- behavior is easy to reason about from config

\### Prompt for coding agent

Implement src/tsl/train/scheduler.py for Transformer Surgery Lab. Add a simple learning-rate scheduler path with warmup and optional cosine decay while keeping the interface straightforward.

\### Validation

Plot or print a short schedule sequence and confirm it matches expectations.

\---

\## Step 4.4 — \`src/tsl/tracking/schema.py\`

\### Why this file matters

Experiment artifacts only become reusable if their structure is consistent.

\### What to implement

Define the canonical shape of:  
\- run metadata  
\- train metrics rows  
\- eval metrics rows  
\- benchmark outputs  
\- compact summary outputs

\### Acceptance criteria

\- schema is stable and dashboard-friendly  
\- enough structure exists to keep files consistent across runs

\### Prompt for coding agent

Implement src/tsl/tracking/schema.py for Transformer Surgery Lab. Define stable artifact schemas or helper structures for run metadata, train metrics, eval metrics, benchmark outputs, and summary files.

\### Validation

Read the definitions and verify they can support every planned downstream consumer.

\---

\## Step 4.5 — \`src/tsl/tracking/writer.py\`

\### Why this file matters

This file turns a training run into a durable experiment.

\### What to implement

Create logic to:  
\- create run directories  
\- save resolved config  
\- save metadata  
\- append train metrics  
\- append eval metrics  
\- save summary outputs

\### Acceptance criteria

\- one run produces a self-contained artifact directory  
\- path naming is stable and readable

\### Prompt for coding agent

Implement src/tsl/tracking/writer.py for Transformer Surgery Lab. It should create run directories, save resolved configs, write metadata and metrics files, and support JSON and JSONL artifact logging cleanly.

\### Validation

Write a fake run directory and inspect the outputs manually.

\---

\## Step 4.6 — \`src/tsl/train/checkpointing.py\`

\### Why this file matters

Checkpoints are essential for reproducibility and evaluation.

\### What to implement

Provide:  
\- save latest checkpoint  
\- save best checkpoint  
\- load checkpoint  
\- store enough information for resume/evaluation

\### Acceptance criteria

\- naming is consistent  
\- restore path is obvious and testable

\### Prompt for coding agent

Implement src/tsl/train/checkpointing.py for Transformer Surgery Lab. Add clear save and load utilities for latest and best checkpoints, keeping the file readable and artifact naming consistent.

\### Validation

Save a small model state and load it back successfully.

\---

\## Step 4.7 — \`src/tsl/train/loop.py\`

\### Why this file matters

Separating train-step and eval-step logic from the high-level trainer makes the code easier to test and reason about.

\### What to implement

Create explicit functions for:  
\- train step  
\- eval step  
\- metric extraction per step

\### Acceptance criteria

\- backward pass logic is clear  
\- no unnecessary coupling to script-level concerns

\### Prompt for coding agent

Implement src/tsl/train/loop.py for Transformer Surgery Lab. Create testable train-step and eval-step functions that handle forward pass, loss computation, backward pass, optimizer stepping, scheduler stepping, and metric extraction.

\### Validation

Run one train step on a toy batch.

\---

\## Step 4.8 — \`src/tsl/train/trainer.py\`

\### Why this file matters

This file coordinates everything but should still remain smaller and clearer than a big framework trainer.

\### What to implement

Build a trainer that manages:  
\- dataloaders  
\- model  
\- optimizer  
\- scheduler  
\- logging  
\- periodic evaluation  
\- checkpointing  
\- summary writing

\### Acceptance criteria

\- explicit control flow  
\- simple enough to inspect top to bottom

\### Prompt for coding agent

Implement src/tsl/train/trainer.py for Transformer Surgery Lab. Build a clean trainer that coordinates dataloaders, model, optimizer, scheduler, tracking, periodic evaluation, and checkpointing. Keep it explicit rather than framework-heavy.

\### Validation

Run a miniature training session and confirm logging, metrics, and checkpoint hooks fire.

\---

\## Step 4.9 — \`scripts/train.py\`

\### Why this file matters

This is the main entry point people will use to run experiments.

\### What to implement

The script should:  
\- load config  
\- set seed and device  
\- prepare dataloaders  
\- build the model  
\- build trainer stack  
\- launch training  
\- log useful runtime info

\### Acceptance criteria

\- one command starts a baseline experiment  
\- startup logs are clean and informative

\### Prompt for coding agent

Implement scripts/train.py for Transformer Surgery Lab. It should load config, set up runtime state, build dataloaders and model, construct the trainer, and launch training with clean logging.

\### Validation

Run a short training job end to end.

\---

\## Step 4.10 — \`configs/train/baseline\_train.yaml\`

\### Why this file matters

The baseline training config should make the first real run easy and disciplined.

\### What to implement

Define:  
\- training step budget  
\- batch size  
\- eval interval  
\- log interval  
\- learning rate  
\- warmup  
\- weight decay  
\- gradient clipping  
\- precision setting if used

\### Acceptance criteria

\- practical for a real first run  
\- not so long that iteration becomes painful

\### Prompt for coding agent

Create configs/train/baseline\_train.yaml for Transformer Surgery Lab with realistic hyperparameters for a small decoder-only model trained on a compact dataset. It should be suitable for the first controlled baseline run.

\### Validation

Use it directly with \`scripts/train.py\`.

\---

\## Step 4.11 — \`configs/experiments/baseline\_layernorm\_sinusoidal\_relu\_mha.yaml\`

\### Why this file matters

This config becomes the anchor for all later comparisons.

\### What to implement

Compose the baseline experiment from:  
\- default config  
\- data config  
\- model config  
\- train config  
\- baseline variant choices

\### Acceptance criteria

\- human-readable run name  
\- minimal duplication  
\- directly runnable

\### Prompt for coding agent

Create configs/experiments/baseline\_layernorm\_sinusoidal\_relu\_mha.yaml for Transformer Surgery Lab as the baseline experiment definition that composes the default, dataset, model, and training configs into one directly runnable baseline setup.

\### Validation

Launch the baseline experiment from this file and confirm the resolved config matches the intended baseline.

\---

\## Phase 4 gate checklist

Do not leave Phase 4 until all of this is true:  
\- a baseline experiment can be launched from one experiment config  
\- metrics are written during training  
\- checkpoints save correctly  
\- run artifacts are readable by humans  
\- the run summary can support later comparison tooling

\---