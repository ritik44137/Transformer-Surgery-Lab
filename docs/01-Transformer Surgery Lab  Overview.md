# Transformer Surgery Lab : Main
\## Purpose

This is the rewritten execution plan for \*\*Transformer Surgery Lab\*\*. It replaces the earlier single long implementation file with a cleaner, stricter, and more realistic build system split across five documents.

The project goal is unchanged: build a \*\*modular, config-driven PyTorch transformer experimentation lab\*\* that proves deep model-architecture understanding through controlled swaps, measurable results, and polished presentation.

What is improved in this rewrite is the execution discipline. The plan now has a tighter scope, stronger experiment controls, cleaner configuration boundaries, more realistic sequencing, and a much higher bar for what counts as “finished.”

\---

\## What the project should feel like when done

A reviewer should be able to look at the repo and conclude all of the following within a few minutes:

\- the author understands transformer internals beyond API usage  
\- the experiments are controlled rather than random  
\- the codebase is clean enough that adding another paper-inspired variant would be easy  
\- the metrics are reproducible and not hand-wavy  
\- the dashboard makes the tradeoffs obvious  
\- the README and lessons learned show real engineering judgment

If the final repo does not communicate those signals, it is not done.

\---

\## The refined scope

\### The core model scope

This project should implement a \*\*small decoder-only transformer\*\* for causal language modeling with the following swappable axes:

\- \*\*Normalization:\*\* LayerNorm vs RMSNorm  
\- \*\*Positional encoding:\*\* Sinusoidal vs RoPE  
\- \*\*Feed-forward block:\*\* ReLU MLP vs SwiGLU MLP  
\- \*\*Attention:\*\* standard multi-head self-attention vs grouped-query attention

\### The experimentation scope

The experiment plan should be built in layers:

1\. \*\*Baseline run\*\*  
  - LayerNorm  
  - sinusoidal  
  - ReLU  
  - MHA

2\. \*\*Single-axis swaps\*\*  
  - change exactly one dimension at a time  
  - keep everything else fixed

3\. \*\*Best-combination runs\*\*  
  - only after single-axis runs are stable and interpreted

\### The presentation scope

The project must ship with:  
\- a polished README  
\- reproducible configs  
\- saved run artifacts  
\- comparison outputs  
\- a Streamlit dashboard  
\- lesson notes for each swap family  
\- at least one clean screenshot for the repo page

\---

\## What this rewrite changes from the earlier plan

\### 1. Scope is tighter where it should be tighter

The project should \*\*not\*\* lose time on peripheral complexity early. That means:  
\- no distributed training work  
\- no giant tokenizer-training rabbit hole in the first pass  
\- no premature KV-cache system unless benchmark needs force it later  
\- no paper-reproduction sprawl before the baseline experiment matrix is complete

\### 2. Fair comparisons are now explicitly enforced

This was not strong enough before. The new plan requires:  
\- a fixed dataset family for the main comparison runs  
\- a fixed tokenizer for the main comparison runs  
\- fixed seed policy  
\- fixed training budget  
\- fixed batch/sequence settings unless a benchmark specifically varies them  
\- explicit handling of parameter-count differences, especially for \*\*SwiGLU\*\* and \*\*GQA\*\*

\### 3. Tokenizer scope is more realistic

For the first polished version, prioritize a \*\*stable tokenizer path\*\* over inventing a tokenizer research side quest. The code should support tokenizer abstraction, but the implementation plan assumes one of these two sane approaches:  
\- use a fixed HuggingFace tokenizer artifact for the main TinyStories runs, or  
\- train one tokenizer once and freeze it across all comparisons

The key rule is that tokenizer variation must not contaminate architecture comparison.

\### 4. Testing is not only an end-stage activity

The earlier sequencing leaned too much toward implementing everything and testing later. The new plan still keeps the repository pragmatic, but it adds earlier validation thinking and a stronger final gate.

\### 5. The dashboard is treated as a product layer, not an afterthought

The dashboard is now planned as a deliberate recruiter-facing surface. It is not just a utility for you. It is part of the repo’s hiring value.

\---

\## Non-negotiable engineering rules

\### Rule 1 — Config-driven means real config-driven

A swap should happen by editing YAML, not by opening Python files and changing branches manually.

\### Rule 2 — One-file steps

Every implementation step in the phase files has exactly one primary file. Small dependent edits are allowed only when necessary to keep the code runnable.

\### Rule 3 — Every phase must close with a gate

Each phase ends only when its gate is satisfied. No skipping ahead because “the rest is probably fine.”

\### Rule 4 — The public surfaces must look polished

The public surfaces are:  
\- README  
\- dashboard  
\- screenshots  
\- experiment summaries  
\- lesson writeups

These are not optional polish extras. They are core deliverables.

\### Rule 5 — Small, legible code wins

The project should feel more like \*\*nanoGPT with strong experiment infrastructure\*\* than like an overbuilt framework.

\---

\## Dataset and experiment policy

\### Main dataset

Use \*\*TinyStories\*\* as the main experiment dataset.

Why:  
\- it is language-model friendly  
\- small enough to run repeatedly  
\- big enough to show meaningful trends  
\- recognizable enough for others to understand

\### Smoke-test dataset

Use a much smaller dataset path or toy text path only for smoke tests and pipeline validation.

\### Comparison policy

For each single-axis comparison, keep fixed:  
\- dataset  
\- tokenizer  
\- sequence length  
\- batch size  
\- optimizer  
\- scheduler  
\- seed or seed set  
\- training step budget  
\- logging cadence

\### Fairness note for SwiGLU

SwiGLU can silently change parameter count if implemented naively. The implementation plan must either:  
\- keep \`d\_ff\` fixed and report parameter differences clearly, or  
\- choose a scaled hidden dimension so the total parameter budget remains approximately comparable

The second option is better if it is implemented cleanly and documented clearly.

\### Fairness note for GQA

GQA must preserve the total query-head count while reducing key/value head count. Benchmark and summary outputs must make that explicit.

\---

\## The final repository should have three layers of value

\### Layer 1 — Core engineering value

A reader sees a clean PyTorch codebase with swappable transformer internals.

\### Layer 2 — Experimental value

A reader sees disciplined experiments with clear metrics and comparable runs.

\### Layer 3 — Communication value

A reader sees concise explanations, a dashboard, and polished narrative outputs that explain what changed and why it matters.

The best version of this project succeeds in all three layers.

\---

\## Build sequence overview

The implementation is split into four detailed phase files plus this master plan.

\### File 1  
\`TSL\_IMPLEMENTATION\_00\_MASTER\_PLAN.md\`

What it contains:  
\- refined scope  
\- engineering rules  
\- global build order  
\- coding-agent operating protocol  
\- final release standard

\### File 2  
\`TSL\_IMPLEMENTATION\_01\_FOUNDATION\_AND\_DATA.md\`

What it contains:  
\- environment scaffolding  
\- Docker and dependency setup  
\- config system  
\- utilities  
\- data pipeline

\### File 3  
\`TSL\_IMPLEMENTATION\_02\_MODEL\_AND\_TRAINING.md\`

What it contains:  
\- baseline model assembly  
\- tracking and artifact contract  
\- training loop  
\- baseline training entry points

\### File 4  
\`TSL\_IMPLEMENTATION\_03\_SWAPS\_EVAL\_DASHBOARD.md\`

What it contains:  
\- RMSNorm, RoPE, SwiGLU, and GQA integration  
\- experiment config matrix  
\- evaluation and benchmark pipeline  
\- dashboard implementation

\### File 5  
\`TSL\_IMPLEMENTATION\_04\_TESTS\_DOCS\_PUBLISH.md\`

What it contains:  
\- tests  
\- lessons learned docs  
\- final README polish  
\- screenshot and publication checklist

\---

\## Global build order

The exact order is:

1\. foundation and environment  
2\. core config and runtime utilities  
3\. data preparation  
4\. baseline transformer modules  
5\. training and artifact writing  
6\. baseline experiment config and training  
7\. architectural swaps  
8\. evaluation and benchmark tooling  
9\. dashboard  
10\. tests, docs, and publication polish

This order is intentional.

If you move the dashboard earlier, it will be fake.  
If you move docs too late, the repo will feel unfinished.  
If you move swaps earlier than baseline training, you will debug four systems at once.

\---

\## Phase gates

\### Gate A — Environment gate

Passes only when:  
\- Docker images build  
\- the dev shell opens  
\- dependency installation works  
\- a minimal command can run inside the container

\### Gate B — Data gate

Passes only when:  
\- dataset preparation runs cleanly  
\- tokenizer path is stable  
\- train and validation dataloaders yield correctly shaped batches

\### Gate C — Baseline model gate

Passes only when:  
\- the model forward pass works  
\- the causal mask is correct  
\- one backward pass succeeds  
\- artifact directories can be created

\### Gate D — Baseline training gate

Passes only when:  
\- a real baseline run trains  
\- metrics are written to disk  
\- checkpoints save correctly  
\- summary artifacts can be consumed downstream

\### Gate E — Swap matrix gate

Passes only when:  
\- all four swap axes are config-selectable  
\- baseline and single-axis runs execute under the same framework  
\- outputs remain schema-compatible

\### Gate F — Evaluation and dashboard gate

Passes only when:  
\- comparison metrics are readable from artifacts  
\- benchmark outputs are generated reliably  
\- the dashboard renders real data clearly

\### Gate G — Publication gate

Passes only when:  
\- tests pass  
\- docs are polished  
\- screenshots exist  
\- README is presentation-ready  
\- the repo can be cloned and understood without live explanation

\---

\## Coding-agent operating protocol

For every single-file step in the detailed phase files, use the same loop.

\### Step loop

1\. give the agent exactly one step prompt  
2\. review the file manually  
3\. run the smallest relevant validation  
4\. fix anything immediately  
5\. commit before moving on

\### Standard prompt prefix

Use this before each file-specific prompt:

\`\`\`text  
You are implementing one file for Transformer Surgery Lab, a polished config-driven PyTorch project for transformer architecture experiments. Follow the requested scope exactly. Keep the code minimal, readable, and production-quality. Do not add unrelated abstractions. Prefer explicit names, clear docstrings, and interview-friendly organization. Assume development happens in WSL Ubuntu 22.04 with Docker. The file must be complete enough to integrate with the current codebase state.  
\`\`\`

\### Review checklist for every generated file

Check these every time:  
\- does the file solve the requested problem directly?  
\- is it smaller and clearer than it needs to be, not larger?  
\- are names consistent with the rest of the repo?  
\- does it introduce hidden assumptions?  
\- is it easy to explain in an interview?  
\- does it make later steps easier rather than harder?

\---

\## Release standard

The project is only “finished” when all of the following are true:

\- the codebase is readable without private context  
\- the baseline and swap experiments are reproducible  
\- the metrics are comparison-ready  
\- the dashboard is genuinely useful  
\- the docs look polished enough for a pinned GitHub repo  
\- the experiment writeups sound like an engineer who measured and reasoned, not guessed

That is the bar the remaining files are designed to hit.

\---