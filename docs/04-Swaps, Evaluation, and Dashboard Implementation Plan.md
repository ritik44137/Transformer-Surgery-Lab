# Swaps, Evaluation, and Dashboard Implementation Plan
\## Purpose

This document covers the differentiating stage of Transformer Surgery Lab:  
\- adding the four core architecture swaps  
\- defining the controlled experiment matrix  
\- building evaluation and benchmark tooling  
\- implementing the recruiter-facing dashboard

This is where the project stops being “a small language model repo” and becomes a \*\*real transformer experimentation lab\*\*.

The most important rule for this stage is simple: \*\*do not add new dimensions of freedom faster than you add measurement\*\*. Every swap should be observable through the same artifact contract and comparison flow.

Every step below has one \*\*primary file\*\*. Small dependent edits are allowed only when necessary to keep the repository coherent.

\---

\## Phase 5 — Architectural Swaps and Experiment Matrix

\### Goal

Add all four planned architecture swap axes in a disciplined way so they remain config-selectable, comparable, and easy to explain.

\### Phase gate

This phase is done only when:  
\- all four swap axes exist in code  
\- the model factory can select them via config  
\- baseline and single-axis runs share the same training pipeline  
\- outputs remain artifact-compatible across variants

\---

\## Swap-order strategy

Do \*\*not\*\* add all swap dimensions at once. The order should be:  
1\. RMSNorm  
2\. RoPE  
3\. SwiGLU  
4\. GQA

Why this order:  
\- RMSNorm is conceptually simple and low-risk  
\- RoPE is conceptually important and benefits from a stable baseline first  
\- SwiGLU introduces parameterization decisions that are easier to reason about once baseline metrics exist  
\- GQA is the most benchmark-sensitive and easiest to misimplement if the rest of the system is still moving

\---

\## Step 5.1 — \`src/tsl/model/norms.py\` update

\### Why this file matters

This is the first true architecture-swap extension to the baseline. It should demonstrate how future variant additions are supposed to look: small, readable, and controlled.

\### What to implement

Extend the file to include:  
\- RMSNorm implementation  
\- a clear common interface story with LayerNorm  
\- any minimal helper logic needed for variant selection

\### Design note

RMSNorm should not feel bolted on. It should sit beside LayerNorm in a way that invites direct comparison.

\### Acceptance criteria

\- LayerNorm path still works unchanged  
\- RMSNorm is compact and mathematically clear  
\- output shapes match baseline expectations

\### Prompt for coding agent

Update src/tsl/model/norms.py in Transformer Surgery Lab to add a clean RMSNorm implementation alongside LayerNorm. Preserve the existing abstraction, keep the math clear, and make the two variants easy to compare and explain.

\### Validation

Run a toy forward pass through both norms and verify shape compatibility and finite outputs.

\---

\## Step 5.2 — \`configs/model/norm\_rmsnorm.yaml\`

\### Why this file matters

The swap is only real if it is selectable by config.

\### What to implement

Create the minimal config fragment required to select RMSNorm.

\### Acceptance criteria

\- small and readable  
\- no duplicated unrelated config values

\### Prompt for coding agent

Create configs/model/norm\_rmsnorm.yaml for Transformer Surgery Lab so the model factory can select RMSNorm entirely through config.

\### Validation

Merge it into the baseline config stack and confirm the resolved config selects RMSNorm.

\---

\## Step 5.3 — \`configs/experiments/rmsnorm\_sinusoidal\_relu\_mha.yaml\`

\### Why this file matters

This is the first single-axis comparison run and should be completely controlled.

\### What to implement

Create a config that changes only normalization relative to the baseline experiment.

\### Acceptance criteria

\- run name is explicit  
\- only the norm dimension changes  
\- it is directly runnable

\### Prompt for coding agent

Create configs/experiments/rmsnorm\_sinusoidal\_relu\_mha.yaml for Transformer Surgery Lab as a strict single-axis swap from the baseline, changing only normalization from LayerNorm to RMSNorm.

\### Validation

Diff it conceptually against the baseline experiment and confirm only one intended axis changes.

\---

\## Step 5.4 — \`src/tsl/model/positional.py\` update

\### Why this file matters

RoPE is one of the strongest conceptual signals in the entire project. This file needs to be especially clean.

\### What to implement

Add:  
\- RoPE helper functions  
\- rotary application to query and key representations  
\- implementation notes clear enough to support interview explanation

\### Design note

This file should be written so you can later explain RoPE using one small vector example. Good naming matters here.

\### Acceptance criteria

\- sinusoidal path still works unchanged  
\- RoPE helper logic is compact and understandable  
\- the code makes relative-position reasoning visible rather than obscure

\### Prompt for coding agent

Update src/tsl/model/positional.py in Transformer Surgery Lab to add rotary position embeddings cleanly. Include helper functions with good naming and comments so the rotation logic is explainable in an interview. Preserve the existing sinusoidal path.

\### Validation

Run a small shape and sanity check on the RoPE transform functions.

\---

\## Step 5.5 — \`configs/model/pos\_rope.yaml\`

\### Why this file matters

This is the config toggle for one of the project’s most important swap dimensions.

\### Prompt for coding agent

Create configs/model/pos\_rope.yaml for Transformer Surgery Lab so RoPE can be selected entirely through config.

\### Validation

Merge it into the baseline stack and verify the resolved config switches only the positional encoding path.

\---

\## Step 5.6 — \`configs/experiments/layernorm\_rope\_relu\_mha.yaml\`

\### Why this file matters

This creates the second controlled single-axis comparison.

\### What to implement

Create the experiment config that changes only positional encoding from sinusoidal to RoPE.

\### Acceptance criteria

\- one-axis change only  
\- run name is explicit and consistent

\### Prompt for coding agent

Create configs/experiments/layernorm\_rope\_relu\_mha.yaml for Transformer Surgery Lab as a strict single-axis swap from the baseline, changing only positional encoding from sinusoidal to RoPE.

\### Validation

Check that no other model or training settings differ from the baseline.

\---

\## Step 5.7 — \`src/tsl/model/feedforward.py\` update

\### Why this file matters

SwiGLU is the swap most likely to create accidental unfairness through hidden-size choices.

\### What to implement

Add:  
\- SwiGLU block implementation  
\- explicit hidden-dimension policy  
\- concise documentation about parameter-count implications

\### Design note

Be explicit about whether the project keeps \`d\_ff\` fixed or adjusts it for approximate parameter comparability. The file should not leave this ambiguous.

\### Acceptance criteria

\- baseline ReLU path still works unchanged  
\- SwiGLU logic is clean and shape-safe  
\- parameterization policy is documented in code comments or docstrings

\### Prompt for coding agent

Update src/tsl/model/feedforward.py in Transformer Surgery Lab to add a SwiGLU variant alongside the baseline ReLU MLP. Make the hidden-dimension and parameterization policy explicit so the comparison is fair and explainable.

\### Validation

Instantiate both feed-forward variants and compare output shapes and parameter counts.

\---

\## Step 5.8 — \`configs/model/ff\_swiglu.yaml\`

\### Why this file matters

It makes the feed-forward swap real and repeatable.

\### Prompt for coding agent

Create configs/model/ff\_swiglu.yaml for Transformer Surgery Lab so the model factory can select the SwiGLU feed-forward path through config.

\### Validation

Load it into a resolved model config and confirm it selects the intended variant.

\---

\## Step 5.9 — \`configs/experiments/layernorm\_sinusoidal\_swiglu\_mha.yaml\`

\### Why this file matters

This creates the third single-axis comparison run.

\### What to implement

Create the experiment config that changes only the feed-forward block from ReLU to SwiGLU.

\### Acceptance criteria

\- one-axis change only  
\- explicit naming  
\- directly runnable

\### Prompt for coding agent

Create configs/experiments/layernorm\_sinusoidal\_swiglu\_mha.yaml for Transformer Surgery Lab as a strict single-axis swap from the baseline, changing only the feed-forward block from ReLU to SwiGLU.

\### Validation

Compare it against the baseline experiment and verify only the intended change exists.

\---

\## Step 5.10 — \`src/tsl/model/attention.py\` update

\### Why this file matters

This is likely the trickiest implementation file after RoPE. It affects both correctness and benchmark meaning.

\### What to implement

Add grouped-query attention while preserving the existing MHA path.

The update should make explicit:  
\- how query-head count relates to key/value-head count  
\- how key/value heads are shared across query heads  
\- how causal masking remains unchanged

\### Design note

Readability matters more than clever optimization here. A reviewer should be able to understand the grouping behavior without reverse engineering dense tensor tricks.

\### Acceptance criteria

\- MHA path still works unchanged  
\- GQA path is clear and shape-safe  
\- grouping assumptions are enforced explicitly

\### Prompt for coding agent

Update src/tsl/model/attention.py in Transformer Surgery Lab to add grouped-query attention alongside standard multi-head attention. Keep the code readable, enforce the relationship between query heads and KV heads clearly, and preserve causal masking behavior.

\### Validation

Run toy inputs through both MHA and GQA variants and confirm shape correctness.

\---

\## Step 5.11 — \`configs/model/attn\_gqa.yaml\`

\### Why this file matters

This config fragment turns the attention swap into a first-class experiment dimension.

\### Prompt for coding agent

Create configs/model/attn\_gqa.yaml for Transformer Surgery Lab so grouped-query attention can be selected from config, including any required KV-head setting.

\### Validation

Load the resolved config and confirm both total heads and KV heads are explicit.

\---

\## Step 5.12 — \`configs/experiments/layernorm\_sinusoidal\_relu\_gqa.yaml\`

\### Why this file matters

This creates the fourth single-axis comparison run.

\### What to implement

Create the experiment config that changes only attention from standard MHA to GQA.

\### Acceptance criteria

\- one-axis change only  
\- naming clearly communicates the variant

\### Prompt for coding agent

Create configs/experiments/layernorm\_sinusoidal\_relu\_gqa.yaml for Transformer Surgery Lab as a strict single-axis swap from the baseline, changing only the attention type from MHA to GQA.

\### Validation

Check it against baseline and confirm only the intended attention axis changes.

\---

\## Step 5.13 — \`configs/experiments/full\_grid.yaml\`

\### Why this file matters

This file defines the official experiment set. Without it, experiment scope tends to drift.

\### What to implement

Create a config or experiment manifest that enumerates:  
\- baseline run  
\- the four single-axis swap runs  
\- optional combined runs clearly separated as second-priority or phase-two runs

\### Acceptance criteria

\- the primary comparison grid is obvious  
\- optional combined experiments do not dilute the main story

\### Prompt for coding agent

Create configs/experiments/full\_grid.yaml for Transformer Surgery Lab that clearly enumerates the official experiment set: the baseline, the four single-axis swaps, and an optional second tier of combined variants. Keep the primary comparison story obvious.

\### Validation

Review it and confirm the experiment story would still be clear to an outsider.

\---

\## Phase 5 gate checklist

Do not leave Phase 5 until all of this is true:  
\- RMSNorm, RoPE, SwiGLU, and GQA are implemented  
\- each swap is config-selectable  
\- baseline and single-axis experiment files exist  
\- experiment naming is stable and readable  
\- artifact compatibility is preserved across runs

\---

\## Phase 6 — Evaluation and Benchmarking

\### Goal

Add the measurement layer that makes the swaps meaningful rather than anecdotal.

\### Phase gate

This phase is done only when:  
\- run artifacts can be read back consistently  
\- comparison metrics exist for quality and efficiency  
\- benchmark outputs are ready for dashboard consumption

\---

\## Metric strategy

The project should track two categories of metrics.

\### Quality metrics

\- training loss  
\- validation loss  
\- perplexity  
\- sample text outputs for qualitative inspection

\### Efficiency metrics

\- parameter count  
\- training throughput  
\- inference throughput  
\- forward latency  
\- short generation latency  
\- peak memory if available

The project’s value comes from showing tradeoffs across both categories, not just one.

\---

\## Step 6.1 — \`src/tsl/eval/perplexity.py\`

\### Why this file matters

Perplexity is the simplest quality metric to expose cleanly beyond raw loss.

\### Prompt for coding agent

Implement src/tsl/eval/perplexity.py for Transformer Surgery Lab. Keep it simple, mathematically clear, and easy to use from evaluation and summary-writing code.

\### Validation

Confirm it transforms validation loss into the expected perplexity value.

\---

\## Step 6.2 — \`src/tsl/eval/generation.py\`

\### Why this file matters

Qualitative samples help make the model outputs legible to humans and useful for README and dashboard storytelling.

\### What to implement

Create a small generation utility for:  
\- prompt-based decoding  
\- max token count  
\- deterministic or temperature-controlled generation as needed  
\- writing or returning samples for artifacts

\### Acceptance criteria

\- simple, not overfeatured  
\- enough for qualitative comparisons and run summaries

\### Prompt for coding agent

Implement src/tsl/eval/generation.py for Transformer Surgery Lab. Add a small autoregressive generation utility suitable for qualitative sample outputs during evaluation, without overcomplicating sampling options.

\### Validation

Generate a short sample from a trained or toy model and confirm the interface is practical.

\---

\## Step 6.3 — \`src/tsl/eval/param\_count.py\`

\### Why this file matters

Parameter count is essential context, especially for SwiGLU and GQA discussions.

\### Prompt for coding agent

Implement src/tsl/eval/param\_count.py for Transformer Surgery Lab. It should compute total and trainable parameter counts cleanly and return results in a dashboard-friendly format.

\### Validation

Compare counts across baseline and at least one swapped model.

\---

\## Step 6.4 — \`src/tsl/eval/throughput.py\`

\### Why this file matters

Throughput is one of the fastest ways to make systems tradeoffs visible.

\### Prompt for coding agent

Implement src/tsl/eval/throughput.py for Transformer Surgery Lab. Add a straightforward utility for measuring tokens-per-second throughput in controlled training or inference settings.

\### Validation

Run it on a known batch shape and confirm the reported value is sensible.

\---

\## Step 6.5 — \`src/tsl/eval/latency.py\`

\### Why this file matters

Latency matters for the attention comparison and for system realism more generally.

\### What to implement

Include utilities for:  
\- forward-pass latency  
\- short autoregressive generation latency

\### Acceptance criteria

\- benchmark procedure is explicit  
\- results are repeatable enough for comparison

\### Prompt for coding agent

Implement src/tsl/eval/latency.py for Transformer Surgery Lab. Include clean utilities for forward-pass latency and short autoregressive generation latency benchmarking.

\### Validation

Run repeated timing and confirm the API reports stable aggregated statistics.

\---

\## Step 6.6 — \`src/tsl/eval/memory.py\`

\### Why this file matters

Memory is one of the central selling points of GQA, so this cannot be missing.

\### What to implement

Provide practical helpers for:  
\- CUDA peak memory reporting when available  
\- graceful CPU fallback behavior

\### Acceptance criteria

\- useful on GPU  
\- does not break on CPU-only environments

\### Prompt for coding agent

Implement src/tsl/eval/memory.py for Transformer Surgery Lab. Add practical helpers for reporting peak memory usage during benchmark runs when CUDA is available, with graceful handling on CPU.

\### Validation

Test on CPU first and, if available, on CUDA.

\---

\## Step 6.7 — \`src/tsl/tracking/reader.py\`

\### Why this file matters

The dashboard and comparison scripts depend on a stable read path as much as training depends on a stable write path.

\### What to implement

Create readers that can:  
\- load run metadata  
\- load train/eval metrics  
\- load benchmark summaries  
\- expose normalized structures for downstream tools

\### Acceptance criteria

\- downstream consumers do not need to know artifact file details  
\- read path is symmetric with writer outputs

\### Prompt for coding agent

Implement src/tsl/tracking/reader.py for Transformer Surgery Lab. It should load run directories, parse summary and metrics artifacts, and expose a clean interface for downstream comparison tools and dashboard code.

\### Validation

Load a baseline run directory and inspect the parsed outputs.

\---

\## Step 6.8 — \`scripts/evaluate.py\`

\### Why this file matters

There should be one obvious command for post-training evaluation.

\### What to implement

The script should:  
\- load a run or checkpoint  
\- compute validation metrics  
\- generate optional text samples  
\- write structured outputs back into the run directory or report path

\### Acceptance criteria

\- explicit and script-like, not framework-y  
\- useful after any completed training run

\### Prompt for coding agent

Implement scripts/evaluate.py for Transformer Surgery Lab. It should load a run or checkpoint, run evaluation, compute validation metrics, optionally generate qualitative text samples, and write structured outputs.

\### Validation

Run it on a saved checkpoint from the baseline experiment.

\---

\## Step 6.9 — \`scripts/benchmark.py\`

\### Why this file matters

This script turns architecture choices into measurable systems tradeoffs.

\### What to implement

The script should:  
\- load a run or checkpoint  
\- compute parameter count  
\- benchmark throughput  
\- benchmark latency  
\- benchmark memory where possible  
\- save results in the canonical artifact format

\### Acceptance criteria

\- one command produces dashboard-ready benchmark outputs  
\- output format matches the tracking schema

\### Prompt for coding agent

Implement scripts/benchmark.py for Transformer Surgery Lab. It should load a run or checkpoint, execute standardized benchmark routines for parameter count, throughput, latency, and memory, and save the results cleanly in the existing artifact schema.

\### Validation

Run it on the baseline run and inspect \`benchmark.json\` or equivalent output.

\---

\## Step 6.10 — \`scripts/compare\_runs.py\`

\### Why this file matters

This script creates the compact comparison layer between raw artifacts and the dashboard.

\### What to implement

Create a script that:  
\- reads multiple run directories  
\- aggregates their summary metrics  
\- exports a compact comparison table or JSON/CSV bundle

\### Acceptance criteria

\- enough for quick CLI-side comparison  
\- directly useful to dashboard loading logic

\### Prompt for coding agent

Implement scripts/compare\_runs.py for Transformer Surgery Lab. It should read multiple experiment run directories, aggregate summary metrics, and export a compact comparison table or file suitable for dashboard use.

\### Validation

Compare baseline with at least one single-axis swap.

\---

\## Phase 6 gate checklist

Do not leave Phase 6 until all of this is true:  
\- quality metrics are readable from run artifacts  
\- efficiency metrics are produced in a consistent schema  
\- runs can be evaluated and benchmarked after training  
\- multiple runs can be compared through a compact output layer

\---

\## Phase 7 — Dashboard

\### Goal

Build a polished visual layer that makes the experiment results understandable in under a minute.

\### Phase gate

This phase is done only when:  
\- the dashboard renders real experiment data  
\- comparisons are visually clear  
\- a recruiter or engineer can understand the value quickly

\---

\## Dashboard design principles

The dashboard should optimize for:  
\- fast comprehension  
\- clear metric tradeoffs  
\- concise interpretation text  
\- visual polish without unnecessary complexity

The dashboard is not trying to be a generic MLOps product. It is trying to make \*\*this project’s experiment story\*\* instantly legible.

\---

\## Step 7.1 — \`dashboard/README.md\`

\### Why this file matters

This file makes the dashboard usable without guesswork.

\### Prompt for coding agent

Write dashboard/README.md for Transformer Surgery Lab. Explain the dashboard purpose, expected data inputs, local launch command, and the main views a user should expect.

\### Validation

Check whether someone unfamiliar with the code could launch the dashboard from this file.

\---

\## Step 7.2 — \`dashboard/loaders.py\`

\### Why this file matters

UI code should not have to understand raw artifact layout.

\### What to implement

Create loading helpers that transform run artifacts into structures suitable for the dashboard pages and charts.

\### Acceptance criteria

\- dashboard pages stay simple  
\- loading logic is resilient to missing optional fields

\### Prompt for coding agent

Implement dashboard/loaders.py for Transformer Surgery Lab. It should load experiment artifacts cleanly and return dashboard-friendly structures for summaries, loss curves, and benchmark metrics.

\### Validation

Point it at a few run directories and inspect the returned structures.

\---

\## Step 7.3 — \`dashboard/metrics.py\`

\### Why this file matters

Derived metrics and display logic should not be scattered across UI components.

\### What to implement

Add helpers for:  
\- formatting values for display  
\- computing derived tradeoff values if needed  
\- consistent labels and units

\### Acceptance criteria

\- the UI layer remains presentation-focused  
\- metric formatting is consistent across the app

\### Prompt for coding agent

Implement dashboard/metrics.py for Transformer Surgery Lab. Add helpers for formatting metrics, computing derived display values where useful, and keeping metric business logic out of the UI layer.

\### Validation

Run the formatting helpers on real sample metrics.

\---

\## Step 7.4 — \`dashboard/plots.py\`

\### Why this file matters

This file determines whether the dashboard looks insightful or generic.

\### What to implement

Create chart builders for:  
\- loss curves  
\- bar comparisons  
\- throughput vs memory scatter plots  
\- latency comparisons  
\- parameter-count views

\### Acceptance criteria

\- plots are readable with small run counts  
\- the project’s tradeoff story becomes visually obvious

\### Prompt for coding agent

Implement dashboard/plots.py for Transformer Surgery Lab. Create clean plotting helpers for loss curves, bar charts, scatter plots, and tradeoff visuals used by the Streamlit app.

\### Validation

Generate each plot using real or representative run data.

\---

\## Step 7.5 — \`dashboard/components.py\`

\### Why this file matters

Reusable components help the app feel consistent and polished.

\### What to implement

Build reusable sections such as:  
\- metric summary cards  
\- run comparison tables  
\- highlighted tradeoff callouts  
\- notes or observations panels

\### Acceptance criteria

\- the app layout feels cohesive  
\- important metrics stand out immediately

\### Prompt for coding agent

Implement dashboard/components.py for Transformer Surgery Lab. Build reusable Streamlit-friendly UI components like metric summaries, comparison tables, and highlighted tradeoff callouts.

\### Validation

Render the components with sample data and check for clarity and visual consistency.

\---

\## Step 7.6 — \`dashboard/app.py\`

\### Why this file matters

This is the final public visualization layer and one of the most important files in the whole repo.

\### What to implement

Build a polished Streamlit app that includes at least:  
\- overview section  
\- run selection and comparison  
\- loss curves  
\- benchmark tradeoff visuals  
\- parameter count and memory context  
\- concise interpretation text or notes

\### Acceptance criteria

\- recruiter-friendly at a glance  
\- technically credible on closer inspection  
\- based on real experiment artifacts, not mocked data in the final version

\### Prompt for coding agent

Implement dashboard/app.py for Transformer Surgery Lab. Build a polished Streamlit dashboard that loads real experiment artifacts and presents run comparisons, loss curves, benchmark tradeoffs, parameter counts, and concise interpretation text. It should feel recruiter-friendly and presentation-ready without being overdesigned.

\### Validation

Launch the dashboard, click through the pages, and confirm the value is obvious in under a minute.

\---

\## Phase 7 gate checklist

Do not leave Phase 7 until all of this is true:  
\- dashboard loads real artifact data  
\- key comparisons are visible in a few clicks or less  
\- the best tradeoffs are obvious from the visuals  
\- the app feels polished enough to screen-record for a portfolio or resume link

\---