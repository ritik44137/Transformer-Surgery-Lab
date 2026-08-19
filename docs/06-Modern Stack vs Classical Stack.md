## Purpose

This document adds one final synthesis experiment to the existing Transformer Surgery Lab roadmap. Its role is to come **after all single-axis runs** so the project ends with one clear, integrated comparison rather than a disconnected set of swap-by-swap results.

The experiment is designed to answer the practical question the rest of the plan naturally leads toward: what happens when the baseline **classical stack** is compared directly against the fully combined **modern stack**?

## Experiment 11 — Modern Stack vs Classical Stack

### Goal

Run one final end-to-end comparison between:

- **Classical stack:** LayerNorm + Sinusoidal + ReLU + MHA
- **Modern stack:** RMSNorm + RoPE + SwiGLU + GQA

This experiment should not replace the single-axis comparisons. It should **synthesize** them. The single-axis runs explain causality one dimension at a time; this final run shows what the combined modernized design looks like as a system.

## Why this experiment matters

The current plan already establishes four controlled single-axis swap runs, then adds evaluation, benchmarking, comparison exports, dashboarding, and evidence-based lesson notes. That structure is strong for measurement, but it still benefits from one concluding experiment that turns the repo into a coherent story.

Without this synthesis step, the project risks ending with “four isolated swap results.” With it, the repo can close with a more useful engineering conclusion: whether the modern bundle of changes is materially better than the classical baseline, and whether the extra implementation complexity was justified.

## Placement in the roadmap

This should be scheduled **after the single-axis swap matrix is complete** and after the evaluation and benchmarking pipeline is already working.

Operationally, this means:

- it belongs **after the primary single-axis runs in Phase 5**
- it depends on the **Phase 6 evaluation and benchmark tooling** already producing consistent metrics
- it should feed directly into the **Phase 9 evidence-based documentation/storytelling layer**

## What to implement

### 1. Add a dedicated modern-stack experiment config

Create one explicit experiment config for the fully combined modern stack. It should enable RMSNorm, RoPE, SwiGLU, and GQA together while preserving the same training pipeline and artifact contract used by the baseline and single-axis runs.

Recommended filename:

- `configs/experiments/rmsnorm_rope_swiglu_gqa.yaml`

### 2. Treat the classical stack as the reference run

The classical stack is already the project baseline configuration in substance: LayerNorm + Sinusoidal + ReLU + MHA. Reuse that run as the classical reference rather than introducing a duplicate config unless naming clarity requires an alias.

### 3. Extend the official experiment manifest

Update the experiment manifest so the final synthesis run is explicitly represented as a concluding comparison, not hidden among optional combinations.

The manifest should make three layers obvious:

- baseline classical run
- four single-axis swap runs
- one final synthesis run: modern stack vs classical stack

### 4. Ensure the comparison layer surfaces the right metrics

For this final comparison, the project should explicitly compare:

- **validation loss**
- **throughput (tokens/sec)**
- **memory usage**
- **parameter count**
- **training stability**

This is consistent with the existing plan’s quality-and-efficiency framing, while adding a stronger emphasis on training stability for the final narrative.

### 5. Define training stability concretely

The comparison output should not leave “training stability” vague. Use a small, explicit summary based on already available run signals, such as:

- whether loss curves are smooth or noisy
- whether optimization shows spikes, divergence, or instability
- whether runs are consistently trainable under the same setup

The exact implementation can be lightweight, but the final comparison must make the stability judgment explainable rather than subjective.

### 6. Add a final written synthesis note

Create one short evidence-based discussion document that answers:

> Which changes produced the biggest gains, and which weren't worth the added complexity?

Recommended filename:

- `experiments/lessons/modern_vs_classical.md`

This note should synthesize the single-axis results and the final combined-stack result. It should read like an engineering conclusion, not marketing copy.

## Acceptance criteria

This feature is complete only when all of the following are true:

- the modern-stack experiment is runnable from a dedicated config
- the classical stack remains the clean reference point
- the experiment manifest shows the final synthesis comparison clearly
- comparison outputs include validation loss, throughput, memory, parameter count, and training stability
- the final discussion note answers the “biggest gains vs added complexity” question using actual run outputs
- the project story now reads as one coherent arc from baseline, to isolated swaps, to final synthesis

## Prompt for coding agent

Add one final synthesis experiment to Transformer Surgery Lab after all single-axis runs. Create a dedicated experiment config for the combined modern stack using RMSNorm, RoPE, SwiGLU, and GQA, and compare it directly against the classical baseline stack of LayerNorm, sinusoidal encoding, ReLU, and MHA. Make sure the comparison explicitly reports validation loss, throughput in tokens per second, memory usage, parameter count, and training stability. Update the experiment manifest so this appears as the concluding synthesis comparison, and add a short evidence-based discussion note answering: “Which changes produced the biggest gains, and which weren't worth the added complexity?” Keep the implementation aligned with the existing artifact schema, evaluation flow, benchmark tooling, and documentation style.

## Validation

Review the final experiment set cold and confirm it tells a complete story:

1. baseline classical stack
2. four controlled single-axis swaps
3. one final modern-vs-classical synthesis comparison
4. one short evidence-based conclusion about gains versus complexity

If that sequence is obvious to an outsider, this addition is doing its job.