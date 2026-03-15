# AttnPathQ

This repository contains the code used for the AttnPathQ paper on diagnosis-guided adaptive attention quantization for vision transformers.

## Scope

The release focuses on the AttnPathQ pipeline studied in the paper:
- placement diagnosis for deciding between `QK` and `QKV`
- adaptive same-setting baseline comparisons
- conditional early-layer coupled rescue
- full-evaluation, selector-stability, and multi-seed scripts

The repository is intentionally limited to the paper-aligned code path rather than the broader exploratory workspace.

## Main files

- `common.py`: shared model loading, data loading, quantization, and evaluation utilities
- `analyze_idea2_placement.py`: placement diagnostics and backbone-level analysis
- `run_idea2_adapted_baselines.py`: same-setting baseline comparison and AttnPathQ evaluation
- `run_idea2_early_coupled_search.py`: conditional rescue search
- `run_idea2_early_coupled_multiseed.py`: multi-seed robustness evaluation
- `run_idea2_selector_stability.py`: selector-stability experiments
- `run_idea2_full.py`: full evaluation entry point
- `diagnose_vitb4_repq_gap.py`: hard-case rescue-gap diagnosis
- `fix_imagenetv2_labels.py`: helper script for ImageNet-V2 label handling

## Environment notes

- The original experiments were run in a local Linux/WSL environment with `timm`, PyTorch, and ImageNet-style evaluation assets.
- Local paths inside the scripts may need to be adjusted before reproduction in a new environment.
- Datasets, checkpoints, and generated results are not included in this release directory.

## What is not included

- ImageNet-1K and ImageNet-V2 data
- model checkpoints
- local experiment outputs
- the broader non-paper workspace

## Release status

This directory is prepared as a GitHub-ready code package. A public remote repository has not yet been attached from the current machine.
