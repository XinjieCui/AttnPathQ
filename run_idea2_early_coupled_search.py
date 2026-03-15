from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch

from common import attention_modules, build_loader, load_model, save_json, seed_everything
from diagnose_vitb4_repq_gap import evaluate_mixed_case
from run_idea2_adapted_baselines import (
    MODEL_SPECS,
    collect_bundle_with_resources,
    evaluate_with_resources,
    mode_tensor_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search early-layer selective coupled configs for idea2")
    parser.add_argument("--dataset-root", type=str, default="/home/cxj/DyadicFold/data/imagenet")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/home/cxj/experiments/vit_quant_5ideas/results/idea2_adapted",
    )
    parser.add_argument("--run-name", type=str, default="vitb4_early_coupled_search")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="vit_base", choices=sorted(MODEL_SPECS))
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--calib-images", type=int, default=128)
    parser.add_argument("--smoke-val-images", type=int, default=5000)
    parser.add_argument("--full-val-images", type=int, default=50000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--topk-full", type=int, default=3)
    return parser.parse_args()


def clone_layer_params(layer_params: dict[int, dict[str, dict]]) -> dict[int, dict[str, dict]]:
    return copy.deepcopy(layer_params)


def apply_overrides(
    base_params: dict[int, dict[str, dict]],
    ref_params: dict[int, dict[str, dict]],
    overrides: dict[int, tuple[str, ...]],
) -> dict[int, dict[str, dict]]:
    mixed = clone_layer_params(base_params)
    for layer, tensor_names in overrides.items():
        for tensor_name in tensor_names:
            mixed[layer][tensor_name] = copy.deepcopy(ref_params[layer][tensor_name])
    return mixed


def candidate_specs() -> list[tuple[str, dict[int, tuple[str, ...]]]]:
    return [
        ("ours_rotated_qk", {}),
        ("repq_style_qkv", {layer: ("q", "k", "v") for layer in range(12)}),
        ("early_l0_kv", {0: ("k", "v")}),
        ("early_l0_qkv", {0: ("q", "k", "v")}),
        ("early_l2_qkv", {2: ("q", "k", "v")}),
        ("early_l4_qkv", {4: ("q", "k", "v")}),
        ("early_l0_kv_l2_qkv", {0: ("k", "v"), 2: ("q", "k", "v")}),
        ("early_l0_kv_l4_qkv", {0: ("k", "v"), 4: ("q", "k", "v")}),
        ("early_l2_qkv_l4_qkv", {2: ("q", "k", "v"), 4: ("q", "k", "v")}),
        ("early_l0_kv_l2_qkv_l4_qkv", {0: ("k", "v"), 2: ("q", "k", "v"), 4: ("q", "k", "v")}),
        ("early_l0_qkv_l2_qkv_l4_qkv", {0: ("q", "k", "v"), 2: ("q", "k", "v"), 4: ("q", "k", "v")}),
    ]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spec = MODEL_SPECS[args.model]

    calib_batch = min(spec["batch_size"], 16)
    calib_loader = build_loader(
        dataset_root=args.dataset_root,
        split="train",
        batch_size=calib_batch,
        num_workers=args.num_workers,
        num_images=args.calib_images,
        seed=args.seed + 17,
        shuffle=False,
        device=device,
    )
    smoke_loader = build_loader(
        dataset_root=args.dataset_root,
        split="val",
        batch_size=spec["batch_size"],
        num_workers=args.num_workers,
        num_images=args.smoke_val_images,
        seed=args.seed,
        shuffle=False,
        device=device,
    )
    full_loader = build_loader(
        dataset_root=args.dataset_root,
        split="val",
        batch_size=spec["batch_size"],
        num_workers=args.num_workers,
        num_images=args.full_val_images,
        seed=args.seed,
        shuffle=False,
        device=device,
    )

    model = load_model(spec["model_name"], spec["checkpoint"], device=device)
    layers = list(range(len(attention_modules(model))))
    bundle, calib_stats = collect_bundle_with_resources(
        model,
        calib_loader,
        device=device,
        layers=layers,
        max_batches=max(1, math.ceil(args.calib_images / calib_batch)),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fp_model = load_model(spec["model_name"], spec["checkpoint"], device=device)
    fp_smoke = evaluate_with_resources(fp_model, smoke_loader, device, f"idea2_selective:{args.model}:fp_smoke")
    del fp_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ours_params, _ = mode_tensor_params(bundle, args.bits, "ours_rotated_qk")
    repq_params, _ = mode_tensor_params(bundle, args.bits, "repq_style_qkv")
    direct_params, _ = mode_tensor_params(bundle, args.bits, "direct_qkv")

    smoke_results: dict[str, dict[str, float | int | None | dict[int, tuple[str, ...]]]] = {}
    baseline_sets = {
        "direct_qkv": direct_params,
    }
    for name, layer_params in baseline_sets.items():
        smoke_results[name] = evaluate_mixed_case(
            spec["model_name"],
            spec["checkpoint"],
            bits=args.bits,
            layer_params=layer_params,
            val_loader=smoke_loader,
            device=device,
            tag=f"idea2_selective:{args.model}:{name}:smoke",
        )

    for name, overrides in candidate_specs():
        if name == "ours_rotated_qk":
            layer_params = ours_params
        elif name == "repq_style_qkv":
            layer_params = repq_params
        else:
            layer_params = apply_overrides(ours_params, repq_params, overrides)
        stats = evaluate_mixed_case(
            spec["model_name"],
            spec["checkpoint"],
            bits=args.bits,
            layer_params=layer_params,
            val_loader=smoke_loader,
            device=device,
            tag=f"idea2_selective:{args.model}:{name}:smoke",
        )
        stats["overrides"] = {str(k): list(v) for k, v in overrides.items()}
        smoke_results[name] = stats

    ranked = sorted(
        ((name, row["top1"]) for name, row in smoke_results.items() if name != "direct_qkv"),
        key=lambda item: item[1],
        reverse=True,
    )
    full_candidates = [name for name, _ in ranked[: max(1, args.topk_full)]]
    if "direct_qkv" not in full_candidates:
        full_candidates.append("direct_qkv")
    if "repq_style_qkv" not in full_candidates:
        full_candidates.append("repq_style_qkv")
    if "ours_rotated_qk" not in full_candidates:
        full_candidates.append("ours_rotated_qk")

    full_results: dict[str, dict[str, float | int | None | dict[int, tuple[str, ...]]]] = {}
    fp_model = load_model(spec["model_name"], spec["checkpoint"], device=device)
    fp_full = evaluate_with_resources(fp_model, full_loader, device, f"idea2_selective:{args.model}:fp_full")
    del fp_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for name in full_candidates:
        if name == "direct_qkv":
            layer_params = direct_params
            overrides: dict[int, tuple[str, ...]] = {}
        elif name == "ours_rotated_qk":
            layer_params = ours_params
            overrides = {}
        elif name == "repq_style_qkv":
            layer_params = repq_params
            overrides = {layer: ("q", "k", "v") for layer in range(12)}
        else:
            overrides = dict(next(spec_item[1] for spec_item in candidate_specs() if spec_item[0] == name))
            layer_params = apply_overrides(ours_params, repq_params, overrides)
        stats = evaluate_mixed_case(
            spec["model_name"],
            spec["checkpoint"],
            bits=args.bits,
            layer_params=layer_params,
            val_loader=full_loader,
            device=device,
            tag=f"idea2_selective:{args.model}:{name}:full",
        )
        stats["overrides"] = {str(k): list(v) for k, v in overrides.items()}
        full_results[name] = stats

    out = {
        "setup": {
            "model": args.model,
            "bits": args.bits,
            "seed": args.seed,
            "calib_images": args.calib_images,
            "smoke_val_images": args.smoke_val_images,
            "full_val_images": args.full_val_images,
            "device": str(device),
            "topk_full": args.topk_full,
        },
        "calibration": calib_stats,
        "fp_smoke": fp_smoke,
        "fp_full": fp_full,
        "smoke_ranked": [{"name": name, "top1": top1} for name, top1 in ranked],
        "smoke_results": smoke_results,
        "full_results": full_results,
    }
    out_path = Path(args.results_dir) / f"{args.run_name}.json"
    save_json(out_path, out)
    print(f"[Done] selective search saved to {out_path}")


if __name__ == "__main__":
    main()
