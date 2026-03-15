from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch

from common import attention_modules, build_loader, load_model, save_json, seed_everything
from diagnose_vitb4_repq_gap import evaluate_mixed_case
from run_idea2_adapted_baselines import MODEL_SPECS, collect_bundle_with_resources, mode_tensor_params


DEFAULT_OVERRIDES = {
    0: ("k", "v"),
    2: ("q", "k", "v"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed full confirmation for early-layer coupled idea2")
    parser.add_argument("--dataset-root", type=str, default="/home/cxj/DyadicFold/data/imagenet")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/home/cxj/experiments/vit_quant_5ideas/results/idea2_adapted",
    )
    parser.add_argument("--run-name", type=str, default="vitb4_early_coupled_multiseed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="vit_base", choices=sorted(MODEL_SPECS))
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--calib-images", type=int, default=128)
    parser.add_argument("--val-images", type=int, default=50000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument(
        "--overrides",
        type=str,
        default="0:kv,2:qkv",
        help="Comma-separated overrides like '0:kv,2:qkv,4:qkv'",
    )
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


def parse_overrides(spec: str) -> dict[int, tuple[str, ...]]:
    token_map = {
        "q": ("q",),
        "k": ("k",),
        "v": ("v",),
        "qk": ("q", "k"),
        "qv": ("q", "v"),
        "kv": ("k", "v"),
        "qkv": ("q", "k", "v"),
    }
    spec = spec.strip()
    if not spec:
        return {}
    out: dict[int, tuple[str, ...]] = {}
    for chunk in spec.split(","):
        layer_text, token_text = chunk.split(":")
        layer = int(layer_text.strip())
        token_key = token_text.strip().lower()
        if token_key not in token_map:
            raise ValueError(f"Unsupported override token: {token_key}")
        out[layer] = token_map[token_key]
    return out


def summarize_metric(rows: list[dict], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": float(mean), "std": float(var ** 0.5)}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spec = MODEL_SPECS[args.model]
    calib_batch = min(spec["batch_size"], 16)
    override_map = parse_overrides(args.overrides)

    all_results: dict[str, dict[str, dict[str, float | int | None]]] = {}

    for seed in args.seeds:
        seed_everything(seed)
        calib_loader = build_loader(
            dataset_root=args.dataset_root,
            split="train",
            batch_size=calib_batch,
            num_workers=args.num_workers,
            num_images=args.calib_images,
            seed=seed + 17,
            shuffle=False,
            device=device,
        )
        val_loader = build_loader(
            dataset_root=args.dataset_root,
            split="val",
            batch_size=spec["batch_size"],
            num_workers=args.num_workers,
            num_images=args.val_images,
            seed=seed,
            shuffle=False,
            device=device,
        )

        model = load_model(spec["model_name"], spec["checkpoint"], device=device)
        layers = list(range(len(attention_modules(model))))
        bundle, _ = collect_bundle_with_resources(
            model,
            calib_loader,
            device=device,
            layers=layers,
            max_batches=max(1, math.ceil(args.calib_images / calib_batch)),
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        direct_params, _ = mode_tensor_params(bundle, args.bits, "direct_qkv")
        ours_params, _ = mode_tensor_params(bundle, args.bits, "ours_rotated_qk")
        repq_params, _ = mode_tensor_params(bundle, args.bits, "repq_style_qkv")
        selective_params = apply_overrides(ours_params, repq_params, override_map)

        seed_results = {
            "direct_qkv": evaluate_mixed_case(
                spec["model_name"],
                spec["checkpoint"],
                bits=args.bits,
                layer_params=direct_params,
                val_loader=val_loader,
                device=device,
                tag=f"idea2_selective:{args.model}:seed{seed}:direct",
            ),
            "ours_rotated_qk": evaluate_mixed_case(
                spec["model_name"],
                spec["checkpoint"],
                bits=args.bits,
                layer_params=ours_params,
                val_loader=val_loader,
                device=device,
                tag=f"idea2_selective:{args.model}:seed{seed}:ours",
            ),
            "repq_style_qkv": evaluate_mixed_case(
                spec["model_name"],
                spec["checkpoint"],
                bits=args.bits,
                layer_params=repq_params,
                val_loader=val_loader,
                device=device,
                tag=f"idea2_selective:{args.model}:seed{seed}:repq",
            ),
            "selective_early_coupled": evaluate_mixed_case(
                spec["model_name"],
                spec["checkpoint"],
                bits=args.bits,
                layer_params=selective_params,
                val_loader=val_loader,
                device=device,
                tag=f"idea2_selective:{args.model}:seed{seed}:selective",
            ),
        }
        all_results[str(seed)] = seed_results

    summary: dict[str, dict[str, float]] = {}
    for method in ("direct_qkv", "ours_rotated_qk", "repq_style_qkv", "selective_early_coupled"):
        rows = [all_results[str(seed)][method] for seed in args.seeds]
        summary[method] = summarize_metric(rows, "top1")
    deltas = [
        all_results[str(seed)]["selective_early_coupled"]["top1"]
        - all_results[str(seed)]["repq_style_qkv"]["top1"]
        for seed in args.seeds
    ]
    delta_mean = sum(deltas) / len(deltas)
    delta_var = sum((delta - delta_mean) ** 2 for delta in deltas) / len(deltas)
    summary["selective_minus_repq"] = {"mean": float(delta_mean), "std": float(delta_var ** 0.5)}

    out = {
        "setup": {
            "model": args.model,
            "bits": args.bits,
            "calib_images": args.calib_images,
            "val_images": args.val_images,
            "seeds": args.seeds,
            "device": str(device),
            "overrides": {str(k): list(v) for k, v in override_map.items()},
        },
        "results": all_results,
        "summary": summary,
    }
    out_path = Path(args.results_dir) / f"{args.run_name}.json"
    save_json(out_path, out)
    print(f"[Done] multiseed selective confirmation saved to {out_path}")


if __name__ == "__main__":
    main()
